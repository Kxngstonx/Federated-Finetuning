"""glue-nlu: GLUE / RoBERTa-Large NLU federated fine-tuning Flower ServerApp."""

import os
from datetime import datetime

import numpy as np
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments

from flowertune_llm.dataset import replace_keys
from flowertune_llm.models import get_parameters, set_parameters
from flowertune_llm.strategies import build_strategy

from fedbench_common.metrics import InstrumentedStrategy, gpu_info
from fedbench_common.resultio import MetricsWriter, write_run_metadata

from glue_nlu.dataset import get_tokenizer, load_centralized_validation
from glue_nlu.models import get_model


def get_evaluate_fn(
    model_cfg, task_name, tokenizer, seq_length, save_path, metrics_writer, eval_freq,
    save_every_round, total_round,
):
    """Centralized eval on the full validation(_matched) split -- FedRot-LoRA's own
    federate.make_global_eval=True protocol -- computing accuracy every `eval_freq` rounds (the
    per-round learning-curve requirement, unaffected by the below), with adapter-checkpoint
    saving (model.save_pretrained) decoupled onto its own, coarser `save_every_round` cadence (+
    always at the final round) -- accuracy computation is cheap (a forward pass over the
    validation set), but writing a checkpoint to disk every one of 250 rounds across a
    480-run sweep (5 tasks x 8 strategies x 4 lrs x 3 seeds) adds up fast, so only a handful of
    checkpoints per run are actually persisted."""
    eval_dataset = load_centralized_validation(task_name, tokenizer, seq_length)
    eval_dataset = eval_dataset.remove_columns(
        [c for c in eval_dataset.column_names if c not in ("input_ids", "attention_mask", "label")]
    )
    eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset.set_format(type="torch")

    def evaluate(server_round: int, parameters, config):
        if server_round == 0 or server_round % eval_freq != 0:
            return 0.0, {}

        model = get_model(model_cfg, task_name)
        set_parameters(model, parameters)

        eval_args = TrainingArguments(
            output_dir=save_path, per_device_eval_batch_size=64, report_to=[]
        )
        trainer = Trainer(model=model, args=eval_args)
        predictions = trainer.predict(eval_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        acc = accuracy_score(eval_dataset["labels"], preds)

        metrics_writer.write_round(server_round, "eval", accuracy=acc)

        if server_round == total_round or server_round % save_every_round == 0:
            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return 0.0, {"accuracy": acc}

    return evaluate


def get_on_fit_config(save_path):
    def fit_config_fn(server_round: int):
        return {"current_round": server_round, "save_path": save_path}

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate reported fit() metrics: weighted mean of train_loss, and pass through the
    (already scalar, round-level) instrumentation fields unweighted-averaged since they're
    per-client-constant-ish quantities (trainable_params) or already round totals (upload_bytes
    is summed server-side in InstrumentedStrategy, not here)."""
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    out = {"train_loss": sum(losses) / sum(examples)}

    trainable = [m["trainable_params"] for _, m in metrics if "trainable_params" in m]
    if trainable:
        out["trainable_params"] = trainable[0]
    train_seconds = [m["client_train_seconds"] for _, m in metrics if "client_train_seconds" in m]
    if train_seconds:
        out["client_train_seconds_mean"] = float(np.mean(train_seconds))
        out["client_train_seconds_max"] = float(np.max(train_seconds))
    flops = [m["est_flops"] for _, m in metrics if "est_flops" in m]
    if flops:
        out["est_flops_total"] = int(sum(flops))
    return out


def server_fn(context: Context):
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    task_name = cfg.dataset.task_name
    aggregation = cfg.strategy.get("aggregation", "fedrot")

    metrics_writer = MetricsWriter(path=os.path.join(save_path, "metrics.jsonl"))
    write_run_metadata(
        save_path,
        {
            **gpu_info(),
            "seed": cfg.get("seed", 0),
            "task": task_name,
            "strategy": aggregation,
            "lr": cfg.train.learning_rate_max,
            # federate.client-num itself isn't read anywhere else in this file (the actual client
            # count comes from the federation's options.num-supernodes, invisible to server_fn's
            # Context) -- logged here purely so aggregate_glue_results.py can distinguish e.g. the
            # main N=3 sweep from a separate N=50 scalability run without them silently averaging
            # together. Keep --run-config federate.client-num in sync with whatever
            # --federation-config options.num-supernodes value is actually passed at invocation.
            "client_num": cfg.federate.client_num,
            "start_time": current_time.isoformat(),
        },
    )

    tokenizer = get_tokenizer(cfg.model.name)
    init_model = get_model(cfg.model, task_name, aggregation)
    init_model_parameters = ndarrays_to_parameters(get_parameters(init_model))

    strategy_kwargs = dict(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        evaluate_fn=get_evaluate_fn(
            cfg.model, task_name, tokenizer, cfg.train.seq_length, save_path,
            metrics_writer, cfg.get("eval", {}).get("freq", 1),
            cfg.train.get("save_every_round", 25), num_rounds,
        ),
    )
    strategy = build_strategy(
        aggregation, model=init_model, cfg=cfg.strategy.get(aggregation, {}), **strategy_kwargs
    )
    strategy = InstrumentedStrategy(strategy, metrics_writer=metrics_writer)

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
