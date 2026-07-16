"""llm-gsm8k: Llama-3-8B federated fine-tuning Flower ServerApp (train GSM8K's own training set,
N=3/IID, eval GSM8K exact-match only). A separate experiment from apps/llm-humaneval's
CodeSearchNet/HumanEval pipeline -- see that app's docstrings for why these were previously
(incorrectly) combined into one CodeSearchNet-trained-then-evaluated-on-both run."""

import json
import os
import subprocess
import sys
from datetime import datetime

from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from omegaconf import DictConfig
import numpy as np
import torch

from flowertune_llm.dataset import replace_keys
from flowertune_llm.models import get_model, get_parameters, set_parameters
from flowertune_llm.strategies import build_strategy

from fedbench_common.metrics import InstrumentedStrategy, gpu_info
from fedbench_common.resultio import MetricsWriter, write_run_metadata


def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path, metrics_writer, run_eval):
    """Checkpoint saving, plus (only at checkpoint-save rounds) invoking GSM8K exact-match
    evaluation as parallel per-GPU subprocesses against the freshly saved checkpoint -- an
    expensive generation-based eval over the full 1319-example test set, infeasible to run every
    round. With total_round=50 and save_every_round=10, checkpoints land at rounds 10, 20, 30, 40,
    and 50 (final round always saved via the `server_round == total_round` branch below)."""

    def evaluate(server_round: int, parameters, config):
        if server_round == 0 or (
            server_round != total_round and server_round % save_every_round != 0
        ):
            return 0.0, {}

        model = get_model(model_cfg)
        set_parameters(model, parameters)
        peft_path = f"{save_path}/peft_{server_round}"
        model.save_pretrained(peft_path)

        if run_eval:
            eval_dir = os.path.dirname(os.path.abspath(__file__))
            # Explicit --data-root rather than relying on eval_gsm8k.py's own
            # os.environ.get("FEDORA_DATA_ROOT", "data") fallback -- see the identical note in
            # llm_humaneval/server_app.py for why (confirmed cwd-relative stray-file bug).
            data_root = os.environ.get(
                "FEDORA_DATA_ROOT", os.path.join(os.path.dirname(save_path), "data")
            )
            # Split the 1319-example test set across every visible physical GPU -- one
            # subprocess per GPU, each scoring an interleaved shard in parallel -- instead of
            # running the full set on a single GPU while every other GPU sits idle. Roughly
            # halves eval wall-clock on this 2-GPU box (confirmed ~9.6h unsharded).
            num_shards = torch.cuda.device_count() if torch.cuda.is_available() else 1
            num_shards = max(num_shards, 1)
            procs = []
            shard_outs = []
            for shard_index in range(num_shards):
                device = f"cuda:{shard_index}" if torch.cuda.is_available() else "cpu"
                shard_out = os.path.join(save_path, f"gsm8k_shard_{server_round}_{shard_index}.json")
                shard_outs.append(shard_out)
                procs.append(subprocess.Popen(
                    [
                        sys.executable, os.path.join(eval_dir, "eval_gsm8k.py"),
                        "--base-model", model_cfg.name, "--peft-path", peft_path,
                        "--data-root", data_root, "--device", device,
                        "--num-shards", str(num_shards), "--shard-index", str(shard_index),
                        "--shard-out", shard_out,
                    ],
                ))
            for proc in procs:
                proc.wait()

            correct, total = 0, 0
            for shard_out in shard_outs:
                if os.path.exists(shard_out):
                    with open(shard_out) as f:
                        result = json.load(f)
                    correct += result["correct"]
                    total += result["total"]
                    os.remove(shard_out)
            if total > 0:
                metrics_writer.write_round(server_round, "gsm8k_eval", gsm8k_exact_match=correct / total)

        return 0.0, {}

    return evaluate


def get_on_fit_config(save_path):
    def fit_config_fn(server_round: int):
        return {"current_round": server_round, "save_path": save_path}

    return fit_config_fn


def fit_weighted_average(metrics):
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
    aggregation = cfg.strategy.get("aggregation", "fedrot")

    metrics_writer = MetricsWriter(path=os.path.join(save_path, "metrics.jsonl"))
    write_run_metadata(
        save_path,
        {
            **gpu_info(),
            "seed": cfg.get("seed", 0),
            "task": "gsm8k",
            "strategy": aggregation,
            "lr": cfg.train.learning_rate_max,
            "start_time": current_time.isoformat(),
        },
    )

    # `cfg.seed` would otherwise only seed the data partitioner -- get_model()'s
    # get_peft_model(...) randomly initializes LoRA A (B is zero-init'd by PEFT) from whatever
    # global torch RNG state happens to exist at process start, which otherwise differs every run
    # even with an identical `seed`. Only the server-side init matters: every client immediately
    # overwrites its own local model via set_parameters() from these same broadcast
    # initial_parameters, so client-side construction is never actually observed. Matches the
    # identical fix in apps/glue-nlu/src/glue_nlu/server_app.py.
    torch.manual_seed(cfg.get("seed", 0))
    init_model = get_model(cfg.model)
    init_model_parameters = ndarrays_to_parameters(get_parameters(init_model))

    strategy_kwargs = dict(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path,
            metrics_writer, cfg.train.get("run_generation_eval", True),
        ),
    )
    strategy = build_strategy(
        aggregation, model=init_model, cfg=cfg.strategy.get(aggregation, {}), **strategy_kwargs
    )
    strategy = InstrumentedStrategy(strategy, metrics_writer=metrics_writer)

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
