"""llm-humaneval: Llama-3-8B federated fine-tuning Flower ServerApp (train CodeSearchNet, eval
HumanEval pass@1 only -- GSM8K is trained/evaluated as a separate experiment, see apps/llm-gsm8k,
since the paper's GSM8K row uses its own N=3/IID training run rather than this CSN checkpoint)."""

import os
import shutil
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
    """Checkpoint saving, plus (only at checkpoint-save rounds, per the confirmed NLG eval-cadence
    decision) invoking HumanEval pass@1 evaluation as parallel per-GPU generation subprocesses
    against the freshly saved checkpoint -- an expensive generation-based eval, infeasible to run
    every one of 100 rounds, so this produces a coarser (5-point: rounds 20, 40, 60, 80, 100)
    learning curve than the GLUE pipeline's per-round curve.

    GSM8K is intentionally NOT evaluated here: cross-checking the official FedRot-LoRA repo's
    yamls confirmed GSM8K is a separate experiment in the paper (its own N=3/IID training run on
    GSM8K's own training data, see apps/llm-gsm8k), not this CodeSearchNet checkpoint evaluated a
    second way -- evaluating GSM8K off a model that never trained on GSM8K-style math reasoning
    was a bug in the original port (see apps/llm-gsm8k's docstrings for the fix)."""

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
            metrics_out = os.path.join(save_path, "metrics.jsonl")
            # Pass --data-root/--samples-out explicitly rather than letting eval_humaneval.py fall
            # back to its own os.environ.get("FEDORA_DATA_ROOT", "data") default: if whatever
            # shell launched `flwr run` never sourced experiments/env.sh, that fallback silently
            # resolves "data" relative to this process's cwd (the repo root), scattering
            # HumanEval.jsonl.gz/humaneval_samples.jsonl there instead of the intended NVMe-backed
            # FEDORA_DATA_ROOT -- confirmed this already happened once (untracked stray artifacts
            # at the repo root).
            data_root = os.environ.get(
                "FEDORA_DATA_ROOT", os.path.join(os.path.dirname(save_path), "data")
            )
            samples_out = os.path.join(save_path, f"humaneval_samples_{server_round}.jsonl")

            # Split the 164 HumanEval problems across every visible physical GPU -- one
            # generation subprocess per GPU, each writing its own shard's samples file in
            # parallel -- instead of running the full set on a single GPU while every other GPU
            # sits idle (mirrors apps/llm-gsm8k's server_app.py sharding).
            num_shards = torch.cuda.device_count() if torch.cuda.is_available() else 1
            num_shards = max(num_shards, 1)
            procs = []
            shard_outs = []
            for shard_index in range(num_shards):
                device = f"cuda:{shard_index}" if torch.cuda.is_available() else "cpu"
                shard_out = os.path.join(
                    save_path, f"humaneval_samples_{server_round}_shard{shard_index}.jsonl"
                )
                shard_outs.append(shard_out)
                procs.append(subprocess.Popen(
                    [
                        sys.executable, os.path.join(eval_dir, "eval_humaneval.py"),
                        "--base-model", model_cfg.name, "--peft-path", peft_path,
                        "--data-root", data_root, "--samples-out", shard_out,
                        "--device", device,
                        "--num-shards", str(num_shards), "--shard-index", str(shard_index),
                    ],
                ))
            for proc in procs:
                proc.wait()

            # Merge every shard's generated samples into the single file score_pass_at_1 expects,
            # then score once (code-execution scoring stays in its own subprocess -- see
            # eval_humaneval.py's SECURITY NOTE -- via --score-only, which skips generation).
            with open(samples_out, "wb") as out_f:
                for shard_out in shard_outs:
                    if os.path.exists(shard_out):
                        with open(shard_out, "rb") as in_f:
                            shutil.copyfileobj(in_f, out_f)
                        os.remove(shard_out)
            subprocess.run(
                [
                    sys.executable, os.path.join(eval_dir, "eval_humaneval.py"),
                    "--peft-path", peft_path, "--data-root", data_root,
                    "--samples-out", samples_out, "--score-only",
                    "--round", str(server_round), "--metrics-out", metrics_out,
                ],
                check=False,
            )

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
            "task": "code_search_net",
            "strategy": aggregation,
            "lr": cfg.train.learning_rate_max,
            "start_time": current_time.isoformat(),
        },
    )

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
