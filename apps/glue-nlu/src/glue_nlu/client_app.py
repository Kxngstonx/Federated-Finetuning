"""glue-nlu: GLUE / RoBERTa-Large NLU federated fine-tuning Flower ClientApp."""

import os
import warnings
from typing import Dict, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from flowertune_llm.dataset import replace_keys
from flowertune_llm.models import get_parameters, set_parameters

from fedbench_common.client_helpers import (
    FREEZE_A_STRATEGIES,
    FROZEN_PARAM_SUBSTRINGS,
    build_fit_metrics,
    capture_fedrot_reference,
    freeze_params,
    get_cached_model,
    maybe_rotate_fedrot,
)
from fedbench_common.metrics import Timer

from glue_nlu.dataset import get_tokenizer, load_data
from glue_nlu.models import get_model

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(NumPyClient):
    """GLUE sequence-classification Flower client. Structurally mirrors
    flowertune_llm/client_app.py::FlowerClient (same rotation/freeze/instrumentation hooks, now
    factored into fedbench_common.client_helpers), but trains with a plain HF Trainer instead of
    trl.SFTTrainer, since GLUE is sequence classification, not causal-LM SFT."""

    def __init__(self, model_cfg, train_cfg, strategy_cfg, task_name, trainset, num_rounds):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.strategy_cfg = strategy_cfg
        self.aggregation = strategy_cfg.get("aggregation", "fedrot")
        self.task_name = task_name
        self.trainset = trainset
        self.num_rounds = num_rounds

        # Reuse an already-loaded model within this actor process across rounds -- see
        # fedbench_common.client_helpers.get_cached_model's docstring. RoBERTa-Large reload is
        # far cheaper than Llama-3-8B's, but the same repeated-construction overhead still applies
        # every round across a 250-round x 480-run sweep, so cache here too for consistency.
        cache_key = (
            "glue_nlu",
            model_cfg.name,
            task_name,
            model_cfg.lora.peft_lora_r,
            model_cfg.lora.peft_lora_alpha,
            model_cfg.lora.peft_lora_dropout,
            model_cfg.get("use_dora", False),
            self.aggregation,
            str(self.device),
        )

        def _build_model():
            model = get_model(model_cfg, task_name, self.aggregation)
            model.to(self.device)
            if self.aggregation in FREEZE_A_STRATEGIES:
                freeze_params(model, FROZEN_PARAM_SUBSTRINGS)
            # PEFT's PeftModelForSequenceClassification unconditionally adds the classifier head
            # to modules_to_save (peft/peft_model.py) for TaskType.SEQ_CLS, keeping it trainable
            # (full fine-tuning, not LoRA) alongside the adapters -- this matches FedRot-LoRA's
            # own protocol, which does NOT freeze the classifier head: every client uploads it
            # each round via get_peft_model_state_dict (any requires_grad=True param, LoRA or
            # not) and it's aggregated same as the LoRA arrays (see fedora.py's plain-FedAvg
            # fallback for the non-LoRA arrays get_peft_model_state_dict returns).
            return model

        self.model = get_cached_model(cache_key, _build_model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        current_round = int(config["current_round"])

        fedrot_ref_arrays = capture_fedrot_reference(self.model, self.aggregation, parameters)

        set_parameters(self.model, parameters)
        if self.aggregation in FREEZE_A_STRATEGIES:
            freeze_params(self.model, FROZEN_PARAM_SUBSTRINGS)

        # Constant LR (no scheduler), matching FedRot-LoRA's own protocol -- see the identical
        # note in llm_humaneval/client_app.py::fit for the upstream-repo confirmation
        # (core/auxiliaries/scheduler_builder.py::get_scheduler returns None for the default
        # cfg.train.scheduler.type=='', which none of FedRot-LoRA's yamls override).
        #
        # optim="sgd" / max_grad_norm=0: FedRot-LoRA's cfg.train.optimizer.type defaults to
        # 'SGD' (core/configs/cfg_training.py) and is never overridden by any GLUE/LLM yaml --
        # the yamls only ever override optimizer.lr, with the `optimizer:` block itself commented
        # out in the base yamls (core/auxiliaries/optimizer_builder.py::get_optimizer just does
        # getattr(torch.optim, 'SGD')(params, lr) with no momentum/weight_decay kwargs). HF
        # TrainingArguments defaults to AdamW, which at these same nominal lr values (grid up to
        # 2e-2) is a much more aggressive per-parameter-normalized step and diverges where SGD's
        # gradient-magnitude-scaled step tolerates it -- confirmed as the root cause of the NLG
        # pipeline's GSM8K collapse (see llm_humaneval/client_app.py's identical note). grad.grad_clip
        # also defaults to -1 (disabled) upstream and is never overridden, unlike HF's own
        # max_grad_norm=1.0 default.
        # learning_rate/output_dir are "" placeholders in self.train_cfg.training_arguments (see
        # apps/glue-nlu/pyproject.toml) meant to be filled in per-round/per-client here -- passed
        # as constructor kwargs alongside **training_arguments this collides (TypeError: multiple
        # values for keyword argument), so set them as attributes after construction instead,
        # matching flowertune_llm/client_app.py's identical pattern.
        training_args = TrainingArguments(
            **self.train_cfg.training_arguments,
            optim="sgd",
            max_grad_norm=0,
        )
        training_args.learning_rate = self.train_cfg.learning_rate_max
        training_args.output_dir = config["save_path"]

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.trainset,
        )

        with Timer() as train_timer:
            results = trainer.train()

        trained = get_parameters(self.model)

        fit_metrics: Dict[str, Scalar] = {"train_loss": results.training_loss}
        if self.aggregation == "fedrot":
            trained, prerot_metrics = maybe_rotate_fedrot(
                self.model, self.strategy_cfg, trained, fedrot_ref_arrays, current_round
            )
            fit_metrics.update(prerot_metrics)

        fit_metrics.update(
            build_fit_metrics(
                self.model,
                trained,
                train_timer,
                seq_length=self.train_cfg.seq_length,
                batch_size=self.train_cfg.training_arguments.per_device_train_batch_size,
                local_steps=self.train_cfg.training_arguments.max_steps,
            )
        )

        return trained, len(self.trainset), fit_metrics


def client_fn(context: Context) -> FlowerClient:
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    task_name = cfg.dataset.task_name
    tokenizer = get_tokenizer(cfg.model.name)
    trainset = load_data(
        partition_id,
        num_partitions,
        task_name,
        tokenizer,
        cfg.train.seq_length,
        cfg.dataset.dirichlet_alpha,
        cfg.get("seed", 0),
    )
    trainset = trainset.remove_columns(
        [c for c in trainset.column_names if c not in ("input_ids", "attention_mask", "label")]
    )
    trainset = trainset.rename_column("label", "labels")
    trainset.set_format(type="torch")

    return FlowerClient(cfg.model, cfg.train, cfg.strategy, task_name, trainset, num_rounds).to_client()


app = ClientApp(client_fn)
