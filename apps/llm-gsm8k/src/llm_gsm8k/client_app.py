"""llm-gsm8k: Llama-3-8B federated fine-tuning Flower ClientApp (trains on GSM8K's own training
set, N=3/IID, evaluated on GSM8K exact-match -- see server_app.py). A separate experiment from
apps/llm-humaneval's CodeSearchNet/HumanEval pipeline; see that app's docstrings for why GSM8K was
previously (incorrectly) evaluated off the CodeSearchNet checkpoint instead of its own training
run."""

import os
import warnings
from typing import Dict, Tuple

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from flowertune_llm.dataset import replace_keys
from flowertune_llm.models import get_model, get_parameters, set_parameters

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

from llm_gsm8k.dataset import formatting_prompts_func, get_tokenizer_and_data_collator, load_data

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(NumPyClient):
    """Structurally identical to llm_humaneval/client_app.py::FlowerClient (same rotation/freeze/
    instrumentation hooks, SGD/no-clip optimizer fix, model caching) -- only the dataset/prompt
    formatting and client count/partitioning differ (GSM8K IID N=3 vs CodeSearchNet non-IID N=6)."""

    def __init__(
        self, model_cfg, train_cfg, strategy_cfg, trainset, tokenizer, data_collator, num_rounds
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.strategy_cfg = strategy_cfg
        self.aggregation = strategy_cfg.get("aggregation", "fedrot")
        self.training_arguments = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset

        # Reuse an already-loaded/quantized model within this actor process across rounds --
        # see fedbench_common.client_helpers.get_cached_model's docstring. Cache key covers every
        # field that determines the model's architecture/weights-on-disk; only the LoRA adapter
        # parameters change round to round, and those get overwritten fresh in fit() regardless.
        cache_key = (
            "llm_gsm8k",
            model_cfg.name,
            model_cfg.quantization,
            model_cfg.gradient_checkpointing,
            model_cfg.lora.peft_lora_r,
            model_cfg.lora.peft_lora_alpha,
            model_cfg.get("use_dora", False),
            self.aggregation,
            str(self.device),
        )

        def _build_model():
            model = get_model(model_cfg)
            model.to(self.device)
            if self.aggregation in FREEZE_A_STRATEGIES:
                freeze_params(model, FROZEN_PARAM_SUBSTRINGS)
            return model

        self.model = get_cached_model(cache_key, _build_model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        current_round = int(config["current_round"])

        if self.aggregation == "flora":
            from flowertune_llm.peft_layers import index_lora_layers

            layers = index_lora_layers(self.model)
            n_layers = len(layers)
            lora_params = parameters[: len(parameters) - n_layers]
            extra = parameters[len(parameters) - n_layers:]
            base_layer_updates = {layer.name: extra[i] for i, layer in enumerate(layers)}
            fedrot_ref_arrays = capture_fedrot_reference(self.model, self.aggregation, lora_params)
            set_parameters(self.model, lora_params, base_layer_updates=base_layer_updates)
        else:
            fedrot_ref_arrays = capture_fedrot_reference(self.model, self.aggregation, parameters)
            set_parameters(self.model, parameters)

        if self.aggregation in FREEZE_A_STRATEGIES:
            freeze_params(self.model, FROZEN_PARAM_SUBSTRINGS)

        if self.device.type == "cpu":
            frozen = self.aggregation in FREEZE_A_STRATEGIES
            for name, param in self.model.named_parameters():
                if frozen and any(s in name for s in FROZEN_PARAM_SUBSTRINGS):
                    continue
                param.requires_grad = True

        # Constant LR (no scheduler) + optim="sgd"/max_grad_norm=0: matching FedRot-LoRA's own
        # protocol -- see the identical, more detailed note in
        # llm_humaneval/client_app.py::fit for the upstream-repo confirmation. The same SGD-vs-
        # AdamW root-cause investigation that fixed apps/llm-humaneval applies here identically
        # (base_rescale.yaml's train.optimizer block is likewise never overridden away from the
        # cfg_training.py SGD default, and grad.grad_clip is likewise never overridden from -1).
        self.training_arguments.learning_rate = self.train_cfg.learning_rate_max
        self.training_arguments.output_dir = config["save_path"]
        self.training_arguments.optim = "sgd"
        self.training_arguments.max_grad_norm = 0

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=formatting_prompts_func,
            data_collator=self.data_collator,
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
                batch_size=self.training_arguments.per_device_train_batch_size,
                local_steps=self.training_arguments.max_steps,
            )
        )

        return trained, len(self.trainset), fit_metrics


def client_fn(context: Context) -> FlowerClient:
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    data_root = os.environ.get("FEDORA_DATA_ROOT", "data")
    seed = cfg.get("seed", 0)
    client_trainset = load_data(partition_id, num_partitions, data_root, seed)
    tokenizer, data_collator = get_tokenizer_and_data_collator(cfg.model.name)

    return FlowerClient(
        cfg.model, cfg.train, cfg.strategy, client_trainset, tokenizer, data_collator, num_rounds
    ).to_client()


app = ClientApp(client_fn)
