"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig

from transformers import TrainingArguments
from trl import SFTTrainer

from flowertune_llm.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from flowertune_llm.models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)
from flowertune_llm.peft_layers import index_lora_layers
from flowertune_llm.strategies.fedrot import rotation_align_optimization

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# Strategies where lora_A (and, when DoRA is enabled, the magnitude vector m) are frozen
# client-side (never trained, never uploaded meaningfully): FedSVD shares an SVD-re-
# orthogonalized A periodically and recomputes m analytically every round; FFALoRA never
# touches A or m after init. Both freeze the *same* substrings -- only the server-side
# strategy (fedsvd.py / ffalora.py) differs in whether m gets recomputed or just cached.
_FREEZE_A_STRATEGIES = ("fedsvd", "ffalora")
_FROZEN_PARAM_SUBSTRINGS = ("lora_A", "lora_magnitude_vector")


def _freeze_params(model, substrings) -> None:
    for name, param in model.named_parameters():
        if any(s in name for s in substrings):
            param.requires_grad = False


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        strategy_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.strategy_cfg = strategy_cfg
        self.aggregation = strategy_cfg.get("aggregation", "fedora")
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset

        # instantiate model
        self.model = get_model(model_cfg)
        # Ensure model is on the correct device
        self.model.to(self.device)

        if self.aggregation in _FREEZE_A_STRATEGIES:
            _freeze_params(self.model, _FROZEN_PARAM_SUBSTRINGS)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        current_round = int(config["current_round"])

        if self.aggregation == "flora":
            # Trailing arrays (one per LoRA layer, in index_lora_layers order) carry the
            # server's evolving float32 master base weight -- see strategies/flora.py.
            layers = index_lora_layers(self.model)
            n_layers = len(layers)
            lora_params = parameters[: len(parameters) - n_layers]
            extra = parameters[len(parameters) - n_layers :]
            base_layer_updates = {layer.name: extra[i] for i, layer in enumerate(layers)}
            set_parameters(self.model, lora_params, base_layer_updates=base_layer_updates)
        else:
            set_parameters(self.model, parameters)

        if self.aggregation in _FREEZE_A_STRATEGIES:
            # set_parameters/set_peft_model_state_dict may reset requires_grad; re-freeze A
            # (and m, if DoRA is enabled) every round after loading the server's parameters.
            _freeze_params(self.model, _FROZEN_PARAM_SUBSTRINGS)

        # FedRot rotates its own just-trained (A, B) against the reference it received THIS
        # round (before it's overwritten by set_parameters above), so capture it up front.
        fedrot_ref_arrays = None
        if self.aggregation == "fedrot":
            layers = index_lora_layers(self.model)
            fedrot_ref_arrays = {layer.name: (parameters[layer.idx_a], parameters[layer.idx_b]) for layer in layers}

        if self.device.type == "cpu":
            # Ensure optimizer updates only parameters that require gradients
            frozen = self.aggregation in _FREEZE_A_STRATEGIES
            for name, param in self.model.named_parameters():
                if frozen and any(s in name for s in _FROZEN_PARAM_SUBSTRINGS):
                    continue
                param.requires_grad = True

        new_lr = cosine_annealing(
            current_round,
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = config["save_path"]

        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        # Do local training
        results = trainer.train()

        trained = get_parameters(self.model)

        if self.aggregation == "fedrot":
            trained = self._maybe_rotate(trained, fedrot_ref_arrays, current_round)

        return (
            trained,
            len(self.trainset),
            {"train_loss": results.training_loss},
        )

    def _maybe_rotate(self, trained: NDArrays, ref_arrays, current_round: int) -> NDArrays:
        fedrot_cfg = self.strategy_cfg.get("fedrot", {})
        if not fedrot_cfg.get("rotate", True) or current_round <= 1:
            return trained  # first round cannot rotate -- no prior reference yet

        initial_share = fedrot_cfg.get("initial_share", "A")
        swap_offset = 0 if initial_share == "A" else 1

        layers = index_lora_layers(self.model)
        trained = list(trained)
        for layer in layers:
            align_matrix = "A" if current_round % 2 != swap_offset else "B"
            ref_a, ref_b = ref_arrays[layer.name]
            ref = ref_a if align_matrix == "A" else ref_b

            a_t = torch.from_numpy(np.asarray(trained[layer.idx_a]))
            b_t = torch.from_numpy(np.asarray(trained[layer.idx_b]))
            ref_t = torch.from_numpy(np.asarray(ref))

            a_new, b_new = rotation_align_optimization(ref_t, align_matrix, a_t, b_t)
            trained[layer.idx_a] = a_new.numpy()
            trained[layer.idx_b] = b_new.numpy()
        return trained


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Let's get the client partition
    client_trainset = load_data(partition_id, num_partitions, cfg.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    return FlowerClient(
        cfg.model,
        cfg.train,
        cfg.strategy,
        client_trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
