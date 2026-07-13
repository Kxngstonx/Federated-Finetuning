import math
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from flwr.common.typing import NDArrays


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with efficient quantization and LoRA tuning.

    Uses smaller models like Mistral-7B, Phi-2, or TinyLLaMA to optimize efficiency.
    """

    # Suggested small model options
    model_choices = {"qwen2-0.5": "Qwen/Qwen2-0.5B"}

    # Choose the model (default: Mistral-7B)
    model_name = model_choices.get(model_cfg.name.lower(), model_cfg.name)

    # Handle different quantization settings
    # device_map is pinned to a single GPU ({"": 0}), not "auto": when this is called from a
    # Ray-isolated client actor (only one GPU visible via CUDA_VISIBLE_DEVICES), "auto" already
    # degenerates to that single visible device, so this is a no-op there. But when called from
    # the (non-Ray-isolated) server process to build init_model, "auto" would see every physical
    # GPU and naively pipeline-split the model's layers across all of them -- harmless for
    # init_model specifically (it's only ever used once to read off initial parameter shapes/
    # values, never for actual forward/generate compute), but still wastes memory on every GPU
    # and, if this function is ever called somewhere that DOES run inference (as
    # apps/llm-gsm8k/eval_gsm8k.py and apps/llm-humaneval/eval_humaneval.py's own separate
    # load_model() used to,
    # before being fixed the same way), causes real ping-pong slowdown during generation.
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        torch_dtype = torch.bfloat16
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = torch.bfloat16
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    elif model_cfg.quantization == 0:
        quantization_config = None  # No quantization
        torch_dtype = torch.float32  # Ensure compatibility with CPU training
        device_map = "cpu"  # Force model to run on CPU
    else:
        raise ValueError(
            f"Use 4-bit, 8-bit, or disable quantization (0). You passed: {model_cfg.quantization}"
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    # RoLoRA (https://arxiv.org/abs/2407.08044): optionally rotate the frozen base weights
    # before LoRA is attached, so LoRA trains against outlier-free activations. Opt-in via
    # model.rolora.rotate; no-op (and independent of strategy.aggregation) otherwise.
    from flowertune_llm.strategies.rolora import apply_rotation

    apply_rotation(model, model_cfg.get("rolora", {}))

    # LoRA / DoRA Configuration (Optimized for Small Models)
    # DoRA (https://arxiv.org/abs/2402.09353) decomposes the LoRA update into
    # magnitude and direction; toggle via model.use-dora in pyproject.toml.
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Add target LoRA layers
        use_dora=model_cfg.get("use_dora", False),
    )

    return get_peft_model(model, peft_config)


def _is_bnb_param4bit(tensor) -> bool:
    """Whether `tensor` is a bitsandbytes 4-bit quantized parameter. Returns False (rather than
    raising) when bitsandbytes isn't relevant to the current model, e.g. an unquantized RoBERTa
    base layer -- see set_parameters's base_layer_updates branch below."""
    try:
        import bitsandbytes as bnb

        return isinstance(tensor, bnb.nn.Params4bit)
    except ImportError:
        return False


def set_parameters(
    model,
    parameters: NDArrays,
    base_layer_updates: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Change the parameters of the model using the given ones.

    `base_layer_updates`, if given, maps a LoRA target module's name (as in
    `peft_layers.index_lora_layers(model)`) to a float32 master weight array (FLoRA only). If the
    module's existing base_layer.weight is bitsandbytes-quantized (Llama/CAUSAL_LM pipeline), the
    master is re-quantized fresh using the exact quantization arguments (blocksize/quant_type/
    compress_statistics/quant_storage) the model's own base layer was already loaded with -- never
    hardcoded -- and swapped in as the module's new frozen `base_layer.weight`. See
    strategies/flora.py for why the master itself must stay exact. If the base layer is NOT
    quantized (e.g. an unquantized RoBERTa/SEQ_CLS model), the float32 master is assigned directly
    (cast to the existing weight's dtype) with no quantization round-trip.
    """
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)

    if base_layer_updates:
        for name, w0_array in base_layer_updates.items():
            module = model.get_submodule(name)
            existing = module.base_layer.weight
            device = existing.device

            if _is_bnb_param4bit(existing):
                import bitsandbytes as bnb

                w0_tensor = torch.from_numpy(w0_array).to(device=device, dtype=torch.float32)
                w0_4bit, quant_state = bnb.functional.quantize_4bit(
                    w0_tensor,
                    blocksize=existing.blocksize,
                    compress_statistics=existing.compress_statistics,
                    quant_type=existing.quant_type,
                    quant_storage=existing.quant_storage,
                )
                new_param = bnb.nn.Params4bit(
                    w0_4bit,
                    requires_grad=False,
                    quant_state=quant_state,
                    blocksize=existing.blocksize,
                    compress_statistics=existing.compress_statistics,
                    quant_type=existing.quant_type,
                    quant_storage=existing.quant_storage,
                    module=module.base_layer,
                    bnb_quantized=True,
                )
                module.base_layer.weight = new_param
            else:
                w0_tensor = torch.from_numpy(w0_array).to(device=device, dtype=existing.dtype)
                module.base_layer.weight = torch.nn.Parameter(w0_tensor, requires_grad=False)


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]
