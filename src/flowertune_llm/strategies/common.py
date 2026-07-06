"""Shared helpers for LoRA-aware federated aggregation strategies.

`attach_w0` is the expensive half of what used to be `_build_layer_refs`: it dequantizes each
LoRA target module's frozen base weight. Only strategies that reconstruct full effective
weights (FeDoRA, FLoRA) need it; strategies that only need to know *where* lora_A/lora_B live
in the flat NDArrays list (FedRot, client-side, every round) use the cheaper
`peft_layers.index_lora_layers` directly and skip dequantization entirely.
"""

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from peft.utils.integrations import dequantize_module_weight

from flowertune_llm.peft_layers import LoraLayerIndex, index_lora_layers

_EPS = 1e-8


@dataclass
class _LoraLayerRef:
    name: str  # module path, e.g. "base_model.model.model.layers.3.self_attn.q_proj"
    idx_a: int  # index of this layer's lora_A in the flat NDArrays list
    idx_b: int  # index of lora_B
    idx_m: "int | None"  # index of lora_magnitude_vector, or None if this layer has no DoRA magnitude key
    W0: torch.Tensor  # (out_features, in_features), cached CPU float32, dequantized, detached
    scaling: float
    r: int


def attach_w0(layers: List[LoraLayerIndex], model: nn.Module) -> List[_LoraLayerRef]:
    """Pull each target module's frozen W0/scaling/r off the live model, once."""
    out = []
    for layer in layers:
        module = model.get_submodule(layer.name)
        w0 = dequantize_module_weight(module.base_layer)
        w0 = w0.detach().to(device="cpu", dtype=torch.float32).clone()
        out.append(
            _LoraLayerRef(
                name=layer.name,
                idx_a=layer.idx_a,
                idx_b=layer.idx_b,
                idx_m=layer.idx_m,
                W0=w0,
                scaling=module.scaling["default"],
                r=module.r["default"],
            )
        )
    return out


def build_layer_refs(model: nn.Module) -> List[_LoraLayerRef]:
    """Convenience wrapper: index_lora_layers(model) + attach_w0(...), for callers that always
    need both (FeDoRA, FLoRA)."""
    layers = index_lora_layers(model)
    if not layers:
        raise ValueError(
            "No LoRA target-module parameters found in the provided model "
            "(get_peft_model_state_dict(model) returned no lora_A/lora_B keys)."
        )
    return attach_w0(layers, model)
