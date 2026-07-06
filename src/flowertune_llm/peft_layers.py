"""Cheap, dequantization-free indexing of LoRA layers within a PEFT model's flat NDArrays.

Shared by strategies that need to locate each LoRA layer's lora_A/lora_B/lora_magnitude_vector
position in the flat parameter list (strategies/common.py for server-side aggregation, and
client_app.py for FedRot's per-round rotation) without paying the cost of dequantizing the
frozen base weight (that cost is only needed by FeDoRA/FLoRA, see strategies/common.py::attach_w0).
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from peft import get_peft_model_state_dict


@dataclass
class LoraLayerIndex:
    name: str  # module path, e.g. "base_model.model.model.layers.3.self_attn.q_proj"
    idx_a: int  # index of this layer's lora_A in the flat NDArrays list
    idx_b: int  # index of lora_B
    idx_m: Optional[int]  # index of lora_magnitude_vector, or None if this layer has no DoRA magnitude key


def index_lora_layers(model) -> List[LoraLayerIndex]:
    """Derive a module -> {A,B,m}-index map from the reference model's PEFT state dict key order.

    Note: get_peft_model_state_dict() strips the ".{adapter_name}" segment from every key
    (peft/utils/save_and_load.py, "# REMOVE ADAPTER NAME"). Since this codebase only ever uses
    the implicit "default" adapter, that means lora_A/lora_B keep a ".weight" suffix (e.g.
    "...q_proj.lora_A.weight") but lora_magnitude_vector loses its suffix entirely (e.g.
    "...q_proj.lora_magnitude_vector", no ".weight"). So these substring matches must not
    require a trailing dot after the token.
    """
    key_order = list(get_peft_model_state_dict(model).keys())

    groups: dict[str, dict[str, int]] = defaultdict(dict)
    for idx, key in enumerate(key_order):
        if ".lora_A" in key:
            groups[key.split(".lora_A")[0]]["A"] = idx
        elif ".lora_B" in key:
            groups[key.split(".lora_B")[0]]["B"] = idx
        elif ".lora_magnitude_vector" in key:
            groups[key.split(".lora_magnitude_vector")[0]]["m"] = idx

    layers = []
    for name, idxs in groups.items():
        layers.append(
            LoraLayerIndex(
                name=name,
                idx_a=idxs["A"],
                idx_b=idxs["B"],
                idx_m=idxs.get("m"),
            )
        )
    return layers
