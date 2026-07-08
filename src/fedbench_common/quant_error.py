"""Quantization round-trip error metric -- FLoRA only, per the experiment plan.

FLoRA (strategies/flora.py) maintains a float32 "master" base weight W0 per LoRA layer on the
server, which the client re-quantizes into a fresh bnb.nn.Params4bit every round (see
flowertune_llm/models.py::set_parameters's base_layer_updates branch). This module measures how
much information that per-round quantize/dequantize round-trip loses: ||Dequant(Quant(W))-W||_F.

Only meaningful when the deployed base model is bitsandbytes 4-bit quantized (the Llama/NLG
pipeline); FLoRA runs against an unquantized RoBERTa base layer (NLU pipeline) have no
quantization round-trip to measure, and quantization_roundtrip_error should not be called there.
"""

import torch


def quantization_roundtrip_error(w0: torch.Tensor, quant_kwargs: dict) -> float:
    """w0: float32 CPU tensor (the current master base weight). quant_kwargs: the exact
    blocksize/compress_statistics/quant_type/quant_storage the base layer was originally loaded
    with (cached once at FLoRA construction time -- see strategies/flora.py::FLoRA.__init__)."""
    import bitsandbytes as bnb

    w0_4bit, quant_state = bnb.functional.quantize_4bit(
        w0,
        blocksize=quant_kwargs["blocksize"],
        compress_statistics=quant_kwargs["compress_statistics"],
        quant_type=quant_kwargs["quant_type"],
        quant_storage=quant_kwargs["quant_storage"],
    )
    dequant = bnb.functional.dequantize_4bit(w0_4bit, quant_state)
    return torch.linalg.norm(dequant - w0, ord="fro").item()
