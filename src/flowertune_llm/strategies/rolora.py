"""RoLoRA (arXiv:2407.08044): a one-time, output-preserving rotation of the frozen base model's
weights, applied before LoRA adapters are attached, so that the activations LoRA trains against
are outlier-free -- which makes the eventually-merged model far more robust to later
weight-activation quantization. Unlike every other strategy in this package, RoLoRA does not
change *how LoRA adapters are aggregated*: plain data-size-weighted FedAvg over lora_A/lora_B
(exactly `fedit.py::FedIT`) is mathematically correct here, since rotation only touches the
frozen base weights, never the trained adapter. The real logic lives in `apply_rotation`, called
once from `models.py::get_model()` *before* `get_peft_model(...)` attaches LoRA -- see
`model.rolora.*` in pyproject.toml. `RoLoRA(FedAvg)` below exists purely so `"rolora"` is
selectable/comparable by name via `strategy.aggregation`, matching this package's convention.

Ported from RoLoRA/src/llamafactory/rotation/{rotation_utils.py,hadamard_utils.py}, scoped down
to what a federated LoRA fine-tuning pipeline needs:

* R1 (global rotation): fuse each RMSNorm's scale into its adjacent linear layers (so the norm
  becomes a pure rescale, "fuse_ln_linear"/"fuse_layer_norms"), then rotate the residual stream
  (embeddings, q/k/v/o, mlp gate/up/down, lm_head) with a random orthogonal matrix Q -- an
  operation that cancels out algebraically across consecutive matmuls and leaves the network's
  function unchanged.
* R4 (down_proj rotation): additionally rotate down_proj's input (intermediate-size) axis with a
  second orthogonal matrix H4, since that's where LLMs' worst activation outliers concentrate;
  a forward_pre_hook rotates down_proj's *runtime* input activations by the same H4 so the
  network's output is unaffected.

Two things the original repo has that this port deliberately drops:

* Fake-quantization simulation (`ActQuantWrapper`/`ActQuantizer`/`WeightQuantizer` in the
  original repo's quant_utils.py, ~430 lines) -- that machinery exists to simulate post-training
  W4A4 accuracy *during* training, a separate concern from FL fine-tuning that this pipeline
  doesn't otherwise do. Only the R4 online-*rotation* hook is ported, not fake quantization.
* The original repo's hardcoded 11-matrix Hadamard table (`get_had12`...`get_had172`, ~3000
  lines) for hidden sizes that are divisible by a specific hardcoded K but aren't themselves a
  power of 2, plus its `fast_hadamard_transform` CUDA-kernel dependency. This port constructs
  Hadamard matrices via pure-PyTorch Sylvester recursive doubling, which only works for
  power-of-2 sizes -- exactly like the original repo's own K=1 fast path, with no precision
  loss, and it covers the actual production model in this repo (Llama-3-8B, hidden_size=4096).
  For sizes that are neither a power of 2 nor divisible by the original's hardcoded K list (e.g.
  the quick-dev default Qwen2-0.5B, hidden_size=896=128x7 -- which the *original repo can't run
  in hadamard mode either*, it hits an assert and crashes), `rotate_mode="hadamard"` falls back
  to `rotate_mode="random"` (QR-based orthogonal) with a logged warning instead of crashing.
* The original repo's unconditional embedding mean-centering step (part of
  `fuse_layer_norms_noreplace`, applied regardless of architecture) is not ported: it's only an
  equivalence-preserving transform for mean-subtracting LayerNorm models (OPT), not for the
  mean-agnostic RMSNorm models this port supports (Llama/Qwen2) -- applying it there actually
  changes the model's output (confirmed empirically, not just in theory: see
  `tests/unit/test_rolora.py`). Since RoLoRA's entire premise here is that rotation is a safe,
  strict no-op appliable identically and independently by every client and the server, this step
  is dropped rather than carried over as a lossy approximation.

Memory note: R4's H4 matrix is dense (intermediate_size x intermediate_size), since it isn't
run through a fast implicit butterfly transform -- for a large intermediate_size (e.g. 14336 on
Llama-3-8B) that's a few hundred MB. It's computed once and the *same* tensor object is shared
(not cloned) across every layer's forward_pre_hook buffer, so total memory is O(1), not
O(num_layers).

Determinism: rotation must be byte-identical across every client and the server, or the
merged model stops being mathematically equivalent to the unrotated one. `model.rolora.seed`
seeds a local `torch.Generator` (not global `torch.manual_seed`) so every process reconstructs
identical Q/H4 matrices independently, without disturbing unrelated RNG state (dropout init,
data shuffling).
"""

import logging
import math
from typing import Optional

import torch
from torch import nn

from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class RoLoRA(FedAvg):
    """Nominal FedAvg subclass -- rotation happens once at model-load time (see
    `apply_rotation`, wired into `models.py::get_model()`), not here. Aggregation itself is
    unaffected by rotation (only the frozen base weights are rotated, never lora_A/lora_B), so
    plain data-size-weighted FedAvg over the flat LoRA state dict is exactly correct, matching
    `fedit.py::FedIT`. This class exists only so `"rolora"` is selectable/comparable by name
    alongside the other strategies."""

    def __init__(self, *, model=None, cfg=None, **kwargs) -> None:
        del model, cfg  # unused; accepted for signature parity with the other strategy factories
        super().__init__(**kwargs)


# --------------------------------------------------------------------------------------------
# Architecture support check
# --------------------------------------------------------------------------------------------

_REQUIRED_LAYER_ATTRS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "input_layernorm",
    "post_attention_layernorm",
)


def _has_attr_path(obj, dotted_path: str) -> bool:
    for part in dotted_path.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def _check_architecture(model) -> None:
    """Raise NotImplementedError (rather than silently no-op'ing) if `model` isn't a
    Llama-family causal LM. Covers every causal-LM app in this repo (llm-gsm8k, llm-humaneval,
    the default Qwen2-0.5B) -- glue-nlu's RoBERTa (encoder-only, no residual-stream/lm_head
    structure matching this shape) is out of scope."""
    base = getattr(model, "model", None)
    ok = (
        base is not None
        and hasattr(base, "embed_tokens")
        and hasattr(base, "layers")
        and len(base.layers) > 0
        and hasattr(base, "norm")
        and hasattr(model, "lm_head")
        and all(_has_attr_path(base.layers[0], attr) for attr in _REQUIRED_LAYER_ATTRS)
    )
    if not ok:
        raise NotImplementedError(
            "rolora.apply_rotation only supports Llama-family causal-LM architectures "
            "(model.model.embed_tokens/layers/norm, model.lm_head, and per-layer "
            "self_attn.{q,k,v,o}_proj / mlp.{gate,up,down}_proj / input_layernorm / "
            f"post_attention_layernorm). Got model type {type(model).__name__}, which doesn't "
            "match this shape -- e.g. encoder-only models like RoBERTa (glue-nlu) aren't "
            "supported."
        )


# --------------------------------------------------------------------------------------------
# bnb-aware weight mutation (base weights may be 4-bit-quantized via BitsAndBytesConfig, loaded
# before rotation runs -- mirrors the dequant/requant pattern already in
# models.py::set_parameters and strategies/common.py::attach_w0)
# --------------------------------------------------------------------------------------------


def _is_bnb_4bit_param(weight) -> bool:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    return isinstance(weight, bnb.nn.Params4bit)


def _dequantize_weight(module: nn.Module) -> torch.Tensor:
    """Return `module.weight` as a float64 CPU-independent tensor, dequantizing first if it's a
    bnb 4-bit packed parameter."""
    weight = module.weight
    if _is_bnb_4bit_param(weight):
        import bitsandbytes as bnb

        return bnb.functional.dequantize_4bit(weight.data, weight.quant_state).double()
    return weight.data.double()


def _requantize_weight_(module: nn.Module, new_weight: torch.Tensor) -> None:
    """Write `new_weight` back into `module.weight` in place, re-quantizing with the exact same
    bnb quant params the module was already loaded with if it's bnb 4-bit, otherwise assigning
    directly (cast to the existing weight's dtype)."""
    weight = module.weight
    if _is_bnb_4bit_param(weight):
        import bitsandbytes as bnb

        packed, quant_state = bnb.functional.quantize_4bit(
            new_weight.to(dtype=torch.float16, device=weight.device),
            blocksize=weight.blocksize,
            compress_statistics=weight.compress_statistics,
            quant_type=weight.quant_type,
            quant_storage=weight.quant_storage,
        )
        module.weight = bnb.nn.Params4bit(
            packed,
            requires_grad=False,
            quant_state=quant_state,
            blocksize=weight.blocksize,
            compress_statistics=weight.compress_statistics,
            quant_type=weight.quant_type,
            quant_storage=weight.quant_storage,
            module=module,
            bnb_quantized=True,
        )
    else:
        module.weight.data = new_weight.to(dtype=weight.dtype, device=weight.device)


# --------------------------------------------------------------------------------------------
# LayerNorm fusion (R1 prerequisite): fold each RMSNorm's scale into its adjacent linear layers,
# then reset the norm's weight to 1.0, so the norm becomes a pure no-op and the residual stream
# can be rotated cleanly. Port of RoLoRA/rotation_utils.py::fuse_ln_linear/fuse_layer_norms_noreplace.
# --------------------------------------------------------------------------------------------


def fuse_ln_linear(layernorm: nn.Module, linear_layers) -> None:
    for linear in linear_layers:
        W_ = _dequantize_weight(linear)
        gamma = layernorm.weight.data.double().to(W_.device)
        _requantize_weight_(linear, W_ * gamma)

        if getattr(layernorm, "bias", None) is not None:
            if linear.bias is None:
                linear.bias = nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64, device=W_.device)
                )
            b = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.data.double())
            linear.bias.data = b.to(linear.bias.dtype)


def _reset_norm_to_identity(norm: nn.Module) -> None:
    with torch.no_grad():
        norm.weight.fill_(1.0)


def fuse_layer_norms(model) -> None:
    # Note: the original repo also mean-centers the token embeddings here, unconditionally,
    # regardless of whether the norm is a mean-subtracting LayerNorm (OPT) or a mean-agnostic
    # RMSNorm (Llama/Qwen2). That's only an equivalence-preserving transform for LayerNorm
    # models: subtracting the embedding mean lets a later linear "bake in" the mean-subtraction
    # LayerNorm itself performs (see `bake_mean_into_linear` in the original repo, applied only
    # on its OPT branch). For RMSNorm models -- the only family this port supports -- there is
    # no such mean-subtraction to absorb, so centering the embeddings actually changes the
    # model's output; it is deliberately NOT ported here, since RoLoRA's entire premise (safe to
    # apply once, identically, on every client and the server) depends on rotation being a
    # strict no-op.
    base = model.model

    for layer in base.layers:
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
        fuse_ln_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        _reset_norm_to_identity(layer.post_attention_layernorm)
        _reset_norm_to_identity(layer.input_layernorm)

    fuse_ln_linear(base.norm, [model.lm_head])
    _reset_norm_to_identity(base.norm)


# --------------------------------------------------------------------------------------------
# Orthogonal / Hadamard matrix construction
# --------------------------------------------------------------------------------------------


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _sylvester_hadamard(n: int) -> torch.Tensor:
    """Pure-PyTorch power-of-2 Hadamard matrix via Sylvester's recursive doubling construction
    (H_1 = [1], H_{2k} = [[H_k, H_k], [H_k, -H_k]]), normalized by 1/sqrt(n). Mathematically
    identical to the original repo's `fast_hadamard_transform`-based K=1 fast path for
    power-of-2 sizes, just materialized as a dense matrix instead of an implicit butterfly
    transform."""
    if not _is_pow2(n):
        raise ValueError(f"_sylvester_hadamard requires a power-of-2 size, got {n}.")
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def random_hadamard_matrix(size: int, generator: torch.Generator) -> torch.Tensor:
    """Randomized Hadamard transform (see QuIP#'s "Randomized Hadamard Transformation"):
    diag(+-1) @ H, with the sign pattern drawn from `generator`. Port of
    RoLoRA/hadamard_utils.py::random_hadamard_matrix, using the dense Sylvester construction
    above instead of the CUDA `fast_hadamard_transform` kernel."""
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64).double() * 2 - 1
    H = _sylvester_hadamard(size)
    return signs.unsqueeze(1) * H  # diag(signs) @ H


def random_orthogonal_matrix(size: int, generator: torch.Generator) -> torch.Tensor:
    """QR-based random orthogonal matrix. Port of RoLoRA/rotation_utils.py::random_orthogonal_matrix."""
    a = torch.randn(size, size, dtype=torch.float64, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size: int, mode: str, generator: torch.Generator) -> torch.Tensor:
    if mode == "random":
        return random_orthogonal_matrix(size, generator)
    if mode == "hadamard":
        if _is_pow2(size):
            return random_hadamard_matrix(size, generator)
        logger.warning(
            "rolora: rotate_mode='hadamard' requested but size=%d is not a power of 2 (and "
            "this port's hardcoded-K-matrix table was intentionally not carried over from the "
            "reference implementation -- see strategies/rolora.py's module docstring). Falling "
            "back to rotate_mode='random' for this rotation.",
            size,
        )
        return random_orthogonal_matrix(size, generator)
    raise ValueError(f"Unknown rotate_mode {mode!r}; expected 'random' or 'hadamard'.")


# --------------------------------------------------------------------------------------------
# Weight rotation (R1: Q on hidden_size; R4: H4 on down_proj's intermediate_size input axis).
# Port of RoLoRA/rotation_utils.py::rotate_embeddings/rotate_attention_inputs/
# rotate_attention_output/rotate_mlp_input/rotate_mlp_output/rotate_head, generalized to drop
# the LLAMA_MODEL/OPT_MODEL branch (Llama-family only) and routed through the bnb-aware
# dequant/requant helpers above.
# --------------------------------------------------------------------------------------------


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    embed = model.model.embed_tokens
    W_ = _dequantize_weight(embed)
    new_w = torch.matmul(W_, Q.to(device=W_.device, dtype=torch.float64))
    _requantize_weight_(embed, new_w)


def rotate_head(model, Q: torch.Tensor) -> None:
    head = model.lm_head
    W_ = _dequantize_weight(head)
    new_w = torch.matmul(W_, Q.to(device=W_.device, dtype=torch.float64))
    _requantize_weight_(head, new_w)


def rotate_attention_inputs(layer, Q: torch.Tensor) -> None:
    for linear in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
        W_ = _dequantize_weight(linear)
        new_w = torch.matmul(W_, Q.to(device=W_.device, dtype=torch.float64))
        _requantize_weight_(linear, new_w)


def rotate_attention_output(layer, Q: torch.Tensor) -> None:
    linear = layer.self_attn.o_proj
    W_ = _dequantize_weight(linear)
    Qd = Q.to(device=W_.device, dtype=torch.float64)
    _requantize_weight_(linear, torch.matmul(Qd.t(), W_))
    if linear.bias is not None:
        b = linear.bias.data.double()
        linear.bias.data = torch.matmul(Qd.t(), b).to(dtype=linear.bias.dtype)


def rotate_mlp_input(layer, Q: torch.Tensor) -> None:
    for linear in (layer.mlp.up_proj, layer.mlp.gate_proj):
        W_ = _dequantize_weight(linear)
        new_w = torch.matmul(W_, Q.to(device=W_.device, dtype=torch.float64))
        _requantize_weight_(linear, new_w)


def rotate_mlp_output(layer, Q: torch.Tensor, H4: torch.Tensor) -> None:
    """Rotate down_proj on both axes: Q on its output axis (feeds the hidden_size residual
    stream, R1) and H4 on its input axis (the intermediate-size MLP activation, R4)."""
    linear = layer.mlp.down_proj
    W_ = _dequantize_weight(linear)
    Qd = Q.to(device=W_.device, dtype=torch.float64)
    H4d = H4.to(device=W_.device, dtype=torch.float64)
    new_w = torch.matmul(torch.matmul(Qd.t(), W_), H4d)
    _requantize_weight_(linear, new_w)
    if linear.bias is not None:
        b = linear.bias.data.double()
        linear.bias.data = torch.matmul(Qd.t(), b).to(dtype=linear.bias.dtype)


# --------------------------------------------------------------------------------------------
# R4 online activation rotation: down_proj's weight had H4 folded into its input axis above, so
# its runtime input activations must also be rotated by H4 for the network's output to stay
# unchanged (W @ H4 applied to x' = H4^T @ x recovers the original W @ x, since H4 is
# orthogonal). Port of RoLoRA/rotation_utils.py::online_rotate/register_online_rotation.
# --------------------------------------------------------------------------------------------


def _online_rotate_pre_hook(module: nn.Module, args):
    x = args[0]
    return (torch.nn.functional.linear(x, module._rolora_h_t),) + args[1:]


def register_online_rotation(module: nn.Module, h_t: torch.Tensor, dtype: torch.dtype) -> None:
    if hasattr(module, "_rolora_h_t"):
        return
    # `h_t` (H4^T) is shared -- the *same* tensor object -- across every layer's down_proj to
    # keep total memory O(1) rather than O(num_layers); see module docstring's memory note.
    module.register_buffer(
        "_rolora_h_t", h_t.to(dtype=dtype, device=module.weight.device), persistent=False
    )
    module.register_forward_pre_hook(_online_rotate_pre_hook)


# --------------------------------------------------------------------------------------------
# Top-level entry point, called from models.py::get_model() before get_peft_model(...) attaches
# LoRA.
# --------------------------------------------------------------------------------------------


def apply_rotation(model, rolora_cfg: Optional[dict]) -> None:
    """Rotate `model`'s frozen weights in place per RoLoRA (R1 + R4), gated by
    `rolora_cfg["rotate"]`. No-op (and no architecture check) if rotation isn't enabled, so this
    is safe to call unconditionally for every app/strategy."""
    rolora_cfg = rolora_cfg or {}
    if not rolora_cfg.get("rotate", False):
        return

    _check_architecture(model)

    seed = int(rolora_cfg.get("seed", 0))
    mode = rolora_cfg.get("rotate_mode", "hadamard")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    fuse_layer_norms(model)

    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    Q = get_orthogonal_matrix(hidden_size, mode, generator)
    H4 = get_orthogonal_matrix(intermediate_size, mode, generator)
    H4_t = H4.t().contiguous()
    norm_dtype = model.model.layers[0].input_layernorm.weight.dtype

    rotate_embeddings(model, Q)
    rotate_head(model, Q)

    for layer in model.model.layers:
        rotate_attention_inputs(layer, Q)
        rotate_attention_output(layer, Q)
        rotate_mlp_input(layer, Q)
        rotate_mlp_output(layer, Q, H4)
        register_online_rotation(layer.mlp.down_proj, H4_t, dtype=norm_dtype)
