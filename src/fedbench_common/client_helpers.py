"""Client-side helpers shared by apps/glue-nlu, apps/llm-humaneval, and apps/llm-gsm8k's
FlowerClient implementations:
FedRot's rotation hook (ported from flowertune_llm/client_app.py, generalized so it doesn't
assume a causal-LM/SFTTrainer setup), the FFA-style freeze-A hook, and the instrumentation
(trainable params, upload bytes, wall-clock, FLOPs) each fit() call needs to report.

flowertune_llm/client_app.py (the original Qwen/Alpaca demo app) keeps its own inline copy of the
rotation/freeze logic unchanged -- these are the same algorithms, just factored out here so the
two new apps don't duplicate ~80 lines of rotation bookkeeping each.
"""

import pickle
from typing import Any, Callable, Dict, Hashable, Optional, Tuple

import numpy as np
import torch

from flwr.common.typing import NDArrays, Scalar

from flowertune_llm.peft_layers import index_lora_layers
from flowertune_llm.strategies.fedrot import rotation_align_optimization
from fedbench_common.metrics import Timer, estimate_lora_flops, ndarrays_nbytes, trainable_param_count

FREEZE_A_STRATEGIES = ("fedsvd", "ffalora")
FROZEN_PARAM_SUBSTRINGS = ("lora_A", "lora_magnitude_vector")
PREROT_A_METRIC_PREFIX = "fedrot_prerot_A::"

# Process-local cache: Flower's simulation engine re-invokes client_fn() (and thus
# FlowerClient.__init__) every time an actor is scheduled to run a client, even across rounds for
# what is logically "the same" model/architecture within a single `flwr run` -- so without this
# cache, every round re-does AutoModelForCausalLM.from_pretrained(...) + bitsandbytes 4-bit
# quantization from scratch (large, CPU/disk-bound, and dwarfs the actual GPU training time for
# small batch sizes), starving the GPU between rounds. Reusing the already-loaded/quantized model
# object across constructions is safe: only the LoRA adapter weights change round to round, and
# those are overwritten fresh every fit() call via set_parameters() regardless of whether the
# underlying model object is new or cached. Keyed per-process (per Ray actor), not shared across
# actors/GPUs.
_MODEL_CACHE: Dict[Hashable, Any] = {}


def get_cached_model(cache_key: Hashable, build_fn: Callable[[], Any]) -> Any:
    """Returns the cached model for `cache_key` if this actor process already built one,
    otherwise calls `build_fn()` once, caches, and returns it."""
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = build_fn()
    return _MODEL_CACHE[cache_key]


def freeze_params(model, substrings=FROZEN_PARAM_SUBSTRINGS) -> None:
    for name, param in model.named_parameters():
        if any(s in name for s in substrings):
            param.requires_grad = False


def capture_fedrot_reference(model, aggregation: str, parameters: NDArrays):
    """Call BEFORE set_parameters(model, parameters) overwrites the model -- captures this
    round's just-received global (A, B) per layer, used as the rotation reference at the end of
    local training. Returns None when aggregation != 'fedrot'."""
    if aggregation != "fedrot":
        return None
    layers = index_lora_layers(model)
    return {layer.name: (parameters[layer.idx_a], parameters[layer.idx_b]) for layer in layers}


def maybe_rotate_fedrot(
    model,
    strategy_cfg,
    trained: NDArrays,
    ref_arrays: Optional[dict],
    current_round: int,
) -> Tuple[NDArrays, Dict[str, Scalar]]:
    """Applies FedRot's hard-rotation Procrustes alignment to `trained`'s (A, B) pairs, using
    `ref_arrays` (this round's just-received global parameters, from capture_fedrot_reference) as
    the alignment reference. Ported from FedRot-LoRA/federatedscope/core/workers/client.py's
    alternating-reference logic (only the "shareAB" mode, matching strategies/fedrot.py).

    Returns (possibly-rotated trained arrays, extra fit()-metrics to report: each layer's
    PRE-rotation A, pickled float16, keyed by PREROT_A_METRIC_PREFIX + layer.name -- consumed by
    FedRot.aggregate_fit's basis-overlap metric, see strategies/fedrot.py). Skips rotation (and
    the metric) on round 1, since there's no prior global reference to align against yet.
    """
    fedrot_cfg = strategy_cfg.get("fedrot", {})
    if not fedrot_cfg.get("rotate", True) or current_round <= 1 or ref_arrays is None:
        return trained, {}

    initial_share = fedrot_cfg.get("initial_share", "A")
    swap_offset = 0 if initial_share == "A" else 1

    layers = index_lora_layers(model)
    trained = list(trained)
    prerot_metrics: Dict[str, Scalar] = {}
    for layer in layers:
        align_matrix = "A" if current_round % 2 != swap_offset else "B"
        ref_a, ref_b = ref_arrays[layer.name]
        ref = ref_a if align_matrix == "A" else ref_b

        a_t = torch.from_numpy(np.asarray(trained[layer.idx_a]))
        b_t = torch.from_numpy(np.asarray(trained[layer.idx_b]))
        ref_t = torch.from_numpy(np.asarray(ref))

        prerot_metrics[f"{PREROT_A_METRIC_PREFIX}{layer.name}"] = pickle.dumps(
            a_t.numpy().astype(np.float16, copy=False)
        )

        a_new, b_new = rotation_align_optimization(ref_t, align_matrix, a_t, b_t)
        trained[layer.idx_a] = a_new.numpy()
        trained[layer.idx_b] = b_new.numpy()
    return trained, prerot_metrics


def build_fit_metrics(
    model,
    trained: NDArrays,
    train_timer: Timer,
    seq_length: int,
    batch_size: int,
    local_steps: int,
) -> Dict[str, Scalar]:
    """Requirements: trainable params, comm bytes (upload), client-side wall-clock, FLOPs.
    Assembled once per fit() call, merged into whatever metrics dict the caller already returns
    (train_loss, plus any fedrot prerot metrics from maybe_rotate_fedrot)."""
    return {
        "trainable_params": trainable_param_count(model),
        "upload_bytes": ndarrays_nbytes(trained),
        "client_train_seconds": train_timer.elapsed,
        "est_flops": estimate_lora_flops(model, seq_length, batch_size, local_steps),
    }
