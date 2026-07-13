"""FeDoRA: DoRA-aware federated aggregation strategy for PEFT LoRA/DoRA adapters."""

from logging import WARNING
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from flowertune_llm.strategies.common import _LoraLayerRef, build_layer_refs
from fedbench_common.subspace_metrics import mean_subspace_overlap

_EPS = 1e-8


def aggregate_dora_layer(
    W0: torch.Tensor,  # (out_features, in_features) float32 CPU
    scaling: float,
    r: int,
    a_list: List[torch.Tensor],  # each (r, in_features) float32
    b_list: List[torch.Tensor],  # each (out_features, r) float32
    m_list: Optional[List[torch.Tensor]],  # each (out_features,) float32, or None
    freqs: List[float],  # per-client weight, sums to 1.0
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Pure-tensor implementation of the 4-step DoRA-aware layer aggregation. All
    inputs/outputs are float32 CPU tensors; numpy/dtype conversion happens in the caller.

    Requires target modules whose out/in dimensions are >= r -- callers must keep
    `target_modules` restricted accordingly (e.g. via pyproject.toml)."""
    is_dora_layer = m_list is not None

    # --- Step 1: reconstruct each client's effective weight, FedAvg them ---
    w_global = torch.zeros_like(W0)
    for i, (freq, a_c, b_c) in enumerate(zip(freqs, a_list, b_list)):
        ba_c = (b_c @ a_c) * scaling
        v_c = W0 + ba_c
        if is_dora_layer:
            m_c = m_list[i]
            row_norm = torch.linalg.norm(v_c, dim=1).clamp_min(_EPS)
            w_c = (m_c / row_norm).unsqueeze(1) * v_c
        else:
            w_c = v_c
        w_global = w_global + freq * w_c

    # --- Step 2: shared direction basis via SVD of stacked A's ---
    a_stack = torch.cat(a_list, dim=0)  # (n_clients * r, in_features)
    _, s, vh = torch.linalg.svd(a_stack, full_matrices=False)
    actual_r = min(r, s.shape[0])  # always == r here; see note in plan
    a_new = vh[:actual_r, :]

    # --- Step 3: fix A_new, closed-form least-squares solve for B_new ---
    m = (w_global - W0) / scaling
    aat = a_new @ a_new.T  # (r, r) Gram matrix, ~= identity
    b_new = torch.linalg.solve(aat, a_new @ m.T).T

    # --- Step 4: new magnitude = row-norm of W_global itself ---
    m_new = torch.linalg.norm(w_global, dim=1) if is_dora_layer else None

    return a_new, b_new, m_new


class FeDoRA(FedAvg):
    """FedAvg variant for PEFT LoRA/DoRA adapters (arXiv:2402.09353).

    Reconstructs each client's full effective weight (base + scaled BA, DoRA-renormalized),
    FedAvg's the reconstructed weights in full weight-space, then re-projects the result back
    onto a shared rank-r LoRA factorization: SVD for the direction (A), closed-form least
    squares for B, and the aggregated weight's own row norms for the new magnitude m.

    Requires every target module to have been built with use_dora=True; raises ValueError at
    construction time otherwise. Also requires target modules whose out/in dimensions are
    >= r -- there is no rank-deficiency fallback; keep `target_modules` restricted accordingly
    (e.g. via pyproject.toml) instead.
    """

    def __init__(self, *, model, cfg=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layers: List[_LoraLayerRef] = build_layer_refs(model)

        if not any(layer.idx_m is not None for layer in self._layers):
            raise ValueError(
                "FeDoRA requires a model built with use_dora=True (found LoRA layers "
                "but zero 'lora_magnitude_vector' keys, i.e. DoRA is not enabled). Set "
                "model.use-dora=true, or switch strategy.aggregation to 'fedavg'."
            )
        # `model` itself is intentionally not retained -- only the lightweight per-layer
        # W0/scaling/r extracted above survives, so the (possibly GPU-resident, quantized)
        # full model is GC-eligible as soon as the caller's reference to it goes away.

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        client_arrays: List[NDArrays] = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        num_examples: List[int] = [fit_res.num_examples for _, fit_res in results]

        aggregated: NDArrays = [None] * len(client_arrays[0])
        layer_overlaps: List[float] = []
        for layer in self._layers:
            a_new, b_new, m_new, overlap = self._aggregate_layer(layer, client_arrays, num_examples)
            aggregated[layer.idx_a] = a_new
            aggregated[layer.idx_b] = b_new
            if layer.idx_m is not None:
                aggregated[layer.idx_m] = m_new
            layer_overlaps.append(overlap)

        # Any index not covered by an indexed LoRA/DoRA layer is a non-LoRA trainable param
        # PEFT still returns from get_peft_model_state_dict -- e.g. glue-nlu's classifier head,
        # auto-added to modules_to_save for TaskType.SEQ_CLS. flowertune_llm's own CAUSAL_LM
        # models have no such params (every trainable array is a LoRA layer), so this was never
        # exercised there; left as `None` here it makes ndarrays_to_parameters's np.save blow up
        # with "Object arrays cannot be saved when allow_pickle=False". Fall back to plain
        # weighted FedAvg for these, matching flora.py's identical fallback for its extra arrays.
        for idx, arr in enumerate(aggregated):
            if arr is None:
                per_client = [([client[idx]], n) for client, n in zip(client_arrays, num_examples)]
                aggregated[idx] = aggregate(per_client)[0]

        parameters_aggregated = ndarrays_to_parameters(aggregated)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Requirement: basis overlap / cosine similarity, FeDoRA -- each client's raw
        # (unrotated) local A_i vs the SVD-derived shared reference A_new computed in
        # aggregate_dora_layer's step 2 (there is no explicit per-client rotation).
        if layer_overlaps:
            metrics_aggregated["fedora_basis_overlap_mean"] = float(np.mean(layer_overlaps))

        return parameters_aggregated, metrics_aggregated

    def _aggregate_layer(
        self,
        layer: _LoraLayerRef,
        client_arrays: List[NDArrays],
        num_examples: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float]:
        total = sum(num_examples)
        freqs = [n / total for n in num_examples]

        dtype_a = client_arrays[0][layer.idx_a].dtype
        dtype_b = client_arrays[0][layer.idx_b].dtype
        dtype_m = client_arrays[0][layer.idx_m].dtype if layer.idx_m is not None else None

        a_list = [torch.from_numpy(arr[layer.idx_a].astype(np.float32, copy=False)) for arr in client_arrays]
        b_list = [torch.from_numpy(arr[layer.idx_b].astype(np.float32, copy=False)) for arr in client_arrays]
        m_list = (
            [torch.from_numpy(arr[layer.idx_m].astype(np.float32, copy=False)) for arr in client_arrays]
            if layer.idx_m is not None
            else None
        )

        a_new, b_new, m_new = aggregate_dora_layer(
            layer.W0, layer.scaling, layer.r, a_list, b_list, m_list, freqs,
        )
        overlap = float(np.mean([mean_subspace_overlap(a_c, a_new) for a_c in a_list]))
        a_out = a_new.numpy().astype(dtype_a, copy=False)
        b_out = b_new.numpy().astype(dtype_b, copy=False)
        m_out = m_new.numpy().astype(dtype_m, copy=False) if m_new is not None else None
        return a_out, b_out, m_out, overlap
