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
from fedbench_common.subspace_metrics import mean_subspace_overlap, overlap_to_misalign_deg

_EPS = 1e-8


def aggregate_dora_layer(
    W0: torch.Tensor,  # (out_features, in_features) float32 CPU
    scaling: float,
    r: int,
    a_list: List[torch.Tensor],  # each (r, in_features) float32
    b_list: List[torch.Tensor],  # each (out_features, r) float32
    m_list: Optional[List[torch.Tensor]],  # each (out_features,) float32, or None
    freqs: List[float],  # per-client weight, sums to 1.0
    diagnostics: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
    """Pure-tensor implementation of the 4-step DoRA-aware layer aggregation. All
    inputs/outputs are float32 CPU tensors; numpy/dtype conversion happens in the caller.

    Requires target modules whose out/in dimensions are >= r -- callers must keep
    `target_modules` restricted accordingly (e.g. via pyproject.toml).

    `diagnostics`, when True, additionally computes and returns (as a 4th tuple element, a dict)
    step 3's Gram-matrix condition number, a row-wise direction-cosine check of the rank-r
    reconstruction against the true aggregated weight, and each client's raw-A overlap with the
    new shared basis -- collapse-investigation instrumentation, off by default so normal runs pay
    none of this cost."""
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

    diag = None
    if diagnostics:
        # aat is a symmetric PSD (r, r) Gram matrix -- eigvalsh is the appropriate, cheaper
        # conditioning measure here (vs. a generic SVD-based cond()).
        eigvals = torch.linalg.eigvalsh(aat).clamp_min(_EPS)
        aat_cond = float((eigvals.max() / eigvals.min()).item())

        # Row-wise direction cosine between the reconstructed direction (W0 + scaling*B_new@A_new)
        # and the true aggregated weight w_global -- NOT a raw ||m - recon|| residual, because `m`
        # is contaminated by DoRA's per-row magnitude renormalization in step 1: w_c =
        # diag(m_c/||v_c||_row) @ (W0 + scaling*B_c@A_c) means w_global - W0 decomposes into a
        # low-rank "direction" term PLUS (D - I) @ W0 (D := the weighted-average per-row magnitude
        # ratio, a diagonal matrix). Row-scaling preserves row space, so that second term has
        # exactly W0's own row space -- generically ~full rank for a real linear layer weight,
        # utterly unreachable by any rank-r a_new regardless of how good the basis is. A raw
        # ||m - recon||/||m|| ratio is dominated by that unreachable term and sits pinned near 1.0
        # whether the round is healthy or collapsing (confirmed empirically: stayed at
        # 1.000003-1.000012 through an actual collapse event). Cosine similarity is scale-invariant
        # per row, so it cancels out the (D-I)@W0 magnitude-scale contamination automatically and
        # isolates just the question step 3 is actually responsible for: does the reconstructed
        # direction point the same way as the true averaged weight, row by row.
        v_new = W0 + scaling * (b_new @ a_new)
        target_norm = w_global.norm(dim=1).clamp_min(_EPS)
        recon_norm = v_new.norm(dim=1).clamp_min(_EPS)
        row_cosine = (w_global * v_new).sum(dim=1) / (target_norm * recon_norm)
        direction_cosine_mean = float(row_cosine.mean().item())
        direction_cosine_min = float(row_cosine.min().item())

        per_client_overlap = [mean_subspace_overlap(a_c, a_new) for a_c in a_list]

        diag = {
            "aat_cond": aat_cond,
            "direction_cosine_mean": direction_cosine_mean,
            "direction_cosine_min": direction_cosine_min,
            "per_client_overlap": per_client_overlap,
        }

    return a_new, b_new, m_new, diag


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
        self._debug_diagnostics: bool = bool((cfg or {}).get("debug_diagnostics", False))

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
        client_ids: List[str] = [proxy.cid for proxy, _ in results]

        aggregated: NDArrays = [None] * len(client_arrays[0])
        layer_overlaps: List[float] = []
        layer_overlaps_min: List[float] = []
        layer_aat_cond: List[float] = []
        layer_direction_cosine: List[float] = []
        layer_direction_cosine_min: List[float] = []
        layer_per_client_overlap: List[List[float]] = []
        for layer in self._layers:
            a_new, b_new, m_new, overlap_mean, overlap_min, diag = self._aggregate_layer(
                layer, client_arrays, num_examples
            )
            aggregated[layer.idx_a] = a_new
            aggregated[layer.idx_b] = b_new
            if layer.idx_m is not None:
                aggregated[layer.idx_m] = m_new
            layer_overlaps.append(overlap_mean)
            layer_overlaps_min.append(overlap_min)
            if diag is not None:
                layer_aat_cond.append(diag["aat_cond"])
                layer_direction_cosine.append(diag["direction_cosine_mean"])
                layer_direction_cosine_min.append(diag["direction_cosine_min"])
                layer_per_client_overlap.append(diag["per_client_overlap"])

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
        #
        # Raw cosine overlap is logged too (for continuity with earlier runs' CSVs/plots), but
        # it's a poor scale to eyeball for anomalies: it saturates near 1.0, so e.g. the
        # collapse-triggering QNLI rounds observed in 2026-07-14's sweep showed overlap dropping
        # only 0.999999 -> ~0.998 (looks tiny) which is actually ~0.08deg -> ~3.2deg of true
        # subspace misalignment (~40x), because arccos is flat near overlap=1. Logging the
        # degree-transform (and the worst single layer, not just the cross-layer mean, since one
        # bad layer can get diluted into ~148 healthy ones) makes that kind of one-round
        # aggregation-triggered collapse visible in the metrics instead of hiding in the noise
        # floor of a mean-of-cosines column.
        if layer_overlaps:
            overlap_mean_all = float(np.mean(layer_overlaps))
            overlap_min_all = float(np.min(layer_overlaps_min))
            metrics_aggregated["fedora_basis_overlap_mean"] = overlap_mean_all
            metrics_aggregated["fedora_basis_overlap_min"] = overlap_min_all
            metrics_aggregated["fedora_basis_misalign_deg_mean"] = overlap_to_misalign_deg(overlap_mean_all)
            metrics_aggregated["fedora_basis_misalign_deg_max"] = overlap_to_misalign_deg(overlap_min_all)

        # Opt-in collapse-investigation diagnostics (strategy.fedora.debug-diagnostics=true) --
        # off by default since per-client breakdowns would otherwise need to be recomputed and
        # logged on every production run for no benefit. `aat_cond` tests (and is expected to
        # rule out) step 3's closed-form solve being numerically ill-conditioned. `direction_cosine`
        # tests whether the SVD-derived shared basis fails to capture the true aggregated weight's
        # *direction* -- deliberately NOT a raw ||m - recon||/||m|| residual, since `m` is
        # contaminated by DoRA's per-row magnitude renormalization (see aggregate_dora_layer's
        # comment): w_global - W0 has a (D-I)@W0 component whose row space equals W0's own
        # (generically ~full rank), structurally unreachable by any rank-r a_new regardless of
        # basis quality, which pinned a raw residual at ~1.0 even during an actual collapse when
        # first tried. Row-wise cosine similarity is scale-invariant, so it cancels that
        # magnitude-scale contamination and isolates just what step 3 is responsible for.
        # `client_overlap_mean/_min` are per-client (not cross-client-mean) so a single outlier
        # client distorting the unweighted SVD stack (step 2 doesn't weight by `freqs` the way
        # step 1's FedAvg does) becomes visible instead of averaged away into ~148 healthy layers
        # times 2 healthy clients.
        if self._debug_diagnostics and layer_aat_cond:
            metrics_aggregated["fedora_aat_cond_mean"] = float(np.mean(layer_aat_cond))
            metrics_aggregated["fedora_aat_cond_max"] = float(np.max(layer_aat_cond))
            metrics_aggregated["fedora_direction_cosine_mean"] = float(np.mean(layer_direction_cosine))
            metrics_aggregated["fedora_direction_cosine_min"] = float(np.min(layer_direction_cosine_min))

            per_client_arr = np.array(layer_per_client_overlap)  # (num_layers, num_clients)
            client_overlap_mean = per_client_arr.mean(axis=0)
            client_overlap_min = per_client_arr.min(axis=0)
            metrics_aggregated["fedora_client_overlap_mean"] = {
                cid: float(v) for cid, v in zip(client_ids, client_overlap_mean)
            }
            metrics_aggregated["fedora_client_overlap_min"] = {
                cid: float(v) for cid, v in zip(client_ids, client_overlap_min)
            }

        return parameters_aggregated, metrics_aggregated

    def _aggregate_layer(
        self,
        layer: _LoraLayerRef,
        client_arrays: List[NDArrays],
        num_examples: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float, float, Optional[dict]]:
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

        a_new, b_new, m_new, diag = aggregate_dora_layer(
            layer.W0, layer.scaling, layer.r, a_list, b_list, m_list, freqs,
            diagnostics=self._debug_diagnostics,
        )
        client_overlaps = [mean_subspace_overlap(a_c, a_new) for a_c in a_list]
        overlap_mean = float(np.mean(client_overlaps))
        overlap_min = float(np.min(client_overlaps))
        a_out = a_new.numpy().astype(dtype_a, copy=False)
        b_out = b_new.numpy().astype(dtype_b, copy=False)
        m_out = m_new.numpy().astype(dtype_m, copy=False) if m_new is not None else None
        return a_out, b_out, m_out, overlap_mean, overlap_min, diag
