"""FedRot: client-side Orthogonal Procrustes rotation-alignment of LoRA (A, B) before upload,
so that averaging the rotated factors doesn't suffer destructive interference from clients'
factors living in different (but BA-equivalent) rotational coordinate frames.

Ported from FedRot-LoRA/federatedscope/rotation_alignment_tools.py::rotation_align_optimization
(framework-agnostic pure PyTorch) and the alternating-reference logic in
FedRot-LoRA/federatedscope/core/workers/client.py. Only the repo's default "shareAB" mode is
supported: both A and B are always shared, rotated first. The rotation itself happens entirely
client-side inside client_app.py::FlowerClient.fit (using the just-received global parameters as
the alignment reference -- no extra communication or cross-round state needed).

Server-side aggregation is a **plain, unweighted arithmetic mean** across selected clients --
deliberately *not* data-size-weighted FedAvg (a chosen deviation from the source repo's own
FederatedScope aggregator, which does weight by client dataset size). This applies uniformly to
every array this pipeline's clients report -- lora_A, lora_B, and (when model.use-dora=true)
the DoRA magnitude vector m, which is trained normally client-side (no freezing) and simply
carried along by the same uniform average, with no special-casing needed.
"""

import pickle
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

from fedbench_common.subspace_metrics import mean_pairwise_subspace_overlap

_PREROT_A_METRIC_PREFIX = "fedrot_prerot_A::"


class FedRot(FedAvg):
    """FedAvg variant that replaces data-size-weighted averaging with a plain unweighted
    arithmetic mean -- FedRot's rotation step happens client-side before upload (see
    client_app.py::FlowerClient.fit); this class only changes how the (already rotated)
    client arrays are combined."""

    def __init__(self, *, model=None, cfg=None, **kwargs) -> None:
        del cfg  # unused; accepted for signature parity with the other strategy factories
        super().__init__(**kwargs)
        # Needed only to recover each LoRA layer's idx_a/name for the basis-overlap metric below;
        # the rotation itself (client_app.py::FlowerClient._maybe_rotate) doesn't need the model
        # server-side at all.
        from flowertune_llm.peft_layers import index_lora_layers

        self._layers = index_lora_layers(model) if model is not None else None

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

        aggregated: NDArrays = []
        for i in range(len(client_arrays[0])):
            dtype = client_arrays[0][i].dtype
            stacked = np.stack([arr[i].astype(np.float32, copy=False) for arr in client_arrays], axis=0)
            aggregated.append(stacked.mean(axis=0).astype(dtype, copy=False))

        parameters_aggregated = ndarrays_to_parameters(aggregated)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        if self._layers:
            overlap_metrics = self._basis_overlap_metrics(client_arrays, results)
            metrics_aggregated.update(overlap_metrics)

        return parameters_aggregated, metrics_aggregated

    def _basis_overlap_metrics(
        self,
        client_arrays: List[NDArrays],
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> "dict[str, Scalar]":
        """Requirement: basis overlap / cosine similarity, FedRot only. Compares client-pair
        A_i/A_j subspaces both post-rotation (already present in client_arrays, the arrays this
        round's aggregate averages) and pre-rotation (each client additionally serializes its
        pre-rotation A into its fit() metrics dict under _PREROT_A_METRIC_PREFIX + layer.name --
        see client_app.py::FlowerClient._maybe_rotate/fit -- and skips this on round 1, since
        there's no rotation reference yet, so pre-rotation overlap is only reported from round 2
        onward). Averaged across layers; per-layer detail is dropped from the top-level scalar
        metrics dict to keep it small, matching the other strategies' metrics granularity."""
        post_overlaps, pre_overlaps = [], []
        for layer in self._layers:
            post_subspaces = [arr[layer.idx_a] for arr in client_arrays]
            post_overlaps.append(mean_pairwise_subspace_overlap(post_subspaces))

            prerot_key = f"{_PREROT_A_METRIC_PREFIX}{layer.name}"
            pre_subspaces = []
            for _, fit_res in results:
                raw = fit_res.metrics.get(prerot_key)
                if raw is None:
                    break
                pre_subspaces.append(pickle.loads(raw))
            if len(pre_subspaces) == len(client_arrays):
                pre_overlaps.append(mean_pairwise_subspace_overlap(pre_subspaces))

        metrics: "dict[str, Scalar]" = {"fedrot_basis_overlap_post": float(np.mean(post_overlaps))}
        if pre_overlaps:
            metrics["fedrot_basis_overlap_pre"] = float(np.mean(pre_overlaps))
        return metrics


def rotation_align_optimization(
    ref: torch.Tensor,
    align_matrix: str,
    updated_a: torch.Tensor,
    updated_b: torch.Tensor,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Solve the Orthogonal Procrustes problem aligning (updated_a, updated_b) to `ref`
    (the client's own previous-round global A or B), and apply the resulting rotation to both
    factors. Preserves the product `updated_b @ updated_a` exactly.

    align_matrix: 'A' or 'B' -- which of the two just-trained factors is compared against `ref`
    to solve for the rotation (the other factor is rotated by the same R for consistency).

    Includes a determinant-flip guard (absent in the source repo's hard-rotation path, present
    only in its soft-rotation variant) to avoid applying a reflection instead of a proper
    rotation when the raw SVD solution has det(U @ Vh) < 0.
    """
    a_dtype, b_dtype = updated_a.dtype, updated_b.dtype

    with torch.no_grad():
        if align_matrix == "A":
            m = torch.matmul(updated_a.to(torch.float32), ref.to(torch.float32).T)
        elif align_matrix == "B":
            m = torch.matmul(updated_b.to(torch.float32).T, ref.to(torch.float32))
        else:
            raise ValueError("align_matrix must be 'A' or 'B'")

        u, _, vh = torch.linalg.svd(m, full_matrices=False)
        r = torch.matmul(u, vh)

        if torch.linalg.det(r) < 0:
            vh = vh.clone()
            vh[-1, :] *= -1
            r = torch.matmul(u, vh)

        rotated_b = torch.matmul(updated_b, r.to(b_dtype))
        rotated_a = torch.matmul(r.T.to(a_dtype), updated_a)

    return rotated_a, rotated_b
