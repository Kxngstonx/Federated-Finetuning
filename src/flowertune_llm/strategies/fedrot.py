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


class FedRot(FedAvg):
    """FedAvg variant that replaces data-size-weighted averaging with a plain unweighted
    arithmetic mean -- FedRot's rotation step happens client-side before upload (see
    client_app.py::FlowerClient.fit); this class only changes how the (already rotated)
    client arrays are combined."""

    def __init__(self, *, model=None, cfg=None, **kwargs) -> None:
        del model, cfg  # unused; accepted for signature parity with the other strategy factories
        super().__init__(**kwargs)

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

        return parameters_aggregated, metrics_aggregated


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
