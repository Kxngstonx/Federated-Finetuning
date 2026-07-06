"""FedSVD: FFA-LoRA (share lora_B only) plus periodic SVD re-orthogonalization of (A, B).

Ported from fed-svd/fedmm/fedavg/server.py::Server.aggregate() (the 'ffa' branch) and
fed-svd/misc/utils.py::reinit_lora. lora_A is never trained by clients (frozen client-side,
see client_app.py); the server FedAvg's the client lora_B updates every round, and every
`recalculate_svd_period` rounds (once past `svd_warmup_steps`) re-derives an orthonormal A and
a matching B from the SVD of the current (B @ A) product -- this is the namesake trick, a
re-basing of the rank-r factorization rather than a full-weight merge.

DoRA magnitude vector (m): when model.use-dora=true, m is frozen client-side (client_app.py,
alongside A) and is never advanced by client-side training. Instead the server recomputes it
analytically every round -- m_new = row_norm(W0 + scaling * (B_out @ A_out)) -- from that
round's just-settled A/B (whether or not SVD re-orthogonalization fired this round), discarding
whatever stale m value clients happened to upload. This mirrors FeDoRA's own step-4 formula
(strategies/fedora.py::aggregate_dora_layer) and is exact because svd_reorthogonalize preserves
the B @ A product, so recomputing m from the *post*-reorthogonalization (A, B) gives the same
row norms as the pre-reorthogonalization effective weight would.
"""

from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

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


def svd_reorthogonalize(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Direct port of fed-svd's reinit_lora, as a pure per-layer tensor function.

    a: (r, in_features) float32, b: (out_features, r) float32.
    Returns a new (orthonormal-rows) A and a matching B such that B_new @ A_new == b @ a.
    """
    r = a.shape[0]
    prod = b @ a
    v, s, uh = torch.linalg.svd(prod, full_matrices=False)
    vr = v[:, :r]
    sr = s[:r]
    uhr = uh[:r]
    a_new = uhr
    b_new = vr @ torch.diag(sr)
    return a_new, b_new


class FedSVD(FedAvg):
    """FedAvg variant implementing FedSVD's FFA + periodic re-orthogonalization."""

    def __init__(self, *, model, cfg=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layers: List[_LoraLayerRef] = build_layer_refs(model)
        cfg = cfg or {}
        self._period = int(cfg.get("recalculate_svd_period", 0))
        self._warmup = int(cfg.get("svd_warmup_steps", 0))
        self._current_a: Dict[str, np.ndarray] = {}

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
        do_svd = bool(self._period) and (server_round % self._period == 0) and (server_round > self._warmup)

        for layer in self._layers:
            b_per_client = [([arr[layer.idx_b]], n) for arr, n in zip(client_arrays, num_examples)]
            b_avg = aggregate(b_per_client)[0]

            if layer.name not in self._current_a:
                # Round 1 (or first time seen): seed from any client's untouched, frozen A.
                self._current_a[layer.name] = client_arrays[0][layer.idx_a].copy()
            a_cur = self._current_a[layer.name]

            if do_svd:
                a_new_t, b_new_t = svd_reorthogonalize(
                    torch.from_numpy(a_cur.astype(np.float32, copy=False)),
                    torch.from_numpy(b_avg.astype(np.float32, copy=False)),
                )
                a_out = a_new_t.numpy().astype(a_cur.dtype, copy=False)
                b_out = b_new_t.numpy().astype(b_avg.dtype, copy=False)
                self._current_a[layer.name] = a_out
            else:
                a_out = a_cur
                b_out = b_avg

            aggregated[layer.idx_a] = a_out
            aggregated[layer.idx_b] = b_out
            if layer.idx_m is not None:
                # m is frozen client-side (client_app.py) and never trained locally; the
                # server is the sole source of truth, recomputing it every round from this
                # round's just-settled (a_out, b_out) -- whatever clients uploaded is ignored.
                dtype_m = client_arrays[0][layer.idx_m].dtype
                w_eff = layer.W0 + layer.scaling * (
                    torch.from_numpy(b_out.astype(np.float32, copy=False))
                    @ torch.from_numpy(a_out.astype(np.float32, copy=False))
                )
                m_new = torch.linalg.norm(w_eff, dim=1)
                aggregated[layer.idx_m] = m_new.numpy().astype(dtype_m, copy=False)

        parameters_aggregated = ndarrays_to_parameters(aggregated)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
