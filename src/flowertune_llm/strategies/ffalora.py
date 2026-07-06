"""FFALoRA: freeze lora_A at its initial Kaiming-uniform value forever, train/share lora_B only.

Unlike FedSVD (strategies/fedsvd.py), there is no periodic SVD re-orthogonalization here --
this is the plain "Freeze-A" baseline. lora_A is frozen client-side (client_app.py) at model
construction and never revisited; the server only ever needs to weighted-average lora_B.

DoRA magnitude vector (m): when model.use-dora=true, m is likewise frozen client-side
(alongside A) at its PEFT-default init (== row_norm(W0) at round 0, since B starts at zero).
Unlike FedSVD, m is never recomputed here -- the server seeds it once from whichever value the
first round's clients report and passes that same cached value through, unchanged, forever
(mirroring exactly how lora_A itself is already cached and passed through below).
"""

from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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

from flowertune_llm.peft_layers import LoraLayerIndex, index_lora_layers


class FFALoRA(FedAvg):
    """FedAvg variant that only ever aggregates lora_B; lora_A is passed through untouched."""

    def __init__(self, *, model, cfg=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layers: List[LoraLayerIndex] = index_lora_layers(model)
        if not self._layers:
            raise ValueError(
                "FFALoRA found no LoRA target-module parameters in the provided model."
            )
        self._current_a: Dict[str, np.ndarray] = {}
        self._current_m: Dict[str, np.ndarray] = {}

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
        for layer in self._layers:
            b_per_client = [([arr[layer.idx_b]], n) for arr, n in zip(client_arrays, num_examples)]
            aggregated[layer.idx_b] = aggregate(b_per_client)[0]

            if layer.name not in self._current_a:
                # lora_A is frozen client-side at its round-0 Kaiming-uniform init and never
                # trained, so every client returns the identical array -- seed once, keep forever.
                self._current_a[layer.name] = client_arrays[0][layer.idx_a].copy()
            aggregated[layer.idx_a] = self._current_a[layer.name]

            if layer.idx_m is not None:
                if layer.name not in self._current_m:
                    # Round 1: seed from any client's frozen, PEFT-default-initialized m.
                    self._current_m[layer.name] = client_arrays[0][layer.idx_m].copy()
                aggregated[layer.idx_m] = self._current_m[layer.name]

        parameters_aggregated = ndarrays_to_parameters(aggregated)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
