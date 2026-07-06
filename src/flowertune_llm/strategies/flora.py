"""FLoRA: merge each round's exact aggregated LoRA update into a float32 master base weight,
then hand clients a freshly re-initialized rank-r adapter for the next round.

Ported from FederatedLLM/fed_utils/model_aggregation.py (the `stacking=True` branch) and
fed-svd/fedmm/fedavg/server.py::Server._agg_flora(). Since every client in this pipeline trains
a homogeneous rank-r adapter (heterogeneous ranks are out of scope), literal concatenation of
clients' A/B across the rank axis is unnecessary: summing `freq_k * scaling * (B_k @ A_k)`
directly is mathematically identical to (block-stack A's and B's, then multiply) and far
simpler to implement.

Quantization design (see plan for the empirical justification): the server keeps `self._w0` as
a float32 CPU tensor per layer that is *never* itself derived from a quantized/dequantized
round-trip after the initial read -- every round it is updated by an exact float32 addition.
The broadcast `Parameters` therefore carry, beyond the usual flat LoRA state dict (length L),
one extra float32 array per LoRA layer (in `index_lora_layers(model)` order) holding the
current master `w0`. The client (models.py::set_parameters) re-quantizes this master into a
fresh `Params4bit` every round using the *same* quantization arguments (blocksize/quant_type/
compress_statistics/quant_storage) the model was originally loaded with -- never hardcoded,
always read off the model's own existing quantized buffer. Clients never send `w0` back; the
server's own master is authoritative and clients only ever return the L-length LoRA state dict.

DoRA magnitude vector (m): when model.use-dora=true, m is trained normally client-side (no
freezing) and, unlike lora_A/lora_B, is *not* part of the merge-and-reset cycle -- it is not
folded into the master base weight and not reset each round. Instead it is treated as an
ordinary federated parameter and combined via plain data-size-weighted FedAvg (the same
`aggregate()` helper FedSVD/FFALoRA use for lora_B), persisting and continuing to train across
rounds independently of each round's freshly re-initialized (A, B) pair.
"""

import math
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


class FLoRA(FedAvg):
    """FedAvg variant implementing FLoRA's exact-reconstruction merge-and-reset aggregation."""

    def __init__(self, *, model, cfg=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layers: List[_LoraLayerRef] = build_layer_refs(model)
        self._w0: Dict[str, torch.Tensor] = {layer.name: layer.W0.clone() for layer in self._layers}

    def initialize_parameters(self, client_manager):
        params = super().initialize_parameters(client_manager)
        if params is None:
            return None
        arrays = parameters_to_ndarrays(params)
        extra = [self._w0[layer.name].numpy().astype(np.float32, copy=False) for layer in self._layers]
        return ndarrays_to_parameters(arrays + extra)

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

        # Clients only ever return the plain L-length LoRA state dict (no w0 echoed back).
        client_arrays: List[NDArrays] = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        num_examples: List[int] = [fit_res.num_examples for _, fit_res in results]
        total = sum(num_examples)
        freqs = [n / total for n in num_examples]

        aggregated_lora: NDArrays = [None] * len(client_arrays[0])
        extra_w0_arrays: List[np.ndarray] = []

        for layer in self._layers:
            a_list = [torch.from_numpy(arr[layer.idx_a].astype(np.float32, copy=False)) for arr in client_arrays]
            b_list = [torch.from_numpy(arr[layer.idx_b].astype(np.float32, copy=False)) for arr in client_arrays]

            delta_w = torch.zeros_like(self._w0[layer.name])
            for freq, a_c, b_c in zip(freqs, a_list, b_list):
                delta_w = delta_w + freq * layer.scaling * (b_c @ a_c)
            # Exact float32 addition -- the master is never derived from a quantized value.
            self._w0[layer.name] = self._w0[layer.name] + delta_w

            out_features, in_features = self._w0[layer.name].shape
            r = layer.r
            a_new = torch.empty(r, in_features, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(a_new, a=math.sqrt(5))
            b_new = torch.zeros(out_features, r, dtype=torch.float32)

            dtype_a = client_arrays[0][layer.idx_a].dtype
            dtype_b = client_arrays[0][layer.idx_b].dtype
            aggregated_lora[layer.idx_a] = a_new.numpy().astype(dtype_a, copy=False)
            aggregated_lora[layer.idx_b] = b_new.numpy().astype(dtype_b, copy=False)
            extra_w0_arrays.append(self._w0[layer.name].numpy().astype(np.float32, copy=False))

            if layer.idx_m is not None:
                # m is trained normally client-side and is NOT part of the merge-and-reset
                # cycle -- just a normal federated parameter, data-size-weighted averaged.
                m_per_client = [([arr[layer.idx_m]], n) for arr, n in zip(client_arrays, num_examples)]
                aggregated_lora[layer.idx_m] = aggregate(m_per_client)[0]

        parameters_aggregated = ndarrays_to_parameters(aggregated_lora + extra_w0_arrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
