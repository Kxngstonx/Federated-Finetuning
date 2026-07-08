"""Measurement instrumentation shared across the experiment pipelines (apps/glue-nlu,
apps/llm-humaneval, apps/llm-gsm8k): trainable-parameter counts, communication byte accounting,
wall-clock timing, an analytical LoRA FLOPs estimate, and GPU environment info.

None of this existed anywhere in federated-finetuning before this port -- FedRot-LoRA's own
generic FederatedScope Monitor (core/monitors/monitor.py) tracked comm bytes/wall-clock/FLOPs in
a framework-specific way that doesn't carry over to Flower; this module re-implements the same
measurements against Flower's NumPyClient/FedAvg API instead.
"""

import time
from typing import Dict, List, Optional, Tuple, Union

import torch

from flwr.common import FitRes, NDArrays, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from flowertune_llm.peft_layers import index_lora_layers


def trainable_param_count(model) -> int:
    """Requirement: trainable parameter count (per client, per round). Call after model
    construction and after any strategy-specific freezing (e.g. FFALoRA/FedSVD's freeze_A) so
    the count reflects what's actually trained this round."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ndarrays_nbytes(arrays: NDArrays) -> int:
    """Requirement: communication parameter/byte count. Size, in bytes, of a flat NDArrays list
    as it would be serialized over the wire (Flower serializes each ndarray roughly at its raw
    buffer size, so `nbytes` is a tight proxy without needing to actually serialize)."""
    return int(sum(a.nbytes for a in arrays))


class Timer:
    """Requirement: client-side training / server-side aggregation wall-clock time.

    Usage: `with Timer() as t: ...; seconds = t.elapsed`.
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start


def gpu_info() -> Dict[str, Union[str, int]]:
    """Requirement: total wall-clock reporting must explicitly name GPU type/count used."""
    if torch.cuda.is_available():
        return {
            "gpu_type": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
        }
    return {"gpu_type": "cpu", "gpu_count": 0}


def estimate_lora_flops(model, seq_length: int, batch_size: int, local_steps: int) -> int:
    """Requirement: FLOPs. Analytical estimate covering only the trainable LoRA branch of each
    target module (forward + backward), NOT the frozen backbone matmuls or non-LoRA layers
    (attention softmax, layernorm, embeddings, etc.) -- deliberately, to avoid depending on
    fvcore/torch.utils.flop_counter (both fragile against PEFT-wrapped + quantized models). This
    excluded backbone cost is architecture-determined and identical across every strategy being
    compared in these experiments, so it does not affect strategy-to-strategy comparisons; it
    would need to be added back in for absolute (cross-architecture) FLOPs comparisons.

    For a LoRA layer with input dim d_in, output dim d_out, rank r, and T = seq_length *
    batch_size tokens per step:
      forward (LoRA branch only): 2*T*d_in*r + 2*T*r*d_out
      backward (weight grads for A, B + input grad through the LoRA branch): ~2x forward
    Total per step ~= 3 * forward. Multiplied by local_steps (this round's local update steps).
    """
    total_forward = 0
    for layer in index_lora_layers(model):
        module = model.get_submodule(layer.name)
        base = module.base_layer
        d_out, d_in = base.weight.shape[0], base.weight.shape[1]
        r = module.r["default"]
        tokens = seq_length * batch_size
        total_forward += 2 * tokens * d_in * r + 2 * tokens * r * d_out
    return int(3 * total_forward * local_steps)


class InstrumentedStrategy(FedAvg):
    """Wraps any strategy built by flowertune_llm.strategies.build_strategy, adding:
      - server-side aggregation wall-clock time (req: server-side wall-clock)
      - server->client communication bytes, per round and cumulative (req: comm bytes)
      - client->server communication bytes, per round and cumulative, read back out of each
        client's already-reported `upload_bytes` fit metric (see client_app.py in both new apps)

    Delegates every other method to the wrapped strategy unchanged, so strategy-specific metrics
    (e.g. fedrot_basis_overlap_*, flora_quant_error_frob, fedora_v1_basis_overlap_mean -- all
    added directly inside the respective strategy's own aggregate_fit, see strategies/fedrot.py,
    flora.py, fedora.py) simply pass through untouched in the merged metrics dict.
    """

    def __init__(self, inner: FedAvg, metrics_writer=None) -> None:
        # Intentionally do not call FedAvg.__init__: this wrapper only ever delegates to `inner`,
        # never uses its own FedAvg behaviour.
        self._inner = inner
        self._cum_upload_bytes = 0
        self._cum_download_bytes = 0
        self._metrics_writer = metrics_writer  # optional fedbench_common.resultio.MetricsWriter

    def initialize_parameters(self, client_manager: ClientManager):
        return self._inner.initialize_parameters(client_manager)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        instructions = self._inner.configure_fit(server_round, parameters, client_manager)
        if instructions:
            broadcast_bytes = ndarrays_nbytes(parameters_to_ndarrays(instructions[0][1].parameters))
            round_download_bytes = broadcast_bytes * len(instructions)
            self._cum_download_bytes += round_download_bytes
            self._last_round_download_bytes = round_download_bytes
        else:
            self._last_round_download_bytes = 0
        return instructions

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        return self._inner.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        round_upload_bytes = sum(int(res.metrics.get("upload_bytes", 0)) for _, res in results)
        self._cum_upload_bytes += round_upload_bytes

        with Timer() as t:
            parameters_aggregated, metrics_aggregated = self._inner.aggregate_fit(
                server_round, results, failures
            )

        metrics_aggregated = dict(metrics_aggregated)
        metrics_aggregated.update(
            {
                "server_aggregate_seconds": t.elapsed,
                "round_upload_bytes": round_upload_bytes,
                "round_download_bytes": getattr(self, "_last_round_download_bytes", 0),
                "cum_upload_bytes": self._cum_upload_bytes,
                "cum_download_bytes": self._cum_download_bytes,
            }
        )
        if self._metrics_writer is not None:
            self._metrics_writer.write_round(server_round, "fit", **metrics_aggregated)
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round: int, results, failures):
        return self._inner.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: Parameters):
        return self._inner.evaluate(server_round, parameters)

    # FedAvg reads these attributes directly in some code paths (e.g. accept_failures);
    # delegate attribute access for anything not explicitly overridden above.
    def __getattr__(self, name):
        return getattr(self._inner, name)
