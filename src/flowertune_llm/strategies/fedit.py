"""FedIT: plain FedAvg baseline, averaging lora_A and lora_B without any special treatment.

This is a purely nominal subclass -- get_parameters/set_parameters (models.py) already expose
only the flat PEFT (LoRA) state dict, so vanilla flwr.server.strategy.FedAvg weighted-averaging
of that flat list *is* FedIT (the naive LoRA-FedAvg baseline from the FLoRA paper). It exists as
its own named class purely so it can be selected/compared by name alongside FedSVD/FLoRA/FedRot/
FFALoRA, with no behavioral difference from strategy.aggregation = "fedavg".
"""

from flwr.server.strategy import FedAvg


class FedIT(FedAvg):
    """Naive FedAvg over the full LoRA state dict (A and B averaged independently, no special
    handling) -- the "FedIT" baseline named in the FLoRA paper."""

    def __init__(self, *, model=None, cfg=None, **kwargs) -> None:
        del model, cfg  # unused; accepted for signature parity with the other strategy factories
        super().__init__(**kwargs)
