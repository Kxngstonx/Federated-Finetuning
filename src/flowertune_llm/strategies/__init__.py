"""Registry of pluggable federated LoRA aggregation strategies.

Extends `server_app.py::server_fn`'s previous `if/elif` on `strategy.aggregation`. To add a new
strategy: implement a `Strategy` subclass accepting `(*, model, cfg, **strategy_kwargs)` in its
own module under `strategies/`, then register it here.
"""

from typing import Callable, Dict

from flwr.server.strategy import FedAvg

from flowertune_llm.strategies.fedit import FedIT
from flowertune_llm.strategies.fedora import FeDoRA
from flowertune_llm.strategies.fedrot import FedRot
from flowertune_llm.strategies.fedsvd import FedSVD
from flowertune_llm.strategies.ffalora import FFALoRA
from flowertune_llm.strategies.flora import FLoRA

STRATEGY_REGISTRY: Dict[str, Callable[..., FedAvg]] = {
    "fedavg": lambda *, model=None, cfg=None, **kwargs: FedAvg(**kwargs),
    "fedora": lambda *, model, cfg=None, **kwargs: FeDoRA(model=model, cfg=cfg, **kwargs),
    "fedsvd": lambda *, model, cfg=None, **kwargs: FedSVD(model=model, cfg=cfg, **kwargs),
    "flora": lambda *, model, cfg=None, **kwargs: FLoRA(model=model, cfg=cfg, **kwargs),
    "fedrot": lambda *, model=None, cfg=None, **kwargs: FedRot(model=model, cfg=cfg, **kwargs),
    "fedit": lambda *, model=None, cfg=None, **kwargs: FedIT(model=model, cfg=cfg, **kwargs),
    "ffalora": lambda *, model, cfg=None, **kwargs: FFALoRA(model=model, cfg=cfg, **kwargs),
}


def build_strategy(name: str, *, model, cfg=None, **strategy_kwargs) -> FedAvg:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy.aggregation={name!r}; expected one of {sorted(STRATEGY_REGISTRY)}."
        )
    return STRATEGY_REGISTRY[name](model=model, cfg=cfg, **strategy_kwargs)
