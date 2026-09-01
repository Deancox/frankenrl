"""Network building blocks - one home for what the legacy `ActorCritics.py` copies did."""

from frankenrl.nn.actors import DeterministicActor, SquashedGaussianActor
from frankenrl.nn.blocks import mlp
from frankenrl.nn.critics import QCritic, TwinQCritic, VCritic

__all__ = [
    "mlp",
    "SquashedGaussianActor",
    "DeterministicActor",
    "QCritic",
    "TwinQCritic",
    "VCritic",
]
