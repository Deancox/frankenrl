"""Agent interface the training loop talks to.

Deliberately tiny so SAC, TD3, PPO and the composed Frankenstein agent all fit:

    act(state, deterministic)  -> np.ndarray in [-1, 1]
    observe(transition)        -> None            (agent owns its buffer)
    on_episode_end()           -> None
    update(step)               -> dict[str, float]  (metrics; empty if it did not learn)
    state_dict() / load_state_dict()
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from frankenrl.buffers.base import Transition


class Agent(abc.ABC):
    @abc.abstractmethod
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        """Return an action in ``[-1, 1]^action_dim`` (env applies ``action_scale``)."""

    @abc.abstractmethod
    def observe(self, transition: Transition, *, log_prob: float = 0.0) -> None:
        """Hand the agent one env step for its buffer."""

    def on_episode_end(self) -> None:
        """Hook for trajectory bootstrapping / burst updates."""

    @abc.abstractmethod
    def update(self, step: int) -> dict[str, float]:
        """Do (up to) one learning iteration. Return metrics; ``{}`` means 'did not learn'."""

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
