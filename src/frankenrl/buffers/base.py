"""Shared types for buffers."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np
import torch


class Transition(NamedTuple):
    """One environment step, as float arrays (batchless)."""

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float          # 1.0 only on a true terminal, NOT on time-limit truncation
    truncated: float = 0.0


class Batch(NamedTuple):
    """A minibatch of transitions as tensors on the target device.

    Optional fields are filled by the on-policy path (advantage estimators write them):
    ``ret`` = target for the value function, ``adv`` = advantage for the policy,
    ``log_prob`` = behaviour-policy log-prob (for PPO importance ratios).
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor
    ret: torch.Tensor | None = None
    adv: torch.Tensor | None = None
    log_prob: torch.Tensor | None = None


class Buffer(Protocol):
    """Uniform interface the training loop depends on."""

    def add(self, t: Transition) -> None: ...
    def on_episode_end(self) -> None: ...
    def ready(self, min_size: int) -> bool: ...
    def __len__(self) -> int: ...
