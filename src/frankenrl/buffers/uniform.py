"""Uniform-sampling replay buffer for off-policy agents (SAC, TD3, M1, ...).

Pre-allocated NumPy ring buffer - the legacy `deque` version re-`np.stack`ed every sample,
which dominated CPU time in the FYP runs.
"""

from __future__ import annotations

import numpy as np
import torch

from frankenrl.buffers.base import Batch, Transition


class UniformReplay:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        *,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)

        self._state = np.zeros((capacity, state_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros((capacity, 1), dtype=np.float32)
        self._next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self._done = np.zeros((capacity, 1), dtype=np.float32)
        self._trunc = np.zeros((capacity, 1), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def add(self, t: Transition) -> None:
        i = self._ptr
        self._state[i] = t.state
        self._action[i] = t.action
        self._reward[i] = t.reward
        self._next_state[i] = t.next_state
        self._done[i] = t.done
        self._trunc[i] = t.truncated
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def on_episode_end(self) -> None:  # noqa: D401 - nothing to do for replay
        """No-op; replay does not need episode boundaries."""

    def ready(self, min_size: int) -> bool:
        return self._size >= min_size

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> Batch:
        if not self.ready(batch_size):
            raise RuntimeError(f"replay has {self._size} < batch_size {batch_size}")
        idx = self._rng.integers(0, self._size, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        return Batch(
            state=to(self._state),
            action=to(self._action),
            reward=to(self._reward),
            next_state=to(self._next_state),
            done=to(self._done),
            truncated=to(self._trunc),
        )
