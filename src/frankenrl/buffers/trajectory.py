"""On-policy rollout buffer for PPO-style / advantage-based updates.

Stores transitions in order plus the behaviour-policy ``log_prob``. Advantage/return
columns are filled by an estimator (see ``frankenrl.advantage``) via ``compute_targets``
before ``get()`` hands back one big ``Batch`` and clears.
"""

from __future__ import annotations

import numpy as np
import torch

from frankenrl.buffers.base import Batch, Transition


class TrajectoryBuffer:
    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self._rows: list[Transition] = []
        self._log_probs: list[float] = []
        self._ret: np.ndarray | None = None
        self._adv: np.ndarray | None = None

    def add(self, t: Transition, *, log_prob: float = 0.0) -> None:
        self._rows.append(t)
        self._log_probs.append(float(log_prob))
        self._ret = self._adv = None  # invalidate any stale targets

    def on_episode_end(self) -> None:  # noqa: D401
        """Episode boundaries are read from ``done``/``truncated`` in ``as_arrays``."""

    def ready(self, min_size: int) -> bool:
        return len(self._rows) >= min_size

    def __len__(self) -> int:
        return len(self._rows)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Column arrays for the advantage estimator. Shapes ``(T, ...)``."""
        if not self._rows:
            raise RuntimeError("trajectory buffer is empty")
        stack = lambda key: np.asarray([getattr(r, key) for r in self._rows], dtype=np.float32)  # noqa: E731
        return {
            "state": np.stack([r.state for r in self._rows]).astype(np.float32),
            "action": np.stack([r.action for r in self._rows]).astype(np.float32),
            "reward": stack("reward").reshape(-1, 1),
            "next_state": np.stack([r.next_state for r in self._rows]).astype(np.float32),
            "done": stack("done").reshape(-1, 1),
            "truncated": stack("truncated").reshape(-1, 1),
            "log_prob": np.asarray(self._log_probs, dtype=np.float32).reshape(-1, 1),
        }

    def set_targets(self, ret: np.ndarray, adv: np.ndarray) -> None:
        n = len(self._rows)
        if ret.shape[0] != n or adv.shape[0] != n:
            raise ValueError(f"targets must have length {n}, got ret={ret.shape} adv={adv.shape}")
        self._ret = ret.astype(np.float32).reshape(-1, 1)
        self._adv = adv.astype(np.float32).reshape(-1, 1)

    def get(self) -> Batch:
        """Return everything as one ``Batch`` and clear. Requires ``set_targets`` first."""
        if self._ret is None or self._adv is None:
            raise RuntimeError("call an advantage estimator's compute_targets() before get()")
        cols = self.as_arrays()
        to = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        batch = Batch(
            state=to(cols["state"]),
            action=to(cols["action"]),
            reward=to(cols["reward"]),
            next_state=to(cols["next_state"]),
            done=to(cols["done"]),
            truncated=to(cols["truncated"]),
            ret=to(self._ret),
            adv=to(self._adv),
            log_prob=to(cols["log_prob"]),
        )
        self.clear()
        return batch

    def clear(self) -> None:
        self._rows.clear()
        self._log_probs.clear()
        self._ret = self._adv = None
