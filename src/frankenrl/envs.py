"""Thin Gymnasium wrapper.

Responsibilities kept deliberately small:
* build the env from an ``EnvConfig`` (with optional time-limit override),
* expose ``state_dim`` / ``action_dim`` / ``action_scale``,
* keep ``done`` (true terminal) and ``truncated`` (time limit) separate all the way to the
  buffer - conflating them is a classic value-target bug.

Action scaling: policies output in ``[-1, 1]``; we multiply by ``action_scale`` (a scalar
for now - per-dim high/low support is a TODO once an env needs it).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from frankenrl.config import EnvConfig


@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: float        # true terminal
    truncated: float   # time-limit / out-of-bounds cutoff
    info: dict


class Env:
    def __init__(self, cfg: EnvConfig, *, seed: int | None = None, render: bool = False):
        kwargs: dict = {}
        if cfg.max_episode_steps is not None:
            kwargs["max_episode_steps"] = cfg.max_episode_steps
        if render:
            kwargs["render_mode"] = "human"
        self._env = gym.make(cfg.id, **kwargs)
        self.cfg = cfg
        self._seed = seed

        obs_space = self._env.observation_space
        act_space = self._env.action_space
        if not isinstance(act_space, gym.spaces.Box):
            raise TypeError(f"{cfg.id} has non-Box action space {act_space}; only continuous supported")
        self.state_dim = int(np.prod(obs_space.shape))
        self.action_dim = int(np.prod(act_space.shape))
        self.action_scale = float(cfg.action_scale)

    def reset(self) -> np.ndarray:
        state, _ = self._env.reset(seed=self._seed)
        self._seed = None  # only seed the first reset
        return np.asarray(state, dtype=np.float32)

    def step(self, action: np.ndarray) -> StepResult:
        scaled = np.clip(action, -1.0, 1.0) * self.action_scale
        next_state, reward, terminated, truncated, info = self._env.step(scaled)
        return StepResult(
            next_state=np.asarray(next_state, dtype=np.float32),
            reward=float(reward),
            done=float(terminated),
            truncated=float(truncated),
            info=info,
        )

    def close(self) -> None:
        self._env.close()
