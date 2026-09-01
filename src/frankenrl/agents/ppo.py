"""Proximal Policy Optimisation - reference on-policy agent.

STATUS: stub. Land after the SAC vertical slice (PLAN.md step 3).

Shape when implemented:
* ``SquashedGaussianActor`` + ``VCritic`` (shared ``NetConfig``).
* ``TrajectoryBuffer``; collect ``cfg.rollout_steps`` transitions, then update.
* ``on_episode_end`` / rollout-full -> ``advantage.gae_targets`` to fill ``ret`` / ``adv``.
* ``cfg.ppo_epochs`` passes of minibatched clipped-surrogate + value MSE + entropy bonus,
  ratio = ``exp(logp_now - batch.log_prob)``, clip ``cfg.ppo_clip``.

Reference: [[PPO]], [[GAE]], and `fyp-corpus-raw/Python/Frankensteins/Standard/Standard.py::PPOAgent`
(with legacy bug #4/#5 from PLAN.md fixed).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from frankenrl.agents.base import Agent
from frankenrl.buffers.base import Transition
from frankenrl.config import AgentConfig


class PPO(Agent):
    def __init__(self, state_dim: int, action_dim: int, action_scale: float, cfg: AgentConfig, **_: Any) -> None:
        raise NotImplementedError("PPO agent not implemented yet - see PLAN.md step 3")

    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray: ...
    def observe(self, transition: Transition, *, log_prob: float = 0.0) -> None: ...
    def update(self, step: int) -> dict[str, float]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
