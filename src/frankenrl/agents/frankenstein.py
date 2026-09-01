"""The composable hybrid agent - the whole point of the package.

STATUS: stub. Land after SAC + TD3 + PPO references exist (PLAN.md step 4).

One class, configured by ``AgentConfig`` fields, reproducing the FYP M-series:

    part            options                                     legacy origin
    --------------  ------------------------------------------  ---------------------------
    actor           squashed_gaussian | deterministic          all
    critics         twin_q (+ optional V baseline)             all
    target_rule     soft (tau) [+ target_policy_noise for DPG]  M1 (TD3-style), M3*
    advantage       onestep | tderror | a2c | expected_sarsa   M3_* replay estimators
                    | mc | gae                                  M2_MC / M2_GAE (trajectory)
    policy_loss     sac (min-Q - alpha*logp)                    M1
                    | dpg (-Q(s, pi(s)))                        TD3-flavoured
                    | ppo_clip (clipped surrogate on adv)       Phase 6 (M6_*)
    policy_delay    int                                         M1/M3 delayed actor
    layernorm       bool                                        LayerNormP* line

Named presets (become ``configs/*.yaml``):
    m1_hybrid       actor=squashed_gaussian, critics=twin_q, advantage=onestep,
                    policy_loss=sac, policy_delay=2, target_rule=soft
                    -> should match reference SAC under a controlled seed
                       (verify: [[SAC equals M1 under a controlled seed]])
    m3_a2c          m1_hybrid + advantage=a2c        (adv = min_Q(s,pi) - V(s))
    m3_tderror      m1_hybrid + advantage=tderror    (adv = min_Q(s,pi) - min_Q_targ(s,pi))
    m3_expsarsa     m1_hybrid + advantage=expected_sarsa (E_a[Q] over n_expected_samples)
    m3_mc           m1_hybrid + advantage=mc         (trajectory buffer + burst updates)
    m3_gae          m1_hybrid + advantage=gae        (trajectory buffer)
    m6_ppo_gae      m3_gae + policy_loss=ppo_clip

Implementation notes:
* if ``advantage.is_trajectory`` -> use ``TrajectoryBuffer`` + ``compute_trajectory_targets``
  in ``on_episode_end`` (+ N burst ``update`` calls, matching the legacy runner);
  else -> ``UniformReplay`` and compute the advantage inside ``_learn`` from the sampled
  ``Batch`` and the agent's own critics.
* the critic update stays SAC-style for every replay variant (that was the FYP design);
  only the *policy*-gradient weight changes with ``advantage``.
* a detached baseline contributes zero gradient - keep baselines under ``no_grad`` /
  ``.detach()`` ([[A detached baseline contributes zero gradient]]).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from frankenrl.agents.base import Agent
from frankenrl.buffers.base import Transition
from frankenrl.config import AgentConfig


class Frankenstein(Agent):
    def __init__(self, state_dim: int, action_dim: int, action_scale: float, cfg: AgentConfig, **_: Any) -> None:
        raise NotImplementedError("Frankenstein agent not implemented yet - see PLAN.md step 4")

    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray: ...
    def observe(self, transition: Transition, *, log_prob: float = 0.0) -> None: ...
    def on_episode_end(self) -> None: ...
    def update(self, step: int) -> dict[str, float]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
