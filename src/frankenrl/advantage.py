"""Advantage / return estimators - the axis the Frankenstein variants vary along.

Two families:

* **trajectory** estimators (``mc``, ``gae``) are pure functions of a rollout plus a
  state-value function. Fully implemented and unit-tested here.
* **replay** estimators (``onestep``, ``tderror``, ``a2c``, ``expected_sarsa``) run inside
  an off-policy agent's policy update, using its critics. They are declared here as specs;
  the maths lives in ``agents/frankenstein.py`` because it needs the agent's networks.
  Legacy references: ``fyp-corpus-raw/Python/Frankensteins/Phase{2,3,5}/P*FrankensteinAgents.py``.

An estimator is selected by name in the config: ``advantage: {name: gae, gamma: 0.99,
gae_lambda: 0.95}``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

TRAJECTORY_ESTIMATORS = ("mc", "gae")
REPLAY_ESTIMATORS = ("onestep", "tderror", "a2c", "expected_sarsa")
ALL_ESTIMATORS = TRAJECTORY_ESTIMATORS + REPLAY_ESTIMATORS

ValueFn = Callable[[np.ndarray], np.ndarray]  # states (T, S) -> values (T, 1)


@dataclass(frozen=True)
class AdvantageSpec:
    name: str
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_expected_samples: int = 10  # expected_sarsa only
    normalize: bool = True        # standardise advantages before the policy loss

    def __post_init__(self) -> None:
        if self.name not in ALL_ESTIMATORS:
            raise ValueError(f"unknown advantage {self.name!r}; have {ALL_ESTIMATORS}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")

    @property
    def is_trajectory(self) -> bool:
        return self.name in TRAJECTORY_ESTIMATORS


def _episode_slices(done: np.ndarray, truncated: np.ndarray) -> list[slice]:
    """Split a flat rollout into per-episode slices at ``done`` or ``truncated`` steps."""
    ends = np.nonzero((done.reshape(-1) > 0) | (truncated.reshape(-1) > 0))[0]
    slices, start = [], 0
    for e in ends:
        slices.append(slice(start, e + 1))
        start = e + 1
    if start < len(done):  # trailing partial episode
        slices.append(slice(start, len(done)))
    return slices


def mc_targets(cols: dict[str, np.ndarray], value_fn: ValueFn, spec: AdvantageSpec):
    """Monte-Carlo return ``G_t`` and advantage ``G_t - V(s_t)``.

    Returns are computed within episode boundaries; a truncated (not terminal) tail is
    bootstrapped with ``V(s_last')`` so we do not pretend a time-limit cut is a real
    terminal (legacy bug #5 in PLAN.md).
    """
    rewards = cols["reward"].reshape(-1)
    done = cols["done"].reshape(-1)
    truncated = cols["truncated"].reshape(-1)
    next_state = cols["next_state"]
    gamma = spec.gamma

    ret = np.zeros_like(rewards)
    for sl in _episode_slices(done, truncated):
        g = 0.0
        if truncated[sl.stop - 1] > 0 and done[sl.stop - 1] == 0:
            g = float(value_fn(next_state[sl.stop - 1: sl.stop]).reshape(-1)[0])
        for i in range(sl.stop - 1, sl.start - 1, -1):
            g = rewards[i] + gamma * g
            ret[i] = g

    values = value_fn(cols["state"]).reshape(-1)
    adv = ret - values
    return ret.reshape(-1, 1), _maybe_normalize(adv, spec).reshape(-1, 1)


def gae_targets(cols: dict[str, np.ndarray], value_fn: ValueFn, spec: AdvantageSpec):
    """Generalised Advantage Estimation. ``ret = adv + V(s)``.

    See the vault note [[GAE]].
    """
    rewards = cols["reward"].reshape(-1)
    done = cols["done"].reshape(-1)
    truncated = cols["truncated"].reshape(-1)
    gamma, lam = spec.gamma, spec.gae_lambda

    values = value_fn(cols["state"]).reshape(-1)
    next_values = value_fn(cols["next_state"]).reshape(-1)

    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - done[t]                    # bootstrap unless a true terminal
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        if truncated[t] > 0:                           # episode boundary: reset the recursion
            gae = delta
        adv[t] = gae
    ret = adv + values
    return ret.reshape(-1, 1), _maybe_normalize(adv, spec).reshape(-1, 1)


TRAJECTORY_FN: dict[str, Callable] = {"mc": mc_targets, "gae": gae_targets}


def _maybe_normalize(adv: np.ndarray, spec: AdvantageSpec) -> np.ndarray:
    if spec.normalize and adv.size > 1:
        return (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv


def compute_trajectory_targets(cols, value_fn, spec: AdvantageSpec):
    if not spec.is_trajectory:
        raise ValueError(f"{spec.name!r} is a replay estimator; handled inside the agent")
    return TRAJECTORY_FN[spec.name](cols, value_fn, spec)
