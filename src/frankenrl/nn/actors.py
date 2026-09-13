"""Policy networks.

`SquashedGaussianActor` is the SAC-style actor used by every Frankenstein variant: a
diagonal Gaussian in pre-squash space, `tanh`-squashed into `[-1, 1] * action_scale`, with
the numerically-stable tanh log-det correction

    log(1 - tanh(x)^2) = 2 * (log 2 - x - softplus(-2x))

(the legacy Phase1-2 used `log(1 - a^2 + eps)`, which underflows near the bounds; LayerNormP3
already switched to this form - we standardise on it).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from frankenrl.nn.blocks import mlp

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(nn.Module):
    """Stochastic policy: ``a = action_scale * tanh(mu(s) + sigma(s) * eps)``."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        *,
        action_scale: float = 1.0,
        layernorm: bool = False,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        if action_scale <= 0:
            raise ValueError(f"action_scale must be > 0, got {action_scale}")
        self.action_dim = action_dim
        self.register_buffer("action_scale", torch.as_tensor(float(action_scale)))

        # trunk outputs features; two linear heads produce mu and log_std
        self.trunk = mlp(
            state_dim, hidden[-1], hidden[:-1] or (hidden[-1],),
            activation=activation, layernorm=layernorm,
        )
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)

    def _distribution(self, state: torch.Tensor) -> Normal:
        x = self.trunk(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mu, log_std.exp())

    def distribution(self, state: torch.Tensor) -> Normal:
        """Public pre-tanh Gaussian accessor (e.g. a closed-form KL between two actors).

        KL divergence is invariant under an identical invertible transform applied to both
        sides, so the exact post-tanh-squash KL between two ``SquashedGaussianActor``s that
        share an ``action_scale`` equals this pre-squash Gaussian KL - see BRO's optimistic
        actor loss in [[BRO - scaling off-policy actor-critic RL with regularized critics
        and optimistic exploration]] section 6, and the cross-check against a Monte-Carlo
        estimate in ``tests/test_bro.py``.
        """
        return self._distribution(state)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, sigma)`` of the pre-squash Gaussian (diagnostics / determin. eval)."""
        dist = self._distribution(state)
        return dist.mean, dist.stddev

    def sample(
        self, state: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob)``.

        ``action`` is squashed and scaled. ``log_prob`` has shape ``(..., 1)`` and already
        includes the tanh Jacobian and the ``action_scale`` Jacobian.
        """
        dist = self._distribution(state)
        pre_tanh = dist.mean if deterministic else dist.rsample()

        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale

        # log_prob of the Gaussian, minus the tanh log-det (stable form), minus the
        # constant scale Jacobian (per dimension).
        log_prob = dist.log_prob(pre_tanh)
        log_prob = log_prob - 2.0 * (math.log(2.0) - pre_tanh - nn.functional.softplus(-2.0 * pre_tanh))
        log_prob = log_prob - torch.log(self.action_scale)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class DeterministicActor(nn.Module):
    """TD3/DDPG actor: ``a = action_scale * tanh(pi(s))``. Exploration noise added outside."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        *,
        action_scale: float = 1.0,
        layernorm: bool = False,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        if action_scale <= 0:
            raise ValueError(f"action_scale must be > 0, got {action_scale}")
        self.action_dim = action_dim
        self.register_buffer("action_scale", torch.as_tensor(float(action_scale)))
        self.net = mlp(
            state_dim, action_dim, hidden,
            activation=activation, layernorm=layernorm, output_activation="tanh",
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.action_scale
