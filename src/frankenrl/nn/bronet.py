"""BroNet: the scaled, residual, quantile-ensemble critic from BRO (Nauman, Ostaszewski,
Jankowski, Milos & Cygan, "Bigger, Regularized, Optimistic", NeurIPS 2024, arXiv:2405.16158).

Stem ``Linear -> LayerNorm -> act`` then ``critic_blocks`` residual blocks (each
``(Linear -> LayerNorm -> act) x2`` with a skip connection), ending in a
``Linear(width -> n_quantiles)`` head - default 2 blocks x width 512, ~7x a standard SAC
critic. Two independent copies form ``QuantileTwinCritic``; BRO drops clipped double-Q, so
the Bellman target uses the elementwise *mean* of the two critics' quantiles, never the
``min`` (see ``agents/bro.py``), while the scalar mean/disagreement reductions here feed the
actor objectives. See [[BRO - scaling off-policy actor-critic RL with regularized critics
and optimistic exploration]] sections 3, 4 and 6 for the full derivation and worked example
this module is tested against (``tests/test_bronet.py``).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from frankenrl.nn.blocks import hard_update, resolve_activation


class BroNetBlock(nn.Module):
    """``(Linear -> LayerNorm -> act) x2`` with a skip connection; in/out width must match.

    No activation after the residual add (pre-activation / ResNet-v2 style): this is what
    makes a zeroed inner branch collapse the block to the exact identity, which is how
    ``tests/test_bronet.py`` verifies the skip is actually wired.
    """

    def __init__(self, width: int, *, activation: str = "mish") -> None:
        super().__init__()
        act_cls = resolve_activation(activation)
        self.fc1 = nn.Linear(width, width)
        self.ln1 = nn.LayerNorm(width)
        self.act1 = act_cls()
        self.fc2 = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        return x + h


def bronet_critic_trunk(
    in_dim: int,
    n_quantiles: int,
    *,
    width: int = 512,
    blocks: int = 2,
    activation: str = "mish",
) -> nn.Sequential:
    """Stem + ``blocks`` x :class:`BroNetBlock`, ending in a ``Linear(width, n_quantiles)``
    quantile head. Used twice, independently initialised, to build the twin critics."""
    act_cls = resolve_activation(activation)
    layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.LayerNorm(width), act_cls()]
    layers.extend(BroNetBlock(width, activation=activation) for _ in range(blocks))
    layers.append(nn.Linear(width, n_quantiles))
    return nn.Sequential(*layers)


def q_mean_from_quantiles(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean over both critics and all quantiles - BRO's ``Q^mu``, the value used everywhere
    clipped double-Q's ``min`` used to be (the actor loss, not the critic's own TD target,
    which regresses full quantile vectors - see ``agents/bro.py::_learn_critic``). Shape
    ``(batch, 1)``."""
    return torch.cat([q1, q2], dim=-1).mean(dim=-1, keepdim=True)


def q_disagreement_from_quantiles(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean ``|q1 - q2|`` across quantiles - the epistemic-uncertainty proxy ``Q^sigma``
    driving the optimistic-exploration actor. Shape ``(batch, 1)``."""
    return (q1 - q2).abs().mean(dim=-1, keepdim=True)


def optimistic_q_from_quantiles(
    q1: torch.Tensor, q2: torch.Tensor, *, optimism_coef: float
) -> torch.Tensor:
    """``Q^mu + optimism_coef * Q^sigma`` - the exploration actor's objective (wiki S6)."""
    return q_mean_from_quantiles(q1, q2) + optimism_coef * q_disagreement_from_quantiles(q1, q2)


def quantile_huber_loss(
    pred: torch.Tensor, target: torch.Tensor, *, kappa: float = 1.0
) -> torch.Tensor:
    """QR-DQN-style pairwise quantile Huber (pinball) loss.

    ``pred``: ``(batch, K)`` current quantile predictions, treated as estimating the fixed,
    evenly spaced fractions ``tau_i = (2i-1)/(2K)`` in ascending order. ``target``:
    ``(batch, K')`` target quantile samples (need not share ``K'`` with ``pred``). Every
    ``(pred_i, target_j)`` pair contributes a Huber-clipped error weighted by
    ``|tau_i - 1{error<0}|``, the asymmetric penalty that makes high (``tau>0.5``) quantiles
    penalise under-prediction more than over-prediction, and vice versa for low quantiles.
    Returns a scalar (mean over batch and all pairs).
    """
    if pred.dim() != 2 or target.dim() != 2:
        raise ValueError("pred and target must be 2D (batch, n_quantiles)")
    k = pred.shape[-1]
    tau = (torch.arange(k, device=pred.device, dtype=pred.dtype) * 2 + 1) / (2 * k)

    pred_i = pred.unsqueeze(2)      # (batch, K, 1)
    target_j = target.unsqueeze(1)  # (batch, 1, K')
    error = target_j - pred_i       # (batch, K, K')

    huber = F.huber_loss(
        pred_i.expand_as(error), target_j.expand_as(error), reduction="none", delta=kappa
    )
    weight = (tau.view(1, -1, 1) - (error.detach() < 0).float()).abs()
    return (weight * huber).mean()


class QuantileTwinCritic(nn.Module):
    """Two independent :func:`bronet_critic_trunk` networks over ``concat(state, action)``."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        width: int = 512,
        blocks: int = 2,
        n_quantiles: int = 100,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        if n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be > 0, got {n_quantiles}")
        self.n_quantiles = n_quantiles
        self.q1 = bronet_critic_trunk(
            state_dim + action_dim, n_quantiles, width=width, blocks=blocks, activation=activation
        )
        self.q2 = bronet_critic_trunk(
            state_dim + action_dim, n_quantiles, width=width, blocks=blocks, activation=activation
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_mean(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return q_mean_from_quantiles(*self(state, action))

    def q_disagreement(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return q_disagreement_from_quantiles(*self(state, action))

    def optimistic_q(
        self, state: torch.Tensor, action: torch.Tensor, *, optimism_coef: float
    ) -> torch.Tensor:
        return optimistic_q_from_quantiles(*self(state, action), optimism_coef=optimism_coef)

    def frozen_target(self) -> "QuantileTwinCritic":
        """Deep copy with grads disabled - the soft-updated target network."""
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        hard_update(target, self)
        return target
