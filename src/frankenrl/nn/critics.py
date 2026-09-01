"""Value networks: ``V(s)``, ``Q(s, a)``, and the twin-Q wrapper used by SAC/TD3.

The twin critic is a single module holding two independent MLPs so one optimiser and one
``.parameters()`` call cover both - and so the "target" is a proper deep-copied twin, not
the legacy `Standard.py` mistake of reusing critic-2 as critic-1's target.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from frankenrl.nn.blocks import hard_update, mlp


class VCritic(nn.Module):
    """State-value ``V(s)``."""

    def __init__(
        self, state_dim: int, hidden: tuple[int, ...] = (256, 256),
        *, layernorm: bool = False, activation: str = "mish",
    ) -> None:
        super().__init__()
        self.net = mlp(state_dim, 1, hidden, activation=activation, layernorm=layernorm)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class QCritic(nn.Module):
    """Action-value ``Q(s, a)`` with ``s`` and ``a`` concatenated at the input."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden: tuple[int, ...] = (256, 256),
        *, layernorm: bool = False, activation: str = "mish",
    ) -> None:
        super().__init__()
        self.net = mlp(
            state_dim + action_dim, 1, hidden, activation=activation, layernorm=layernorm
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class TwinQCritic(nn.Module):
    """Two independent ``Q`` networks. ``forward`` returns both; ``q_min`` returns the min."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden: tuple[int, ...] = (256, 256),
        *, layernorm: bool = False, activation: str = "mish",
    ) -> None:
        super().__init__()
        self.q1 = QCritic(state_dim, action_dim, hidden, layernorm=layernorm, activation=activation)
        self.q2 = QCritic(state_dim, action_dim, hidden, layernorm=layernorm, activation=activation)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(state, action)
        return torch.min(q1, q2)

    def frozen_target(self) -> "TwinQCritic":
        """Deep copy with grads disabled - the soft-updated target network."""
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        hard_update(target, self)
        return target
