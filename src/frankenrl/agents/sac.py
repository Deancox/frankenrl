"""Soft Actor-Critic - the reference implementation everything else is checked against.

Corrects the legacy `Standard.py::SACAgent`:
* the target critic is a frozen deep-copy, **soft-updated only** - never optimised
  (legacy created a `critic_target_opt` and MSE-trained it);
* twin critics are two independent networks (`TwinQCritic`), not "critic vs critic_target";
* optional automatic entropy-temperature tuning (`alpha`), per [[Automatic entropy-temperature tuning]].

See [[SAC]] and [[Soft policy iteration]].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from frankenrl.agents.base import Agent
from frankenrl.buffers.base import Transition
from frankenrl.buffers.uniform import UniformReplay
from frankenrl.config import AgentConfig
from frankenrl.nn.actors import SquashedGaussianActor
from frankenrl.nn.blocks import soft_update
from frankenrl.nn.critics import TwinQCritic


class SAC(Agent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float,
        cfg: AgentConfig,
        *,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.action_dim = action_dim
        self._updates = 0

        net = cfg.net
        self.actor = SquashedGaussianActor(
            state_dim, action_dim, tuple(net.hidden),
            action_scale=action_scale, layernorm=net.layernorm, activation=net.activation,
        ).to(self.device)
        self.critics = TwinQCritic(
            state_dim, action_dim, tuple(net.hidden),
            layernorm=net.layernorm, activation=net.activation,
        ).to(self.device)
        self.critic_target = self.critics.frozen_target().to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=cfg.lr)

        # entropy temperature
        self.autotune = cfg.autotune_entropy
        if self.autotune:
            target_entropy = cfg.target_entropy
            self.target_entropy = float(-action_dim if target_entropy is None else target_entropy)
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self._log_alpha], lr=cfg.lr)
        else:
            self._fixed_alpha = float(cfg.entropy_coef)

        self.buffer = UniformReplay(
            cfg.buffer_capacity, state_dim, action_dim, device=self.device, seed=seed
        )

    # --------------------------------------------------------------- properties
    @property
    def alpha(self) -> torch.Tensor:
        return self._log_alpha.exp() if self.autotune else torch.tensor(self._fixed_alpha, device=self.device)

    # --------------------------------------------------------------- Agent API
    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(s, deterministic=deterministic)
        # actor returns scaled action; env re-applies scale, so normalise back to [-1, 1]
        return (action.squeeze(0) / self.actor.action_scale).cpu().numpy()

    def observe(self, transition: Transition, *, log_prob: float = 0.0) -> None:
        self.buffer.add(transition)

    def update(self, step: int) -> dict[str, float]:
        if step < self.cfg.warmup_steps or not self.buffer.ready(self.cfg.batch_size):
            return {}
        metrics: dict[str, float] = {}
        for _ in range(self.cfg.updates_per_step):
            metrics = self._learn()
            self._updates += 1
        return metrics

    def _learn(self) -> dict[str, float]:
        b = self.buffer.sample(self.cfg.batch_size)
        gamma, alpha = self.cfg.gamma, self.alpha

        # --- critic target: r + gamma * (1 - done) * (min Q_targ(s', a') - alpha log pi(a'|s'))
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(b.next_state)
            next_action = next_action / self.actor.action_scale  # critics see [-1, 1] actions
            q_next = self.critic_target.q_min(b.next_state, next_action) - alpha * next_logp
            q_target = b.reward + gamma * (1.0 - b.done) * q_next

        q1, q2 = self.critics(b.state, b.action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # --- actor: maximise E[min Q(s, a) - alpha log pi(a|s)]
        pi_action, logp = self.actor.sample(b.state)
        pi_action = pi_action / self.actor.action_scale
        q_pi = self.critics.q_min(b.state, pi_action)
        actor_loss = (alpha.detach() * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # --- temperature
        alpha_loss_val = 0.0
        if self.autotune:
            alpha_loss = -(self._log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.item())

        soft_update(self.critic_target, self.critics, self.cfg.tau)

        return {
            "loss/critic": float(critic_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "loss/alpha": alpha_loss_val,
            "alpha": float(self.alpha.item()),
            "logp": float(logp.mean().item()),
        }

    # --------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "updates": self._updates,
        }
        if self.autotune:
            state["log_alpha"] = self._log_alpha.detach().cpu()
            state["alpha_opt"] = self.alpha_opt.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self._updates = state.get("updates", 0)
        if self.autotune and "log_alpha" in state:
            with torch.no_grad():
                self._log_alpha.copy_(state["log_alpha"].to(self.device))
            self.alpha_opt.load_state_dict(state["alpha_opt"])
