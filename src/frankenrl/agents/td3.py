"""Twin Delayed DDPG - reference implementation.

DDPG + the three TD3 fixes: twin clipped critics, delayed actor updates, target-policy
smoothing. See [[TD3]], [[Target-policy smoothing]], [[Delayed policy updates]].
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
from frankenrl.nn.actors import DeterministicActor
from frankenrl.nn.blocks import hard_update, soft_update
from frankenrl.nn.critics import TwinQCritic


class TD3(Agent):
    def __init__(
        self, state_dim: int, action_dim: int, action_scale: float, cfg: AgentConfig,
        *, device: str | torch.device = "cpu", seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.action_dim = action_dim
        self._it = 0
        self._rng = np.random.default_rng(seed)

        net = cfg.net
        mk_actor = lambda: DeterministicActor(  # noqa: E731
            state_dim, action_dim, tuple(net.hidden),
            action_scale=1.0, layernorm=net.layernorm, activation=net.activation,
        ).to(self.device)
        self.actor = mk_actor()
        self.actor_target = mk_actor()
        hard_update(self.actor_target, self.actor)
        self.actor_target.requires_grad_(False)

        self.critics = TwinQCritic(
            state_dim, action_dim, tuple(net.hidden),
            layernorm=net.layernorm, activation=net.activation,
        ).to(self.device)
        self.critic_target = self.critics.frozen_target().to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=cfg.lr)
        self.buffer = UniformReplay(
            cfg.buffer_capacity, state_dim, action_dim, device=self.device, seed=seed
        )

    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).squeeze(0).cpu().numpy()
        if not deterministic:
            a = a + self._rng.normal(0.0, self.cfg.exploration_noise, size=self.action_dim)
        return np.clip(a, -1.0, 1.0)

    def observe(self, transition: Transition, *, log_prob: float = 0.0) -> None:
        self.buffer.add(transition)

    def update(self, step: int) -> dict[str, float]:
        if step < self.cfg.warmup_steps or not self.buffer.ready(self.cfg.batch_size):
            return {}
        metrics: dict[str, float] = {}
        for _ in range(self.cfg.updates_per_step):
            metrics = self._learn()
        return metrics

    def _learn(self) -> dict[str, float]:
        self._it += 1
        b = self.buffer.sample(self.cfg.batch_size)
        c = self.cfg

        with torch.no_grad():
            noise = (torch.randn_like(b.action) * c.target_policy_noise).clamp(
                -c.target_noise_clip, c.target_noise_clip
            )
            next_action = (self.actor_target(b.next_state) + noise).clamp(-1.0, 1.0)
            q_next = self.critic_target.q_min(b.next_state, next_action)
            q_target = b.reward + c.gamma * (1.0 - b.done) * q_next

        q1, q2 = self.critics(b.state, b.action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics = {"loss/critic": float(critic_loss.item())}
        if self._it % c.policy_delay == 0:
            actor_loss = -self.critics.q1(b.state, self.actor(b.state)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            soft_update(self.actor_target, self.actor, c.tau)
            soft_update(self.critic_target, self.critics, c.tau)
            metrics["loss/actor"] = float(actor_loss.item())
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critics": self.critics.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "it": self._it,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critics.load_state_dict(state["critics"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self._it = state.get("it", 0)
