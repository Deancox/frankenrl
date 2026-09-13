"""BRO: Bigger, Regularized, Optimistic (Nauman et al., NeurIPS 2024, arXiv:2405.16158).

SAC's actor-critic skeleton with four changes layered on: a BroNet quantile-ensemble critic
(~7x a standard SAC critic, `nn/bronet.py`), AdamW weight decay + a full-parameter reset
schedule, a high critic replay ratio (`cfg.updates_per_step`, reused rather than duplicated),
and a dual-actor optimistic-exploration scheme. Clipped double-Q is dropped entirely: the
Bellman target uses the elementwise *mean* of the twin critics' quantile vectors, never the
min - see `_learn_critic`. That is what frees the twin-critic disagreement signal to drive a
second, exploration-only actor instead of supplying pessimism.

Two actors:
* `pi_p` (update actor) supplies the bootstrap action for the TD target and is trained with
  the ordinary SAC actor loss, `Q^mu` (the critic mean) in place of `Q_min`.
* `pi_o` (exploration actor) is the *only* one `act()` samples from - its own stochastic
  sampling is already the exploration mechanism (no extra noise needed, unlike TD3's
  deterministic actor). It is trained to maximise `Q^mu + optimism_coef * Q^sigma`
  (`QuantileTwinCritic.optimistic_q`), KL-regularised back toward `pi_p` in closed form
  (`SquashedGaussianActor.distribution`) rather than by sampling - see
  [[BRO - scaling off-policy actor-critic RL with regularized critics and optimistic
  exploration]] section 6 for the derivation this implements, including the worked example
  this module's critic (`nn/bronet.py`) is unit-tested against.

Several hyperparameters (`optimism_coef`, `kl_coef`, `weight_decay`, the exact quantile-loss
form, and the assumption that both actors update once per outer `update()` call rather than
once per inner critic step) are this codebase's own picks, not confirmed literature values -
see `config.py::BroConfig`'s docstring and `configs/bro_pendulum.yaml`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.distributions import kl_divergence

from frankenrl.agents.base import Agent
from frankenrl.buffers.base import Transition
from frankenrl.buffers.uniform import UniformReplay
from frankenrl.config import AgentConfig
from frankenrl.nn.actors import SquashedGaussianActor
from frankenrl.nn.blocks import hard_update, reinit_module_, soft_update
from frankenrl.nn.bronet import QuantileTwinCritic, quantile_huber_loss


class BRO(Agent):
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
        self._reset_steps = set(cfg.bro.reset_schedule)
        self._resets_done: set[int] = set()

        net, bro = cfg.net, cfg.bro
        self._mk_actor = lambda: SquashedGaussianActor(
            state_dim, action_dim, tuple(net.hidden),
            action_scale=action_scale, layernorm=net.layernorm, activation=net.activation,
        ).to(self.device)
        self.pi_p = self._mk_actor()
        self.pi_o = self._mk_actor()

        self.critics = QuantileTwinCritic(
            state_dim, action_dim,
            width=bro.critic_width, blocks=bro.critic_blocks,
            n_quantiles=bro.n_quantiles, activation=net.activation,
        ).to(self.device)
        self.critic_target = self.critics.frozen_target().to(self.device)

        self._build_optimizers()

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

    def _build_optimizers(self) -> None:
        wd = self.cfg.bro.weight_decay
        self.pi_p_opt = torch.optim.AdamW(self.pi_p.parameters(), lr=self.cfg.lr, weight_decay=wd)
        self.pi_o_opt = torch.optim.AdamW(self.pi_o.parameters(), lr=self.cfg.lr, weight_decay=wd)
        self.critic_opt = torch.optim.AdamW(self.critics.parameters(), lr=self.cfg.lr, weight_decay=wd)

    # --------------------------------------------------------------- properties
    @property
    def alpha(self) -> torch.Tensor:
        return self._log_alpha.exp() if self.autotune else torch.tensor(self._fixed_alpha, device=self.device)

    # --------------------------------------------------------------- Agent API
    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        """Samples from `pi_o` only - `pi_p`'s parameters are never touched here."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.pi_o.sample(s, deterministic=deterministic)
        return (action.squeeze(0) / self.pi_o.action_scale).cpu().numpy()

    def observe(self, transition: Transition, *, log_prob: float = 0.0) -> None:
        self.buffer.add(transition)

    def update(self, step: int) -> dict[str, float]:
        reset_triggered = self._maybe_reset(step)
        if step < self.cfg.warmup_steps or not self.buffer.ready(self.cfg.batch_size):
            return {"reset": 1.0} if reset_triggered else {}

        metrics: dict[str, float] = {}
        for _ in range(self.cfg.updates_per_step):
            metrics = self._learn_critic()
            self._updates += 1
        metrics.update(self._learn_actors())
        if reset_triggered:
            metrics["reset"] = 1.0
        return metrics

    # --------------------------------------------------------------- reset schedule
    def _maybe_reset(self, step: int) -> bool:
        if step not in self._reset_steps or step in self._resets_done:
            return False
        reinit_module_(self.pi_p)
        reinit_module_(self.pi_o)
        reinit_module_(self.critics)
        hard_update(self.critic_target, self.critics)
        if self.cfg.bro.reset_optimizer_state:
            self._build_optimizers()
            if self.autotune:
                self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_opt = torch.optim.Adam([self._log_alpha], lr=self.cfg.lr)
        self._resets_done.add(step)
        return True

    # --------------------------------------------------------------- learning
    def _learn_critic(self) -> dict[str, float]:
        b = self.buffer.sample(self.cfg.batch_size)
        gamma, alpha = self.cfg.gamma, self.alpha

        # target quantiles: r + gamma * (1 - done) * (mean(q1_targ, q2_targ) - alpha * logp)
        # "mean, not min" is the CDQ removal - see class docstring. pi_p supplies the
        # bootstrap action (the "update actor" role); this stays a full (batch, K) quantile
        # vector, a genuine distributional Bellman backup, not the scalar Q^mu reduction
        # (that reduction is only for the actor objectives, in `_learn_actors`).
        with torch.no_grad():
            next_action, next_logp = self.pi_p.sample(b.next_state)
            next_action = next_action / self.pi_p.action_scale
            q1_next, q2_next = self.critic_target(b.next_state, next_action)
            q_next = 0.5 * (q1_next + q2_next) - alpha * next_logp
            target_quantiles = b.reward + gamma * (1.0 - b.done) * q_next

        q1, q2 = self.critics(b.state, b.action)
        kappa = self.cfg.bro.quantile_huber_kappa
        critic_loss = quantile_huber_loss(q1, target_quantiles, kappa=kappa) \
            + quantile_huber_loss(q2, target_quantiles, kappa=kappa)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        soft_update(self.critic_target, self.critics, self.cfg.tau)
        return {"loss/critic": float(critic_loss.item())}

    def _learn_actors(self) -> dict[str, float]:
        b = self.buffer.sample(self.cfg.batch_size)
        alpha = self.alpha

        # pi_p: ordinary SAC actor loss with Q^mu (mean over both critics/quantiles) standing
        # in for Q_min - the wiki doc does not give pi_p's loss explicitly; this is the
        # natural reading of "supplies the action sampled for the TD target" plus "the update
        # actor keeps the ordinary Q-target".
        pi_p_action, pi_p_logp = self.pi_p.sample(b.state)
        pi_p_action = pi_p_action / self.pi_p.action_scale
        q_mu = self.critics.q_mean(b.state, pi_p_action)
        pi_p_loss = (alpha.detach() * pi_p_logp - q_mu).mean()
        self.pi_p_opt.zero_grad(set_to_none=True)
        pi_p_loss.backward()
        self.pi_p_opt.step()

        alpha_loss_val = 0.0
        if self.autotune:
            alpha_loss = -(self._log_alpha * (pi_p_logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.item())

        # pi_o: maximise Q^mu + beta*Q^sigma, KL-tied back to pi_p (closed-form Gaussian KL
        # in pre-tanh space - exact, not an approximation, since both actors share
        # `action_scale` and the tanh squash cancels identically on both sides of the ratio).
        pi_o_action, _ = self.pi_o.sample(b.state)
        pi_o_action = pi_o_action / self.pi_o.action_scale
        q_opt = self.critics.optimistic_q(b.state, pi_o_action, optimism_coef=self.cfg.bro.optimism_coef)
        with torch.no_grad():
            pi_p_dist = self.pi_p.distribution(b.state)
        pi_o_dist = self.pi_o.distribution(b.state)
        kl = kl_divergence(pi_p_dist, pi_o_dist).sum(dim=-1, keepdim=True)
        pi_o_loss = -(q_opt - self.cfg.bro.kl_coef * kl).mean()
        self.pi_o_opt.zero_grad(set_to_none=True)
        pi_o_loss.backward()
        self.pi_o_opt.step()

        return {
            "loss/pi_p": float(pi_p_loss.item()),
            "loss/pi_o": float(pi_o_loss.item()),
            "loss/alpha": alpha_loss_val,
            "alpha": float(self.alpha.item()),
            "kl/pi_p_pi_o": float(kl.mean().item()),
        }

    # --------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict[str, Any]:
        state = {
            "pi_p": self.pi_p.state_dict(),
            "pi_o": self.pi_o.state_dict(),
            "critics": self.critics.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "pi_p_opt": self.pi_p_opt.state_dict(),
            "pi_o_opt": self.pi_o_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "updates": self._updates,
            "resets_done": sorted(self._resets_done),
        }
        if self.autotune:
            state["log_alpha"] = self._log_alpha.detach().cpu()
            state["alpha_opt"] = self.alpha_opt.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.pi_p.load_state_dict(state["pi_p"])
        self.pi_o.load_state_dict(state["pi_o"])
        self.critics.load_state_dict(state["critics"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.pi_p_opt.load_state_dict(state["pi_p_opt"])
        self.pi_o_opt.load_state_dict(state["pi_o_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self._updates = state.get("updates", 0)
        self._resets_done = set(state.get("resets_done", []))
        if self.autotune and "log_alpha" in state:
            with torch.no_grad():
                self._log_alpha.copy_(state["log_alpha"].to(self.device))
            self.alpha_opt.load_state_dict(state["alpha_opt"])
