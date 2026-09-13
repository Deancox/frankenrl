#!/usr/bin/env python
"""BRO: Bigger, Regularized, Optimistic - completely self-contained, single-file reference.

Nauman, Ostaszewski, Jankowski, Milos & Cygan, "Bigger, Regularized, Optimistic: scaling for
compute and sample-efficient continuous control", NeurIPS 2024 (arXiv:2405.16158).

SAC's actor-critic skeleton with four changes: a BroNet residual quantile-ensemble critic
(~7x a standard SAC critic), AdamW weight decay + a full-parameter reset schedule, a high
critic replay ratio, and a dual-actor optimistic-exploration scheme that drops clipped
double-Q entirely (the target uses the elementwise *mean* of the twin critics' quantiles,
never the min). No dependency on the rest of this repo - this file alone is the whole
algorithm (CleanRL-style; this is the same algorithm as ../src/frankenrl/agents/bro.py +
nn/bronet.py, inlined and duplicated on purpose rather than imported, so this file is
readable standalone). See DeanVault/Wiki/RL/BRO - scaling off-policy actor-critic RL with
regularized critics and optimistic exploration.md for the full derivation and worked example
this file's critic aggregation functions are checked against (see test_standalone_smoke.py).

pi_p (update actor) supplies the bootstrap action for the TD target and is trained with the
ordinary SAC actor loss (Q^mu in place of Q_min). pi_o (exploration actor) is the only one
`act()` ever samples from; it is trained to maximise Q^mu + optimism_coef*Q^sigma,
KL-regularised back toward pi_p in closed form (pre-tanh Gaussian KL - exact, since both
actors share the same tanh/action_scale transform, which cancels in the log-ratio).

`optimism_coef`, `kl_coef`, `weight_decay`, and the exact quantile-loss form are this
codebase's own picks, not confirmed literature values - see the wiki doc's ambiguity notes.

Usage:
    uv run python standalone/bro.py --env Pendulum-v1 --seed 0 --total-steps 60000
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


# --------------------------------------------------------------------------- args

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=60_000)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--actor-hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--updates-per-step", type=int, default=10, help="BRO's critic replay ratio (paper default 10, 'Fast' variant 2)")
    p.add_argument("--critic-width", type=int, default=512)
    p.add_argument("--critic-blocks", type=int, default=2)
    p.add_argument("--n-quantiles", type=int, default=100)
    p.add_argument("--quantile-kappa", type=float, default=1.0)
    p.add_argument("--reset-schedule", type=int, nargs="*", default=[15_000, 50_000, 250_000, 500_000, 750_000, 1_000_000])
    p.add_argument("--optimism-coef", type=float, default=0.5, help="beta^o")
    p.add_argument("--kl-coef", type=float, default=0.1, help="tau, KL(pi_p||pi_o) weight")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW, actors + critic")
    p.add_argument("--no-autotune", action="store_true")
    p.add_argument("--entropy-coef", type=float, default=0.2, help="used only if --no-autotune")
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- shared helpers

def _orthogonal_(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for w in hidden:
        layers += [nn.Linear(prev, w), nn.Mish()]
        prev = w
    layers.append(nn.Linear(prev, out_dim))
    net = nn.Sequential(*layers)
    net.apply(lambda m: _orthogonal_(m, math.sqrt(2)))
    return net


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.copy_(sp)


def reinit_module_(module: nn.Module, *, orthogonal_gain: float = math.sqrt(2)) -> None:
    """BRO's primacy-bias reset: re-initialise every Linear/LayerNorm submodule in place."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=orthogonal_gain)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            m.reset_parameters()


# --------------------------------------------------------------------------- actor

class SquashedGaussianActor(nn.Module):
    """a = action_bias + action_scale * tanh(mu(s) + sigma(s) * eps)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: list[int], action_scale: torch.Tensor, action_bias: torch.Tensor):
        super().__init__()
        self.trunk = mlp(state_dim, hidden[-1], hidden[:-1] or [hidden[-1]])
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def distribution(self, state: torch.Tensor) -> Normal:
        x = self.trunk(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mu, log_std.exp())

    def sample(self, state: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        pre_tanh = dist.mean if deterministic else dist.rsample()
        squashed = torch.tanh(pre_tanh)
        zero_centred_action = squashed * self.action_scale
        log_prob = dist.log_prob(pre_tanh)
        log_prob = log_prob - 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_prob = log_prob - torch.log(self.action_scale)
        return zero_centred_action, log_prob.sum(dim=-1, keepdim=True)

    def to_env_action(self, zero_centred_action: torch.Tensor) -> torch.Tensor:
        return zero_centred_action + self.action_bias


# --------------------------------------------------------------------------- BroNet critic

class BroNetBlock(nn.Module):
    """(Linear -> LayerNorm -> act) x2 with a skip connection; no activation after the add,
    so a zeroed inner branch collapses the block to the exact identity."""

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.ln1 = nn.LayerNorm(width)
        self.act1 = nn.Mish()
        self.fc2 = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        return x + h


def bronet_critic_trunk(in_dim: int, n_quantiles: int, *, width: int, blocks: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.LayerNorm(width), nn.Mish()]
    layers.extend(BroNetBlock(width) for _ in range(blocks))
    layers.append(nn.Linear(width, n_quantiles))
    return nn.Sequential(*layers)


def q_mean_from_quantiles(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean over both critics and all quantiles - BRO's Q^mu (the actor objective's value;
    the critic's own TD target regresses full quantile vectors, see BRO._learn_critic)."""
    return torch.cat([q1, q2], dim=-1).mean(dim=-1, keepdim=True)


def q_disagreement_from_quantiles(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean |q1 - q2| across quantiles - the epistemic-uncertainty proxy Q^sigma."""
    return (q1 - q2).abs().mean(dim=-1, keepdim=True)


def optimistic_q_from_quantiles(q1: torch.Tensor, q2: torch.Tensor, *, optimism_coef: float) -> torch.Tensor:
    return q_mean_from_quantiles(q1, q2) + optimism_coef * q_disagreement_from_quantiles(q1, q2)


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, *, kappa: float = 1.0) -> torch.Tensor:
    """QR-DQN-style pairwise quantile Huber loss. `pred`: (batch, K), estimating the fixed,
    evenly spaced fractions tau_i = (2i-1)/(2K). `target`: (batch, K'), need not match K."""
    k = pred.shape[-1]
    tau = (torch.arange(k, device=pred.device, dtype=pred.dtype) * 2 + 1) / (2 * k)
    pred_i = pred.unsqueeze(2)      # (batch, K, 1)
    target_j = target.unsqueeze(1)  # (batch, 1, K')
    error = target_j - pred_i
    huber = F.huber_loss(pred_i.expand_as(error), target_j.expand_as(error), reduction="none", delta=kappa)
    weight = (tau.view(1, -1, 1) - (error.detach() < 0).float()).abs()
    return (weight * huber).mean()


class QuantileTwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, *, width: int, blocks: int, n_quantiles: int):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.q1 = bronet_critic_trunk(state_dim + action_dim, n_quantiles, width=width, blocks=blocks)
        self.q2 = bronet_critic_trunk(state_dim + action_dim, n_quantiles, width=width, blocks=blocks)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_mean(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return q_mean_from_quantiles(*self(state, action))

    def optimistic_q(self, state: torch.Tensor, action: torch.Tensor, *, optimism_coef: float) -> torch.Tensor:
        return optimistic_q_from_quantiles(*self(state, action), optimism_coef=optimism_coef)


# --------------------------------------------------------------------------- replay buffer

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device, seed: int):
        self.capacity = capacity
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, done: float) -> None:
        i = self.ptr
        self.state[i], self.action[i], self.reward[i] = s, a, r
        self.next_state[i], self.done[i] = s2, done
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        return to(self.state), to(self.action), to(self.reward), to(self.next_state), to(self.done)


# --------------------------------------------------------------------------- agent

class BRO:
    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self._reset_steps = set(args.reset_schedule)
        self._resets_done: set[int] = set()

        action_scale = torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32, device=device)
        action_bias = torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32, device=device)
        self._mk_actor = lambda: SquashedGaussianActor(state_dim, action_dim, args.actor_hidden, action_scale, action_bias).to(device)
        self.pi_p = self._mk_actor()
        self.pi_o = self._mk_actor()

        self.critics = QuantileTwinCritic(state_dim, action_dim, width=args.critic_width, blocks=args.critic_blocks, n_quantiles=args.n_quantiles).to(device)
        self.critic_target = copy.deepcopy(self.critics)
        self.critic_target.requires_grad_(False)

        self._build_optimizers()

        self.autotune = not args.no_autotune
        if self.autotune:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        else:
            self.fixed_alpha = float(args.entropy_coef)

    def _build_optimizers(self) -> None:
        wd = self.args.weight_decay
        self.pi_p_opt = torch.optim.AdamW(self.pi_p.parameters(), lr=self.args.lr, weight_decay=wd)
        self.pi_o_opt = torch.optim.AdamW(self.pi_o.parameters(), lr=self.args.lr, weight_decay=wd)
        self.critic_opt = torch.optim.AdamW(self.critics.parameters(), lr=self.args.lr, weight_decay=wd)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp() if self.autotune else torch.tensor(self.fixed_alpha, device=self.device)

    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        """Samples from pi_o only - pi_p's parameters are never touched here."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.pi_o.sample(s, deterministic=deterministic)
        return self.pi_o.to_env_action(a).squeeze(0).cpu().numpy()

    def maybe_reset(self, step: int) -> bool:
        if step not in self._reset_steps or step in self._resets_done:
            return False
        reinit_module_(self.pi_p)
        reinit_module_(self.pi_o)
        reinit_module_(self.critics)
        hard_update(self.critic_target, self.critics)
        self._build_optimizers()
        if self.autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.args.lr)
        self._resets_done.add(step)
        return True

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for _ in range(self.args.updates_per_step):
            metrics = self._learn_critic(buffer)
        metrics.update(self._learn_actors(buffer))
        return metrics

    def _learn_critic(self, buffer: ReplayBuffer) -> dict[str, float]:
        s, a, r, s2, done = buffer.sample(self.args.batch_size)
        a = a - self.pi_p.action_bias  # critics operate in the zero-centred action space
        gamma, alpha = self.args.gamma, self.alpha

        # target quantiles: r + gamma*(1-done)*(mean(q1_targ, q2_targ) - alpha*logp) - "mean,
        # not min" is the CDQ removal; a genuine (batch, K) distributional Bellman backup,
        # not a scalar broadcast (that reduction is only for the actor objectives below).
        with torch.no_grad():
            next_action, next_logp = self.pi_p.sample(s2)
            q1_next, q2_next = self.critic_target(s2, next_action)
            q_next = 0.5 * (q1_next + q2_next) - alpha * next_logp
            target_quantiles = r + gamma * (1.0 - done) * q_next

        q1, q2 = self.critics(s, a)
        critic_loss = quantile_huber_loss(q1, target_quantiles, kappa=self.args.quantile_kappa) \
            + quantile_huber_loss(q2, target_quantiles, kappa=self.args.quantile_kappa)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        soft_update(self.critic_target, self.critics, self.args.tau)
        return {"loss/critic": float(critic_loss.item())}

    def _learn_actors(self, buffer: ReplayBuffer) -> dict[str, float]:
        s, *_ = buffer.sample(self.args.batch_size)
        alpha = self.alpha

        pi_p_action, pi_p_logp = self.pi_p.sample(s)
        q_mu = self.critics.q_mean(s, pi_p_action)
        pi_p_loss = (alpha.detach() * pi_p_logp - q_mu).mean()
        self.pi_p_opt.zero_grad(set_to_none=True)
        pi_p_loss.backward()
        self.pi_p_opt.step()

        alpha_loss_val = 0.0
        if self.autotune:
            alpha_loss = -(self.log_alpha * (pi_p_logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.item())

        pi_o_action, _ = self.pi_o.sample(s)
        q_opt = self.critics.optimistic_q(s, pi_o_action, optimism_coef=self.args.optimism_coef)
        with torch.no_grad():
            pi_p_dist = self.pi_p.distribution(s)
        pi_o_dist = self.pi_o.distribution(s)
        kl = kl_divergence(pi_p_dist, pi_o_dist).sum(dim=-1, keepdim=True)
        pi_o_loss = -(q_opt - self.args.kl_coef * kl).mean()
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


# --------------------------------------------------------------------------- train loop

def evaluate(agent: BRO, env_id: str, episodes: int, seed: int) -> float:
    env = gym.make(env_id)
    total = 0.0
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + 10_000 + ep)
        done = trunc = False
        while not (done or trunc):
            action = agent.act(state, deterministic=True)
            state, r, done, trunc, _ = env.step(action)
            total += r
    env.close()
    return total / episodes


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low, action_high = env.action_space.low, env.action_space.high

    agent = BRO(state_dim, action_dim, action_low, action_high, args, device)
    buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim, device, args.seed)

    print(f"[bro] {args.env} S={state_dim} A={action_dim} device={device} seed={args.seed} "
          f"critic_width={args.critic_width} replay_ratio={args.updates_per_step}")

    step = 0
    episode = 0
    t0 = time.time()
    state, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    while step < args.total_steps:
        reset_now = agent.maybe_reset(step)
        if reset_now:
            print(f"[bro] step {step:>8} | full parameter reset")

        if step < args.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = float(terminated)
        buffer.add(state, action, reward, next_state, done)
        ep_reward += reward
        state = next_state
        step += 1

        metrics: dict[str, float] = {}
        if step >= args.warmup_steps and buffer.ready(args.batch_size):
            metrics = agent.update(buffer)

        if terminated or truncated:
            episode += 1
            state, _ = env.reset(seed=args.seed + episode)
            if episode % 10 == 0:
                print(f"[bro] ep {episode:>5} step {step:>8} | reward {ep_reward:8.1f}")
            ep_reward = 0.0

        if step % args.eval_every == 0:
            ev = evaluate(agent, args.env, args.eval_episodes, args.seed)
            sps = step / max(time.time() - t0, 1e-9)
            print(f"[bro] step {step:>8} | eval {ev:8.1f} | {sps:6.0f} sps" + (f" | critic {metrics.get('loss/critic', float('nan')):.3f}" if metrics else ""))

    env.close()
    print(f"[bro] done: {episode} episodes, {step} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    train(parse_args())
