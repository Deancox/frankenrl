#!/usr/bin/env python
"""Soft Actor-Critic (SAC) - completely self-contained, single-file reference.

Haarnoja, Zhou, Abbeel & Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
with a Stochastic Actor", ICML 2018 (arXiv:1801.01290); Haarnoja et al., "Soft Actor-Critic
Algorithms and Applications", 2018 (arXiv:1812.05905, automatic temperature tuning).

Maximum-entropy off-policy actor-critic for continuous control: a squashed-Gaussian actor
trained with the reparameterisation trick, twin critics with a clipped-double-Q (min) target,
and an automatically-tuned entropy temperature. No dependency on the rest of this repo (no
`frankenrl` imports) - this file alone is the whole algorithm, on purpose (CleanRL-style):
read top to bottom, no framework indirection. See DeanVault/Wiki/RL/SAC.md for the full
derivation this implementation follows.

Usage:
    uv run python standalone/sac.py --env Pendulum-v1 --seed 0 --total-steps 60000
"""

from __future__ import annotations

import argparse
import math
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--entropy-coef", type=float, default=0.2, help="used only if --no-autotune")
    p.add_argument("--no-autotune", action="store_true", help="fix entropy coef instead of learning it")
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


# --------------------------------------------------------------------------- networks

def mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for w in hidden:
        layers += [nn.Linear(prev, w), nn.Mish()]
        prev = w
    layers.append(nn.Linear(prev, out_dim))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
    return net


class SquashedGaussianActor(nn.Module):
    """a = action_scale * tanh(mu(s) + sigma(s) * eps), stable tanh log-det correction."""

    def __init__(self, state_dim: int, action_dim: int, hidden: list[int], action_scale: torch.Tensor):
        super().__init__()
        self.trunk = mlp(state_dim, hidden[-1], hidden[:-1] or [hidden[-1]])
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)
        self.register_buffer("action_scale", action_scale)

    def distribution(self, state: torch.Tensor) -> Normal:
        x = self.trunk(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mu, log_std.exp())

    def sample(self, state: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        pre_tanh = dist.mean if deterministic else dist.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale
        log_prob = dist.log_prob(pre_tanh)
        log_prob = log_prob - 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_prob = log_prob - torch.log(self.action_scale)
        return action, log_prob.sum(dim=-1, keepdim=True)


class TwinQCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: list[int]):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, 1, hidden)
        self.q2 = mlp(state_dim + action_dim, 1, hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(s, a)
        return torch.min(q1, q2)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


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

class SAC:
    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        action_scale = torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32, device=device)
        self.action_bias = torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32, device=device)

        self.actor = SquashedGaussianActor(state_dim, action_dim, args.hidden, action_scale).to(device)
        self.critics = TwinQCritic(state_dim, action_dim, args.hidden).to(device)
        self.critic_target = TwinQCritic(state_dim, action_dim, args.hidden).to(device)
        self.critic_target.load_state_dict(self.critics.state_dict())
        self.critic_target.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=args.lr)

        self.autotune = not args.no_autotune
        if self.autotune:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        else:
            self.fixed_alpha = float(args.entropy_coef)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp() if self.autotune else torch.tensor(self.fixed_alpha, device=self.device)

    def _to_env_action(self, scaled_action: torch.Tensor) -> torch.Tensor:
        """Actor already applies action_scale (half-range); add the bias term for asymmetric bounds."""
        return scaled_action + self.action_bias

    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor.sample(s, deterministic=deterministic)
        return self._to_env_action(a).squeeze(0).cpu().numpy()

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for _ in range(self.args.updates_per_step):
            metrics = self._learn(buffer)
        return metrics

    def _learn(self, buffer: ReplayBuffer) -> dict[str, float]:
        s, a, r, s2, done = buffer.sample(self.args.batch_size)
        a = a - self.action_bias  # critics operate in the actor's zero-centred action space
        gamma, alpha = self.args.gamma, self.alpha

        with torch.no_grad():
            next_a, next_logp = self.actor.sample(s2)
            q_next = self.critic_target.q_min(s2, next_a) - alpha * next_logp
            q_target = r + gamma * (1.0 - done) * q_next

        q1, q2 = self.critics(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        pi_a, logp = self.actor.sample(s)
        q_pi = self.critics.q_min(s, pi_a)
        actor_loss = (alpha.detach() * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss_val = 0.0
        if self.autotune:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.item())

        soft_update(self.critic_target, self.critics, self.args.tau)
        return {
            "loss/critic": float(critic_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "loss/alpha": alpha_loss_val,
            "alpha": float(self.alpha.item()),
        }


# --------------------------------------------------------------------------- train loop

def evaluate(agent: SAC, env_id: str, episodes: int, seed: int) -> float:
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

    agent = SAC(state_dim, action_dim, action_low, action_high, args, device)
    buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim, device, args.seed)

    print(f"[sac] {args.env} S={state_dim} A={action_dim} device={device} seed={args.seed}")

    step = 0
    episode = 0
    t0 = time.time()
    state, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    while step < args.total_steps:
        if step < args.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = float(terminated)  # not truncated - time-limits are not true terminals
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
                print(f"[sac] ep {episode:>5} step {step:>8} | reward {ep_reward:8.1f}")
            ep_reward = 0.0

        if step % args.eval_every == 0:
            ev = evaluate(agent, args.env, args.eval_episodes, args.seed)
            sps = step / max(time.time() - t0, 1e-9)
            print(f"[sac] step {step:>8} | eval {ev:8.1f} | {sps:6.0f} sps" + (f" | critic {metrics.get('loss/critic', float('nan')):.3f}" if metrics else ""))

    env.close()
    print(f"[sac] done: {episode} episodes, {step} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    train(parse_args())
