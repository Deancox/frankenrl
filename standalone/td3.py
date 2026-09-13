#!/usr/bin/env python
"""Twin Delayed DDPG (TD3) - completely self-contained, single-file reference.

Fujimoto, van Hoof & Meger, "Addressing Function Approximation Error in Actor-Critic
Methods", ICML 2018 (arXiv:1802.09477).

DDPG + three fixes for its overestimation/instability: clipped double-Q (min of twin
critics), target-policy smoothing (clipped noise on the bootstrap action), and delayed actor
+ target updates (every `policy_delay` critic steps). No dependency on the rest of this repo
- this file alone is the whole algorithm (CleanRL-style). See DeanVault/Wiki/RL/TD3.md for
the full derivation this implementation follows.

Usage:
    uv run python standalone/td3.py --env Pendulum-v1 --seed 0 --total-steps 60000
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
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--target-policy-noise", type=float, default=0.2)
    p.add_argument("--target-noise-clip", type=float, default=0.5)
    p.add_argument("--exploration-noise", type=float, default=0.1)
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

def mlp(in_dim: int, out_dim: int, hidden: list[int], *, output_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for w in hidden:
        layers += [nn.Linear(prev, w), nn.Mish()]
        prev = w
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation)
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
    return net


class DeterministicActor(nn.Module):
    """a = action_bias + action_scale * tanh(pi(s)). Exploration noise added outside."""

    def __init__(self, state_dim: int, action_dim: int, hidden: list[int], action_scale: torch.Tensor, action_bias: torch.Tensor):
        super().__init__()
        self.net = mlp(state_dim, action_dim, hidden, output_activation=nn.Tanh())
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.action_scale + self.action_bias


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


def hard_update(target: nn.Module, source: nn.Module) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.copy_(sp)


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

class TD3:
    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        action_scale = (self.action_high - self.action_low) / 2.0
        action_bias = (self.action_high + self.action_low) / 2.0

        self.actor = DeterministicActor(state_dim, action_dim, args.hidden, action_scale, action_bias).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.requires_grad_(False)

        self.critics = TwinQCritic(state_dim, action_dim, args.hidden).to(device)
        self.critic_target = copy.deepcopy(self.critics)
        self.critic_target.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=args.lr)
        self._it = 0
        self._rng = np.random.default_rng(0)

    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).squeeze(0).cpu().numpy()
        if not deterministic:
            scale = (self.action_high - self.action_low).cpu().numpy() / 2.0
            a = a + self._rng.normal(0.0, self.args.exploration_noise * scale, size=a.shape)
        return np.clip(a, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for _ in range(self.args.updates_per_step):
            metrics = self._learn(buffer)
        return metrics

    def _learn(self, buffer: ReplayBuffer) -> dict[str, float]:
        self._it += 1
        c = self.args
        s, a, r, s2, done = buffer.sample(c.batch_size)

        with torch.no_grad():
            action_range = self.action_high - self.action_low
            noise = (torch.randn_like(a) * c.target_policy_noise * (action_range / 2.0)).clamp(
                -c.target_noise_clip * (action_range / 2.0), c.target_noise_clip * (action_range / 2.0)
            )
            next_action = (self.actor_target(s2) + noise).clamp(self.action_low, self.action_high)
            q_next = self.critic_target.q_min(s2, next_action)
            q_target = r + c.gamma * (1.0 - done) * q_next

        q1, q2 = self.critics(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics = {"loss/critic": float(critic_loss.item())}
        if self._it % c.policy_delay == 0:
            actor_loss = -self.critics.q1(torch.cat([s, self.actor(s)], dim=-1)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            soft_update(self.actor_target, self.actor, c.tau)
            soft_update(self.critic_target, self.critics, c.tau)
            metrics["loss/actor"] = float(actor_loss.item())
        return metrics


# --------------------------------------------------------------------------- train loop

def evaluate(agent: TD3, env_id: str, episodes: int, seed: int) -> float:
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

    agent = TD3(state_dim, action_dim, action_low, action_high, args, device)
    buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim, device, args.seed)

    print(f"[td3] {args.env} S={state_dim} A={action_dim} device={device} seed={args.seed}")

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
                print(f"[td3] ep {episode:>5} step {step:>8} | reward {ep_reward:8.1f}")
            ep_reward = 0.0

        if step % args.eval_every == 0:
            ev = evaluate(agent, args.env, args.eval_episodes, args.seed)
            sps = step / max(time.time() - t0, 1e-9)
            print(f"[td3] step {step:>8} | eval {ev:8.1f} | {sps:6.0f} sps" + (f" | critic {metrics.get('loss/critic', float('nan')):.3f}" if metrics else ""))

    env.close()
    print(f"[td3] done: {episode} episodes, {step} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    train(parse_args())
