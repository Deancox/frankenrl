#!/usr/bin/env python
"""TD3 + SimBa backbone - completely self-contained, single-file reference.

**No paper covers this exact combination.** Lee, Hwang, Kim, Kim, Tai, Subramanian, Wurman,
Choo, Stone & Seno, "SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement
Learning" (ICLR 2025 spotlight, arXiv:2410.09754) applies the SimBa architecture to SAC and
DDPG, not TD3. This file is this codebase's own engineered extrapolation: DDPG+SimBa's exact
architecture and hyperparameters (Table 8 of the paper - critic 2 blocks x 512, actor 1
block x 128, AdamW weight decay 1e-2, replay ratio 2) with TD3's three fixes (Fujimoto, van
Hoof & Meger, ICML 2018, arXiv:1802.09477) layered on top: clipped double-Q, target-policy
smoothing, delayed actor updates. See DeanVault/Wiki/RL/"SimBa - simplicity bias for scaling
reinforcement learning.md" section 3 for why this combination is explicitly unpublished, and
"SAC + SimBa backbone" (standalone/simba_sac.py) for the SimBa architecture itself and the
paper's own SAC recipe, which mostly turns clipped double-Q OFF once the architecture is in
place - an open question this file resolves by KEEPING TD3's clipped double-Q, on the
reasoning that TD3 without it is arguably no longer TD3 in any meaningful sense; flagged
here, not silently decided.

Architecture: running-statistics observation normalisation (RSNorm), a pre-LayerNorm
residual block with an inverted-bottleneck inner MLP and an unnormalised skip, a final
post-LayerNorm before the output heads - identical stack to simba_sac.py, just feeding a
deterministic actor instead of a squashed-Gaussian one.

No dependency on the rest of this repo (CleanRL-style, single file).

Usage:
    uv run python standalone/simba_td3.py --env Pendulum-v1 --seed 0 --total-steps 60000
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
    p.add_argument("--critic-width", type=int, default=512)
    p.add_argument("--critic-blocks", type=int, default=2)
    p.add_argument("--actor-width", type=int, default=128)
    p.add_argument("--actor-blocks", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=5e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--updates-per-step", type=int, default=2, help="SimBa's default replay ratio")
    p.add_argument("--policy-delay", type=int, default=2, help="TD3's actor-update delay")
    p.add_argument("--target-policy-noise", type=float, default=0.2)
    p.add_argument("--target-noise-clip", type=float, default=0.5)
    p.add_argument("--exploration-noise", type=float, default=0.1)
    p.add_argument("--rsnorm-eps", type=float, default=1e-8)
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


def _orthogonal_(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.copy_(sp)


# --------------------------------------------------------------------------- RSNorm + SimBa block
# (identical to simba_sac.py - duplicated on purpose, see that file's header on why)

class RunningNorm(nn.Module):
    """Welford/Chan running mean-variance, updated from batches of real observations."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot_count)
        self.count.copy_(tot_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class SimbaBlock(nn.Module):
    """Pre-LayerNorm residual block, inverted bottleneck (x4), UNNORMALISED skip."""

    def __init__(self, width: int, expansion: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * expansion)
        self.fc2 = nn.Linear(width * expansion, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(self.ln(x)))
        h = self.fc2(h)
        return x + h


def simba_stack(width: int, blocks: int) -> nn.ModuleList:
    return nn.ModuleList(SimbaBlock(width) for _ in range(blocks))


# --------------------------------------------------------------------------- actor / critic

class SimbaDeterministicActor(nn.Module):
    """a = action_bias + action_scale * tanh(pi(s)). Exploration noise added outside, as in
    plain TD3 - SimBa's stochasticity-free actor needs it, unlike simba_sac.py's Gaussian."""

    def __init__(self, state_dim: int, action_dim: int, *, width: int, blocks: int,
                 action_scale: torch.Tensor, action_bias: torch.Tensor, obs_norm: RunningNorm):
        super().__init__()
        self.obs_norm = obs_norm
        self.embed = nn.Linear(state_dim, width)
        self.blocks = simba_stack(width, blocks)
        self.post_ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, action_dim)
        self.apply(lambda m: _orthogonal_(m, math.sqrt(2)))
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.embed(self.obs_norm(state))
        for block in self.blocks:
            x = block(x)
        x = self.post_ln(x)
        return torch.tanh(self.head(x)) * self.action_scale + self.action_bias


class SimbaQCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, *, width: int, blocks: int, obs_norm: RunningNorm):
        super().__init__()
        self.obs_norm = obs_norm
        self.embed = nn.Linear(state_dim + action_dim, width)
        self.blocks = simba_stack(width, blocks)
        self.post_ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        self.apply(lambda m: _orthogonal_(m, math.sqrt(2)))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.obs_norm(state), action], dim=-1)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_ln(x)
        return self.head(x)


class SimbaTwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, *, width: int, blocks: int, obs_norm: RunningNorm):
        super().__init__()
        self.q1 = SimbaQCritic(state_dim, action_dim, width=width, blocks=blocks, obs_norm=obs_norm)
        self.q2 = SimbaQCritic(state_dim, action_dim, width=width, blocks=blocks, obs_norm=obs_norm)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(s, a), self.q2(s, a)

    def q_min(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(s, a)
        return torch.min(q1, q2)


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

class SimbaTD3:
    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        action_scale = (self.action_high - self.action_low) / 2.0
        action_bias = (self.action_high + self.action_low) / 2.0

        self.obs_norm = RunningNorm(state_dim, eps=args.rsnorm_eps).to(device)

        mk_actor = lambda: SimbaDeterministicActor(  # noqa: E731
            state_dim, action_dim, width=args.actor_width, blocks=args.actor_blocks,
            action_scale=action_scale, action_bias=action_bias, obs_norm=self.obs_norm,
        ).to(device)
        self.actor = mk_actor()
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.obs_norm = self.obs_norm  # share live stats, don't freeze a copy
        hard_update(self.actor_target, self.actor)
        self.actor_target.requires_grad_(False)

        self.critics = SimbaTwinQ(
            state_dim, action_dim, width=args.critic_width, blocks=args.critic_blocks, obs_norm=self.obs_norm,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critics)
        self.critic_target.q1.obs_norm = self.obs_norm
        self.critic_target.q2.obs_norm = self.obs_norm
        hard_update(self.critic_target, self.critics)
        self.critic_target.requires_grad_(False)

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.critic_opt = torch.optim.AdamW(self.critics.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self._it = 0
        self._rng = np.random.default_rng(0)

    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        if not deterministic:
            self.obs_norm.update(torch.as_tensor(state, dtype=torch.float32, device=self.device))
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
            actor_loss = -self.critics.q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            soft_update(self.actor_target, self.actor, c.tau)
            soft_update(self.critic_target, self.critics, c.tau)
            metrics["loss/actor"] = float(actor_loss.item())
        return metrics


# --------------------------------------------------------------------------- train loop

def evaluate(agent: SimbaTD3, env_id: str, episodes: int, seed: int) -> float:
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

    agent = SimbaTD3(state_dim, action_dim, action_low, action_high, args, device)
    buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim, device, args.seed)

    print(f"[simba-td3] {args.env} S={state_dim} A={action_dim} device={device} seed={args.seed}")

    step = 0
    episode = 0
    t0 = time.time()
    state, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    while step < args.total_steps:
        if step < args.warmup_steps:
            action = env.action_space.sample()
            agent.obs_norm.update(torch.as_tensor(state, dtype=torch.float32, device=device))
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
                print(f"[simba-td3] ep {episode:>5} step {step:>8} | reward {ep_reward:8.1f}")
            ep_reward = 0.0

        if step % args.eval_every == 0:
            ev = evaluate(agent, args.env, args.eval_episodes, args.seed)
            sps = step / max(time.time() - t0, 1e-9)
            print(f"[simba-td3] step {step:>8} | eval {ev:8.1f} | {sps:6.0f} sps" + (f" | critic {metrics.get('loss/critic', float('nan')):.3f}" if metrics else ""))

    env.close()
    print(f"[simba-td3] done: {episode} episodes, {step} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    train(parse_args())
