#!/usr/bin/env python
"""TD7: For SALE - completely self-contained, single-file reference.

Fujimoto, Chang, Smith, Gu, Precup & Meger, "For SALE: State-Action Representation Learning
for Deep Reinforcement Learning", NeurIPS 2023 (arXiv:2306.02451).

TD3 (kept unchanged: twin critics, clipped double-Q, target-policy smoothing, delayed actor
updates) plus four additions the paper glosses as "TD3+4 additions", hence "TD7": SALE
(self-supervised state-action embeddings, trained on their own next-state-prediction loss,
fed to the actor/critic as FROZEN, periodically-refreshed side-channel features - never as a
replacement for raw state/action), LAP (Loss-Adjusted Prioritized replay: priority
max(|delta|^0.4, 1), paired with a Huber critic loss so no importance-sampling correction is
needed), policy checkpoints (decouple the policy that generates training data from the one
that gets evaluated, gated on a pessimistic worst-episode-in-a-batch criterion), and value
clipping against a running Q-target range (an offline-RL-style extrapolation-error fix,
needed because the embedded state-action space is effectively larger than TD3's raw one).
TD3's Polyak-averaged target update is dropped in favour of a hard copy every
`target_update_rate` steps, applied to the critic target, actor target, and both frozen
encoder copies together - one shared clock for everything "stale but stable".

No dependency on the rest of this repo (CleanRL-style, single file). See DeanVault/Wiki/RL/
"TD7 - state-action representation learning, LAP replay and policy checkpoints on TD3.md"
for the full derivation and the load-bearing fact this implementation is built around: no
gradient from the critic/actor loss ever reaches the encoder, and no gradient from the
encoder loss ever reaches the actor/critic - three optimisers, three disjoint parameter
sets, connected only by frozen, detached embedding values flowing forward.

Simplifications vs. the official `sfujim/TD7` code, flagged because the exact constants were
not recovered from the paper text (see the wiki doc's Caveats / the research report's
Unresolved section): RSNorm's epsilon (1e-8, a conventional default); the Huber loss delta
(1.0, matched to `min_priority`); the policy-checkpoint schedule (`max_eps_before_update`
step-increase and `train_and_reset` burst length are approximated, not the exact official
schedule). Online mode only (the offline BC term, lambda=0, is present but inert by default).

Usage:
    uv run python standalone/td7.py --env Pendulum-v1 --seed 0 --total-steps 60000
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
    p.add_argument("--zs-dim", type=int, default=256)
    p.add_argument("--hdim", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--target-update-rate", type=int, default=250)
    p.add_argument("--policy-freq", type=int, default=2)
    p.add_argument("--target-policy-noise", type=float, default=0.2)
    p.add_argument("--target-noise-clip", type=float, default=0.5)
    p.add_argument("--exploration-noise", type=float, default=0.1)
    p.add_argument("--lap-alpha", type=float, default=0.4)
    p.add_argument("--lap-min-priority", type=float, default=1.0)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--bc-lambda", type=float, default=0.0, help="offline-only BC weight; 0 = online")
    p.add_argument("--checkpoint-steps-before", type=int, default=20_000, help="simplified schedule, see module docstring")
    p.add_argument("--checkpoint-reset-weight", type=float, default=0.9)
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


def hard_update(target: nn.Module, source: nn.Module) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.copy_(sp)


def avg_l1_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """AvgL1Norm: rescale to unit mean-absolute-value per row, so an embedding sits at the
    same scale as the raw, similarly-normalised features it's concatenated with."""
    return x / (x.abs().mean(dim=-1, keepdim=True) + eps)


# --------------------------------------------------------------------------- SALE encoder

class Encoder(nn.Module):
    """f(s) = zs (AvgL1Norm'd), g(zs, a) = zsa (not normalised). Three independent copies
    exist in TD7 (live, fixed, fixed_target) - see the module docstring."""

    def __init__(self, state_dim: int, action_dim: int, zs_dim: int, hdim: int):
        super().__init__()
        self.zs_mlp = nn.Sequential(
            nn.Linear(state_dim, hdim), nn.ELU(),
            nn.Linear(hdim, hdim), nn.ELU(),
            nn.Linear(hdim, zs_dim),
        )
        self.zsa_mlp = nn.Sequential(
            nn.Linear(zs_dim + action_dim, hdim), nn.ELU(),
            nn.Linear(hdim, hdim), nn.ELU(),
            nn.Linear(hdim, zs_dim),
        )
        self.apply(lambda m: _orthogonal_(m, math.sqrt(2)))

    def zs(self, state: torch.Tensor) -> torch.Tensor:
        return avg_l1_norm(self.zs_mlp(state))

    def zsa(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.zsa_mlp(torch.cat([zs, action], dim=-1))


# --------------------------------------------------------------------------- actor / critic

class Actor(nn.Module):
    """a = tanh(...); input = [AvgL1Norm(Linear(s)), zs]. Scaled to the env's action range
    by the agent wrapper, not here (mirrors the paper's own [-1,1]-normalised-action convention)."""

    def __init__(self, state_dim: int, action_dim: int, zs_dim: int, hdim: int):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        self.apply(lambda m: _orthogonal_(m, math.sqrt(2)))

    def forward(self, state: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        a = avg_l1_norm(self.l0(state))
        a = torch.cat([a, zs], dim=-1)
        a = F.elu(self.l1(a))
        a = F.elu(self.l2(a))
        return torch.tanh(self.l3(a))


class QNet(nn.Module):
    """input = [AvgL1Norm(Linear(s,a)), zsa, zs] -> scalar Q."""

    def __init__(self, state_dim: int, action_dim: int, zs_dim: int, hdim: int):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)
        self.apply(lambda m: _orthogonal_(m, math.sqrt(2)))

    def forward(self, state: torch.Tensor, action: torch.Tensor, zsa: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        q = avg_l1_norm(self.q01(sa))
        q = torch.cat([q, zsa, zs], dim=-1)
        q = F.elu(self.q1(q))
        q = F.elu(self.q2(q))
        return self.q3(q)


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, zs_dim: int, hdim: int):
        super().__init__()
        self.qa = QNet(state_dim, action_dim, zs_dim, hdim)
        self.qb = QNet(state_dim, action_dim, zs_dim, hdim)

    def forward(self, s, a, zsa, zs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qa(s, a, zsa, zs), self.qb(s, a, zsa, zs)


# --------------------------------------------------------------------------- LAP replay buffer

class SumTree:
    """Fixed-capacity binary sum-tree: O(log n) priority update and sampling.

    Internally rounds up to the next power of two - the "leaves start at index `capacity`"
    layout only forms a valid complete binary tree when `capacity` itself is a power of two
    (e.g. the default buffer size, 1_000_000, is not). Unused padding leaves are never
    `.set()`, stay at priority 0, and are therefore never sampled.
    """

    def __init__(self, data_capacity: int):
        self.data_capacity = data_capacity
        self.capacity = 1
        while self.capacity < data_capacity:
            self.capacity *= 2
        self.tree = np.zeros(2 * self.capacity)

    def set(self, idx: int, priority: float) -> None:
        i = idx + self.capacity
        self.tree[i] = priority
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def total(self) -> float:
        return float(self.tree[1])

    def sample_index(self, value: float) -> int:
        i = 1
        while i < self.capacity:
            left = 2 * i
            i = left if self.tree[left] >= value else left + 1
            if self.tree[left] < value:
                value -= self.tree[left]
        return i - self.capacity


class LAPReplayBuffer:
    """Loss-Adjusted Prioritized replay: priority = max(|delta|^alpha, 1), no IS-weight
    correction (paired with a Huber critic loss instead - see the wiki doc section 5)."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device,
                 seed: int, alpha: float, min_priority: float):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.min_priority = min_priority
        self.rng = np.random.default_rng(seed)
        self.tree = SumTree(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_priority = min_priority  # new transitions seed at the current max (standard PER convention)

    def add(self, s, a, r, s2, done: float) -> None:
        i = self.ptr
        self.state[i], self.action[i], self.reward[i] = s, a, r
        self.next_state[i], self.done[i] = s2, done
        self.tree.set(i, self.max_priority ** self.alpha)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int):
        total = self.tree.total()
        segment = total / batch_size
        idx = np.array([
            self.tree.sample_index(self.rng.uniform(segment * i, segment * (i + 1)))
            for i in range(batch_size)
        ])
        idx = np.clip(idx, 0, self.size - 1)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        batch = (to(self.state), to(self.action), to(self.reward), to(self.next_state), to(self.done))
        return idx, batch

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.maximum(np.abs(td_errors), self.min_priority) ** self.alpha
        for i, p in zip(idx, priorities):
            self.tree.set(int(i), float(p))
        self.max_priority = max(self.max_priority, float(priorities.max()))


# --------------------------------------------------------------------------- policy checkpoints

class Checkpointer:
    """Decouples the data-collecting policy from the evaluated/deployed one. Simplified
    schedule vs. the official code (see module docstring): `max_eps_before_update` steps
    from 1 to `late_eps` once `steps_before` env steps have passed, rather than the exact
    official ramp."""

    def __init__(self, steps_before: int, reset_weight: float, late_eps: int = 10):
        self.steps_before = steps_before
        self.reset_weight = reset_weight
        self.late_eps = late_eps
        self.best_min_return = -1e8
        self._min_return_cur = 1e8
        self._eps_since_update = 0
        self.max_eps_before_update = 1

    def episode_end(self, ep_return: float, step: int) -> bool | None:
        """Returns True if this batch of episodes should be checkpointed, False if the live
        policy should instead train harder before re-assessing, None if not decision time yet."""
        if step >= self.steps_before:
            self.max_eps_before_update = self.late_eps
        self._min_return_cur = min(self._min_return_cur, ep_return)
        self._eps_since_update += 1
        if self._eps_since_update < self.max_eps_before_update:
            return None
        passed = self._min_return_cur >= self.best_min_return
        if passed:
            self.best_min_return = self._min_return_cur
        self._eps_since_update = 0
        self._min_return_cur = 1e8
        return passed

    def maybe_decay(self, step: int) -> None:
        if self.steps_before > 0 and step % self.steps_before == 0:
            self.best_min_return *= self.reset_weight


# --------------------------------------------------------------------------- agent

class TD7:
    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        zs_dim, hdim = args.zs_dim, args.hdim
        mk_enc = lambda: Encoder(state_dim, action_dim, zs_dim, hdim).to(device)  # noqa: E731
        self.encoder = mk_enc()
        self.fixed_encoder = mk_enc()
        self.fixed_encoder_target = mk_enc()
        hard_update(self.fixed_encoder, self.encoder)
        hard_update(self.fixed_encoder_target, self.encoder)
        self.fixed_encoder.requires_grad_(False)
        self.fixed_encoder_target.requires_grad_(False)

        mk_actor = lambda: Actor(state_dim, action_dim, zs_dim, hdim).to(device)  # noqa: E731
        self.actor = mk_actor()
        self.actor_target = mk_actor()
        hard_update(self.actor_target, self.actor)
        self.actor_target.requires_grad_(False)

        mk_critic = lambda: TwinQ(state_dim, action_dim, zs_dim, hdim).to(device)  # noqa: E731
        self.critics = mk_critic()
        self.critic_target = mk_critic()
        hard_update(self.critic_target, self.critics)
        self.critic_target.requires_grad_(False)

        # policy checkpoint: a separate snapshot, evaluated instead of the live actor/encoder
        self.checkpoint_actor = mk_actor()
        self.checkpoint_encoder = mk_enc()
        hard_update(self.checkpoint_actor, self.actor)
        hard_update(self.checkpoint_encoder, self.encoder)
        self.checkpoint_encoder.requires_grad_(False)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=args.lr)

        self.checkpointer = Checkpointer(args.checkpoint_steps_before, args.checkpoint_reset_weight)

        # running Q-target range for value clipping (extrapolation-error control, wiki §4)
        self._q_max, self._q_min = -1e8, 1e8
        self.max_target, self.min_target = 1e8, -1e8  # open until the first 250-step freeze

        self._it = 0
        self._rng = np.random.default_rng(0)

    def _to_env_action(self, tanh_action: torch.Tensor) -> torch.Tensor:
        return tanh_action * self.action_scale + self.action_bias

    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False, use_checkpoint: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if use_checkpoint:
            zs = self.checkpoint_encoder.zs(s)
            a = self.checkpoint_actor(s, zs)
        else:
            zs = self.encoder.zs(s)  # LIVE encoder for data collection, not the frozen copy
            a = self.actor(s, zs)
        env_action = self._to_env_action(a).squeeze(0).cpu().numpy()
        if not deterministic:
            noise_scale = self.args.exploration_noise * self.action_scale.cpu().numpy()
            env_action = env_action + self._rng.normal(0.0, noise_scale, size=env_action.shape)
        return np.clip(env_action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())

    def update(self, buffer: LAPReplayBuffer) -> dict[str, float]:
        self._it += 1
        c = self.args
        idx, (s, a, r, s2, done) = buffer.sample(c.batch_size)
        # the actor outputs tanh(...) in [-1,1]; buffer actions are real env actions, so map
        # them back to that same normalised space for every network that consumes an action
        # (must match `next_action`'s space below, which comes straight from the actor).
        a_norm = (a - self.action_bias) / self.action_scale

        # 1) encoder update - own optimizer, own loss, no RL gradient anywhere in it
        with torch.no_grad():
            next_zs_target = self.encoder.zs(s2)
        zs = self.encoder.zs(s)
        pred_zs = self.encoder.zsa(zs, a_norm)
        encoder_loss = F.mse_loss(pred_zs, next_zs_target)
        self.encoder_opt.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_opt.step()

        # 2) critic update - FROZEN encoder embeddings only, value-clipped clipped-double-Q target
        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(s)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, a_norm)
            next_fixed_zs = self.fixed_encoder_target.zs(s2)

            # smoothing noise lives in the actor's own normalised [-1,1] action space
            noise = (torch.randn_like(a_norm) * c.target_policy_noise).clamp(
                -c.target_noise_clip, c.target_noise_clip
            )
            next_action = (self.actor_target(s2, next_fixed_zs) + noise).clamp(-1.0, 1.0)
            next_zsa = self.fixed_encoder_target.zsa(next_fixed_zs, next_action)

            q1_next, q2_next = self.critic_target(s2, next_action, next_zsa, next_fixed_zs)
            q_next = torch.min(q1_next, q2_next)
            self._q_max = max(self._q_max, float(q_next.max()))
            self._q_min = min(self._q_min, float(q_next.min()))
            q_target = r + c.gamma * (1.0 - done) * q_next.clamp(self.min_target, self.max_target)

        q1, q2 = self.critics(s, a_norm, fixed_zsa, fixed_zs)
        critic_loss = F.huber_loss(q1, q_target, delta=c.huber_delta) + F.huber_loss(q2, q_target, delta=c.huber_delta)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        td_error = torch.max((q1 - q_target).abs(), (q2 - q_target).abs()).detach().cpu().numpy().squeeze(-1)
        buffer.update_priorities(idx, td_error)

        metrics = {"loss/critic": float(critic_loss.item()), "loss/encoder": float(encoder_loss.item())}

        # 3) actor update - every policy_freq steps, same frozen embeddings, no grad into f,g
        if self._it % c.policy_freq == 0:
            pi_action = self.actor(s, fixed_zs)
            pi_zsa = self.fixed_encoder.zsa(fixed_zs, pi_action)
            q_pi = self.critics.qa(s, pi_action, pi_zsa, fixed_zs)
            actor_loss = -q_pi.mean()
            if c.bc_lambda > 0.0:
                bc_weight = c.bc_lambda / q_pi.abs().mean().detach().clamp(min=1e-6)
                actor_loss = actor_loss + bc_weight * F.mse_loss(pi_action, a_norm)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            metrics["loss/actor"] = float(actor_loss.item())

        # 4) hard-copy schedule - critic/actor targets AND both encoder copies, one shared clock
        if self._it % c.target_update_rate == 0:
            hard_update(self.critic_target, self.critics)
            hard_update(self.actor_target, self.actor)
            hard_update(self.fixed_encoder_target, self.fixed_encoder)  # old fixed -> target
            hard_update(self.fixed_encoder, self.encoder)               # live -> new fixed
            self.max_target, self.min_target = self._q_max, self._q_min
            metrics["target_hard_copy"] = 1.0

        return metrics

    def checkpoint_if_due(self, ep_return: float, step: int) -> str | None:
        """Feed one episode's return in; returns 'checkpointed', 'reset', or None."""
        self.checkpointer.maybe_decay(step)
        decision = self.checkpointer.episode_end(ep_return, step)
        if decision is None:
            return None
        if decision:
            hard_update(self.checkpoint_actor, self.actor)
            hard_update(self.checkpoint_encoder, self.fixed_encoder)
            return "checkpointed"
        return "reset"


# --------------------------------------------------------------------------- train loop

def evaluate(agent: TD7, env_id: str, episodes: int, seed: int) -> float:
    """Uses the CHECKPOINTED actor + its associated frozen encoder, not the live one -
    this is the load-bearing detail the wiki doc flags: "the policy TD7 reports" and "the
    policy TD7 is training" are different objects."""
    env = gym.make(env_id)
    total = 0.0
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + 10_000 + ep)
        done = trunc = False
        while not (done or trunc):
            action = agent.act(state, deterministic=True, use_checkpoint=True)
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

    agent = TD7(state_dim, action_dim, action_low, action_high, args, device)
    buffer = LAPReplayBuffer(args.buffer_size, state_dim, action_dim, device, args.seed,
                              alpha=args.lap_alpha, min_priority=args.lap_min_priority)

    print(f"[td7] {args.env} S={state_dim} A={action_dim} device={device} seed={args.seed}")

    step = 0
    episode = 0
    t0 = time.time()
    state, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    while step < args.total_steps:
        if step < args.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(state)  # live actor + live encoder, exploration noise on

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
            outcome = agent.checkpoint_if_due(ep_reward, step)
            if outcome:
                print(f"[td7] ep {episode:>5} step {step:>8} | {outcome} (best_min_return={agent.checkpointer.best_min_return:.1f})")
            state, _ = env.reset(seed=args.seed + episode)
            if episode % 10 == 0:
                print(f"[td7] ep {episode:>5} step {step:>8} | reward {ep_reward:8.1f}")
            ep_reward = 0.0

        if step % args.eval_every == 0:
            ev = evaluate(agent, args.env, args.eval_episodes, args.seed)
            sps = step / max(time.time() - t0, 1e-9)
            print(f"[td7] step {step:>8} | eval(checkpoint) {ev:8.1f} | {sps:6.0f} sps" + (f" | critic {metrics.get('loss/critic', float('nan')):.3f}" if metrics else ""))

    env.close()
    print(f"[td7] done: {episode} episodes, {step} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    train(parse_args())
