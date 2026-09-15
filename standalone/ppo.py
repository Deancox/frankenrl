#!/usr/bin/env python
"""PPO (Proximal Policy Optimization) - completely self-contained, single-file reference.

Schulman, Wolski, Dhariwal, Radford & Klimov, "Proximal Policy Optimization Algorithms",
2017 (arXiv:1707.06347). The on-policy counterpart to this suite's off-policy scripts
(sac.py, td3.py, bro.py, td7.py, simba_sac.py) - included so the standalone suite can run a
genuinely on-policy baseline alongside the off-policy ones for the Gadi comparison.

Unbounded diagonal Gaussian policy (state-dependent mean, a single learned log-std vector
shared across states - not tanh-squashed, unlike every off-policy actor in this suite; PPO's
own exploration comes from the Gaussian's width, not a bounded-action reparameterisation) +
a separate state-value critic, both plain MLPs. Fixed-length rollouts (`--rollout-steps`,
collected in lockstep across however many episodes fall inside that window) -> GAE
advantages/returns (same recursion as `frankenrl.advantage.gae_targets`: bootstrap V(s') at
every non-terminal step, including a mid-episode time-limit truncation, but reset the GAE
lookahead's temporal accumulation at that truncation boundary - PLAN.md's legacy bug #5) ->
`--ppo-epochs` passes of shuffled minibatches with the clipped surrogate objective, a clipped
value loss (max of the plain and clipped-around-the-rollout-time-value MSE, CleanRL's
`clip_vloss` convention - `--clip-eps` doubles as both the policy ratio clip and the value
clip range, matching CleanRL's default of reusing one constant for both), an entropy bonus,
and gradient-norm clipping.

Implementation details pinned to specific choices, each a well-established convention from
the reference PPO implementations (CleanRL, OpenAI Baselines / "the 37 implementation
details of PPO"), not the paper text itself, which underspecifies most of them: orthogonal
init with gain sqrt(2) on hidden layers, gain 0.01 on the policy mean head (starts the policy
near-linear/low-variance-of-outputs so early updates don't blow up the KL) and gain 1.0 on
the value head; advantages normalised per-minibatch (not once over the whole rollout); the
env is stepped with the action CLIPPED to its bounds, but log-prob/ratio computations use the
raw, unclipped sampled action (both choices are the common convention, not the only
defensible one - e.g. SB3 does the same, other implementations differ).

No dependency on the rest of this repo (CleanRL-style, single file).

Usage:
    uv run python standalone/ppo.py --env Pendulum-v1 --seed 0 --total-steps 60000
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


# --------------------------------------------------------------------------- args

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=60_000)
    p.add_argument("--rollout-steps", type=int, default=2_048)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64], help="PPO's own convention: smaller nets than the off-policy scripts")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
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

def mlp(in_dim: int, out_dim: int, hidden: list[int], *, final_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for w in hidden:
        layers += [nn.Linear(prev, w), nn.Tanh()]
        prev = w
    layers.append(nn.Linear(prev, out_dim))
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(layers[-1].weight, gain=final_gain)  # policy mean: 0.01, value: 1.0
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Unbounded diagonal Gaussian: a ~ N(mu(s), exp(log_std)). log_std is a single learned
    vector, shared across all states (CleanRL's convention, not state-dependent as in the
    squashed actors elsewhere in this suite)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: list[int]):
        super().__init__()
        self.mu_net = mlp(state_dim, action_dim, hidden, final_gain=0.01)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def distribution(self, state: torch.Tensor) -> Normal:
        mu = self.mu_net(state)
        std = self.log_std.expand_as(mu).exp()
        return Normal(mu, std)

    def sample(self, state: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """log_prob and entropy of a GIVEN (already-sampled) action, for the PPO ratio."""
        dist = self.distribution(state)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class VCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: list[int]):
        super().__init__()
        self.net = mlp(state_dim, 1, hidden, final_gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# --------------------------------------------------------------------------- rollout buffer + GAE

class RolloutBuffer:
    """Fixed-length, in-order storage for one rollout. `compute_gae` mirrors
    `frankenrl.advantage.gae_targets`: bootstraps V(s') at every non-terminal step (including
    a mid-episode time-limit truncation - legacy bug #5), but resets the GAE recursion's
    temporal accumulation at a truncation boundary since the "next" step starts a new episode."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((capacity, 1), dtype=np.float32)
        self.log_prob = np.zeros((capacity, 1), dtype=np.float32)
        self.value = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0

    def add(self, s, a, r, s2, done: float, truncated: float, log_prob: float, value: float) -> None:
        i = self.ptr
        self.state[i], self.action[i], self.reward[i] = s, a, r
        self.next_state[i], self.done[i], self.truncated[i] = s2, done, truncated
        self.log_prob[i], self.value[i] = log_prob, value
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.capacity

    def compute_gae(self, critic: VCritic, device: torch.device, *, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        n = self.ptr
        with torch.no_grad():
            next_values = critic(torch.as_tensor(self.next_state[:n], device=device)).cpu().numpy().reshape(-1)
        values = self.value[:n].reshape(-1)
        rewards = self.reward[:n].reshape(-1)
        done = self.done[:n].reshape(-1)
        truncated = self.truncated[:n].reshape(-1)

        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in range(n - 1, -1, -1):
            nonterminal = 1.0 - done[t]
            delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            if truncated[t] > 0:
                gae = delta
            adv[t] = gae
        ret = adv + values
        return ret.reshape(-1, 1), adv.reshape(-1, 1)

    def get(self, ret: np.ndarray, adv: np.ndarray):
        n = self.ptr
        to = lambda a: torch.as_tensor(a[:n], device=self.device)  # noqa: E731
        return to(self.state), to(self.action), to(self.log_prob), to(self.value), \
            torch.as_tensor(ret, device=self.device), torch.as_tensor(adv, device=self.device)

    def clear(self) -> None:
        self.ptr = 0


# --------------------------------------------------------------------------- agent

class PPO:
    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)

        self.actor = GaussianActor(state_dim, action_dim, args.hidden).to(device)
        self.critic = VCritic(state_dim, args.hidden).to(device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=args.lr, eps=1e-5)

    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        """Deterministic eval only - returns the clipped env action (no log_prob/value needed)."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor.sample(s, deterministic=deterministic)
        return a.clamp(self.action_low, self.action_high).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act_for_rollout(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Returns (env_action [clipped], raw_action [unclipped, stored for the ratio],
        log_prob, value). See module docstring for why the clip/no-clip split."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        raw_action, log_prob = self.actor.sample(s)
        value = self.critic(s)
        env_action = raw_action.clamp(self.action_low, self.action_high)
        return (env_action.squeeze(0).cpu().numpy(), raw_action.squeeze(0).cpu().numpy(),
                float(log_prob.item()), float(value.item()))

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        c = self.args
        ret, adv = buffer.compute_gae(self.critic, self.device, gamma=c.gamma, lam=c.gae_lambda)
        states, actions, old_log_probs, old_values, ret_t, adv_t = buffer.get(ret, adv)
        n = states.shape[0]

        metrics: dict[str, float] = {}
        idx = np.arange(n)
        for _epoch in range(c.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, c.minibatch_size):
                mb = idx[start:start + c.minibatch_size]
                mb_states, mb_actions = states[mb], actions[mb]
                mb_old_logp, mb_ret, mb_adv = old_log_probs[mb], ret_t[mb], adv_t[mb]
                mb_old_values = old_values[mb]

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)  # per-minibatch normalisation

                new_logp, entropy = self.actor.evaluate(mb_states, mb_actions)
                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1.0 - c.clip_eps, 1.0 + c.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(mb_states)
                v_loss_unclipped = (value - mb_ret).pow(2)
                v_clipped = mb_old_values + (value - mb_old_values).clamp(-c.clip_eps, c.clip_eps)
                v_loss_clipped = (v_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), c.max_grad_norm)
                self.opt.step()

                metrics = {
                    "loss/policy": float(policy_loss.item()),
                    "loss/value": float(value_loss.item()),
                    "entropy": float(entropy.mean().item()),
                }

        buffer.clear()
        return metrics


# --------------------------------------------------------------------------- train loop

def evaluate(agent: PPO, env_id: str, episodes: int, seed: int) -> float:
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

    agent = PPO(state_dim, action_dim, action_low, action_high, args, device)
    buffer = RolloutBuffer(args.rollout_steps, state_dim, action_dim, device)

    print(f"[ppo] {args.env} S={state_dim} A={action_dim} device={device} seed={args.seed} rollout={args.rollout_steps}")

    step = 0
    episode = 0
    t0 = time.time()
    state, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    next_eval = args.eval_every
    metrics: dict[str, float] = {}
    while step < args.total_steps:
        env_action, raw_action, log_prob, value = agent.act_for_rollout(state)
        next_state, reward, terminated, truncated, _ = env.step(env_action)
        done = float(terminated)
        buffer.add(state, raw_action, reward, next_state, done, float(truncated), log_prob, value)
        ep_reward += reward
        state = next_state
        step += 1

        if terminated or truncated:
            episode += 1
            state, _ = env.reset(seed=args.seed + episode)
            if episode % 10 == 0:
                print(f"[ppo] ep {episode:>5} step {step:>8} | reward {ep_reward:8.1f}")
            ep_reward = 0.0

        if buffer.full():
            metrics = agent.update(buffer)

        if step >= next_eval:
            ev = evaluate(agent, args.env, args.eval_episodes, args.seed)
            sps = step / max(time.time() - t0, 1e-9)
            print(f"[ppo] step {step:>8} | eval {ev:8.1f} | {sps:6.0f} sps" + (f" | policy {metrics.get('loss/policy', float('nan')):.3f}" if metrics else ""))
            next_eval += args.eval_every

    env.close()
    print(f"[ppo] done: {episode} episodes, {step} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    train(parse_args())
