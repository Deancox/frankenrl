#!/usr/bin/env python
"""SimBa-SAC + BRO + TD7: a combined off-policy actor-critic - completely self-contained,
single-file reference. **Unpublished combination** - no paper tests this mixture; every
grafting decision below is this codebase's own engineering choice, flagged where it departs
from a source paper's own design.

Base algorithm: SAC (max-entropy objective, reparameterised squashed-Gaussian actor(s), twin
critics) - the lineage BRO and SimBa-SAC both already share, NOT TD7's deterministic TD3
lineage. This is the starting point per instruction ("start with the SAC variant of SimBa"):
`simba_sac.py`'s backbone (RSNorm input normalisation + SimBa's pre-LN residual block) is the
one architecture used for every network in this file - actor(s) and critics alike. TD7's own
input-normalisation scheme (AvgL1Norm on a learned first linear layer) is dropped in favour of
RSNorm everywhere, including inside the SALE encoder, so the whole agent shares one
normalisation scheme rather than two competing ones.

What each source paper contributes, concretely:

- **SimBa** (backbone): `RunningNorm` (RSNorm) + `SimbaBlock` (pre-LN, x4 inverted-bottleneck,
  UNNORMALISED skip) for actor and critic trunks. AdamW weight decay throughout (SimBa and BRO
  already agree on this). Clipped double-Q stays off by default (`Q^mu` = mean of the twin
  critics, matching both SimBa's and BRO's "off" convention), consistent with the note below.
- **BRO** (regularisation + exploration): the dual-actor scheme - `pi_p` (proposal actor,
  ordinary SAC objective against `Q^mu`) and `pi_o` (exploration actor, the ONLY one `act()`
  samples, trained to maximise `Q^mu + optimism_coef * Q^sigma` minus a closed-form KL pull
  back toward `pi_p`) - plus the full-parameter reset schedule and BRO's default high critic
  replay ratio (`updates_per_step`, default 10). Deviation: BRO's `Q^sigma` is disagreement
  across a quantile ensemble; this file has no quantile critic (SimBa's critic is a plain
  scalar-Q twin), so `Q^sigma` here is `|Q1 - Q2|` - twin-critic disagreement, the same
  epistemic-uncertainty proxy REDQ-style ensembles use (see
  `DeanVault/Concepts/REDQ high update-to-data ratio.md`), not BRO's own quantile spread.
- **TD7** (representation + replay + eval policy): the SALE encoder (`f(s)=zs`,
  `g(zs,a)=zsa`, self-supervised next-zs prediction loss, its own Adam optimiser, ZERO
  gradient from the RL losses ever reaching it) is grafted in as extra concatenated features -
  `zs` into every actor's embed input, `zs` AND `zsa` into every critic's - alongside the
  RSNorm-normalised raw state, not replacing it. Three encoder copies (live / fixed /
  fixed_target) on TD7's own hard-copy refresh clock (`target_update_rate`), independent of
  the critics' ordinary Polyak averaging (`tau`) - SimBa/BRO's soft target update is kept for
  critics, TD7's hard-copy is kept ONLY for the encoder pair, because SALE's own rationale
  (frozen, periodically-refreshed features) is what motivated a hard-copy clock in the first
  place, not a stylistic TD7 vs. SAC choice. LAP (Loss-Adjusted Prioritised replay, sum-tree
  based) replaces the plain uniform buffer, paired with a Huber critic loss so no
  importance-sampling correction is needed - same design as `td7.py`. Policy checkpoints
  decouple the data-collecting/deployed policy from the one being trained: `pi_o` + the
  `fixed_encoder` snapshot are what get checkpointed and evaluated, matching TD7's own
  single-actor convention (mirrored onto `pi_o` since it is the actor that touches the env).

Deliberately out of scope (kept the surface area bounded): TD7's target-policy smoothing and
delayed actor updates (both TD3/deterministic-policy-specific; a stochastic Gaussian actor
already supplies its own smoothing via sampling noise); TD7's value-clipping against a running
Q-target range (an offline-RL extrapolation-error fix; SAC's entropy regularisation already
bounds targets differently, and this file is online-only); the encoder and RSNorm running
statistics are NEVER touched by the reset schedule - both represent accumulated, slowly-built
knowledge (a learned representation, running observation statistics) that BRO's primacy-bias
argument was never aimed at; resetting them would just discard signal for no stated benefit.

Every "own choice, not a literature value" from `bro.py`, `td7.py`, and `simba_sac.py`
inherits into this file unchanged (`optimism_coef`, `kl_coef`, `weight_decay`, reset schedule,
LAP alpha/min-priority, Huber delta, checkpoint schedule, RSNorm epsilon) - see those three
scripts' own module docstrings and `standalone/README.md`'s "Untuned hyperparameters" section
for the full list. Treat any run of this file as a correctness/plumbing check, not a benchmark
result: it has not been tuned, and no ablation has been run to confirm any one of the three
graftings actually helps here.

No dependency on the rest of this repo (CleanRL-style, single file).

Usage:
    uv run python standalone/simba_bro_td7.py --env Pendulum-v1 --seed 0 --total-steps 60000
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
    # SimBa backbone (actor + critic trunks)
    p.add_argument("--critic-width", type=int, default=512)
    p.add_argument("--critic-blocks", type=int, default=2)
    p.add_argument("--actor-width", type=int, default=128)
    p.add_argument("--actor-blocks", type=int, default=1)
    p.add_argument("--rsnorm-eps", type=float, default=1e-8)
    # SALE encoder (TD7)
    p.add_argument("--zs-dim", type=int, default=256)
    p.add_argument("--encoder-hdim", type=int, default=256)
    p.add_argument("--encoder-lr", type=float, default=3e-4, help="TD7's own encoder lr, decoupled from --lr")
    p.add_argument("--target-update-rate", type=int, default=250, help="hard-refresh clock for the encoder pair only")
    # optimisation
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=5e-3, help="Polyak rate for the critic target only")
    p.add_argument("--lr", type=float, default=1e-4, help="actor(s)/critic/alpha lr, matches SimBa")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW, actors + critic")
    p.add_argument("--updates-per-step", type=int, default=10, help="BRO's critic replay ratio")
    # LAP replay (TD7)
    p.add_argument("--lap-alpha", type=float, default=0.4)
    p.add_argument("--lap-min-priority", type=float, default=1.0)
    p.add_argument("--huber-delta", type=float, default=1.0)
    # BRO dual-actor + resets
    p.add_argument("--optimism-coef", type=float, default=0.5, help="beta^o")
    p.add_argument("--kl-coef", type=float, default=0.1, help="tau, KL(pi_p||pi_o) weight")
    p.add_argument("--reset-schedule", type=int, nargs="*", default=[15_000, 50_000, 250_000, 500_000, 750_000, 1_000_000])
    p.add_argument("--no-autotune", action="store_true")
    p.add_argument("--entropy-coef", type=float, default=0.2, help="used only if --no-autotune")
    # TD7 policy checkpoints
    p.add_argument("--checkpoint-steps-before", type=int, default=20_000, help="simplified schedule, see td7.py's module docstring")
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


# --------------------------------------------------------------------------- shared helpers

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


def reinit_module_(module: nn.Module, *, orthogonal_gain: float = math.sqrt(2)) -> None:
    """BRO's primacy-bias reset: re-initialise every Linear/LayerNorm submodule in place.
    Never applied to the encoder or the RSNorm running stats - see module docstring."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=orthogonal_gain)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            m.reset_parameters()


def avg_l1_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """AvgL1Norm (TD7): rescale to unit mean-absolute-value per row, so an embedding sits at
    the same scale as the RSNorm-normalised features it gets concatenated with."""
    return x / (x.abs().mean(dim=-1, keepdim=True) + eps)


# --------------------------------------------------------------------------- RSNorm (SimBa)

class RunningNorm(nn.Module):
    """Welford/Chan running mean-variance. Shared by every network in this file (actor(s),
    critics, and the SALE encoder) so there is exactly one normalisation scheme, not two
    competing ones - see module docstring."""

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


# --------------------------------------------------------------------------- SimBa block

class SimbaBlock(nn.Module):
    """Pre-LayerNorm residual block, inverted bottleneck (x4 expansion), UNNORMALISED skip:
    x_{l+1} = x_l + MLP(LayerNorm(x_l)) - see simba_sac.py for the full rationale."""

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


# --------------------------------------------------------------------------- SALE encoder (TD7)

class Encoder(nn.Module):
    """f(s) = zs (AvgL1Norm'd), g(zs, a) = zsa (not normalised). Consumes RSNorm-normalised
    state (not TD7's own AvgL1Norm-on-a-linear-layer scheme - see module docstring). Three
    independent copies exist (live, fixed, fixed_target), exactly as in td7.py."""

    def __init__(self, state_dim: int, action_dim: int, zs_dim: int, hdim: int, obs_norm: RunningNorm):
        super().__init__()
        self.obs_norm = obs_norm
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
        return avg_l1_norm(self.zs_mlp(self.obs_norm(state)))

    def zsa(self, zs: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
        return self.zsa_mlp(torch.cat([zs, action_norm], dim=-1))


# --------------------------------------------------------------------------- actor / critic

class HybridActor(nn.Module):
    """SimBa trunk (RSNorm -> embed -> SimbaBlocks -> post-LN) with `zs` concatenated into the
    embed input. Outputs tanh(pre_tanh) DIRECTLY in [-1, 1] (TD7's normalised-action
    convention), not multiplied by action_scale internally - see `HybridAgent.to_env_action`."""

    def __init__(self, state_dim: int, action_dim: int, zs_dim: int, *, width: int, blocks: int, obs_norm: RunningNorm):
        super().__init__()
        self.obs_norm = obs_norm
        self.embed = nn.Linear(state_dim + zs_dim, width)
        self.blocks = simba_stack(width, blocks)
        self.post_ln = nn.LayerNorm(width)
        self.mu_head = nn.Linear(width, action_dim)
        self.log_std_head = nn.Linear(width, action_dim)
        self.apply(lambda m: _orthogonal_(m, math.sqrt(2)))

    def _trunk(self, state: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.obs_norm(state), zs], dim=-1)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.post_ln(x)

    def distribution(self, state: torch.Tensor, zs: torch.Tensor) -> Normal:
        x = self._trunk(state, zs)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mu, log_std.exp())

    def sample(self, state: torch.Tensor, zs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state, zs)
        pre_tanh = dist.mean if deterministic else dist.rsample()
        action_norm = torch.tanh(pre_tanh)  # in [-1, 1], unscaled
        log_prob = dist.log_prob(pre_tanh)
        log_prob = log_prob - 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        return action_norm, log_prob.sum(dim=-1, keepdim=True)


class HybridQCritic(nn.Module):
    """SimBa trunk with `[action_norm, zsa, zs]` concatenated into the embed input alongside
    RSNorm-normalised state - a plain scalar-Q critic (no quantile ensemble; see module
    docstring for how BRO's Q^sigma is adapted to this)."""

    def __init__(self, state_dim: int, action_dim: int, zs_dim: int, *, width: int, blocks: int, obs_norm: RunningNorm):
        super().__init__()
        self.obs_norm = obs_norm
        self.embed = nn.Linear(state_dim + action_dim + 2 * zs_dim, width)
        self.blocks = simba_stack(width, blocks)
        self.post_ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        self.apply(lambda m: _orthogonal_(m, math.sqrt(2)))

    def forward(self, state: torch.Tensor, action_norm: torch.Tensor, zsa: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.obs_norm(state), action_norm, zsa, zs], dim=-1)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_ln(x)
        return self.head(x)


class HybridTwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, zs_dim: int, *, width: int, blocks: int, obs_norm: RunningNorm):
        super().__init__()
        self.q1 = HybridQCritic(state_dim, action_dim, zs_dim, width=width, blocks=blocks, obs_norm=obs_norm)
        self.q2 = HybridQCritic(state_dim, action_dim, zs_dim, width=width, blocks=blocks, obs_norm=obs_norm)

    def forward(self, s, a_norm, zsa, zs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(s, a_norm, zsa, zs), self.q2(s, a_norm, zsa, zs)

    def q_mean(self, s, a_norm, zsa, zs) -> torch.Tensor:
        q1, q2 = self(s, a_norm, zsa, zs)
        return 0.5 * (q1 + q2)

    def q_disagreement(self, s, a_norm, zsa, zs) -> torch.Tensor:
        """|Q1 - Q2|: this file's substitute for BRO's quantile-ensemble Q^sigma - see module
        docstring."""
        q1, q2 = self(s, a_norm, zsa, zs)
        return (q1 - q2).abs()

    def optimistic_q(self, s, a_norm, zsa, zs, *, optimism_coef: float) -> torch.Tensor:
        q1, q2 = self(s, a_norm, zsa, zs)
        return 0.5 * (q1 + q2) + optimism_coef * (q1 - q2).abs()


# --------------------------------------------------------------------------- LAP replay buffer (TD7)

class SumTree:
    """Fixed-capacity binary sum-tree: O(log n) priority update and sampling - identical to
    td7.py's, see its docstring for the power-of-two padding note."""

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
    correction (paired with a Huber critic loss instead) - identical to td7.py's."""

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
        self.max_priority = min_priority

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


# --------------------------------------------------------------------------- policy checkpoints (TD7)

class Checkpointer:
    """Decouples the data-collecting policy (`pi_o`) from the evaluated/deployed one -
    identical to td7.py's, see its docstring for the simplified-schedule caveat."""

    def __init__(self, steps_before: int, reset_weight: float, late_eps: int = 10):
        self.steps_before = steps_before
        self.reset_weight = reset_weight
        self.late_eps = late_eps
        self.best_min_return = -1e8
        self._min_return_cur = 1e8
        self._eps_since_update = 0
        self.max_eps_before_update = 1

    def episode_end(self, ep_return: float, step: int) -> bool | None:
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

class HybridAgent:
    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self._reset_steps = set(args.reset_schedule)
        self._resets_done: set[int] = set()

        self.action_scale = torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32, device=device)
        self.action_bias = torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32, device=device)

        self.obs_norm = RunningNorm(state_dim, eps=args.rsnorm_eps).to(device)

        zs_dim = args.zs_dim
        mk_enc = lambda: Encoder(state_dim, action_dim, zs_dim, args.encoder_hdim, self.obs_norm).to(device)  # noqa: E731
        self.encoder = mk_enc()
        self.fixed_encoder = mk_enc()
        self.fixed_encoder_target = mk_enc()
        hard_update(self.fixed_encoder, self.encoder)
        hard_update(self.fixed_encoder_target, self.encoder)
        self.fixed_encoder.requires_grad_(False)
        self.fixed_encoder_target.requires_grad_(False)

        mk_actor = lambda: HybridActor(state_dim, action_dim, zs_dim, width=args.actor_width, blocks=args.actor_blocks, obs_norm=self.obs_norm).to(device)  # noqa: E731
        self.pi_p = mk_actor()
        self.pi_o = mk_actor()

        self.critics = HybridTwinQ(state_dim, action_dim, zs_dim, width=args.critic_width, blocks=args.critic_blocks, obs_norm=self.obs_norm).to(device)
        self.critic_target = copy.deepcopy(self.critics)
        self.critic_target.q1.obs_norm = self.obs_norm
        self.critic_target.q2.obs_norm = self.obs_norm
        hard_update(self.critic_target, self.critics)
        self.critic_target.requires_grad_(False)

        # policy checkpoint: snapshot of pi_o + the frozen encoder it was trained against
        self.checkpoint_actor = mk_actor()
        self.checkpoint_encoder = mk_enc()
        hard_update(self.checkpoint_actor, self.pi_o)
        hard_update(self.checkpoint_encoder, self.fixed_encoder)
        self.checkpoint_encoder.requires_grad_(False)
        self.checkpointer = Checkpointer(args.checkpoint_steps_before, args.checkpoint_reset_weight)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self._build_optimizers()

        self.autotune = not args.no_autotune
        if self.autotune:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        else:
            self.fixed_alpha = float(args.entropy_coef)

        self._it = 0

    def _build_optimizers(self) -> None:
        wd = self.args.weight_decay
        self.pi_p_opt = torch.optim.AdamW(self.pi_p.parameters(), lr=self.args.lr, weight_decay=wd)
        self.pi_o_opt = torch.optim.AdamW(self.pi_o.parameters(), lr=self.args.lr, weight_decay=wd)
        self.critic_opt = torch.optim.AdamW(self.critics.parameters(), lr=self.args.lr, weight_decay=wd)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp() if self.autotune else torch.tensor(self.fixed_alpha, device=self.device)

    def to_env_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        return action_norm * self.action_scale + self.action_bias

    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False, use_checkpoint: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not deterministic and not use_checkpoint:
            self.obs_norm.update(s)
        if use_checkpoint:
            zs = self.checkpoint_encoder.zs(s)
            a, _ = self.checkpoint_actor.sample(s, zs, deterministic=True)
        else:
            zs = self.encoder.zs(s)  # LIVE encoder for data collection, not the frozen copy
            a, _ = self.pi_o.sample(s, zs, deterministic=deterministic)
        env_action = self.to_env_action(a).squeeze(0).cpu().numpy()
        return np.clip(env_action, self.action_bias.cpu().numpy() - self.action_scale.cpu().numpy(),
                        self.action_bias.cpu().numpy() + self.action_scale.cpu().numpy())

    def maybe_reset(self, step: int) -> bool:
        """BRO's primacy-bias reset - actors + critics only. The encoder, its optimiser, and
        the RSNorm running stats are intentionally never reset (see module docstring)."""
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

    def update(self, buffer: LAPReplayBuffer) -> dict[str, float]:
        self._it += 1
        metrics = self._learn_encoder(buffer)

        crit_metrics: dict[str, float] = {}
        for _ in range(self.args.updates_per_step):
            crit_metrics = self._learn_critic(buffer)
        metrics.update(crit_metrics)
        metrics.update(self._learn_actors(buffer))

        if self._it % self.args.target_update_rate == 0:
            hard_update(self.fixed_encoder_target, self.fixed_encoder)  # old fixed -> target
            hard_update(self.fixed_encoder, self.encoder)               # live -> new fixed
            metrics["encoder_hard_refresh"] = 1.0

        return metrics

    def _learn_encoder(self, buffer: LAPReplayBuffer) -> dict[str, float]:
        _, (s, a, _r, s2, _done) = buffer.sample(self.args.batch_size)
        a_norm = (a - self.action_bias) / self.action_scale
        with torch.no_grad():
            next_zs_target = self.encoder.zs(s2)
        zs = self.encoder.zs(s)
        pred_zs = self.encoder.zsa(zs, a_norm)
        encoder_loss = F.mse_loss(pred_zs, next_zs_target)
        self.encoder_opt.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_opt.step()
        return {"loss/encoder": float(encoder_loss.item())}

    def _learn_critic(self, buffer: LAPReplayBuffer) -> dict[str, float]:
        c = self.args
        idx, (s, a, r, s2, done) = buffer.sample(c.batch_size)
        a_norm = (a - self.action_bias) / self.action_scale
        gamma, alpha = c.gamma, self.alpha

        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(s)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, a_norm)

            next_fixed_zs = self.fixed_encoder_target.zs(s2)
            next_action, next_logp = self.pi_p.sample(s2, next_fixed_zs)
            next_zsa = self.fixed_encoder_target.zsa(next_fixed_zs, next_action)
            q1_next, q2_next = self.critic_target(s2, next_action, next_zsa, next_fixed_zs)
            q_next = 0.5 * (q1_next + q2_next) - alpha * next_logp  # mean, not min - clipped-double-Q off (SimBa/BRO convention)
            q_target = r + gamma * (1.0 - done) * q_next

        q1, q2 = self.critics(s, a_norm, fixed_zsa, fixed_zs)
        critic_loss = F.huber_loss(q1, q_target, delta=c.huber_delta) + F.huber_loss(q2, q_target, delta=c.huber_delta)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        soft_update(self.critic_target, self.critics, c.tau)

        td_error = torch.max((q1 - q_target).abs(), (q2 - q_target).abs()).detach().cpu().numpy().squeeze(-1)
        buffer.update_priorities(idx, td_error)

        return {"loss/critic": float(critic_loss.item())}

    def _learn_actors(self, buffer: LAPReplayBuffer) -> dict[str, float]:
        _, (s, *_rest) = buffer.sample(self.args.batch_size)
        alpha = self.alpha

        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(s)

        pi_p_action, pi_p_logp = self.pi_p.sample(s, fixed_zs)
        pi_p_zsa = self.fixed_encoder.zsa(fixed_zs, pi_p_action)
        q_mu = self.critics.q_mean(s, pi_p_action, pi_p_zsa, fixed_zs)
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

        pi_o_action, _ = self.pi_o.sample(s, fixed_zs)
        pi_o_zsa = self.fixed_encoder.zsa(fixed_zs, pi_o_action)
        q_opt = self.critics.optimistic_q(s, pi_o_action, pi_o_zsa, fixed_zs, optimism_coef=self.args.optimism_coef)
        with torch.no_grad():
            pi_p_dist = self.pi_p.distribution(s, fixed_zs)
        pi_o_dist = self.pi_o.distribution(s, fixed_zs)
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

    def checkpoint_if_due(self, ep_return: float, step: int) -> str | None:
        """Feed one episode's return in; returns 'checkpointed', 'reset', or None."""
        self.checkpointer.maybe_decay(step)
        decision = self.checkpointer.episode_end(ep_return, step)
        if decision is None:
            return None
        if decision:
            hard_update(self.checkpoint_actor, self.pi_o)
            hard_update(self.checkpoint_encoder, self.fixed_encoder)
            return "checkpointed"
        return "reset"


# --------------------------------------------------------------------------- train loop

def evaluate(agent: HybridAgent, env_id: str, episodes: int, seed: int) -> float:
    """Uses the CHECKPOINTED pi_o + its associated frozen encoder, not the live one - the
    policy this agent reports and the policy it is training are different objects, exactly
    as in td7.py."""
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

    agent = HybridAgent(state_dim, action_dim, action_low, action_high, args, device)
    buffer = LAPReplayBuffer(args.buffer_size, state_dim, action_dim, device, args.seed,
                              alpha=args.lap_alpha, min_priority=args.lap_min_priority)

    print(f"[simba-bro-td7] {args.env} S={state_dim} A={action_dim} device={device} seed={args.seed} "
          f"critic_width={args.critic_width} replay_ratio={args.updates_per_step}")

    step = 0
    episode = 0
    t0 = time.time()
    state, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    while step < args.total_steps:
        reset_now = agent.maybe_reset(step)
        if reset_now:
            print(f"[simba-bro-td7] step {step:>8} | full parameter reset")

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
            outcome = agent.checkpoint_if_due(ep_reward, step)
            if outcome:
                print(f"[simba-bro-td7] ep {episode:>5} step {step:>8} | {outcome} (best_min_return={agent.checkpointer.best_min_return:.1f})")
            state, _ = env.reset(seed=args.seed + episode)
            if episode % 10 == 0:
                print(f"[simba-bro-td7] ep {episode:>5} step {step:>8} | reward {ep_reward:8.1f}")
            ep_reward = 0.0

        if step % args.eval_every == 0:
            ev = evaluate(agent, args.env, args.eval_episodes, args.seed)
            sps = step / max(time.time() - t0, 1e-9)
            print(f"[simba-bro-td7] step {step:>8} | eval(checkpoint) {ev:8.1f} | {sps:6.0f} sps" + (f" | critic {metrics.get('loss/critic', float('nan')):.3f}" if metrics else ""))

    env.close()
    print(f"[simba-bro-td7] done: {episode} episodes, {step} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    train(parse_args())
