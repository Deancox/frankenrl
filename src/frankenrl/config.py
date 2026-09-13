"""Typed config: YAML file -> nested dataclasses, with ``key.path=value`` CLI overrides.

Every run resolves to one ``RunConfig``; it is serialised into the checkpoint so a result
is always traceable to the exact settings that produced it
(see [[Self-describing RL checkpoints]]).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


@dataclass
class EnvConfig:
    id: str = "Pendulum-v1"
    action_scale: float = 1.0
    normalize_obs: bool = False
    max_episode_steps: int | None = None


@dataclass
class NetConfig:
    hidden: tuple[int, ...] = (256, 256)
    activation: str = "mish"
    layernorm: bool = False


@dataclass
class AdvantageConfig:
    name: str = "gae"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_expected_samples: int = 10
    normalize: bool = True


@dataclass
class BroConfig:
    """BRO-specific hyperparameters (Nauman et al., NeurIPS 2024, arXiv:2405.16158).

    See [[BRO - scaling off-policy actor-critic RL with regularized critics and optimistic
    exploration]]. ``optimism_coef``/``kl_coef``/``weight_decay`` are not pinned down by the
    paper as read for that document - defaults below are this codebase's own picks, not
    literature values; expect to sweep them (`scripts/sweep.py`).
    """

    critic_width: int = 512
    critic_blocks: int = 2
    n_quantiles: int = 100
    # env-step checkpoints at which every network is fully re-initialised (buffer kept).
    # paper's own 1M-step schedule; rescale per task step budget (see configs/bro_pendulum.yaml).
    reset_schedule: tuple[int, ...] = (15_000, 50_000, 250_000, 500_000, 750_000, 1_000_000)
    reset_optimizer_state: bool = True
    optimism_coef: float = 0.5        # beta^o: exploration actor's disagreement bonus weight
    kl_coef: float = 0.1              # tau: KL(pi_p || pi_o) regulariser weight
    weight_decay: float = 1e-2        # AdamW, actors + critic
    quantile_huber_kappa: float = 1.0


@dataclass
class AgentConfig:
    kind: str = "sac"                 # sac | td3 | ppo | frankenstein | bro
    gamma: float = 0.99
    tau: float = 0.005               # Polyak factor for target nets
    lr: float = 3e-4
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    warmup_steps: int = 1_000        # random actions before learning
    updates_per_step: int = 1        # also BRO's critic replay ratio (paper default: 10)
    # SAC / entropy (also used by BRO's pi_p)
    entropy_coef: float = 0.2
    autotune_entropy: bool = True
    target_entropy: float | None = None
    # TD3
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    exploration_noise: float = 0.1
    # PPO / on-policy
    rollout_steps: int = 2048
    ppo_epochs: int = 10
    ppo_clip: float = 0.2
    # Frankenstein composition
    policy_loss: str = "sac"         # sac | dpg | ppo_clip
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    net: NetConfig = field(default_factory=NetConfig)
    bro: BroConfig = field(default_factory=BroConfig)


@dataclass
class TrainConfig:
    total_steps: int = 300_000
    eval_every_steps: int = 10_000
    eval_episodes: int = 5
    log_every_steps: int = 1_000
    checkpoint_every_steps: int = 50_000
    max_episodes: int | None = None  # legacy runs count episodes; either bound works


@dataclass
class RunConfig:
    label: str = "run"
    seed: int = 0
    device: str = "auto"             # auto | cpu | cuda
    out_dir: str = "runs"
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --------------------------------------------------------------------------- loading


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return data
    # `from __future__ import annotations` makes f.type a string; resolve to real objects.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"{cls.__name__} has no field {key!r}")
        ftype = hints.get(key)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _from_dict(ftype, value)
        elif get_origin(ftype) is tuple and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load a YAML file into a ``RunConfig``, then apply ``a.b.c=value`` overrides."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _from_dict(RunConfig, raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key.path=value, got {item!r}")
        dotted, value = item.split("=", 1)
        _apply_override(cfg, dotted.split("."), yaml.safe_load(value))
    return cfg


def _apply_override(obj: Any, path: list[str], value: Any) -> None:
    for part in path[:-1]:
        obj = getattr(obj, part)
    leaf = path[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"no config field at {'.'.join(path)}")
    current = getattr(obj, leaf)
    if isinstance(current, tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(obj, leaf, value)


def to_dict(cfg: Any) -> dict[str, Any]:
    """Plain-dict form for serialising into checkpoints / logs."""
    return dataclasses.asdict(cfg)
