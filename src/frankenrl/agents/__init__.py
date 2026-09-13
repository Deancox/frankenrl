"""Agent registry + factory. ``build_agent(cfg.kind, ...)`` is the only entry point callers need."""

from __future__ import annotations

from frankenrl.agents.base import Agent
from frankenrl.agents.bro import BRO
from frankenrl.agents.frankenstein import Frankenstein
from frankenrl.agents.ppo import PPO
from frankenrl.agents.sac import SAC
from frankenrl.agents.td3 import TD3
from frankenrl.config import AgentConfig

_REGISTRY: dict[str, type[Agent]] = {
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "frankenstein": Frankenstein,
    "bro": BRO,
}


def build_agent(
    kind: str,
    state_dim: int,
    action_dim: int,
    action_scale: float,
    cfg: AgentConfig,
    *,
    device: str = "cpu",
    seed: int | None = None,
) -> Agent:
    if kind not in _REGISTRY:
        raise ValueError(f"unknown agent {kind!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[kind](
        state_dim, action_dim, action_scale, cfg, device=device, seed=seed
    )


__all__ = ["Agent", "SAC", "TD3", "PPO", "Frankenstein", "BRO", "build_agent"]
