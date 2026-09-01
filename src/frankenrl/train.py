"""Single seeded training run. One env, one agent, step budget or episode budget.

The loop is agent-agnostic: it never branches on agent type - buffers and
``on_episode_end`` absorb the differences (fixing the legacy ``if label == "M3_MonteCarlo"``
ladder in ``P3run.py``).
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

from frankenrl.agents import build_agent
from frankenrl.buffers.base import Transition
from frankenrl.config import RunConfig, to_dict
from frankenrl.envs import Env
from frankenrl.logging import RunLogger
from frankenrl.seeding import seed_everything


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def evaluate(agent, env_cfg, *, episodes: int, seed: int) -> float:
    """Mean deterministic-policy return over ``episodes``."""
    env = Env(env_cfg, seed=seed + 10_000)
    total = 0.0
    try:
        for _ in range(episodes):
            state = env.reset()
            done = trunc = False
            ep_r = 0.0
            while not (done or trunc):
                action = agent.act(state, deterministic=True)
                r = env.step(action)
                state, done, trunc = r.next_state, bool(r.done), bool(r.truncated)
                ep_r += r.reward
            total += ep_r
    finally:
        env.close()
    return total / episodes


def train(cfg: RunConfig) -> RunLogger:
    device = _resolve_device(cfg.device)
    seed_everything(cfg.seed)

    env = Env(cfg.env, seed=cfg.seed)
    agent = build_agent(
        cfg.agent.kind, env.state_dim, env.action_dim, env.action_scale,
        cfg.agent, device=device, seed=cfg.seed,
    )
    logger = RunLogger(cfg.out_dir, cfg.label, to_dict(cfg))

    print(
        f"[frankenrl] {cfg.label}: {cfg.agent.kind} on {cfg.env.id} "
        f"(S={env.state_dim} A={env.action_dim}) device={device} seed={cfg.seed}"
    )

    step = 0
    episode = 0
    t0 = time.time()
    tconf = cfg.train
    try:
        while step < tconf.total_steps:
            if tconf.max_episodes is not None and episode >= tconf.max_episodes:
                break
            state = env.reset()
            done = trunc = False
            ep_reward = 0.0

            while not (done or trunc):
                if step < cfg.agent.warmup_steps:
                    action = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
                else:
                    action = agent.act(state)

                res = env.step(action)
                agent.observe(
                    Transition(
                        state=state, action=action, reward=res.reward,
                        next_state=res.next_state, done=res.done, truncated=res.truncated,
                    )
                )
                metrics = agent.update(step)

                state = res.next_state
                done, trunc = bool(res.done), bool(res.truncated)
                ep_reward += res.reward
                step += 1

                if step % tconf.log_every_steps == 0 and metrics:
                    sps = step / max(time.time() - t0, 1e-9)
                    logger.log_scalars({**metrics, "perf/steps_per_sec": sps}, step)
                if step % tconf.eval_every_steps == 0:
                    ev = evaluate(agent, cfg.env, episodes=tconf.eval_episodes, seed=cfg.seed)
                    logger.log_scalars({"eval/return": ev}, step)
                    print(f"[frankenrl] step {step:>8} | eval {ev:8.1f}")
                if step % tconf.checkpoint_every_steps == 0:
                    logger.save_checkpoint(step, agent.state_dict())

            agent.on_episode_end()
            logger.log_episode(ep_reward, step)
            episode += 1
            if episode % 10 == 0:
                print(f"[frankenrl] ep {episode:>5} step {step:>8} | reward {ep_reward:8.1f}")
    finally:
        logger.save_checkpoint(step, agent.state_dict())
        logger.finish()
        env.close()

    print(f"[frankenrl] done: {episode} episodes, {step} steps, {time.time() - t0:.0f}s")
    return logger


def train_from_file(path: str, overrides: list[str] | None = None, *, label: str | None = None):
    from frankenrl.config import load_config

    cfg = load_config(path, overrides)
    if label is not None:
        cfg = replace(cfg, label=label)
    return train(cfg)
