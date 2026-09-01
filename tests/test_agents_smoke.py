"""Build each real agent and run a few learning steps on a tiny synthetic MDP.

No Gym dependency: a 3-state / 1-action linear-ish env is enough to exercise shapes,
buffer wiring, target updates and the checkpoint round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from frankenrl.agents import build_agent
from frankenrl.buffers.base import Transition
from frankenrl.config import AgentConfig
from frankenrl.seeding import seed_everything

S, A = 3, 1


def _rollout(agent, steps: int) -> None:
    rng = np.random.default_rng(0)
    state = rng.standard_normal(S).astype(np.float32)
    for t in range(steps):
        action = agent.act(state)
        assert action.shape == (A,)
        assert np.all(np.abs(action) <= 1.0 + 1e-5)
        nxt = (0.9 * state + 0.1 * rng.standard_normal(S)).astype(np.float32)
        done = float(t % 25 == 24)
        agent.observe(
            Transition(state=state, action=action.astype(np.float32),
                       reward=float(-np.sum(state**2)), next_state=nxt,
                       done=done, truncated=0.0)
        )
        agent.update(t)
        state = rng.standard_normal(S).astype(np.float32) if done else nxt
        if done:
            agent.on_episode_end()


@pytest.mark.parametrize("kind", ["sac", "td3"])
def test_agent_builds_learns_and_checkpoints(kind):
    seed_everything(0)
    cfg = AgentConfig(kind=kind, batch_size=32, warmup_steps=20, buffer_capacity=2000)
    cfg.net.hidden = (32, 32)
    agent = build_agent(kind, S, A, action_scale=1.0, cfg=cfg, device="cpu", seed=0)

    _rollout(agent, steps=120)  # past warmup -> at least one real update

    state = agent.state_dict()
    clone = build_agent(kind, S, A, action_scale=1.0, cfg=cfg, device="cpu", seed=1)
    clone.load_state_dict(state)  # must not raise

    a1 = agent.act(np.zeros(S, dtype=np.float32), deterministic=True)
    a2 = clone.act(np.zeros(S, dtype=np.float32), deterministic=True)
    assert np.allclose(a1, a2, atol=1e-5)


@pytest.mark.parametrize("kind", ["ppo", "frankenstein"])
def test_unimplemented_agents_raise_clearly(kind):
    cfg = AgentConfig(kind=kind)
    with pytest.raises(NotImplementedError):
        build_agent(kind, S, A, 1.0, cfg)
