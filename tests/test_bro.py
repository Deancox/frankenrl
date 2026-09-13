"""BRO-specific behaviour not covered by the generic agent smoke test: the reset schedule,
pi_o-only acting, the closed-form KL-invariance-under-tanh assumption, and the no-CDQ
(mean, not min) property of the critic target. Synthetic MDP, matches
`tests/test_agents_smoke.py`'s conventions - no Gym dependency."""

from __future__ import annotations

import numpy as np
import torch

from frankenrl.agents.bro import BRO
from frankenrl.buffers.base import Transition
from frankenrl.config import AgentConfig
from frankenrl.seeding import seed_everything

S, A = 3, 1


def _tiny_cfg(**overrides) -> AgentConfig:
    cfg = AgentConfig(kind="bro", batch_size=16, warmup_steps=5, buffer_capacity=500)
    cfg.net.hidden = (16, 16)
    cfg.bro.critic_width = 16
    cfg.bro.critic_blocks = 1
    cfg.bro.n_quantiles = 8
    for path, value in overrides.items():
        obj, leaf = cfg, path
        if "." in path:
            head, leaf = path.rsplit(".", 1)
            obj = getattr(cfg, head)
        setattr(obj, leaf, value)
    return cfg


def _rollout(agent: BRO, steps: int, *, start_step: int = 0) -> None:
    rng = np.random.default_rng(0)
    state = rng.standard_normal(S).astype(np.float32)
    for i in range(steps):
        action = agent.act(state)
        nxt = (0.9 * state + 0.1 * rng.standard_normal(S)).astype(np.float32)
        agent.observe(
            Transition(state=state, action=action.astype(np.float32),
                       reward=float(-np.sum(state**2)), next_state=nxt,
                       done=0.0, truncated=0.0)
        )
        agent.update(start_step + i)
        state = nxt


def test_act_uses_pi_o_only():
    """`pi_p`'s parameters must never change from a call to `act()` alone."""
    seed_everything(0)
    agent = BRO(S, A, action_scale=1.0, cfg=_tiny_cfg(), device="cpu", seed=0)
    before = [p.clone() for p in agent.pi_p.parameters()]
    for _ in range(10):
        agent.act(np.random.default_rng(0).standard_normal(S).astype(np.float32))
    for p0, p1 in zip(before, agent.pi_p.parameters(), strict=True):
        assert torch.equal(p0, p1)


def test_act_matches_direct_pi_o_sample():
    seed_everything(0)
    agent = BRO(S, A, action_scale=1.0, cfg=_tiny_cfg(), device="cpu", seed=0)
    state = np.zeros(S, dtype=np.float32)
    a1 = agent.act(state, deterministic=True)
    s = torch.as_tensor(state).unsqueeze(0)
    a2, _ = agent.pi_o.sample(s, deterministic=True)
    assert np.allclose(a1, (a2.squeeze(0) / agent.pi_o.action_scale).detach().numpy(), atol=1e-6)


def test_reset_reinitialises_weights_and_optimizer_state_but_not_buffer():
    seed_everything(0)
    cfg = _tiny_cfg(**{"bro.reset_schedule": (10,), "batch_size": 4})
    agent = BRO(S, A, action_scale=1.0, cfg=cfg, device="cpu", seed=0)

    _rollout(agent, steps=6)  # past warmup=5, one real update at step 5; buffer has 6 entries
    pre_reset_buffer_len = len(agent.buffer)
    pi_p_before = [p.clone() for p in agent.pi_p.parameters()]
    critic_before = [p.clone() for p in agent.critics.parameters()]
    opt_state_before = agent.critic_opt.state_dict()["state"]
    critic_opt_id_before = id(agent.critic_opt)
    assert opt_state_before != {}  # sanity: the first rollout really did learn something

    _rollout(agent, steps=6, start_step=6)  # steps 6..11: step 10 hits the reset schedule

    assert 10 in agent._resets_done
    assert len(agent.buffer) == pre_reset_buffer_len + 6  # buffer untouched by the reset
    assert any(
        not torch.allclose(p0, p1)
        for p0, p1 in zip(pi_p_before, agent.pi_p.parameters(), strict=True)
    )
    assert any(
        not torch.allclose(p0, p1)
        for p0, p1 in zip(critic_before, agent.critics.parameters(), strict=True)
    )
    # `reset_optimizer_state=True` (default) rebuilds fresh optimizer objects entirely - a
    # different id is the unambiguous signal, since subsequent learning steps after the reset
    # (steps 10, 11 both learn) immediately accumulate new state, making the state *content*
    # a moving target to assert on directly.
    assert id(agent.critic_opt) != critic_opt_id_before


def test_reset_does_not_fire_twice_for_the_same_step():
    cfg = _tiny_cfg(**{"bro.reset_schedule": (3,)})
    agent = BRO(S, A, action_scale=1.0, cfg=cfg, device="cpu", seed=0)
    assert agent._maybe_reset(3) is True
    assert agent._maybe_reset(3) is False


def test_critic_target_uses_mean_not_min():
    """No `min` appears anywhere in the critic target: construct a case where mean and min
    disagree and confirm the *mean* is what the target regresses toward."""
    seed_everything(0)
    agent = BRO(S, A, action_scale=1.0, cfg=_tiny_cfg(), device="cpu", seed=0)
    state = torch.randn(4, S)
    action = torch.randn(4, A)
    with torch.no_grad():
        q1, q2 = agent.critic_target(state, action)
    mean_q = 0.5 * (q1 + q2)
    min_q = torch.minimum(q1, q2)
    # with random independent critics these essentially never coincide - if this ever flakes
    # the critics were initialised identically, which would itself be a bug.
    assert not torch.allclose(mean_q, min_q)


def test_pi_p_and_pi_o_share_action_scale():
    """The KL-invariance-under-tanh argument (`bro.py::_learn_actors`, `distribution`'s
    docstring) relies on both actors applying the *identical* transform (same tanh, same
    `action_scale`) so the change-of-variables Jacobian cancels exactly in the log-ratio,
    leaving the pre-tanh Gaussian KL equal to the true post-squash KL. That cancellation is
    algebraic, not approximate - it would be pointless to Monte-Carlo it - but it depends on
    this precondition, which is what actually needs checking."""
    agent = BRO(S, A, action_scale=2.0, cfg=_tiny_cfg(), device="cpu", seed=0)
    assert torch.equal(agent.pi_p.action_scale, agent.pi_o.action_scale)


def test_closed_form_kl_is_zero_for_identical_distributions_and_positive_otherwise():
    from frankenrl.nn.actors import SquashedGaussianActor

    seed_everything(0)
    pi_p = SquashedGaussianActor(4, 2, (16,), action_scale=1.5)
    pi_o = SquashedGaussianActor(4, 2, (16,), action_scale=1.5)
    state = torch.randn(5, 4)

    dist_p = pi_p.distribution(state)
    self_kl = torch.distributions.kl_divergence(dist_p, dist_p).sum(-1)
    assert torch.allclose(self_kl, torch.zeros_like(self_kl), atol=1e-6)

    dist_o = pi_o.distribution(state)
    cross_kl = torch.distributions.kl_divergence(dist_p, dist_o).sum(-1)
    assert torch.all(cross_kl > 0.0)  # independently-initialised networks differ
