"""BroNet block/critic - shapes, the identity-collapse property, and the wiki's own
worked-example arithmetic (the one piece of BRO math this repo is checked against a fixed,
by-hand-computed number for)."""

from __future__ import annotations

import pytest
import torch

from frankenrl.nn.blocks import reinit_module_
from frankenrl.nn.bronet import (
    BroNetBlock,
    QuantileTwinCritic,
    bronet_critic_trunk,
    optimistic_q_from_quantiles,
    q_disagreement_from_quantiles,
    q_mean_from_quantiles,
    quantile_huber_loss,
)


def test_bronet_block_preserves_width():
    block = BroNetBlock(16)
    out = block(torch.randn(4, 16))
    assert out.shape == (4, 16)


def test_bronet_block_zero_branch_is_identity():
    """Zeroing the inner (fc1, fc2) branch must collapse the block to the identity - this is
    what proves the skip connection is wired, not merely shape-compatible."""
    block = BroNetBlock(8)
    with torch.no_grad():
        for p in (block.fc1.weight, block.fc1.bias, block.fc2.weight, block.fc2.bias):
            p.zero_()
    x = torch.randn(5, 8)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_critic_trunk_output_shape():
    trunk = bronet_critic_trunk(6, 10, width=32, blocks=2)
    out = trunk(torch.randn(3, 6))
    assert out.shape == (3, 10)


def test_quantile_twin_critic_forward_shapes():
    critic = QuantileTwinCritic(4, 2, width=16, blocks=1, n_quantiles=5)
    q1, q2 = critic(torch.randn(7, 4), torch.randn(7, 2))
    assert q1.shape == (7, 5)
    assert q2.shape == (7, 5)


def test_worked_example_from_the_wiki_doc():
    """Q1=10, Q2=6 (constant across quantiles) -> Q^mu=8, Q^sigma=4, and with beta=0.5,
    Q^o=10 - the exact numbers in the wiki doc's worked example (section 6)."""
    q1 = torch.full((3, 100), 10.0)
    q2 = torch.full((3, 100), 6.0)
    assert torch.allclose(q_mean_from_quantiles(q1, q2), torch.full((3, 1), 8.0))
    assert torch.allclose(q_disagreement_from_quantiles(q1, q2), torch.full((3, 1), 4.0))
    assert torch.allclose(
        optimistic_q_from_quantiles(q1, q2, optimism_coef=0.5), torch.full((3, 1), 10.0)
    )


def test_critic_q_mean_disagreement_optimistic_q_agree_with_pure_functions():
    critic = QuantileTwinCritic(3, 1, width=16, blocks=1, n_quantiles=4)
    s, a = torch.randn(6, 3), torch.randn(6, 1)
    q1, q2 = critic(s, a)
    assert torch.allclose(critic.q_mean(s, a), q_mean_from_quantiles(q1, q2))
    assert torch.allclose(critic.q_disagreement(s, a), q_disagreement_from_quantiles(q1, q2))
    assert torch.allclose(
        critic.optimistic_q(s, a, optimism_coef=0.3),
        optimistic_q_from_quantiles(q1, q2, optimism_coef=0.3),
    )


def test_frozen_target_matches_and_is_detached():
    critic = QuantileTwinCritic(3, 1, width=16, blocks=1, n_quantiles=4)
    target = critic.frozen_target()
    for p in target.parameters():
        assert not p.requires_grad
    s, a = torch.randn(2, 3), torch.randn(2, 1)
    q1, q2 = critic(s, a)
    tq1, tq2 = target(s, a)
    assert torch.allclose(q1, tq1) and torch.allclose(q2, tq2)


def test_quantile_huber_loss_zero_for_matching_constant_distribution():
    """The pairwise loss compares every (pred_i, target_j) pair, so it is only zero when the
    predicted and target quantile functions are identical *as distributions* - a constant
    vector (a point mass) matching another constant vector at the same value, not any two
    elementwise-equal-but-non-constant arrays (those still have nonzero off-diagonal terms)."""
    pred = torch.full((4, 5), 2.5)
    assert quantile_huber_loss(pred, pred).item() == 0.0


def test_quantile_huber_loss_monotonic_in_distance():
    target = torch.zeros(1, 1)
    near = torch.full((1, 1), 0.5)
    far = torch.full((1, 1), 2.0)
    loss_near = quantile_huber_loss(near, target).item()
    loss_far = quantile_huber_loss(far, target).item()
    assert 0.0 < loss_near < loss_far


def test_quantile_huber_loss_penalises_underestimation_more_at_high_quantiles():
    """K=2 -> tau = (0.25, 0.75). Hold quantile 1 (tau=0.25) exactly matched (target=pred=c)
    so it contributes zero loss, and perturb quantile 2 (tau=0.75) by +-delta. Underestimating
    a high quantile (pred < target) must cost exactly 3x what overestimating it by the same
    delta costs, since weight = tau / (1 - tau) = 0.75 / 0.25 = 3 in the quadratic (kappa=1)
    region of the Huber loss."""
    c, delta = 1.0, 0.1
    target = torch.tensor([[c, c]])
    pred_under = torch.tensor([[c, c - delta]])
    pred_over = torch.tensor([[c, c + delta]])
    loss_under = quantile_huber_loss(pred_under, target, kappa=1.0).item()
    loss_over = quantile_huber_loss(pred_over, target, kappa=1.0).item()
    assert loss_under == pytest.approx(3.0 * loss_over, rel=1e-5)


def test_reinit_module_changes_linear_weights():
    block = BroNetBlock(8)
    linears = [m for m in block.modules() if isinstance(m, torch.nn.Linear)]
    before = [(lin.weight.clone(), lin.bias.clone()) for lin in linears]
    reinit_module_(block)
    for (w0, b0), lin in zip(before, linears, strict=True):
        assert not torch.allclose(w0, lin.weight)
        assert w0.shape == lin.weight.shape and b0.shape == lin.bias.shape


def test_reinit_module_resets_layernorm_to_default():
    block = BroNetBlock(8)
    with torch.no_grad():
        block.ln1.weight.fill_(3.0)
        block.ln1.bias.fill_(2.0)
    reinit_module_(block)
    assert torch.allclose(block.ln1.weight, torch.ones_like(block.ln1.weight))
    assert torch.allclose(block.ln1.bias, torch.zeros_like(block.ln1.bias))
