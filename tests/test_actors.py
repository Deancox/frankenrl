"""The squashed-Gaussian log-prob is the part most often gotten wrong; pin it down."""

import math

import torch
from torch.distributions import Normal

from frankenrl.nn.actors import DeterministicActor, SquashedGaussianActor
from frankenrl.seeding import seed_everything


def test_sample_shapes_and_bounds():
    seed_everything(0)
    actor = SquashedGaussianActor(6, 3, (32, 32), action_scale=2.0)
    s = torch.randn(16, 6)
    a, logp = actor.sample(s)
    assert a.shape == (16, 3)
    assert logp.shape == (16, 1)
    assert a.abs().max() <= 2.0 + 1e-5


def test_deterministic_is_tanh_of_mean():
    actor = SquashedGaussianActor(4, 2, (16,), action_scale=1.0)
    s = torch.randn(8, 4)
    mu, _ = actor(s)
    a, _ = actor.sample(s, deterministic=True)
    assert torch.allclose(a, torch.tanh(mu), atol=1e-6)


def test_logprob_matches_change_of_variables():
    """log pi(a) == log N(x) - sum log(1 - tanh(x)^2) - dim*log(scale), computed the naive way."""
    torch.manual_seed(0)
    actor = SquashedGaussianActor(3, 2, (16,), action_scale=1.5)
    s = torch.randn(32, 3)
    # reproduce the internal pre-tanh sample deterministically
    dist: Normal = actor._distribution(s)
    x = dist.mean  # deterministic path -> pre_tanh == mean
    a, logp = actor.sample(s, deterministic=True)

    naive = dist.log_prob(x)
    naive = naive - torch.log(1 - torch.tanh(x).pow(2) + 1e-6)
    naive = naive - math.log(1.5)
    naive = naive.sum(dim=-1, keepdim=True)

    assert torch.allclose(logp, naive, atol=1e-4)


def test_logprob_has_grad():
    actor = SquashedGaussianActor(3, 2, (16,))
    s = torch.randn(8, 3)
    _, logp = actor.sample(s)
    logp.mean().backward()
    assert all(p.grad is not None for p in actor.parameters())


def test_deterministic_actor_bounded():
    actor = DeterministicActor(4, 2, (16,), action_scale=3.0)
    a = actor(torch.randn(10, 4) * 50)
    assert a.abs().max() <= 3.0 + 1e-5
