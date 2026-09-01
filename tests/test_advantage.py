"""Pin the trajectory estimators against hand-computed values and known GAE limits."""

import numpy as np

from frankenrl.advantage import AdvantageSpec, gae_targets, mc_targets


def _cols(rewards, dones, truncs=None, sdim=2):
    n = len(rewards)
    truncs = truncs if truncs is not None else [0.0] * n
    return {
        "state": np.zeros((n, sdim), dtype=np.float32),
        "action": np.zeros((n, 1), dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        "next_state": np.zeros((n, sdim), dtype=np.float32),
        "done": np.asarray(dones, dtype=np.float32).reshape(-1, 1),
        "truncated": np.asarray(truncs, dtype=np.float32).reshape(-1, 1),
    }


def test_mc_returns_single_episode():
    rewards = [1.0, 2.0, 3.0]
    cols = _cols(rewards, [0, 0, 1])
    spec = AdvantageSpec(name="mc", gamma=0.5, normalize=False)
    ret, _ = mc_targets(cols, value_fn=lambda s: np.zeros((len(s), 1)), spec=spec)
    # G2=3 ; G1=2+0.5*3=3.5 ; G0=1+0.5*3.5=2.75
    assert np.allclose(ret.reshape(-1), [2.75, 3.5, 3.0])


def test_mc_resets_between_episodes():
    cols = _cols([1, 1, 1, 1], [1, 0, 0, 1])
    spec = AdvantageSpec(name="mc", gamma=1.0, normalize=False)
    ret, _ = mc_targets(cols, value_fn=lambda s: np.zeros((len(s), 1)), spec=spec)
    assert np.allclose(ret.reshape(-1), [1.0, 3.0, 2.0, 1.0])


def test_gae_lambda_zero_is_one_step_td_error():
    rewards = [1.0, 1.0, 1.0]
    cols = _cols(rewards, [0, 0, 1])
    v = lambda s: np.full((len(s), 1), 0.5)  # constant value
    spec = AdvantageSpec(name="gae", gamma=0.9, gae_lambda=0.0, normalize=False)
    _, adv = gae_targets(cols, v, spec)
    # delta_t = r + gamma*V(s') - V(s); last step done -> no bootstrap
    expected = [1 + 0.9 * 0.5 - 0.5, 1 + 0.9 * 0.5 - 0.5, 1 + 0.0 - 0.5]
    assert np.allclose(adv.reshape(-1), expected, atol=1e-6)


def test_gae_return_equals_adv_plus_value():
    cols = _cols([0.1, -0.2, 0.3, 0.4], [0, 0, 0, 1])
    v = lambda s: np.linspace(0, 1, len(s)).reshape(-1, 1)
    spec = AdvantageSpec(name="gae", gamma=0.99, gae_lambda=0.95, normalize=False)
    ret, adv = gae_targets(cols, v, spec)
    assert np.allclose(ret, adv + v(cols["state"]), atol=1e-6)


def test_normalized_advantages_are_standardised():
    cols = _cols([1.0, 2.0, 3.0, 4.0, 5.0], [0, 0, 0, 0, 1])
    spec = AdvantageSpec(name="gae", gamma=0.99, gae_lambda=0.95, normalize=True)
    _, adv = gae_targets(cols, lambda s: np.zeros((len(s), 1)), spec)
    assert abs(adv.mean()) < 1e-6
    assert abs(adv.std() - 1.0) < 1e-3
