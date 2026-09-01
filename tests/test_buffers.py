import numpy as np
import pytest

from frankenrl.buffers.base import Transition
from frankenrl.buffers.trajectory import TrajectoryBuffer
from frankenrl.buffers.uniform import UniformReplay


def _t(i: int) -> Transition:
    return Transition(
        state=np.full(3, i, dtype=np.float32),
        action=np.full(2, i, dtype=np.float32),
        reward=float(i),
        next_state=np.full(3, i + 1, dtype=np.float32),
        done=0.0,
        truncated=0.0,
    )


def test_replay_ring_and_sample_shapes():
    buf = UniformReplay(capacity=8, state_dim=3, action_dim=2, seed=0)
    assert not buf.ready(1)
    for i in range(10):  # overflow the ring
        buf.add(_t(i))
    assert len(buf) == 8
    b = buf.sample(4)
    assert b.state.shape == (4, 3)
    assert b.action.shape == (4, 2)
    assert b.reward.shape == (4, 1)
    assert b.done.shape == (4, 1)


def test_replay_raises_when_not_ready():
    buf = UniformReplay(4, 3, 2)
    with pytest.raises(RuntimeError):
        buf.sample(4)


def test_trajectory_requires_targets_before_get():
    buf = TrajectoryBuffer()
    buf.add(_t(0), log_prob=-1.0)
    with pytest.raises(RuntimeError):
        buf.get()


def test_trajectory_roundtrip_and_clear():
    buf = TrajectoryBuffer()
    for i in range(5):
        buf.add(_t(i), log_prob=-0.5 * i)
    cols = buf.as_arrays()
    assert cols["state"].shape == (5, 3)
    assert cols["log_prob"].shape == (5, 1)
    buf.set_targets(np.arange(5.0), np.arange(5.0) - 2)
    batch = buf.get()
    assert batch.ret.shape == (5, 1)
    assert batch.adv.shape == (5, 1)
    assert len(buf) == 0  # cleared
