"""Experience buffers with a uniform API so the training loop never branches on agent type.

Contract (see `base.Buffer`):
    add(Transition)        record one env step
    on_episode_end()       hook (GAE/MC bootstrapping happens here for the trajectory buffer)
    ready(n)               enough data to learn from?
    sample(n) / get()      pull a Batch (random for replay, all-then-clear for on-policy)
"""

from frankenrl.buffers.base import Batch, Buffer, Transition
from frankenrl.buffers.trajectory import TrajectoryBuffer
from frankenrl.buffers.uniform import UniformReplay

__all__ = ["Transition", "Batch", "Buffer", "UniformReplay", "TrajectoryBuffer"]
