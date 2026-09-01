"""One call to make a run reproducible.

See the vault note [[Controlled-seed diffing as an RL debugging method]] - matched seeds are
how we show two agents are the same algorithm.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeededRNGs:
    """Handles for callers that want an explicit generator rather than global state."""

    seed: int
    numpy: np.random.Generator
    torch: torch.Generator


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> SeededRNGs:
    """Seed python, numpy and torch (CPU + CUDA).

    Args:
        seed: base seed.
        deterministic_torch: if True, force deterministic cuDNN kernels. Slower; only worth
            it when chasing a divergence between two supposedly-identical runs.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # opt-in; raises if an op has no deterministic implementation
        torch.use_deterministic_algorithms(True, warn_only=True)

    g = torch.Generator()
    g.manual_seed(seed)
    return SeededRNGs(seed=seed, numpy=np.random.default_rng(seed), torch=g)
