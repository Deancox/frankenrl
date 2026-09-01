"""Load and aggregate ``*_rewards.csv`` across seeds/variants for the report figures.

STATUS: skeleton. Fill in once there are real runs to plot (PLAN.md step 6). Works on both
the legacy FYP layout (`FYP_Results/<phase>/<label>_rewards.csv`) and the new one
(`runs/<label>_s<seed>/<label>_rewards.csv`).

Needs `.[analysis]` (pandas, matplotlib, scipy).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def find_reward_csvs(root: str | Path) -> dict[str, list[Path]]:
    """Group ``*_rewards.csv`` under ``root`` by variant label (seed suffix stripped)."""
    groups: dict[str, list[Path]] = {}
    for csv in Path(root).rglob("*_rewards.csv"):
        label = re.sub(r"_s\d+$", "", csv.parent.name) or csv.stem.replace("_rewards", "")
        groups.setdefault(label, []).append(csv)
    return groups


def load_curve(csv: str | Path) -> np.ndarray:
    """One run: 1-D array of per-episode reward (tolerates the legacy header row)."""
    return np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=1)


def aggregate(csvs: list[Path], *, min_len: int | None = None) -> dict[str, np.ndarray]:
    """Stack seeds to equal length; return mean / std / n per episode index."""
    curves = [load_curve(c) for c in csvs]
    n = min_len or min(len(c) for c in curves)
    stack = np.vstack([c[:n] for c in curves])
    return {"mean": stack.mean(0), "std": stack.std(0), "n": len(curves), "episodes": np.arange(n)}


# TODO: moving_average(), plot_comparison(labels, root, window=50, ci=0.95),
#       welch_ttest(a, b) for "is variant X better than baseline" claims.
