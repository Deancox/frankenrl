"""Run logging: the legacy CSV + PNG contract, plus optional TensorBoard.

Output layout matches the FYP runs so old analysis keeps working:

    <out_dir>/<label>/
        config.yaml          resolved RunConfig
        <label>_rewards.csv   header `reward`, one row per completed episode
        <label>_metrics.png   raw + moving-average curve
        checkpoints/step_<n>.pt

Checkpoints embed the resolved config + git SHA + env id
(see [[Self-describing RL checkpoints]]).
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class RunLogger:
    def __init__(self, out_dir: str | Path, label: str, config: dict[str, Any], *, tensorboard: bool = False):
        self.dir = Path(out_dir) / label
        self.dir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.config = config
        self._episode_rewards: list[float] = []

        (self.dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        self._csv_path = self.dir / f"{label}_rewards.csv"
        self._csv_path.write_text("reward\n", encoding="utf-8")

        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.dir / "tb"))
            except ImportError:
                print("[frankenrl] tensorboard not installed; skipping (pip install '.[logging]')")

    def log_episode(self, reward: float, step: int) -> None:
        self._episode_rewards.append(float(reward))
        with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([reward])
        if self._tb is not None:
            self._tb.add_scalar("episode/reward", reward, step)

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        if self._tb is not None:
            for k, v in scalars.items():
                self._tb.add_scalar(k, v, step)

    def save_checkpoint(self, step: int, state: dict[str, Any]) -> Path:
        ckpt_dir = self.dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / f"step_{step}.pt"
        torch.save(
            {
                "step": step,
                "git_sha": _git_sha(),
                "config": self.config,
                "state": state,
            },
            path,
        )
        return path

    def finish(self) -> None:
        self._write_plot()
        (self.dir / "summary.json").write_text(
            json.dumps(
                {
                    "label": self.label,
                    "episodes": len(self._episode_rewards),
                    "last_10_mean": _mean(self._episode_rewards[-10:]),
                    "best_episode": max(self._episode_rewards, default=None),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self._tb is not None:
            self._tb.close()

    def _write_plot(self, window: int = 50) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return
        r = np.asarray(self._episode_rewards, dtype=float)
        if r.size == 0:
            return
        plt.figure(figsize=(12, 6))
        plt.plot(r, alpha=0.3, label="raw")
        if r.size >= window:
            ma = np.convolve(r, np.ones(window) / window, mode="valid")
            plt.plot(range(window - 1, r.size), ma, linewidth=2, label=f"MA({window})")
        plt.title(f"{self.label} training")
        plt.xlabel("episode")
        plt.ylabel("total reward")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.dir / f"{self.label}_metrics.png")
        plt.close()


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None
