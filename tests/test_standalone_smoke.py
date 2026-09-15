"""Smoke tests for the self-contained standalone/*.py scripts - run each script's own
`train()` with a tiny step budget on Pendulum-v1 and confirm it doesn't crash or NaN.

These scripts intentionally do not import the `frankenrl` package (CleanRL-style: each is a
single, standalone file). Loading them here by path is purely a test-harness concern - it
does not make the scripts themselves depend on anything; `python standalone/x.py` still runs
each one on its own.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

STANDALONE = Path(__file__).resolve().parents[1] / "standalone"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"standalone_{name}", STANDALONE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON_ARGV = [
    "--env", "Pendulum-v1",
    "--total-steps", "200",
    "--warmup-steps", "50",
    "--eval-every", "100",
    "--eval-episodes", "1",
    "--batch-size", "16",
    "--buffer-size", "500",
]


@pytest.mark.parametrize(
    "name,argv",
    [
        ("sac", COMMON_ARGV + ["--hidden", "16", "16"]),
        ("td3", COMMON_ARGV + ["--hidden", "16", "16"]),
        (
            "bro",
            COMMON_ARGV
            + [
                "--actor-hidden", "16", "16",
                "--critic-width", "16",
                "--critic-blocks", "1",
                "--n-quantiles", "4",
                "--updates-per-step", "1",
                "--reset-schedule", "150",
            ],
        ),
        (
            "td7",
            COMMON_ARGV
            + [
                "--hdim", "16",
                "--zs-dim", "8",
                "--checkpoint-steps-before", "100",
                "--target-update-rate", "50",
            ],
        ),
        (
            "simba_sac",
            COMMON_ARGV
            + [
                "--critic-width", "32", "--critic-blocks", "1",
                "--actor-width", "16", "--actor-blocks", "1",
                "--updates-per-step", "1",
            ],
        ),
        (
            "simba_td3",
            COMMON_ARGV
            + [
                "--critic-width", "32", "--critic-blocks", "1",
                "--actor-width", "16", "--actor-blocks", "1",
                "--updates-per-step", "1",
            ],
        ),
        (
            "simba_bro_td7",
            COMMON_ARGV
            + [
                "--critic-width", "32", "--critic-blocks", "1",
                "--actor-width", "16", "--actor-blocks", "1",
                "--zs-dim", "8", "--encoder-hdim", "16",
                "--updates-per-step", "1",
                "--target-update-rate", "50",
                "--reset-schedule", "150",
                "--checkpoint-steps-before", "100",
            ],
        ),
        (
            # PPO is on-policy - its CLI has no --warmup-steps/--batch-size/--buffer-size,
            # so it can't extend COMMON_ARGV like the off-policy scripts above.
            "ppo",
            [
                "--env", "Pendulum-v1",
                "--total-steps", "200",
                "--rollout-steps", "64",
                "--minibatch-size", "16",
                "--ppo-epochs", "2",
                "--eval-every", "100",
                "--eval-episodes", "1",
                "--hidden", "16", "16",
            ],
        ),
    ],
)
def test_script_runs_without_crashing_or_nan(name, argv, monkeypatch, capsys):
    module = _load(name)
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.train(module.parse_args())
    out = capsys.readouterr().out
    assert "nan" not in out.lower()
    assert "done:" in out
