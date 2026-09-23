"""standalone/bro.py's checkpoint/resume feature: periodic saves survive a restart, and a
mismatched (e.g. differently-sized) checkpoint fails safe rather than crashing the run."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

STANDALONE = Path(__file__).resolve().parents[1] / "standalone"


def _load_bro():
    spec = importlib.util.spec_from_file_location("standalone_bro_ckpt", STANDALONE / "bro.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TINY_NET = ["--actor-hidden", "16", "16", "--critic-width", "16", "--critic-blocks", "1", "--n-quantiles", "4"]


def _argv(tmp_path, total_steps: int, extra: list[str]) -> list[str]:
    return [
        "--env", "Pendulum-v1", "--seed", "0",
        "--total-steps", str(total_steps), "--warmup-steps", "20",
        "--eval-every", "1000", "--eval-episodes", "1",
        "--batch-size", "16", "--buffer-size", "500",
        "--updates-per-step", "1",
        "--checkpoint-dir", str(tmp_path), "--checkpoint-every", "50",
        *TINY_NET, *extra,
    ]


def test_resume_picks_up_from_saved_step(tmp_path, monkeypatch, capsys):
    bro = _load_bro()

    monkeypatch.setattr(sys, "argv", ["prog", *_argv(tmp_path, 100, [])])
    bro.train(bro.parse_args())
    ckpt_path = tmp_path / "bro_Pendulum-v1_s0.pt"
    assert ckpt_path.exists()

    monkeypatch.setattr(sys, "argv", ["prog", *_argv(tmp_path, 200, [])])
    bro.train(bro.parse_args())
    out = capsys.readouterr().out
    m = re.search(r"resumed from .+ at step (\d+)", out)
    assert m is not None, out
    resumed_step = int(m.group(1))
    assert 0 < resumed_step <= 100  # a real checkpoint step, not a fresh start


def test_no_checkpoint_flag_skips_saving(tmp_path, monkeypatch):
    bro = _load_bro()
    monkeypatch.setattr(sys, "argv", ["prog", *_argv(tmp_path, 100, ["--no-checkpoint"])])
    bro.train(bro.parse_args())
    assert list(tmp_path.iterdir()) == []


def test_mismatched_checkpoint_fails_safe_not_crash(tmp_path, monkeypatch, capsys):
    bro = _load_bro()

    # save a checkpoint with one critic width...
    small = ["--actor-hidden", "16", "16", "--critic-width", "16", "--critic-blocks", "1", "--n-quantiles", "4"]
    argv = [
        "--env", "Pendulum-v1", "--seed", "0",
        "--total-steps", "60", "--warmup-steps", "20",
        "--eval-every", "1000", "--eval-episodes", "1",
        "--batch-size", "16", "--buffer-size", "500", "--updates-per-step", "1",
        "--checkpoint-dir", str(tmp_path), "--checkpoint-every", "50",
        *small,
    ]
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    bro.train(bro.parse_args())
    assert (tmp_path / "bro_Pendulum-v1_s0.pt").exists()

    # ...then try to resume with a DIFFERENT critic width - must not raise.
    different = ["--actor-hidden", "16", "16", "--critic-width", "32", "--critic-blocks", "1", "--n-quantiles", "4"]
    argv2 = [
        "--env", "Pendulum-v1", "--seed", "0",
        "--total-steps", "60", "--warmup-steps", "20",
        "--eval-every", "1000", "--eval-episodes", "1",
        "--batch-size", "16", "--buffer-size", "500", "--updates-per-step", "1",
        "--checkpoint-dir", str(tmp_path), "--checkpoint-every", "50",
        *different,
    ]
    monkeypatch.setattr(sys, "argv", ["prog", *argv2])
    bro.train(bro.parse_args())  # would raise RuntimeError on a shape mismatch if not handled
    out = capsys.readouterr().out
    assert "WARNING" in out and "starting fresh" in out


def test_final_checkpoint_saved_on_natural_completion(tmp_path, monkeypatch):
    bro = _load_bro()
    # checkpoint_every larger than total_steps - only the final save should produce a file
    argv = _argv(tmp_path, 60, [])
    argv[argv.index("--checkpoint-every") + 1] = "1000"
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    bro.train(bro.parse_args())
    assert (tmp_path / "bro_Pendulum-v1_s0.pt").exists()
