from pathlib import Path

import pytest

from frankenrl.config import load_config, to_dict

REPO = Path(__file__).resolve().parents[1]


def test_load_base_config():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg.agent.kind == "sac"
    assert cfg.env.id == "Pendulum-v1"
    assert cfg.agent.net.hidden == (256, 256)  # list -> tuple


def test_all_shipped_configs_load():
    for path in (REPO / "configs").glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.label
        assert to_dict(cfg)["agent"]["kind"] in {"sac", "td3", "ppo", "frankenstein", "bro"}


def test_cli_overrides():
    cfg = load_config(
        REPO / "configs" / "base.yaml",
        ["seed=7", "agent.lr=0.001", "agent.net.hidden=[64, 64]", "env.id=BipedalWalker-v3"],
    )
    assert cfg.seed == 7
    assert cfg.agent.lr == 0.001
    assert cfg.agent.net.hidden == (64, 64)
    assert cfg.env.id == "BipedalWalker-v3"


def test_unknown_field_rejected():
    with pytest.raises(KeyError):
        load_config(REPO / "configs" / "base.yaml", ["agent.nonsense=1"])


def test_bro_reset_schedule_loads_as_tuple():
    """`reset_schedule` is a `tuple[int, ...]` field, same as `net.hidden` - a YAML list
    should convert generically, not via a name-specific special case."""
    cfg = load_config(REPO / "configs" / "bro_pendulum.yaml")
    assert cfg.agent.kind == "bro"
    assert isinstance(cfg.agent.bro.reset_schedule, tuple)
    assert cfg.agent.bro.reset_schedule == tuple(sorted(cfg.agent.bro.reset_schedule))
