# frankenrl

Composable hybrid reinforcement-learning agents. One `FrankensteinAgent` assembled from
swappable parts — squashed-Gaussian or deterministic **actor**, twin-clipped **critics**,
a **target-update rule**, an **advantage estimator** (`onestep` / `tderror` / `a2c` / `mc` /
`expected_sarsa` / `gae`), and a **policy loss** (`sac` / `dpg` / `ppo_clip`). SAC, TD3 and
PPO fall out as configs; so do the FYP "M1…M6" variants.

Consolidates the scattered `Frankensteins/Phase*` + `LayerNorm*` FYP code into one tested,
config-driven package. Local dev on a laptop/GPU; full sweeps as PBS job arrays on NCI Gadi.

## Quickstart

```bash
uv sync
uv run python -m frankenrl.run --config configs/sac_pendulum.yaml --seed 0
uv run pytest
```

## Status

Scaffolded 2026-09-01. Primitives (nn / buffers / advantage / config / seeding) land first,
then a correct SAC vertical slice, then the variants migrate in as configs. See `PLAN.md`
and `CLAUDE.md`.
