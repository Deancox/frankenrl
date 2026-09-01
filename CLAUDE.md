# Project: frankenrl

Consolidated, installable framework for **composable hybrid RL agents** - the "Frankenstein"
line from the ELEC4840 FYP (*Reinforcement Learning in Robotic Simulations*), unified into
one package. Agents are assembled from swappable parts (actor, critic set, target-update
rule, advantage estimator, policy loss); the legacy M1/M2/M3/M6 zoo becomes **configs, not
classes**. Local dev + NCI Gadi PBS for real runs.

A code project in the **Garage** workspace (`C:\Users\deanc\Garage\`). House rules and the
shared agents/commands/hooks load from `~/.claude/CLAUDE.md`.

## Scope & goal

- **Goal:** replace ~15 drifting copies of `ActorCritics.py` / `ReplayBuffer` / plotting
  scattered across `Frankensteins/Phase*`, `LayerNorm*`, `NewStandard`, ... with one tested,
  config-driven package. Reproduce the FYP results, then extend (PPO-clip-in-TD3, etc.).
- **In scope:** the `frankenrl` package; reference SAC/TD3/PPO; the composable Frankenstein
  agent; buffers; advantage estimators; seeded training loop; CSV + TensorBoard logging;
  local + PBS runners; per-variant YAML configs; tests; a results-analysis layer.
- **Out of scope:** new environments/physics; sim-to-real; the FYP report prose (lives in
  `fyp-corpus-raw/FYP Report/`); rewriting history - legacy code stays read-only reference.
- **Status (2026-09-01):** scaffold + primitives done (27 tests green). **SAC vertical slice
  verified** — Pendulum-v1 seed 0, eval settled ~-80 in 60k CPU steps. Next: PPO reference,
  then `FrankensteinAgent` + the 5 advantage estimators, then verify `m1 ≈ SAC` under a
  matched seed, then a Gadi BipedalWalker sweep.

## Source material (read-only reference)

All under **`../../fyp-corpus-raw/`** (i.e. `Garage/fyp-corpus-raw/`):

- **Legacy code:** `fyp-corpus-raw/Python/Frankensteins/` (Phase1-7, Standard, LayerNormP2-6,
  NewStandard, FailPhase3) and `Python/ActorCritic/Individual/Agents.py` (2429-line monolith).
  `fyp-corpus-raw/Frankensteins/` is a trimmed copy of Phase1-7 + Standard.
- **Results:** `Python/FYP_Results/` and `Python/MasterComp/` - `*_rewards.csv` per
  `<env>/<label>`, one row per episode.
- **Report:** `fyp-corpus-raw/FYP Report/Dean_Cox_FYP_A.pdf` (Part A). Ch.6 future-work =
  the brief for the hybrids.
- **Theory + findings:** the vault RL wiki - `..\..\DeanVault\INDEX.md` -> *Reinforcement
  learning*. Start at `[[Frankenstein agents]]`, `[[SAC]]`, `[[TD3]]`, `[[GAE]]`,
  `[[Baselines and the advantage function]]`, `[[LayerNorm in RL critics and actors]]`,
  `[[A detached baseline contributes zero gradient]]`, `[[SAC equals M1 under a controlled seed]]`.

## Stack

- **Python**: Gadi runs `python3/3.11.7`; local dev on 3.12/3.13 is fine (CI should pin
  3.11 for parity). `uv` for the local venv (`.python-version` = 3.13, the machine's
  python.org build - the uv-managed pythons are flaky here).
- **PyTorch 2.x** (Gadi `pytorch/2.12.0`), **Gymnasium** (`box2d` for BipedalWalker,
  `mujoco` for Humanoid), NumPy, PyYAML, TensorBoard.
- Test: **pytest**. Style: `~/.claude/rules` - immutability, small files, comprehensive
  error handling, no bare mutation of shared state.

## Layout

```
src/frankenrl/
  nn/         blocks (MLP + optional LayerNorm/Mish/orthogonal), actors, critics
  buffers/    uniform (off-policy), trajectory (on-policy), returns (MC), gae
  advantage.py   estimator registry: onestep | tderror | a2c | mc | expected_sarsa | gae
  agents/     base (ABC) | sac | td3 | ppo | frankenstein (composed from parts + a config)
  config.py   dataclasses + YAML load/merge/CLI-override
  envs.py     gym.make wrapper (action scaling, optional obs norm)
  seeding.py  one call seeds python/numpy/torch/env
  train.py    single seeded run + logging          eval.py
  logging.py  CSV + TensorBoard; self-describing checkpoints
configs/      base.yaml + one per legacy variant (sac_bipedal, m1_hybrid_bipedal, m3_gae_*, ...)
scripts/      run.py (CLI) | sweep.py (local multi-seed) | gadi/train_suite.pbs + submit.sh
tests/        actors (log-prob vs analytic) | buffers | advantage (GAE vs known) | agent smoke
analysis/     load *_rewards.csv across seeds -> CI bands, comparison figures
```

## Design rules

- **One agent, many configs.** `FrankensteinAgent` takes `{actor, critics, target_rule,
  advantage, policy_loss, ...}`; M1 = twin-clipped-Q + entropy + delayed actor + 1-step
  target; M3_GAE = same but `advantage: gae`; M6 = `policy_loss: ppo_clip`. No per-variant
  classes.
- **Every run is seeded and self-describing.** Checkpoints embed the resolved config +
  git SHA + env id (see `[[Self-describing RL checkpoints]]`).
- **Match the legacy output contract** so old analysis still works:
  `<out>/<label>/<label>_rewards.csv` (header `reward`, one row/episode) + `<label>_metrics.png`.
- **Fix, don't port, the known bugs:** SAC target critic is soft-updated only (never
  optimised - the legacy `critic_target_opt` is wrong); on-policy plotting must not read a
  cleared buffer; `ep % N` save guards must be `== 0`; the runner must not branch on string
  labels - buffers expose a uniform API.

## Garage tools

`..\..\grunt.ps1` for boilerplate/commit messages | `..\..\jask.ps1` for quick RL-theory
lookups against the vault | `~/.claude` agents (`planner`, `tdd-guide`, `code-reviewer`).
Durable RL knowledge learned here -> the vault wiki, not this repo.
