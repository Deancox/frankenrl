# Standalone baselines

Completely self-contained, single-file reference implementations - CleanRL-style. Each
script is the *whole* algorithm: its own networks, replay buffer, training loop, and CLI.
None of them import the `frankenrl` package (`src/frankenrl/`) or each other; some
duplication across files is deliberate, so any one script can be read top to bottom or
copied elsewhere without pulling in the rest of this repo. Only dependency: this repo's own
`torch`/`gymnasium`/`numpy` (already in `pyproject.toml`).

For the composable, config-driven version of these same algorithms (shared primitives, YAML
configs, checkpointing, TensorBoard logging), see `src/frankenrl/agents/` instead - that's
the version to extend into the `FrankensteinAgent` hybrid family. These scripts are for fast,
readable, side-by-side comparison and hacking on one algorithm in isolation.

| Script | Algorithm | Paper | Published combination? |
|---|---|---|---|
| `sac.py` | SAC | Haarnoja et al., ICML 2018 (arXiv:1801.01290) | yes |
| `td3.py` | TD3 | Fujimoto, van Hoof & Meger, ICML 2018 (arXiv:1802.09477) | yes |
| `td7.py` | TD7 ("TD3+4 additions": SALE, LAP, checkpoints, offline BC) | Fujimoto et al., "For SALE", NeurIPS 2023 | yes |
| `bro.py` | BRO | Nauman et al., NeurIPS 2024 (arXiv:2405.16158) | yes |
| `simba_sac.py` | SAC + SimBa backbone | Lee et al., "SimBa: Simplicity Bias for Scalable RL", ICLR 2025 | yes (paper tests SAC) |
| `simba_td3.py` | TD3 + SimBa backbone | same SimBa paper | **no** — paper tests DDPG, not TD3; this file layers TD3's three fixes onto the DDPG+SimBa recipe as an engineered extrapolation, flagged in its own docstring |

(Exact arXiv ids for TD7 and SimBa are pinned down in each script's own module docstring and
the corresponding vault research report, not repeated here to avoid a second place to get a
citation wrong.)

Each script's module docstring gives the full citation and a one-paragraph algorithm summary;
the corresponding `DeanVault/Wiki/RL/` document has the full derivation.

## Usage

```bash
uv run python standalone/sac.py --env Pendulum-v1 --seed 0 --total-steps 60000
uv run python standalone/bro.py --env Pendulum-v1 --updates-per-step 10
```

`--help` on any script lists its full CLI (every hyperparameter is a flag, no YAML). Run
`--total-steps` small (a few hundred) with small `--hidden`/`--critic-width` for a fast
crash/NaN check rather than a real training run - see `tests/test_standalone_smoke.py`.

## Untuned hyperparameters

BRO's `--optimism-coef`, `--kl-coef`, `--weight-decay`, and `--reset-schedule` defaults are
this codebase's own picks, not confirmed literature values (see the BRO wiki doc's ambiguity
notes). TD7's checkpoint schedule (`--checkpoint-steps-before`) and SimBa's RSNorm epsilon
(`--rsnorm-eps`) are similarly this codebase's own picks — the official constants weren't
recovered from the paper text, only from partial code reading; see each script's module
docstring and the corresponding vault wiki doc's Caveats section for the full list. Treat a
run with default flags as a correctness check, not a benchmark result — tune with
`scripts/sweep.py` (or your own sweep over these scripts directly, since they don't share
that infrastructure) before drawing any real conclusion.

## Running on NCI Gadi

`scripts/gadi/train_standalone_suite.pbs` + `standalone_suite.txt` submit these scripts as a
PBS job array, parallel to `scripts/gadi/train_suite.pbs` (which only drives the framework's
YAML-config agents — no use for TD7/SimBa-SAC/SimBa-TD3, which have no framework port).

```bash
ENV=Ant-v5 SEED=0 qsub -v ENV,SEED,TOTAL_STEPS -J 0-5 scripts/gadi/train_standalone_suite.pbs   # one array index per line of standalone_suite.txt
./scripts/gadi/submit_standalone.sh "0 1 2"                                                      # or: several seeds, one env, at once
./scripts/gadi/submit_standalone_multi.sh "Ant-v5 Humanoid-v5" "0 1 42 975206 928980"            # or: several envs x seeds in one call
```

The `-v` flag is not optional - `VAR=value qsub ...` only sets `VAR` for the `qsub` client
process, not the job itself; PBS Pro needs `-v VAR` (bare name) to forward it through. Both
wrapper scripts already do this correctly.

Defaults to `Ant-v5`, 1M steps. Logs land at
`/scratch/$PBS_O_PROJECT/$USER/frankenrl-standalone/<script>_<env>_s<seed>.log`. Needs the
`mujoco` extra installed on the Gadi environment for MuJoCo tasks
(`uv sync --extra mujoco`, or `--all-extras` for box2d/analysis too).

## A note on correctness vs. faithfulness

Every script has been smoke-tested (`tests/test_standalone_smoke.py`) on a real Gymnasium
env with tiny step budgets — confirms nothing crashes or NaNs, not that hyperparameters are
tuned or that every implementation detail exactly matches the authors' own code. TD7 in
particular has a materially larger implementation surface (three networks, three optimisers,
a priority-replay sum-tree, a stateful checkpoint subsystem) than the others; its checkpoint
timing and BC-term weighting are explicitly simplified versions of the official schedule —
see `td7.py`'s module docstring for the specific deviations.
