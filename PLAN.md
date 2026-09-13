# frankenrl — build plan

Migration of the FYP "Frankenstein" corpus into one config-driven package.

## Legacy → new mapping

| Legacy | New |
|---|---|
| `Frankensteins/*/ActorCritics.py` (x15) | `frankenrl/nn/{blocks,actors,critics}.py` — one copy, `layernorm: bool` flag |
| `ReplayBuffer` / `MCReplayBuffer` / `GAEReplayBuffer` (copied per folder) | `frankenrl/buffers/{uniform,returns,gae,trajectory}.py`, uniform API |
| `plot_actor_critic_results` (copied per folder) | `frankenrl/logging.py` + `analysis/` |
| `Standard.py::{VanillaACAgent,PPOAgent,SACAgent,TD3Agent}` | `frankenrl/agents/{sac,td3,ppo}.py` (reference, corrected) |
| `FrankensteinM1` | config `configs/m1_hybrid_*.yaml` (twin-clipped-Q + entropy + delayed actor + 1-step target) |
| `FrankensteinM2_A2C / M3_A2C` | `advantage: a2c` |
| `FrankensteinM2_TDError / M3_TDError` | `advantage: tderror` |
| `FrankensteinM2_MonteCarlo / M3_MonteCarlo` | `advantage: mc` |
| `FrankensteinM2_ExpectedSarsa / M3_ExpectedSarsa` | `advantage: expected_sarsa` |
| `FrankensteinM2_GAE / M3_GAE` | `advantage: gae` |
| `Phase6/*_PPO*` | `policy_loss: ppo_clip` on top of the above |
| `LayerNormP*` | `nn.layernorm: true` in the config |
| `NewStandard` | reference agents + `nn.layernorm: true` |
| `train_suite.pbs` (`#PBS -J 0-5`, label map) | `scripts/gadi/train_suite.pbs` — array index → config path, no string branching |
| `P{n}run.py` `if label == "M3_MonteCarlo": ...` ladder | buffer polymorphism: `buffer.add(step)` + `buffer.on_episode_end()` |

## Known bugs to fix (not port)

1. `Standard.py::SACAgent` creates `critic_target_opt` and does `F.mse_loss(q2, q_target)`
   on the *target* — the target must be soft-updated only. Also `q2` there is the target
   net, so its "critic 2" is not an independent twin.
2. `Srun.py::train()` runs an env loop that never calls `push`/`update` — it trains nothing.
3. `P3run.py`: `if ep % 100:` (truthy) — saves every episode *except* multiples of 100.
   Want `if ep % 100 == 0`.
4. `Standard.py::PPOAgent.train` plots `[m[3] for m in self.memory]` after `self.memory=[]`.
5. GAE in `Standard.py::PPOAgent.update` bootstraps `next_val = 0` past buffer end even when
   the episode was truncated (not terminal) — should bootstrap V(s_T).
6. Actor `sample()` log-prob: Phase1-2 use `log(1 - a^2 + 1e-6)`; LayerNormP3 already uses
   the stable `2*(log2 - x - softplus(-2x))` form — standardise on the stable one.

## Order of work

1. ~~**Primitives + tests**~~ ✅ `nn/`, `buffers/`, `advantage.py`, `config.py`, `seeding.py`,
   `envs.py`. 27 tests green (actor log-prob vs analytic; GAE vs hand-computed; MC returns;
   buffer shapes; config load/override; agent smoke + checkpoint round-trip).
2. ~~**SAC vertical slice**~~ ✅ `agents/sac.py` + `train.py` + `configs/sac_pendulum.yaml`.
   Pendulum-v1, seed 0, 60k steps CPU (~30 min): turned around by ~ep 35, **eval
   (deterministic) settled ~-80**, best episode -0.4. Reference SAC confirmed correct.
   TODO: add a tiny-budget CI smoke.
3. **TD3, PPO** references + configs; smoke tests. (TD3 code written, not yet run on a real env.)
4. **`FrankensteinAgent`** — compose from parts; reproduce `m1_hybrid` == SAC-ish; add the
   5 advantage estimators as configs; verify `[[SAC equals M1 under a controlled seed]]`.
5. **PBS runner** + `sweep.py`; one BipedalWalker sweep end-to-end on Gadi.
6. **`analysis/`** — multi-seed aggregation, CI bands, the report comparison figures.
7. ~~**BRO**~~ ✅ `nn/bronet.py` (BroNet residual quantile critic) + `agents/bro.py` +
   `configs/bro_{pendulum,fast_pendulum}.yaml`. Standalone reference agent (not a
   `FrankensteinAgent` config — BRO's mechanisms don't map onto that vocabulary; see
   `agents/bro.py`'s module docstring). Not part of the FYP legacy corpus, doesn't block
   step 4. 19 new tests (`test_bronet.py`, `test_bro.py`, plus `"bro"` added to the generic
   smoke/config test parametrizations). Manual verification: short (3k-step, replay-ratio-2,
   reset-at-500/1500) Pendulum-v1 CLI run completed without crashing or NaNs (exit 0, 15
   episodes / 3000 steps / 663s CPU; eval returns -870 → -1161 → -261 — noisy and not a
   benchmark at this step budget with two resets in the first 1500 steps, but no divergence
   or NaN). Confirms the full pipeline end to end; correctness only, not tuning.
   `optimism_coef`/`kl_coef`/`weight_decay`/the
   reset schedule are this codebase's own untuned picks (`config.py::BroConfig`'s
   docstring), not the paper's literature values — tune with `scripts/sweep.py` before
   drawing any real conclusion from a run.
8. ~~**Standalone baselines**~~ ✅ `standalone/{sac,td3,td7,bro,simba_sac,simba_td3}.py` —
   completely self-contained, single-file (CleanRL-style) reference implementations, no
   `frankenrl` package imports, one algorithm per file. TD7 and SimBa required fresh vault
   research (`Research/2026-09-13-td7-for-sale-state-action-representation-learning.md`,
   `Research/2026-09-13-simba-simplicity-bias-scaling-rl.md`) and new wiki docs
   (`Wiki/RL/TD7 - ...md`, `Wiki/RL/SimBa - ...md`) — neither existed in the vault before.
   `simba_td3.py` is flagged explicitly as an unpublished combination: the SimBa paper tests
   DDPG, not TD3; this file layers TD3's three fixes onto the DDPG+SimBa recipe by analogy.
   6 new tests (`tests/test_standalone_smoke.py`), all real-env smoke runs, no crashes/NaNs.
   Separate from the composable `src/frankenrl/agents/` line — see `standalone/README.md`
   for why both exist and when to use which.
   **Proof pass, 2026-09-14:** all six also verified on `Ant-v5` (105-dim obs, 8-dim action —
   substantially higher-dimensional than the Pendulum smoke tests, so a real check for
   shape/broadcast bugs), 3000 steps each, small nets, exit 0, no crashes/NaNs across the
   board. `mujoco`/`box2d`/`analysis` extras installed locally (`uv sync --all-extras`) to
   make this possible.
9. ~~**Gadi PBS for the standalone suite**~~ ✅ `scripts/gadi/{train_standalone_suite.pbs,
   standalone_suite.txt, submit_standalone.sh}` — parallel to the existing `train_suite.pbs`
   (which only drives the framework's YAML-config agents), since TD7/SimBa-SAC/SimBa-TD3
   have no framework port. One job-array task per `standalone/` script, CLI flags only (no
   config system). Defaults to `Ant-v5`, 1M steps, seed 0 — override via `ENV`/`TOTAL_STEPS`/
   `SEED` env vars or `scripts/gadi/submit_standalone.sh "<seeds>"`. **Not yet run on Gadi
   itself** — the proof pass above confirms correctness locally (small nets, 3k steps, CPU);
   real throughput/walltime at full BRO/TD7 default sizes (critic width 512, BRO replay
   ratio 10) on Gadi's GPU is unmeasured. If 8h walltime proves tight for BRO/TD7 at 1M
   steps, either raise `#PBS -l walltime` or lower `TOTAL_STEPS` per submission.

## Validation targets (from the FYP results)

- `Pendulum-v1`: SAC settles ~ -150..-200 MA.
- `BipedalWalker-v3`: "solved" ~ 300; the hybrids in the FYP got partial learning — match
  or beat the legacy `*_rewards.csv` curves under matched seeds/episodes.
- `Humanoid-v5`: slow; just needs directional learning + no divergence.
