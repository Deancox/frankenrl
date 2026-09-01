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

## Validation targets (from the FYP results)

- `Pendulum-v1`: SAC settles ~ -150..-200 MA.
- `BipedalWalker-v3`: "solved" ~ 300; the hybrids in the FYP got partial learning — match
  or beat the legacy `*_rewards.csv` curves under matched seeds/episodes.
- `Humanoid-v5`: slow; just needs directional learning + no divergence.
