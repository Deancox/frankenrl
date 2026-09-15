#!/bin/bash
# Submit the standalone suite across multiple ENVIRONMENTS and seeds in one call - e.g. the
# full overnight Ant + Humanoid sweep. Each (env, seed) pair gets its own PBS job array
# covering all 6 algorithms (standalone_suite.txt), so log labels never collide across envs
# (train_standalone_suite.pbs names each log "<script>_<env>_s<seed>.log").
#
#   ./scripts/gadi/submit_standalone_multi.sh "Ant-v5 Humanoid-v5" "0 1 42 975206 928980"
#   TOTAL_STEPS=500000 ./scripts/gadi/submit_standalone_multi.sh "Ant-v5 Humanoid-v5" "0 1 42 975206 928980"
#
# Cost check before you submit: N_envs x N_seeds x 6 algorithms = total jobs, each requesting
# up to 8h walltime x 1 GPU (#PBS -l walltime/ngpus in train_standalone_suite.pbs). Two envs x
# five seeds = 10 job arrays x 6 tasks = 60 jobs, up to 480 GPU-hours of *requested* walltime
# (actual usage will be less if runs finish early, but this is what gets reserved against your
# qy44 allocation) - lower TOTAL_STEPS or trim SEEDS/ENVS if that's more than you want to burn
# on a first full-scale run, especially since nothing has been verified at 1M-step scale on
# Gadi yet (see the smoke test in train_standalone_suite.pbs's header comment).
set -euo pipefail
ENVS="${1:-Ant-v5}"
SEEDS="${2:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
N=$(grep -cvE '^\s*(#|$)' scripts/gadi/standalone_suite.txt)

for e in $ENVS; do
    for s in $SEEDS; do
        echo "submitting standalone suite (${N} runs) - env=${e} seed=${s} total_steps=${TOTAL_STEPS}"
        GYM_ENV="$e" SEED="$s" TOTAL_STEPS="$TOTAL_STEPS" \
            qsub -v GYM_ENV,SEED,TOTAL_STEPS -J "0-$((N - 1))" scripts/gadi/train_standalone_suite.pbs
    done
done
