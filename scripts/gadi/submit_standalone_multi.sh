#!/bin/bash
# Submit the standalone suite across multiple ENVIRONMENTS and seeds in one call - e.g. the
# full overnight Ant + Humanoid sweep. Each (env, seed) pair gets its own PBS job array
# covering every algorithm in SUITE (default: standalone_suite.txt, all 7 standalone/
# scripts; set SUITE=scripts/gadi/compare_suite.txt for just the 6-algorithm comparison), so
# log labels never collide across envs (train_standalone_suite.pbs names each log
# "<script>_<env>_s<seed>.log").
#
#   ./scripts/gadi/submit_standalone_multi.sh "Ant-v5 Humanoid-v5" "0 1 42 975206 928980"
#   TOTAL_STEPS=500000 ./scripts/gadi/submit_standalone_multi.sh "Ant-v5 Humanoid-v5" "0 1 42 975206 928980"
#   SUITE=scripts/gadi/compare_suite.txt ./scripts/gadi/submit_standalone_multi.sh "Ant-v5" "0 1 42"
#
# Cost check before you submit: N_envs x N_seeds x N_algorithms(SUITE) = total jobs, each
# requesting up to 24h walltime x 1 GPU (#PBS -l walltime/ngpus in train_standalone_suite.pbs
# - raised from the original 8h after SimBa-SAC and BRO both needed more of it than that on
# the first real run; verify 24h is actually within gpuvolta-exec's own cap before relying on
# it, since PBS rejects an over-cap request at submission time with no compute spent, but a
# silently-lower cap you didn't check would still be a surprise). Two envs x five seeds x 7
# algorithms = 10 job arrays x 7 tasks = 70 jobs, up to ~1680 GPU-hours of *requested*
# walltime (actual usage will be less if runs finish early, but this is what gets reserved
# against your qy44 allocation) - lower TOTAL_STEPS or trim SEEDS/ENVS/SUITE if that's more
# than you want to burn on a first full-scale run.
set -euo pipefail
ENVS="${1:-Ant-v5}"
SEEDS="${2:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
SUITE="${SUITE:-scripts/gadi/standalone_suite.txt}"
N=$(grep -cvE '^\s*(#|$)' "$SUITE")

for e in $ENVS; do
    for s in $SEEDS; do
        echo "submitting ${SUITE} (${N} runs) - env=${e} seed=${s} total_steps=${TOTAL_STEPS}"
        GYM_ENV="$e" SEED="$s" TOTAL_STEPS="$TOTAL_STEPS" SUITE="$SUITE" \
            qsub -v GYM_ENV,SEED,TOTAL_STEPS,SUITE -J "0-$((N - 1))" scripts/gadi/train_standalone_suite.pbs
    done
done
