#!/bin/bash
# Convenience wrapper: submit the standalone suite across several seeds, ONE environment.
#   ./scripts/gadi/submit_standalone.sh "0 1 2"
#   GYM_ENV=Ant-v5 TOTAL_STEPS=1000000 ./scripts/gadi/submit_standalone.sh "0 1 2"
# SUITE picks which suite file to submit - defaults to all 7 standalone/ scripts
# (standalone_suite.txt); set it to compare_suite.txt for just the 6-algorithm
# TD7/BRO/SimBa-SAC + PPO/SAC/TD3 comparison:
#   SUITE=scripts/gadi/compare_suite.txt ./scripts/gadi/submit_standalone.sh "0 1 2"
# For multiple environments in one call (e.g. Ant + Humanoid overnight), use
# submit_standalone_multi.sh instead.
#
# Reference 5-seed spread used for the first Ant-v5 comparison run (2026-09-14):
#   0        - canonical default; what SAC/TD3/BRO/SimBa/TD7 each report their "seed 0" against
#   1        - simplest adjacent seed
#   42       - classic "arbitrary but memorable" ML seed, deliberately not a paper default
#   975206   - drawn fresh from the OS CSPRNG (python -c "import secrets; secrets.randbelow(1_000_000)")
#   928980   - drawn fresh from the OS CSPRNG, same call
#   ./scripts/gadi/submit_standalone.sh "0 1 42 975206 928980"
set -euo pipefail
SEEDS="${1:-0}"
SUITE="${SUITE:-scripts/gadi/standalone_suite.txt}"
N=$(grep -cvE '^\s*(#|$)' "$SUITE")
for s in $SEEDS; do
    echo "submitting ${SUITE} (${N} runs) at seed ${s}, env ${GYM_ENV:-Ant-v5}"
    SEED="$s" GYM_ENV="${GYM_ENV:-Ant-v5}" TOTAL_STEPS="${TOTAL_STEPS:-1000000}" SUITE="$SUITE" \
        qsub -v SEED,GYM_ENV,TOTAL_STEPS,SUITE -J "0-$((N - 1))" scripts/gadi/train_standalone_suite.pbs
done
