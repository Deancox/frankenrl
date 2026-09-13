#!/bin/bash
# Convenience wrapper: submit the standalone suite across several seeds.
#   ./scripts/gadi/submit_standalone.sh "0 1 2"
#   ENV=Ant-v5 TOTAL_STEPS=1000000 ./scripts/gadi/submit_standalone.sh "0 1 2"
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
N=$(grep -cvE '^\s*(#|$)' scripts/gadi/standalone_suite.txt)
for s in $SEEDS; do
    echo "submitting standalone suite (${N} runs) at seed ${s}, env ${ENV:-Ant-v5}"
    SEED="$s" ENV="${ENV:-Ant-v5}" TOTAL_STEPS="${TOTAL_STEPS:-1000000}" \
        qsub -J "0-$((N - 1))" scripts/gadi/train_standalone_suite.pbs
done
