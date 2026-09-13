#!/bin/bash
# Convenience wrapper: submit the standalone suite across several seeds.
#   ./scripts/gadi/submit_standalone.sh "0 1 2"
#   ENV=Ant-v5 TOTAL_STEPS=1000000 ./scripts/gadi/submit_standalone.sh "0 1 2"
set -euo pipefail
SEEDS="${1:-0}"
N=$(grep -cvE '^\s*(#|$)' scripts/gadi/standalone_suite.txt)
for s in $SEEDS; do
    echo "submitting standalone suite (${N} runs) at seed ${s}, env ${ENV:-Ant-v5}"
    SEED="$s" ENV="${ENV:-Ant-v5}" TOTAL_STEPS="${TOTAL_STEPS:-1000000}" \
        qsub -J "0-$((N - 1))" scripts/gadi/train_standalone_suite.pbs
done
