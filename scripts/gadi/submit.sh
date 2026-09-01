#!/bin/bash
# Convenience wrapper: submit the suite across several seeds.
#   ./scripts/gadi/submit.sh "0 1 2"
set -euo pipefail
SEEDS="${1:-0}"
N=$(grep -cvE '^\s*(#|$)' scripts/gadi/suite.txt)
for s in $SEEDS; do
    echo "submitting suite (${N} runs) at seed ${s}"
    SEED="$s" qsub -J "0-$((N - 1))" scripts/gadi/train_suite.pbs
done
