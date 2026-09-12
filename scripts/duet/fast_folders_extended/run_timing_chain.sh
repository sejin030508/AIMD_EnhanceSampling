#!/usr/bin/env bash
# Run one checkpoint-timing arm end to end on a single GPU.
#
# Usage: run_timing_chain.sh <tag>            e.g. cp2550
#
# The cell runner derives its output stage from the molecule and stride alone,
# so two timings would write to the same directory and overwrite the completed
# pilot.  Each arm therefore gets its own SMALL_PROTEIN_OUTPUT_ROOT; the
# original pilot output tree is never touched.
set -u

TAG="${1:?usage: run_timing_chain.sh <tag>}"
ROOT="${DUET_PROJECT_ROOT:-/workspace/sejin/AI_MD_NSMC_phase_b_recovery}"
STRIDE=128

cd "$ROOT" || exit 1
. small_protein_pilot/small_protein_env.sh

export SMALL_PROTEIN_CONFIG_SUBDIR="configs_${TAG}"
export SMALL_PROTEIN_OUTPUT_ROOT="$ROOT/outputs/small_protein_timing_variation/$TAG"
mkdir -p "$SMALL_PROTEIN_OUTPUT_ROOT/_logs"

LOG="$SMALL_PROTEIN_OUTPUT_ROOT/_logs/chain.log"
echo "=== $TAG started $(date -Is) on $(hostname) ===" >> "$LOG"
echo "config subdir: $SMALL_PROTEIN_CONFIG_SUBDIR" >> "$LOG"
echo "output root:   $SMALL_PROTEIN_OUTPUT_ROOT" >> "$LOG"

for molecule in trpcage bba; do
    for seed in 211 223; do
        cell="$molecule/$seed"
        echo "--- $cell start $(date -Is)" >> "$LOG"
        start=$(date +%s)
        bash small_protein_pilot/run_small_protein_cell.sh \
            "$molecule" "$STRIDE" duet "$seed" production \
            >> "$SMALL_PROTEIN_OUTPUT_ROOT/_logs/${molecule}_seed${seed}.log" 2>&1
        status=$?
        echo "--- $cell exit=$status elapsed=$(( $(date +%s) - start ))s" >> "$LOG"
    done
done

echo "=== $TAG finished $(date -Is) ===" >> "$LOG"
