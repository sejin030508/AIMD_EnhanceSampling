#!/usr/bin/env bash
# Run PVB small-protein cells on one GPU.
#
# Usage: run_pvb_chain.sh <molecule> <method> [seeds...]
#
# PVB's extra dependencies live in an isolated --target directory rather than in
# the conda environment, so PVB_SITE has to be set before the env script is
# sourced -- the cell runner re-sources it and would otherwise drop the path.
set -u

MOLECULE="${1:?usage: run_pvb_chain.sh <molecule> <method> [seeds...]}"
METHOD="${2:?}"
shift 2
SEEDS=("${@:-211 223}")
[ $# -eq 0 ] && SEEDS=(211 223)

ROOT="${DUET_PROJECT_ROOT:-/workspace/sejin/AI_MD_NSMC_phase_b_recovery}"
cd "$ROOT" || exit 1

export PVB_SITE=/workspace/sejin/pvb_assets/site
. small_protein_pilot/small_protein_env.sh
export SMALL_PROTEIN_CONFIG_SUBDIR=configs_pvb
export SMALL_PROTEIN_OUTPUT_ROOT="$ROOT/outputs/pvb_small_protein"
mkdir -p "$SMALL_PROTEIN_OUTPUT_ROOT/_logs"

LOG="$SMALL_PROTEIN_OUTPUT_ROOT/_logs/chain_${MOLECULE}_${METHOD}.log"
echo "=== $MOLECULE/$METHOD started $(date -Is) on $(hostname) ===" >> "$LOG"

for seed in "${SEEDS[@]}"; do
    echo "--- seed $seed start $(date -Is)" >> "$LOG"
    start=$(date +%s)
    bash small_protein_pilot/run_small_protein_cell.sh \
        "$MOLECULE" 128 "$METHOD" "$seed" production \
        >> "$SMALL_PROTEIN_OUTPUT_ROOT/_logs/${MOLECULE}_${METHOD}_seed${seed}.log" 2>&1
    echo "--- seed $seed exit=$? elapsed=$(( $(date +%s) - start ))s" >> "$LOG"
done

echo "=== $MOLECULE/$METHOD finished $(date -Is) ===" >> "$LOG"
