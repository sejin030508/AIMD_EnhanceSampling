#!/usr/bin/env bash
# Run a queue of PVB fast-folding cells on one GPU.
#
# Usage: run_pvb_full.sh <queue-name> <spec> [<spec> ...]
#        spec = <timing-tag>:<molecule>:<method>:<seed>
#
# Cells already carrying small_protein_metrics.json are skipped by the cell
# runner itself, so a queue can be re-run after an interruption without
# repeating finished work.  Each timing keeps its own output root because the
# cell runner derives its stage name from molecule and stride alone.
set -u

QUEUE="${1:?usage: run_pvb_full.sh <queue-name> <tag:molecule:method:seed> ...}"
shift

ROOT="${DUET_PROJECT_ROOT:-/workspace/sejin/AI_MD_NSMC_phase_b_recovery}"
cd "$ROOT" || exit 1
export PVB_SITE=/workspace/sejin/pvb_assets/site
. small_protein_pilot/small_protein_env.sh

QUEUE_LOG="$ROOT/outputs/pvb_full/_queues"
mkdir -p "$QUEUE_LOG"
LOG="$QUEUE_LOG/${QUEUE}.log"
echo "=== queue $QUEUE started $(date -Is) on $(hostname), $# cells ===" >> "$LOG"

for spec in "$@"; do
    IFS=':' read -r tag molecule method seed <<< "$spec"
    export SMALL_PROTEIN_CONFIG_SUBDIR="configs_pvb_${tag}"
    export SMALL_PROTEIN_OUTPUT_ROOT="$ROOT/outputs/pvb_full/${tag}"
    mkdir -p "$SMALL_PROTEIN_OUTPUT_ROOT/_logs"

    start=$(date +%s)
    bash small_protein_pilot/run_small_protein_cell.sh \
        "$molecule" 128 "$method" "$seed" production \
        >> "$SMALL_PROTEIN_OUTPUT_ROOT/_logs/${molecule}_${method}_seed${seed}.log" 2>&1
    status=$?
    echo "$spec exit=$status elapsed=$(( $(date +%s) - start ))s" >> "$LOG"

    # Native-contact Q is a sidecar, so a failure here must not fail the cell.
    run_dir="$SMALL_PROTEIN_OUTPUT_ROOT/$molecule/stride128_t32/$method/seed_$seed"
    if [ -f "$run_dir/small_protein_metrics.json" ]; then
        "$SMALL_PROTEIN_EVAL_PYTHON" \
            scripts/fast_folders_extended/compute_native_contacts.py \
            --run-dir "$run_dir" \
            --prepared-dir "$SMALL_PROTEIN_DATA_ROOT/prepared/$molecule" \
            >> "$SMALL_PROTEIN_OUTPUT_ROOT/_logs/native_contacts.log" 2>&1 || true
    fi
done

echo "=== queue $QUEUE finished $(date -Is) ===" >> "$LOG"
