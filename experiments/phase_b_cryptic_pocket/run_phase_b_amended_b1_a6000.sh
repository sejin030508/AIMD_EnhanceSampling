#!/usr/bin/env bash
set -euo pipefail

# Protein-local B1 launcher introduced by the 2026-09-08 amendment.  It never
# reads the obsolete global all_ready_for_b1 gate.
protein="${1:?usage: run_phase_b_amended_b1_a6000.sh prmt5|prmt6}"
PROJECT_ROOT=/home/sejin/AI_MD_NSMC
PYTHON=/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python
export DUET_PROJECT_ROOT="$PROJECT_ROOT"
export DUET_ASSET_ROOT=/mnt/ssd0/sejin/confrover_mh_steering
export PHASE_B_DATA_ROOT=/mnt/ssd0/sejin/phase_b_pockets
export PHASE_B_OUTPUT_ROOT=/mnt/ssd0/sejin/phase_b_pockets/outputs
export PYTHONPATH="$PROJECT_ROOT/src"

cd "$PROJECT_ROOT"
LOG_ROOT="$PHASE_B_OUTPUT_ROOT/_launcher"
mkdir -p "$LOG_ROOT"
CONFIG="configs/phase_b_pockets/${protein}_b1_amended.yaml"

case "$protein" in
  prmt5)
    jq -e '.proteins[] | select(.protein == "prmt5") | .ready_for_b1 == true' \
      "$PHASE_B_OUTPUT_ROOT/preflight_status.json" >/dev/null || {
        echo "PRMT5 strict readiness is not true; B1 not started." >&2; exit 2; }
    ;;
  prmt6)
    jq -e '.exploratory_pilot_readiness.decision == "proceed" or .exploratory_pilot_readiness.decision == "exploratory_proceed"' \
      "$PHASE_B_OUTPUT_ROOT/prmt6/diagnostics/b0_expanded_seed_41.json" >/dev/null || {
        echo "PRMT6 exploratory readiness is not a proceed decision; B1 not started." >&2; exit 3; }
    ;;
  *) echo "Unsupported protein: $protein" >&2; exit 64 ;;
esac

LOCK="$LOG_ROOT/b1_${protein}_amended_running.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Another amended $protein B1 launcher holds $LOCK" >&2
  exit 4
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

run_cell() {
  local method="$1" seed="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m confmh.duet.phase_b_runner \
    --config "$CONFIG" --method "$method" --seed "$seed" --device cuda:0 \
    >"$LOG_ROOT/${protein}_amended_${method}_seed${seed}_gpu${gpu}.log" 2>&1
}

run_seed() {
  local seed="$1" failed=0
  local pids=()
  run_cell frozen "$seed" 0 & pids+=("$!")
  run_cell outer_only "$seed" 1 & pids+=("$!")
  run_cell complete_nested "$seed" 2 & pids+=("$!")
  run_cell duet "$seed" 3 & pids+=("$!")
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]]
}

run_seed 17
run_seed 29
"$PYTHON" -m confmh.duet.phase_b_runner --config "$CONFIG" --summarize
