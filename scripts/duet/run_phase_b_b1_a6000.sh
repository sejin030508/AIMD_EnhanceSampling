#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/sejin/AI_MD_NSMC
PYTHON=/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python
export DUET_PROJECT_ROOT="$PROJECT_ROOT"
export DUET_ASSET_ROOT=/mnt/ssd0/sejin/confrover_mh_steering
export PHASE_B_DATA_ROOT=/mnt/ssd0/sejin/phase_b_pockets
export PHASE_B_OUTPUT_ROOT=/mnt/ssd0/sejin/phase_b_pockets/outputs
export PYTHONPATH="$PROJECT_ROOT/src"

cd "$PROJECT_ROOT"
STATUS="$PHASE_B_OUTPUT_ROOT/preflight_status.json"
if [[ ! -f "$STATUS" ]]; then
  echo "Missing preflight audit: $STATUS" >&2
  exit 2
fi
if ! jq -e '.all_ready_for_b1 == true and .b1_started == false' "$STATUS" >/dev/null; then
  echo "Phase B preflight is not complete; B1 was not started." >&2
  exit 3
fi

LOG_ROOT="$PHASE_B_OUTPUT_ROOT/_launcher"
mkdir -p "$LOG_ROOT"
LOCK="$LOG_ROOT/b1_running.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Another Phase-B B1 launcher holds $LOCK" >&2
  exit 4
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

run_cell() {
  local protein="$1"
  local method="$2"
  local seed="$3"
  local gpu="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m confmh.duet.phase_b_runner \
    --config "configs/phase_b_pockets/${protein}_b1.yaml" \
    --method "$method" --seed "$seed" --device cuda:0 \
    >"$LOG_ROOT/${protein}_${method}_seed${seed}_gpu${gpu}.log" 2>&1
}

run_protein() {
  local protein="$1"
  local failed=0
  local pids=()
  run_cell "$protein" frozen 17 0 &
  pids+=("$!")
  run_cell "$protein" complete_nested 17 1 &
  pids+=("$!")
  run_cell "$protein" duet 17 2 &
  pids+=("$!")
  run_cell "$protein" frozen 29 3 &
  pids+=("$!")
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" -eq 0 ]] || return 1
  pids=()
  run_cell "$protein" complete_nested 29 0 &
  pids+=("$!")
  run_cell "$protein" duet 29 1 &
  pids+=("$!")
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" -eq 0 ]] || return 1
}

# The protocol fixes the protein execution order.
run_protein prmt5
run_protein prmt6
"$PYTHON" -m confmh.duet.phase_b_runner \
  --config configs/phase_b_pockets/prmt5_b1.yaml --summarize
