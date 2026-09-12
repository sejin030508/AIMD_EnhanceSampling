#!/usr/bin/env bash
set -u

protein="${1:?usage: run_phase_b_recovery_mb1.sh prmt5|prmt6 gpu-id...}"
shift
if [[ "$#" -lt 1 ]]; then
  echo "At least one GPU id is required" >&2
  exit 64
fi
gpus=("$@")
PROJECT_ROOT="${DUET_PROJECT_ROOT:?}"
PYTHON="${PHASE_B_PYTHON:?}"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
cd "$PROJECT_ROOT" || exit 1

config="configs/phase_b_pockets/${protein}_b1_recovery_mb1.yaml"
log_root="$PHASE_B_OUTPUT_ROOT/_launcher/recovery_mb1_${protein}"
mkdir -p "$log_root"
lock="$log_root/running.lock"
if ! mkdir "$lock" 2>/dev/null; then
  echo "Recovery launcher already active: $lock" >&2
  exit 4
fi
trap 'rmdir "$lock" 2>/dev/null || true' EXIT

case "$protein" in
  prmt5)
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert any(p.get("protein")=="prmt5" and p.get("ready_for_b1") is True for p in d["proteins"])' \
      "$PHASE_B_PROVENANCE_ROOT/preflight_status.json" || exit 2
    ;;
  prmt6)
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["exploratory_pilot_readiness"]["decision"] in {"proceed","exploratory_proceed"}' \
      "$PHASE_B_PROVENANCE_ROOT/prmt6/diagnostics/b0_expanded_seed_41.json" || exit 3
    ;;
  *) echo "Unsupported protein: $protein" >&2; exit 64 ;;
esac

run_cell() {
  local method="$1" seed="$2" gpu="$3"
  local run_dir="$PHASE_B_OUTPUT_ROOT/$protein/b1_recovery_mb1/$method/seed_$seed"
  local log="$log_root/${method}_seed${seed}_gpu${gpu}.log"
  if [[ -f "$run_dir/metrics.json" ]]; then
    echo "skip complete $method seed $seed" >>"$log_root/status.log"
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    echo "refusing incomplete existing run directory: $run_dir" >&2
    return 10
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m confmh.duet.phase_b_runner \
    --config "$config" --method "$method" --seed "$seed" --device cuda:0 \
    >"$log" 2>&1
}

methods=(frozen outer_only complete_nested duet)
failed=0
for seed in 17 29; do
  index=0
  while [[ "$index" -lt "${#methods[@]}" ]]; do
    pids=()
    labels=()
    for gpu in "${gpus[@]}"; do
      [[ "$index" -lt "${#methods[@]}" ]] || break
      method="${methods[$index]}"
      run_cell "$method" "$seed" "$gpu" &
      pids+=("$!")
      labels+=("$method")
      index=$((index + 1))
    done
    for slot in "${!pids[@]}"; do
      if ! wait "${pids[$slot]}"; then
        echo "failed ${labels[$slot]} seed $seed" >>"$log_root/status.log"
        failed=1
      else
        echo "complete ${labels[$slot]} seed $seed" >>"$log_root/status.log"
      fi
    done
  done
done
exit "$failed"
