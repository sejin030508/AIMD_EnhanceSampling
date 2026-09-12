#!/usr/bin/env bash
set -u

protein="${1:?usage: run_phase_b_recovery_cells_mb1.sh protein seed gpu method...}"
seed="${2:?}"
gpu="${3:?}"
shift 3
if [[ "$#" -lt 1 ]]; then exit 64; fi
methods=("$@")
PROJECT_ROOT="${DUET_PROJECT_ROOT:?}"
PYTHON="${PHASE_B_PYTHON:?}"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
cd "$PROJECT_ROOT" || exit 1

case "$protein" in
  prmt5)
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert any(p.get("protein")=="prmt5" and p.get("ready_for_b1") is True for p in d["proteins"])' \
      "$PHASE_B_PROVENANCE_ROOT/preflight_status.json" || exit 2 ;;
  prmt6)
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["exploratory_pilot_readiness"]["decision"] in {"proceed","exploratory_proceed"}' \
      "$PHASE_B_PROVENANCE_ROOT/prmt6/diagnostics/b0_expanded_seed_41.json" || exit 3 ;;
  *) exit 64 ;;
esac

config="configs/phase_b_pockets/${protein}_b1_recovery_mb1.yaml"
log_root="$PHASE_B_OUTPUT_ROOT/_launcher/recovery_mb1_${protein}_subset"
mkdir -p "$log_root"
failed=0
for method in "${methods[@]}"; do
  run_dir="$PHASE_B_OUTPUT_ROOT/$protein/b1_recovery_mb1/$method/seed_$seed"
  log="$log_root/${method}_seed${seed}_gpu${gpu}.log"
  if [[ -f "$run_dir/metrics.json" ]]; then
    echo "skip complete $method seed $seed" >>"$log_root/status.log"
    continue
  fi
  if [[ -d "$run_dir" ]]; then
    echo "refusing incomplete existing run directory: $run_dir" >>"$log_root/status.log"
    failed=1
    continue
  fi
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m confmh.duet.phase_b_runner \
      --config "$config" --method "$method" --seed "$seed" --device cuda:0 \
      >"$log" 2>&1; then
    echo "complete $method seed $seed" >>"$log_root/status.log"
  else
    echo "failed $method seed $seed" >>"$log_root/status.log"
    failed=1
  fi
done
exit "$failed"
