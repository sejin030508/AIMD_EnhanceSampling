#!/usr/bin/env bash
set -eu

protein="${1:?usage: run_phase_b4_cell.sh protein strength gpu}"
strength="${2:?}"
gpu="${3:?}"
case "$protein" in
  prmt6|smarca2|pi3ka_observed_mask) ;;
  pi3ka)
    echo "pi3ka is blocked: 8TSB lacks reward-region backbone residues 943-950" >&2
    exit 65
    ;;
  *) exit 64 ;;
esac
case "$strength" in 16|32) ;; *) exit 64 ;; esac

project="${DUET_PROJECT_ROOT:?}"
python="${PHASE_B_PYTHON:?}"
root="${PHASE_B_OUTPUT_ROOT:?}/B4_pocket_search"
stage="b4_duet_a${strength}_t24_nofloor_cp075_090"
run_dir="$root/$protein/$stage/duet/seed_103"
log_root="$root/_launcher/b4_duet_t24_nofloor_cp075_090"
mkdir -p "$log_root"
if [[ -f "$run_dir/metrics.json" ]]; then
  echo "skip complete $protein a=$strength" >>"$log_root/status.log"
  exit 0
fi
if [[ -d "$run_dir" ]]; then
  echo "refusing incomplete existing run directory: $run_dir" >&2
  exit 2
fi

export PYTHONPATH="$project/src"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
cd "$project"
if CUDA_VISIBLE_DEVICES="$gpu" "$python" -m confmh.duet.phase_b_runner \
    --config "configs/phase_b_pockets/${protein}_b4_duet_t24.yaml" \
    --method duet --seed 103 --device cuda:0 \
    --stage "$stage" --reward-coefficient "$strength" \
    >"$log_root/${protein}_a${strength}_seed103_gpu${gpu}.log" 2>&1; then
  echo "complete $protein a=$strength" >>"$log_root/status.log"
else
  echo "failed $protein a=$strength" >>"$log_root/status.log"
  exit 1
fi
