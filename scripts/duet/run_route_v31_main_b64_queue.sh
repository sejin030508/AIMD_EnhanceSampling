#!/usr/bin/env bash
set -euo pipefail

WORKER="${1:?usage: run_route_v31_main_b64_queue.sh worker_name}"
PROJECT_ROOT="${DUET_PROJECT_ROOT:-/workspace/sejin/AI_MD_NSMC_v31_stage}"
ASSET_ROOT="${DUET_ASSET_ROOT:-/workspace/sejin/confrover_mh_steering}"
PYTHON_BIN="${DUET_PYTHON_BIN:-/workspace/sejin/.conda/envs/confrover-mh/bin/python}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/duet_md/protein_benchmark/route_v3_1"
STATE_ROOT="${OUTPUT_ROOT}/_main_b64_queue"

# Finish every main method for one protein before moving to the next protein.
# Ordered is the primary task and is exhausted before the endpoint task.
PROTEINS=(6jv8_A 7bwf_B 6tly_A 7s86_A 6rrv_A 6gus_A 6q9c_A 7rm7_A 7aex_A 7p46_A)
TASKS=(ordered endpoint)
SEEDS=(20261001 20261002 20261003 20261004 20261005)

# Frozen anchors the comparison.  The primary Complete-vs-DuET comparison is
# completed next, before the two decomposition baselines.
METHODS=(frozen complete_nested duet outer_only inner_only)

export DUET_PROJECT_ROOT="${PROJECT_ROOT}"
export DUET_ASSET_ROOT="${ASSET_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${STATE_ROOT}/locks" "${STATE_ROOT}/done"

run_seed_block() {
  local protein="$1"
  local task="$2"
  local seed="$3"
  local config="${PROJECT_ROOT}/configs/duet/protein_benchmark/route_v3_1/${protein}_main.yaml"
  local protein_output="${OUTPUT_ROOT}/${protein}"
  local method metrics

  echo "BLOCK_START worker=${WORKER} protein=${protein} task=${task} seed=${seed} utc=$(date -u +%FT%TZ)"
  for method in "${METHODS[@]}"; do
    metrics="${protein_output}/${task}/${method}/seed_${seed}/metrics.json"
    if [[ -s "${metrics}" ]]; then
      echo "SKIP_COMPLETE worker=${WORKER} protein=${protein} task=${task} method=${method} seed=${seed}"
      continue
    fi
    echo "RUN_START worker=${WORKER} protein=${protein} task=${task} method=${method} seed=${seed} utc=$(date -u +%FT%TZ)"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/duet/run_protocol_v2.py" \
      --config "${config}" \
      --methods "${method}" \
      --tasks "${task}" \
      --seed "${seed}" \
      --output-directory "${protein_output}" \
      --resume
    if [[ ! -s "${metrics}" ]]; then
      echo "RUN_FAILED_MISSING_METRICS protein=${protein} task=${task} method=${method} seed=${seed}" >&2
      return 1
    fi
    echo "RUN_DONE worker=${WORKER} protein=${protein} task=${task} method=${method} seed=${seed} utc=$(date -u +%FT%TZ)"
  done
  touch "${STATE_ROOT}/done/${protein}.${task}.${seed}.done"
  echo "BLOCK_DONE worker=${WORKER} protein=${protein} task=${task} seed=${seed} utc=$(date -u +%FT%TZ)"
}

echo "QUEUE_START worker=${WORKER} utc=$(date -u +%FT%TZ) budget=64 proteins=${PROTEINS[*]}"
for protein in "${PROTEINS[@]}"; do
  for task in "${TASKS[@]}"; do
    while :; do
      done_count=0
      claimed=0
      for seed in "${SEEDS[@]}"; do
        done_file="${STATE_ROOT}/done/${protein}.${task}.${seed}.done"
        if [[ -f "${done_file}" ]]; then
          done_count=$((done_count + 1))
          continue
        fi
        lock_file="${STATE_ROOT}/locks/${protein}.${task}.${seed}.lock"
        if flock -n 9; then
          # Recheck after acquiring the lock because the other worker may have
          # completed the block between the first check and flock.
          if [[ ! -f "${done_file}" ]]; then
            claimed=1
            run_seed_block "${protein}" "${task}" "${seed}"
          fi
          flock -u 9
          break
        fi 9>"${lock_file}"
      done
      if [[ "${done_count}" -eq "${#SEEDS[@]}" ]]; then
        echo "TASK_DONE worker=${WORKER} protein=${protein} task=${task} utc=$(date -u +%FT%TZ)"
        break
      fi
      if [[ "${claimed}" -eq 0 ]]; then
        sleep 30
      fi
    done
  done
  echo "PROTEIN_DONE worker=${WORKER} protein=${protein} utc=$(date -u +%FT%TZ)"
done
echo "QUEUE_DONE worker=${WORKER} utc=$(date -u +%FT%TZ)"
