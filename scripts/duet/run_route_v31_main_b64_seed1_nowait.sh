#!/usr/bin/env bash
set -euo pipefail

WORKER="${1:?usage: run_route_v31_main_b64_seed1_nowait.sh worker_name}"
PROJECT_ROOT="${DUET_PROJECT_ROOT:-/workspace/sejin/AI_MD_NSMC_v31_stage}"
ASSET_ROOT="${DUET_ASSET_ROOT:-/workspace/sejin/confrover_mh_steering}"
PYTHON_BIN="${DUET_PYTHON_BIN:-/workspace/sejin/.conda/envs/confrover-mh/bin/python}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/duet_md/protein_benchmark/route_v3_1"
STATE_ROOT="${OUTPUT_ROOT}/_main_b64_seed1_nowait"
SEED=20261001

PROTEINS=(6jv8_A 7bwf_B 6tly_A 7s86_A 6rrv_A 6gus_A 6q9c_A 7rm7_A 7aex_A 7p46_A)
TASKS=(ordered endpoint)
METHODS=(frozen complete_nested duet outer_only inner_only)

export DUET_PROJECT_ROOT="${PROJECT_ROOT}"
export DUET_ASSET_ROOT="${ASSET_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${STATE_ROOT}/locks"

run_unit_if_claimed() {
  local protein="$1"
  local task="$2"
  local method="$3"
  local protein_output="${OUTPUT_ROOT}/${protein}"
  local metrics="${protein_output}/${task}/${method}/seed_${SEED}/metrics.json"
  local lock="${STATE_ROOT}/locks/${protein}.${task}.${method}.${SEED}.lock"
  local config="${PROJECT_ROOT}/configs/duet/protein_benchmark/route_v3_1/${protein}_main.yaml"

  exec 9>"${lock}"
  if ! flock -n 9; then
    exec 9>&-
    return 75
  fi
  if [[ -s "${metrics}" ]]; then
    flock -u 9
    exec 9>&-
    return 0
  fi

  echo "RUN_START worker=${WORKER} protein=${protein} task=${task} method=${method} seed=${SEED} utc=$(date -u +%FT%TZ)"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/duet/run_protocol_v2.py" \
    --config "${config}" \
    --methods "${method}" \
    --tasks "${task}" \
    --seed "${SEED}" \
    --output-directory "${protein_output}" \
    --resume
  if [[ ! -s "${metrics}" ]]; then
    echo "RUN_FAILED_MISSING_METRICS protein=${protein} task=${task} method=${method} seed=${SEED}" >&2
    exit 1
  fi
  echo "RUN_DONE worker=${WORKER} protein=${protein} task=${task} method=${method} seed=${SEED} utc=$(date -u +%FT%TZ)"
  flock -u 9
  exec 9>&-
  return 0
}

echo "QUEUE_START worker=${WORKER} seed=${SEED} scheduling=global_nowait utc=$(date -u +%FT%TZ)"
while :; do
  pending=0
  claimed=0
  for protein in "${PROTEINS[@]}"; do
    for task in "${TASKS[@]}"; do
      for method in "${METHODS[@]}"; do
        metrics="${OUTPUT_ROOT}/${protein}/${task}/${method}/seed_${SEED}/metrics.json"
        if [[ -s "${metrics}" ]]; then
          continue
        fi
        pending=1
        if run_unit_if_claimed "${protein}" "${task}" "${method}"; then
          claimed=1
          break 3
        else
          status=$?
          if [[ "${status}" -ne 75 ]]; then
            exit "${status}"
          fi
        fi
      done
    done
  done

  if [[ "${pending}" -eq 0 ]]; then
    echo "QUEUE_DONE worker=${WORKER} utc=$(date -u +%FT%TZ)"
    exit 0
  fi
  if [[ "${claimed}" -eq 0 ]]; then
    sleep 30
  fi
done
