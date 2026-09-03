#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: $0 METHOD PARENT_QUEUE_PID_FILE}"
PARENT_QUEUE_PID_FILE="${2:?usage: $0 METHOD PARENT_QUEUE_PID_FILE}"

case "${METHOD}" in
  complete_nested|duet) ;;
  *)
    echo "unsupported method: ${METHOD}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="/workspace/sejin/AI_MD_NSMC"
CONFIG="configs/duet/7lp1_v2_pilot_ordered.yaml"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/duet_md/extra_proteins/7lp1_A/protocol_v2"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source scripts/duet/env.sh

metrics_path() {
  local output_dir="$1"
  local seed="$2"
  printf '%s/ordered/%s/seed_%s/metrics.json' "${output_dir}" "${METHOD}" "${seed}"
}

run_cell() {
  local outer_k="$1"
  local inner_m="$2"
  local seed="$3"
  local output_dir="$4"
  local metrics
  metrics="$(metrics_path "${output_dir}" "${seed}")"

  if [[ -s "${metrics}" ]]; then
    echo "$(date -Is) skip completed ${METHOD} K${outer_k} M${inner_m} seed=${seed}"
    return 0
  fi

  echo "$(date -Is) start ${METHOD} K${outer_k} M${inner_m} seed=${seed}"
  "${DUET_PYTHON}" scripts/duet/run_protocol_v2.py \
    --config "${CONFIG}" \
    --methods "${METHOD}" \
    --seed "${seed}" \
    --outer-k "${outer_k}" \
    --inner-m "${inner_m}" \
    --checkpoint-progress 0.95 \
    --outer-resampling-ess-fraction 0.50 \
    --output-directory "${output_dir}" \
    --resume

  if [[ ! -s "${metrics}" ]]; then
    echo "missing metrics after ${METHOD} K${outer_k} M${inner_m} seed=${seed}: ${metrics}" >&2
    return 1
  fi
  echo "$(date -Is) complete ${METHOD} K${outer_k} M${inner_m} seed=${seed}"
}

# Do not compete with the already-running base queue. Zombie processes are
# treated as finished because detached pod jobs may remain unreaped by PID 1.
if [[ -s "${PARENT_QUEUE_PID_FILE}" ]]; then
  parent_pid="$(tr -cd '0-9' < "${PARENT_QUEUE_PID_FILE}")"
  while [[ -n "${parent_pid}" ]]; do
    parent_stat="$(ps -o stat= -p "${parent_pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -z "${parent_stat}" || "${parent_stat}" == Z* ]]; then
      break
    fi
    sleep 30
  done
fi

# Re-enter the base queue once to resume any interrupted cell and verify that
# every seed-20--22 result exists before extending the evaluation.
bash scripts/duet/run_7lp1_overnight_queue.sh "${METHOD}" /dev/null

# Bring all primary allocations to five predeclared matched seeds.
for seed in 20260923 20260924; do
  run_cell 4 8 "${seed}" "${OUTPUT_BASE}/main/K4_M8_p95"
  run_cell 8 4 "${seed}" "${OUTPUT_BASE}/main/K8_M4_p95"
  run_cell 8 8 "${seed}" "${OUTPUT_BASE}/confirmation/K8_M8_p95"
done

# Bring both fixed-budget-64 allocation diagnostics to the same five seeds.
for seed in 20260922 20260923 20260924; do
  run_cell 16 4 "${seed}" "${OUTPUT_BASE}/factorial/K16_M4_p95"
  run_cell 4 16 "${seed}" "${OUTPUT_BASE}/factorial/K4_M16_p95"
done

echo "$(date -Is) five-seed extension complete for ${METHOD}"
