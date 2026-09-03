#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: $0 METHOD CURRENT_PID_FILE}"
CURRENT_PID_FILE="${2:?usage: $0 METHOD CURRENT_PID_FILE}"

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

CURRENT_OUTPUT="${OUTPUT_BASE}/main/K4_M8_p95"
CURRENT_METRICS="$(metrics_path "${CURRENT_OUTPUT}" 20260920)"

if [[ ! -s "${CURRENT_METRICS}" && -s "${CURRENT_PID_FILE}" ]]; then
  current_pid="$(tr -cd '0-9' < "${CURRENT_PID_FILE}")"
  while [[ -n "${current_pid}" ]]; do
    if [[ -s "${CURRENT_METRICS}" ]]; then
      break
    fi
    current_stat="$(ps -o stat= -p "${current_pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -z "${current_stat}" || "${current_stat}" == Z* ]]; then
      break
    fi
    sleep 30
  done
fi

# Recover the already-running matched cell if it ended without a complete result.
run_cell 4 8 20260920 "${CURRENT_OUTPUT}"

# Priority 1: test whether the K=4 outer population is the immediate bottleneck.
run_cell 8 4 20260920 "${OUTPUT_BASE}/main/K8_M4_p95"

# Priority 2: test whether increasing both outer and inner populations rescues DuET.
run_cell 8 8 20260920 "${OUTPUT_BASE}/confirmation/K8_M8_p95"

# Priority 3: repeat all primary population shapes on two fresh, matched seeds.
for seed in 20260921 20260922; do
  run_cell 4 8 "${seed}" "${OUTPUT_BASE}/main/K4_M8_p95"
  run_cell 8 4 "${seed}" "${OUTPUT_BASE}/main/K8_M4_p95"
  run_cell 8 8 "${seed}" "${OUTPUT_BASE}/confirmation/K8_M8_p95"
done

# Priority 4: decompose a fixed population budget of 64 into outer-heavy and
# inner-heavy allocations. These are exploratory mechanism cells, not headline
# comparisons selected after seeing their outcomes.
for seed in 20260920 20260921; do
  run_cell 16 4 "${seed}" "${OUTPUT_BASE}/factorial/K16_M4_p95"
  run_cell 4 16 "${seed}" "${OUTPUT_BASE}/factorial/K4_M16_p95"
done

echo "$(date -Is) overnight queue complete for ${METHOD}"
