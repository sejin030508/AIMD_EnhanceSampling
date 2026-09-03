#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?usage: $0 complete_host|duet_host}"
case "${ROLE}" in
  complete_host|duet_host) ;;
  *)
    echo "unsupported role: ${ROLE}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="/workspace/sejin/AI_MD_NSMC"
CONFIG="configs/duet/7lp1_v2_pilot_ordered.yaml"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/duet_md/extra_proteins/7lp1_A/protocol_v2"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source scripts/duet/env.sh

run_cell() {
  local method="$1"
  local seed="$2"
  local output_dir="$3"
  local schedule="$4"
  local metrics="${output_dir}/ordered/${method}/seed_${seed}/metrics.json"

  if [[ -s "${metrics}" ]]; then
    echo "$(date -Is) skip completed ${method} seed=${seed} schedule=${schedule}"
    return 0
  fi

  echo "$(date -Is) start ${method} K16 M4 seed=${seed} schedule=${schedule}"
  if [[ "${schedule}" == "p95" ]]; then
    "${DUET_PYTHON}" scripts/duet/run_protocol_v2.py \
      --config "${CONFIG}" \
      --methods "${method}" \
      --seed "${seed}" \
      --outer-k 16 \
      --inner-m 4 \
      --checkpoint-progress 0.95 \
      --outer-resampling-ess-fraction 0.50 \
      --output-directory "${output_dir}" \
      --resume
  else
    "${DUET_PYTHON}" scripts/duet/run_protocol_v2.py \
      --config "${CONFIG}" \
      --methods "${method}" \
      --seed "${seed}" \
      --outer-k 16 \
      --inner-m 4 \
      --checkpoint-progresses 0.85 0.95 \
      --outer-resampling-ess-fraction 0.50 \
      --output-directory "${output_dir}" \
      --resume
  fi

  if [[ ! -s "${metrics}" ]]; then
    echo "missing metrics after ${method} seed=${seed}: ${metrics}" >&2
    return 1
  fi
  echo "$(date -Is) complete ${method} seed=${seed} schedule=${schedule}"
}

CONFIRMATION="${OUTPUT_BASE}/multicheckpoint/confirmation/K16_M4"
DEVELOPMENT="${OUTPUT_BASE}/multicheckpoint/development/K16_M4_p85_p95"

if [[ "${ROLE}" == "complete_host" ]]; then
  # Priority 1: untouched strong baseline on fresh confirmation seeds.
  for seed in 20260925 20260926; do
    run_cell complete_nested "${seed}" "${CONFIRMATION}" p95
  done
  # Priority 2: paired schedule ablation against existing seed-20/22 results.
  for seed in 20260920 20260922; do
    run_cell duet "${seed}" "${DEVELOPMENT}" p85_p95
  done
else
  # Priority 1: multi-checkpoint DuET on the same fresh confirmation seeds.
  for seed in 20260925 20260926; do
    run_cell duet "${seed}" "${CONFIRMATION}" p85_p95
  done
  # Priority 2: paired schedule ablation against existing seed-21/23 results.
  for seed in 20260921 20260923; do
    run_cell duet "${seed}" "${DEVELOPMENT}" p85_p95
  done
fi

echo "$(date -Is) multicheckpoint final queue complete role=${ROLE}"
