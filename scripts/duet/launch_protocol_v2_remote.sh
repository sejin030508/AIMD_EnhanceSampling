#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 LABEL CONFIG METHOD [METHOD ...]" >&2
  exit 2
fi

LABEL="$1"
CONFIG="$2"
shift 2

source "$(dirname "$0")/env.sh"
cd "${DUET_PROJECT_ROOT}"

LOG_DIR="${DUET_PROJECT_ROOT}/outputs/duet_md/protocol_v2/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${LABEL}.log"
PID_PATH="${LOG_DIR}/${LABEL}.pid"

nohup "${DUET_PYTHON}" scripts/duet/run_protocol_v2.py \
  --config "${CONFIG}" --methods "$@" \
  >"${LOG_PATH}" 2>&1 </dev/null &
PID="$!"
printf '%s\n' "${PID}" >"${PID_PATH}"
printf 'pid=%s log=%s\n' "${PID}" "${LOG_PATH}"
