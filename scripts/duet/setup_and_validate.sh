#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"${DUET_PYTHON}" -m compileall -q "${DUET_PROJECT_ROOT}/src"
"${DUET_PYTHON}" -m pytest -q "${DUET_PROJECT_ROOT}/tests"
"${DUET_PYTHON}" -m confmh.duet.cli validate \
  --config "${DUET_PROJECT_ROOT}/configs/duet/exact_toy.yaml" "$@"
"${DUET_PYTHON}" -m confmh.duet.cli prepare-programs \
  --config "${DUET_PROJECT_ROOT}/configs/duet/prepare_6j56_programs.yaml" \
  --dry-run

