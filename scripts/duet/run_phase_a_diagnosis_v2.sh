#!/usr/bin/env bash
set -euo pipefail

DUET_PROJECT_ROOT="${DUET_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DUET_PROJECT_ROOT
export PYTHONPATH="${DUET_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m confmh.duet.phase_a_diagnosis_v2 \
  --config "${DUET_PROJECT_ROOT}/configs/duet/phase_a_diagnosis_v2.yaml" \
  "$@"
