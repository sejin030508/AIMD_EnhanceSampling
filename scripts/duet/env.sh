#!/usr/bin/env bash
set -euo pipefail

DUET_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DUET_PROJECT_ROOT="${DUET_PROJECT_ROOT:-$(cd "${DUET_SCRIPT_DIR}/../.." && pwd)}"
export DUET_ASSET_ROOT="${DUET_ASSET_ROOT:-/workspace/sejin/confrover_mh_steering}"
export PYTHONPATH="${DUET_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${DUET_PYTHON:-}" ]]; then
  if [[ -x /workspace/sejin/.conda/envs/confrover-mh/bin/python ]]; then
    DUET_PYTHON=/workspace/sejin/.conda/envs/confrover-mh/bin/python
  else
    DUET_PYTHON=python3
  fi
fi
export DUET_PYTHON
