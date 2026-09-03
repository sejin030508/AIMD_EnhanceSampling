#!/usr/bin/env bash
set -euo pipefail

DUET_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DUET_PROJECT_ROOT="${DUET_PROJECT_ROOT:-$(cd "${DUET_SCRIPT_DIR}/../.." && pwd)}"
if [[ -z "${DUET_ASSET_ROOT:-}" ]]; then
  for candidate in \
    /workspace/sejin/confrover_mh_steering \
    /mnt/ssd0/sejin/confrover_mh_steering \
    /home/sejin/confrover_mh_steering
  do
    if [[ -d "${candidate}" ]]; then
      DUET_ASSET_ROOT="${candidate}"
      break
    fi
  done
fi
: "${DUET_ASSET_ROOT:?Could not locate the shared ConfRover asset root}"
export DUET_ASSET_ROOT
export PYTHONPATH="${DUET_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${DUET_PYTHON:-}" ]]; then
  for candidate in \
    /workspace/sejin/.conda/envs/confrover-mh/bin/python \
    /mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python
  do
    if [[ -x "${candidate}" ]]; then
      DUET_PYTHON="${candidate}"
      break
    fi
  done
  DUET_PYTHON="${DUET_PYTHON:-python3}"
fi
export DUET_PYTHON
