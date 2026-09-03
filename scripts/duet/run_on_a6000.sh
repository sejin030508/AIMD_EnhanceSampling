#!/usr/bin/env bash
set -euo pipefail

export DUET_PROJECT_ROOT="${DUET_PROJECT_ROOT:-/home/sejin/AI_MD_NSMC}"
export DUET_ASSET_ROOT="${DUET_ASSET_ROOT:-/home/sejin/confrover_mh_steering}"
export DUET_PYTHON="${DUET_PYTHON:-/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/ssd0/sejin/cache}"
export PYTHONPATH="${DUET_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

cd "${DUET_PROJECT_ROOT}"
exec "$@"
