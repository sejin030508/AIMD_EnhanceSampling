#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${DUET_PROJECT_ROOT:-/home/sejin/AI_MD_NSMC}"
ASSET_ROOT="${DUET_ASSET_ROOT:-/home/sejin/confrover_mh_steering}"
CONDA_BIN="${CONDA_BIN:-/home/sejin/miniconda3/bin/conda}"
ENV_PREFIX="${DUET_ENV_PREFIX:-/mnt/ssd0/sejin/conda_envs/confrover-mh}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/mnt/ssd0/sejin/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/ssd0/sejin/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/ssd0/sejin/cache}"

mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.10 pip
fi

PYTHON="${ENV_PREFIX}/bin/python"

"${PYTHON}" -m pip install --upgrade pip wheel setuptools==80.9.0
"${PYTHON}" -m pip install -e "${ASSET_ROOT}/external/ConfRover"
"${PYTHON}" -m pip install --no-build-isolation \
  "openfold @ git+https://github.com/aqlaboratory/openfold.git@c587b06e8a9655f30112932693c9e715664ebe41"
"${PYTHON}" -m pip install -r "${ASSET_ROOT}/requirements-overlay.txt"
"${PYTHON}" -m pip install -e "${ASSET_ROOT}"
"${PYTHON}" -m pip install -e "${PROJECT_ROOT}[dev]"

"${PYTHON}" -c "import confrover, confmh, mdtraj, torch; print('python_ok'); print(torch.__version__, torch.version.cuda); print(confrover.__file__); print(confmh.__file__)"
