#!/usr/bin/env bash
set -euo pipefail

export DUET_PROJECT_ROOT="${DUET_PROJECT_ROOT:-/home/sejin/AI_MD_NSMC}"
export DUET_ASSET_ROOT="${DUET_ASSET_ROOT:-/home/sejin/confrover_mh_steering}"
export DUET_PYTHON="${DUET_PYTHON:-/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python}"
export CUDA_VISIBLE_DEVICES=2
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/ssd0/sejin/cache}"
export PYTHONPATH="${DUET_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${DUET_PROJECT_ROOT}"

"${DUET_PYTHON}" scripts/duet/extract_atlas_case_frames.py \
  --topology "${DUET_ASSET_ROOT}/data/atlas/7lp1_A/7lp1_A.pdb" \
  --trajectory "${DUET_ASSET_ROOT}/data/atlas/7lp1_A/7lp1_A_prod_R2_fit.xtc" \
  --start-frame 5000 \
  --end-frame 7048 \
  --start-output "${DUET_ASSET_ROOT}/data/atlas/7lp1_A/7lp1_A_R2F5000_start.pdb" \
  --end-output "${DUET_ASSET_ROOT}/data/atlas/7lp1_A/7lp1_A_R2F7048_end.pdb" \
  --manifest-output "${DUET_ASSET_ROOT}/data/atlas/7lp1_A/7LP1-A-R2F5000S256_manifest.json"

"${DUET_PYTHON}" -m confmh.duet.cli prepare-programs \
  --config configs/duet/prepare_7lp1_programs.yaml

"${DUET_PYTHON}" -m confmh.duet.cli preflight \
  --config configs/duet/7lp1_preflight.yaml \
  --resume

"${DUET_PYTHON}" -m confmh.duet.cli run \
  --config configs/duet/7lp1_factorial.yaml \
  --resume

"${DUET_PYTHON}" -m confmh.duet.cli run-grid \
  --config configs/duet/7lp1_km_grid.yaml \
  --resume
