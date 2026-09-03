#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?usage: $0 GPU_ID SHARD_ID [abl1|smarca2]}"
SHARD_ID="${2:?usage: $0 GPU_ID SHARD_ID [abl1|smarca2]}"
CASE_NAME="${3:-abl1}"

if [[ ! "${GPU_ID}" =~ ^[0-3]$ ]]; then
  echo "GPU_ID must be one of 0, 1, 2, or 3" >&2
  exit 2
fi
if [[ "${SHARD_ID}" != "0" && "${SHARD_ID}" != "1" ]]; then
  echo "SHARD_ID must be 0 or 1" >&2
  exit 2
fi
if [[ "${CASE_NAME}" != "abl1" && "${CASE_NAME}" != "smarca2" ]]; then
  echo "CASE_NAME must be abl1 or smarca2" >&2
  exit 2
fi

export DUET_PROJECT_ROOT="${DUET_PROJECT_ROOT:-/home/sejin/AI_MD_NSMC}"
export DUET_ASSET_ROOT="${DUET_ASSET_ROOT:-/home/sejin/confrover_mh_steering}"
export DUET_PYTHON="${DUET_PYTHON:-/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/ssd0/sejin/cache}"
export PYTHONPATH="${DUET_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${CASE_NAME}" == "abl1" ]]; then
  CONFIG="configs/duet/case_studies/abl1_dfg_flip_production.yaml"
else
  CONFIG="configs/duet/case_studies/smarca2_cryptic_production.yaml"
fi

SEEDS=(20261001 20261002 20261003 20261004 20261005)
METHODS=(frozen complete_nested duet)
LOG_ROOT="${DUET_PROJECT_ROOT}/outputs/duet_md/case_studies/_launcher/${CASE_NAME}/gpu${GPU_ID}"
mkdir -p "${LOG_ROOT}"
cd "${DUET_PROJECT_ROOT}"

job_index=0
for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    if (( job_index % 2 == SHARD_ID )); then
      log="${LOG_ROOT}/${method}_seed_${seed}.log"
      echo "[$(date -Iseconds)] start ${CASE_NAME} ${method} seed=${seed}" | tee -a "${log}"
      "${DUET_PYTHON}" scripts/duet/run_protocol_v2.py \
        --config "${CONFIG}" \
        --methods "${method}" \
        --seed "${seed}" \
        --resume >>"${log}" 2>&1
      echo "[$(date -Iseconds)] done ${CASE_NAME} ${method} seed=${seed}" | tee -a "${log}"
    fi
    job_index=$((job_index + 1))
  done
done
