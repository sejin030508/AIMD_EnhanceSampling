#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?usage: $0 GPU_ID SHARD_ID SUPPORT_METHOD}"
SHARD_ID="${2:?usage: $0 GPU_ID SHARD_ID SUPPORT_METHOD}"
SUPPORT_METHOD="${3:?usage: $0 GPU_ID SHARD_ID SUPPORT_METHOD}"

if [[ ! "${GPU_ID}" =~ ^[0-3]$ ]]; then
  echo "GPU_ID must be one of 0, 1, 2, or 3" >&2
  exit 2
fi
if [[ "${SHARD_ID}" != "0" && "${SHARD_ID}" != "1" ]]; then
  echo "SHARD_ID must be 0 or 1" >&2
  exit 2
fi
if [[ "${SUPPORT_METHOD}" != "complete_nested" && "${SUPPORT_METHOD}" != "duet" ]]; then
  echo "SUPPORT_METHOD must be complete_nested or duet" >&2
  exit 2
fi

export DUET_PROJECT_ROOT="${DUET_PROJECT_ROOT:-/home/sejin/AI_MD_NSMC}"
export DUET_ASSET_ROOT="${DUET_ASSET_ROOT:-/home/sejin/confrover_mh_steering}"
export DUET_PYTHON="${DUET_PYTHON:-/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/ssd0/sejin/cache}"
export PYTHONPATH="${DUET_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="configs/duet/case_studies/abl1_dfg_flip_production.yaml"
SUPPORT_ROOT="outputs/duet_md/case_studies/abl1_dfg_flip/support_gate_budget8"
SUPPORT_METRICS="${SUPPORT_ROOT}/endpoint/${SUPPORT_METHOD}/seed_20260930/metrics.json"

cd "${DUET_PROJECT_ROOT}"
echo "[$(date -Iseconds)] support gate start: gpu=${GPU_ID} method=${SUPPORT_METHOD}"
"${DUET_PYTHON}" scripts/duet/run_protocol_v2.py \
  --config "${CONFIG}" \
  --methods "${SUPPORT_METHOD}" \
  --seed 20260930 \
  --outer-k 4 \
  --inner-m 2 \
  --output-directory "${SUPPORT_ROOT}" \
  --resume

"${DUET_PYTHON}" - "${SUPPORT_METRICS}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    metrics = json.load(handle)

required = {
    "decoder_nfe": metrics.get("decoder_nfe"),
    "structural_validity_rate": metrics.get("structural_validity_rate"),
    "path_structural_validity_rate": metrics.get("path_structural_validity_rate"),
}
if not all(value is not None and math.isfinite(float(value)) for value in required.values()):
    raise SystemExit(f"support gate has missing/non-finite metrics: {required}")
if float(required["decoder_nfe"]) <= 0:
    raise SystemExit(f"support gate has invalid decoder NFE: {required}")
if float(required["structural_validity_rate"]) < 1.0:
    raise SystemExit(f"support gate failed endpoint validity: {required}")
if float(required["path_structural_validity_rate"]) < 1.0:
    raise SystemExit(f"support gate failed path validity: {required}")
print(f"support gate passed: {required}")
PY

echo "[$(date -Iseconds)] production shard start: gpu=${GPU_ID} shard=${SHARD_ID}"
bash scripts/duet/case_studies/launch_phase_e_a6000.sh "${GPU_ID}" "${SHARD_ID}" abl1
echo "[$(date -Iseconds)] production shard complete: gpu=${GPU_ID} shard=${SHARD_ID}"
