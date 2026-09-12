#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?usage: run_route_v31_frozen_support_queue.sh gpu1|gpu2}"
PROJECT_ROOT="${DUET_PROJECT_ROOT:-/workspace/sejin/AI_MD_NSMC_v31_stage}"
ASSET_ROOT="${DUET_ASSET_ROOT:-/workspace/sejin/confrover_mh_steering}"
PYTHON_BIN="${DUET_PYTHON_BIN:-/workspace/sejin/.conda/envs/confrover-mh/bin/python}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/duet_md/protein_benchmark/route_v3_1_support_gate/frozen_k48_final"
SEEDS=(20260993 20260994)

case "${ROLE}" in
  gpu1)
    # Prioritize the primary task with the weakest pilot validity, then the
    # remaining shorter primary proteins.
    PROTEINS=(6tly_A 6jv8_A 7bwf_B 7s86_A 6rrv_A)
    ;;
  gpu2)
    # Prioritize the only K=8 A->B observation, then the remaining primary and
    # predeclared reserve proteins.
    PROTEINS=(6q9c_A 6gus_A 7rm7_A 7aex_A 7p46_A)
    ;;
  *)
    echo "unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

export DUET_PROJECT_ROOT="${PROJECT_ROOT}"
export DUET_ASSET_ROOT="${ASSET_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "QUEUE_START role=${ROLE} utc=$(date -u +%FT%TZ) K=48 M=1 seeds=${SEEDS[*]} proteins=${PROTEINS[*]}"
for protein in "${PROTEINS[@]}"; do
  config="${PROJECT_ROOT}/configs/duet/protein_benchmark/route_v3_1/${protein}_main.yaml"
  protein_output="${OUTPUT_ROOT}/${protein}"
  for seed in "${SEEDS[@]}"; do
    metrics="${protein_output}/ordered/frozen/seed_${seed}/metrics.json"
    if [[ -s "${metrics}" ]]; then
      echo "SKIP_COMPLETE protein=${protein} seed=${seed} metrics=${metrics}"
      continue
    fi
    echo "RUN_START role=${ROLE} protein=${protein} seed=${seed} utc=$(date -u +%FT%TZ)"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/duet/run_protocol_v2.py" \
      --config "${config}" \
      --methods frozen \
      --tasks ordered \
      --seed "${seed}" \
      --outer-k 48 \
      --inner-m 1 \
      --output-directory "${protein_output}" \
      --resume
    if [[ ! -s "${metrics}" ]]; then
      echo "RUN_FAILED_MISSING_METRICS protein=${protein} seed=${seed}" >&2
      exit 1
    fi
    echo "RUN_DONE role=${ROLE} protein=${protein} seed=${seed} utc=$(date -u +%FT%TZ)"
  done
done
echo "QUEUE_DONE role=${ROLE} utc=$(date -u +%FT%TZ)"
