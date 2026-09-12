#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/sejin/AI_MD_NSMC
PYTHON=/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python
export DUET_PROJECT_ROOT="$PROJECT_ROOT"
export DUET_ASSET_ROOT=/mnt/ssd0/sejin/confrover_mh_steering
export PHASE_B_DATA_ROOT=/mnt/ssd0/sejin/phase_b_pockets
export PHASE_B_OUTPUT_ROOT=/mnt/ssd0/sejin/phase_b_pockets/outputs
export PYTHONPATH="$PROJECT_ROOT/src"

cd "$PROJECT_ROOT"

# PRMT5 already passed these gates. Complete the missing PRMT6 preparation first.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/duet/case_studies/prepare_confrover_repr.py \
  --config configs/phase_b_pockets/prmt6_b1.yaml

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m confmh.duet.phase_b_runner \
  --config configs/phase_b_pockets/prmt6_b1.yaml \
  --method duet --seed 17 --device cuda:0 \
  --stage memory_gate_k1_m4_t1 --horizon 1 --outer-k 1 --inner-m 4

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m confmh.duet.phase_b_runner \
  --config configs/phase_b_pockets/prmt6_b1.yaml \
  --method duet --seed 17 --device cuda:0 \
  --stage memory_gate_k4_m4_t2 --horizon 2 --outer-k 4 --inner-m 4

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m confmh.duet.phase_b_runner \
  --config configs/phase_b_pockets/prmt6_b0.yaml \
  --method frozen --seed 17 --device cuda:0

"$PYTHON" scripts/duet/audit_phase_b_preflight.py \
  --data-root "$PHASE_B_DATA_ROOT" \
  --output-root "$PHASE_B_OUTPUT_ROOT"

jq -e '.all_ready_for_b1 == true and .b1_started == false' \
  "$PHASE_B_OUTPUT_ROOT/preflight_status.json" >/dev/null

# The guarded B1 launcher runs PRMT5 first and PRMT6 second, with all preregistered
# methods and seeds. It writes one log per run under outputs/_launcher.
bash scripts/duet/run_phase_b_b1_a6000.sh
