#!/usr/bin/env bash
set -euo pipefail

# Sequential continuation: wait only for the PRMT5 GPU allocation to finish,
# then run the independent PRMT6 diagnostic.  No PRMT5 outcome changes PRMT6
# config, reward, checkpoint, or readiness rule.
PROJECT_ROOT=/home/sejin/AI_MD_NSMC
PYTHON=/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python
export DUET_PROJECT_ROOT="$PROJECT_ROOT"
export DUET_ASSET_ROOT=/mnt/ssd0/sejin/confrover_mh_steering
export PHASE_B_DATA_ROOT=/mnt/ssd0/sejin/phase_b_pockets
export PHASE_B_OUTPUT_ROOT=/mnt/ssd0/sejin/phase_b_pockets/outputs
export PYTHONPATH="$PROJECT_ROOT/src"
cd "$PROJECT_ROOT"
LOG_ROOT="$PHASE_B_OUTPUT_ROOT/_launcher"
mkdir -p "$LOG_ROOT"
while [[ -d "$LOG_ROOT/b1_prmt5_amended_running.lock" ]]; do sleep 60; done

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m confmh.duet.phase_b_runner \
  --config configs/phase_b_pockets/prmt6_b0.yaml \
  --method frozen --seed 41 --stage b0_expanded_diagnostic \
  --horizon 4 --outer-k 16 --inner-m 1 --device cuda:0 \
  >"$LOG_ROOT/prmt6_b0_expanded_seed41_gpu0.log" 2>&1

"$PYTHON" scripts/duet/diagnose_phase_b_prmt6_b0.py \
  --run-dir "$PHASE_B_OUTPUT_ROOT/prmt6/b0_expanded_diagnostic/frozen/seed_41" \
  --manifest "$PHASE_B_DATA_ROOT/prepared/prmt6/manifest.json" \
  --mapping "$PHASE_B_DATA_ROOT/prepared/prmt6/residue_mapping.csv" \
  --reference-npz "$PHASE_B_DATA_ROOT/prepared/prmt6/pocket_references_atom37.npz" \
  --output "$PHASE_B_OUTPUT_ROOT/prmt6/diagnostics/b0_expanded_seed_41.json" \
  --pilot-readiness \
  >"$LOG_ROOT/prmt6_b0_expanded_seed41_diagnosis.log" 2>&1

decision=$(jq -r '.exploratory_pilot_readiness.decision' "$PHASE_B_OUTPUT_ROOT/prmt6/diagnostics/b0_expanded_seed_41.json")
if [[ "$decision" == proceed || "$decision" == exploratory_proceed ]]; then
  exec bash scripts/duet/run_phase_b_amended_b1_a6000.sh prmt6
fi
echo "PRMT6 B1 held after diagnostic decision: $decision" >&2
