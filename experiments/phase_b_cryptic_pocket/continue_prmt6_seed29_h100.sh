#!/usr/bin/env bash
set -u

while kill -0 3775361 2>/dev/null; do sleep 60; done
root=/workspace/sejin/phase_b_pockets_recovery/outputs
count=$(find "$root/prmt5/b1_recovery_mb1" -name metrics.json 2>/dev/null | wc -l)
if [[ "$count" -ne 8 ]]; then
  echo "PRMT5 recovery completed $count/8; PRMT6 seed29 reallocation not started." >&2
  exit 1
fi
export DUET_PROJECT_ROOT=/workspace/sejin/AI_MD_NSMC_phase_b_recovery
export DUET_ASSET_ROOT=/workspace/sejin/confrover_mh_steering
export PHASE_B_DATA_ROOT=/workspace/sejin/phase_b_pockets_recovery
export PHASE_B_OUTPUT_ROOT=/workspace/sejin/phase_b_pockets_recovery/outputs
export PHASE_B_PROVENANCE_ROOT=/workspace/sejin/phase_b_pockets_recovery/provenance
export PHASE_B_PYTHON=/workspace/sejin/.conda/envs/confrover-mh/bin/python
exec bash "$DUET_PROJECT_ROOT/scripts/duet/run_phase_b_recovery_cells_mb1.sh" \
  prmt6 29 0 frozen outer_only complete_nested duet
