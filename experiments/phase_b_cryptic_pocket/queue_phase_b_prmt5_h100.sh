#!/usr/bin/env bash
set -u

# Do not contend with the pre-existing route-v3.1 job.  Launch recovery only
# after the H100 has no compute process.
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
  sleep 60
done
export DUET_PROJECT_ROOT=/workspace/sejin/AI_MD_NSMC_phase_b_recovery
export DUET_ASSET_ROOT=/workspace/sejin/confrover_mh_steering
export PHASE_B_DATA_ROOT=/workspace/sejin/phase_b_pockets_recovery
export PHASE_B_OUTPUT_ROOT=/workspace/sejin/phase_b_pockets_recovery/outputs
export PHASE_B_PROVENANCE_ROOT=/workspace/sejin/phase_b_pockets_recovery/provenance
export PHASE_B_PYTHON=/workspace/sejin/.conda/envs/confrover-mh/bin/python
exec bash "$DUET_PROJECT_ROOT/scripts/duet/run_phase_b_recovery_mb1.sh" prmt5 0
