#!/usr/bin/env bash
set -u

while kill -0 644131 2>/dev/null || kill -0 644132 2>/dev/null; do sleep 60; done
root=/mnt/ssd0/sejin/phase_b_pockets/outputs
if [[ ! -f "$root/prmt6/b1_recovery_mb1/frozen/seed_17/metrics.json" || \
      ! -f "$root/prmt6/b1_recovery_mb1/outer_only/seed_17/metrics.json" ]]; then
  echo "Initial PRMT6 seed17 pair did not both complete; nested pair not started." >&2
  exit 1
fi
common=(
  DUET_PROJECT_ROOT=/home/sejin/AI_MD_NSMC
  DUET_ASSET_ROOT=/mnt/ssd0/sejin/confrover_mh_steering
  PHASE_B_DATA_ROOT=/mnt/ssd0/sejin/phase_b_pockets
  PHASE_B_OUTPUT_ROOT=/mnt/ssd0/sejin/phase_b_pockets/outputs
  PHASE_B_PROVENANCE_ROOT=/mnt/ssd0/sejin/phase_b_pockets/outputs
  PHASE_B_PYTHON=/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python
)
env "${common[@]}" bash /home/sejin/AI_MD_NSMC/scripts/duet/run_phase_b_recovery_cells_mb1.sh \
  prmt6 17 0 complete_nested &
left=$!
env "${common[@]}" bash /home/sejin/AI_MD_NSMC/scripts/duet/run_phase_b_recovery_cells_mb1.sh \
  prmt6 17 3 duet &
right=$!
failed=0
wait "$left" || failed=1
wait "$right" || failed=1
exit "$failed"
