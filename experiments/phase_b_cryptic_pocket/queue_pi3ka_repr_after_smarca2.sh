#!/usr/bin/env bash
set -eu

project=/workspace/sejin/AI_MD_NSMC_phase_b_recovery
output_root=/workspace/sejin/phase_b_pockets_recovery/outputs
smarca2_log="$output_root/B4_pocket_search/_setup/smarca2_repr.log"
pi3ka_log="$output_root/B4_pocket_search/_setup/pi3ka_repr.log"

# The MSA generator uses a shared fixed .tmp/batch_0 directory, so the two
# new sequences must be prepared serially even though they have separate GPUs.
while ! grep -q 'representation_ready=' "$smarca2_log" 2>/dev/null; do
  if grep -q 'Traceback (most recent call last)' "$smarca2_log" 2>/dev/null; then
    printf '%s\n' 'SMARCA2 representation failed; PI3Kalpha preparation not started.' >"$pi3ka_log"
    exit 1
  fi
  sleep 30
done

cd "$project"
export DUET_ASSET_ROOT=/workspace/sejin/confrover_mh_steering
export PYTHONPATH="$project/src"
export CUDA_VISIBLE_DEVICES=0
exec /workspace/sejin/.conda/envs/confrover-mh/bin/python \
  scripts/duet/case_studies/prepare_confrover_repr.py \
  --config configs/phase_b_pockets/pi3ka_b4_duet_t24.HOLD.yaml \
  >"$pi3ka_log" 2>&1
