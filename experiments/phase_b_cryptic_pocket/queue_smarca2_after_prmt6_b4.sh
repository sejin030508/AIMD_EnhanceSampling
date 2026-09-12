#!/usr/bin/env bash
set -eu

project=/workspace/sejin/AI_MD_NSMC_phase_b_recovery
data_root=/workspace/sejin/phase_b_pockets_recovery
output_root="$data_root/outputs"
search_root="$output_root/B4_pocket_search"
repr_pid="${1:?usage: queue_smarca2_after_prmt6_b4.sh smarca2_repr_pid}"
repr_log="$search_root/_setup/smarca2_repr.log"
queue_log="$search_root/_setup/smarca2_queue.log"

while kill -0 "$repr_pid" 2>/dev/null; do
  repr_state="$(ps -o stat= -p "$repr_pid" 2>/dev/null || true)"
  if [[ "$repr_state" == Z* ]]; then
    break
  fi
  sleep 30
done
if ! grep -q 'representation_ready=' "$repr_log"; then
  printf '%s\n' 'SMARCA2 representation preparation failed; scientific runs not started.' >>"$queue_log"
  exit 1
fi
printf '%s\n' 'SMARCA2 representation ready; waiting for both PRMT6 B4 cells.' >>"$queue_log"

prmt_a16="$search_root/prmt6/b4_duet_a16_t24_nofloor_cp075_090/duet/seed_103/metrics.json"
prmt_a32="$search_root/prmt6/b4_duet_a32_t24_nofloor_cp075_090/duet/seed_103/metrics.json"
while [[ ! -f "$prmt_a16" || ! -f "$prmt_a32" ]]; do
  if grep -q '^failed prmt6 ' "$search_root/_launcher/b4_duet_t24_nofloor_cp075_090/status.log" 2>/dev/null; then
    printf '%s\n' 'PRMT6 B4 chain failed; SMARCA2 scientific runs not started.' >>"$queue_log"
    exit 1
  fi
  sleep 60
done

export DUET_PROJECT_ROOT="$project"
export DUET_ASSET_ROOT=/workspace/sejin/confrover_mh_steering
export PHASE_B_DATA_ROOT="$data_root"
export PHASE_B_OUTPUT_ROOT="$output_root"
export PHASE_B_PYTHON=/workspace/sejin/.conda/envs/confrover-mh/bin/python
printf '%s\n' 'PRMT6 B4 chain complete; starting SMARCA2 a=16 then a=32.' >>"$queue_log"
exec bash "$project/scripts/duet/run_phase_b4_protein_chain.sh" smarca2 0
