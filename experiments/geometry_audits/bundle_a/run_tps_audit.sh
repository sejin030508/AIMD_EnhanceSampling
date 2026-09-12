#!/bin/sh
set -eu
PROTEIN="$1"
SHARD_INDEX="${2:-0}"
SHARD_COUNT="${3:-1}"
ROOT=/workspace/sejin/AI_MD_NSMC_phase_b_recovery
PAYLOAD="$ROOT/bundle_a_audit_payload"
export PYTHONPATH="$PAYLOAD:$ROOT/small_protein_pilot"
exec /workspace/sejin/.conda/envs/tps-dps-eval/bin/python "$PAYLOAD/audit_tps_hits.py" \
  --protein "$PROTEIN" --root "$ROOT" --output "$ROOT/outputs/bundle_a_cross_clock_audit/tps_audit" \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"
