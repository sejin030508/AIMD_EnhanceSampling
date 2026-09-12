#!/bin/sh
set -eu
PROTEIN="$1"
ARM="$2"
GPU_INDEX="${3:-0}"
ROOT=/workspace/sejin/AI_MD_NSMC_phase_b_recovery
PAYLOAD="$ROOT/bundle_a_audit_payload"
OUT="$ROOT/outputs/bundle_a_cross_clock_audit/sampler_contrast"
export PYTHONPATH="$PAYLOAD:$PAYLOAD/base:$ROOT/src:/workspace/sejin/confrover_mh_steering/external/ConfRover/src"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
exec /workspace/sejin/.conda/envs/confrover-mh/bin/python "$PAYLOAD/run_sampler_contrast.py" \
  --protein "$PROTEIN" --arm "$ARM" --output-root "$OUT" --device cuda:0
