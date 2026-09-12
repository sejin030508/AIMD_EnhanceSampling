#!/usr/bin/env bash

export DUET_PROJECT_ROOT="${DUET_PROJECT_ROOT:-/workspace/sejin/AI_MD_NSMC_phase_b_recovery}"
export DUET_ASSET_ROOT="${DUET_ASSET_ROOT:-/workspace/sejin/confrover_mh_steering}"
export SMALL_PROTEIN_CODE_ROOT="${SMALL_PROTEIN_CODE_ROOT:-$DUET_PROJECT_ROOT/small_protein_pilot}"
export SMALL_PROTEIN_DATA_ROOT="${SMALL_PROTEIN_DATA_ROOT:-$DUET_PROJECT_ROOT/data/small_protein_transition_pilot}"
export SMALL_PROTEIN_OUTPUT_ROOT="${SMALL_PROTEIN_OUTPUT_ROOT:-$DUET_PROJECT_ROOT/outputs/small_protein_transition_pilot}"
export TPS_DPS_ROOT="${TPS_DPS_ROOT:-$DUET_PROJECT_ROOT/external/tps-dps}"
export CONFROVER_PYTHON="${CONFROVER_PYTHON:-/workspace/sejin/.conda/envs/confrover-mh/bin/python}"
export SMALL_PROTEIN_EVAL_PYTHON="${SMALL_PROTEIN_EVAL_PYTHON:-/workspace/sejin/.conda/envs/tps-dps-eval/bin/python}"
export PYTHONPATH="$DUET_PROJECT_ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
