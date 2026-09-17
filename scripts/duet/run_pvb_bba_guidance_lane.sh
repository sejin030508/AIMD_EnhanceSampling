#!/usr/bin/env bash
# Run complete same-seed blocks for the fixed PVB/BBA guidance matrix.
set -u

if [ "$#" -lt 1 ]; then
  echo "usage: $0 SEED [SEED ...]" >&2
  exit 2
fi

ROOT="${GUIDED_PVB_ROOT:-/workspace/sejin/AI_MD_NSMC_guided_pvb_perf}"
PYTHON_BIN="${GUIDED_PVB_PYTHON:-/workspace/sejin/.conda/envs/confrover-mh/bin/python}"
CONFIG="$ROOT/configs/duet/development/pvb_bba_guidance_tuning_20260917.yaml"
OUTPUT_ROOT="$ROOT/outputs/pvb_bba_guidance_tuning_20260917"
LOG_ROOT="$OUTPUT_ROOT/_launcher"
NO_NEW_RUN_AFTER_EPOCH="${NO_NEW_RUN_AFTER_EPOCH:-0}"
CONDITIONS=(G2 C0 D0 G1 G3 G4 G5)

mkdir -p "$LOG_ROOT"
export PYTHONPATH="$ROOT/src:/workspace/sejin/pvb_assets/site"
export GEOMSTATS_BACKEND=pytorch

lane="$(hostname)_$$"
status_file="$LOG_ROOT/${lane}.tsv"
printf 'condition\tseed\tstart_epoch\tend_epoch\telapsed_s\texit_code\n' > "$status_file"

for seed in "$@"; do
  for condition in "${CONDITIONS[@]}"; do
    now="$(date +%s)"
    if [ "$NO_NEW_RUN_AFTER_EPOCH" -gt 0 ] && [ "$now" -ge "$NO_NEW_RUN_AFTER_EPOCH" ]; then
      printf 'deadline\t%s\t%s\t%s\t0\t75\n' "$seed" "$now" "$now" >> "$status_file"
      touch "$LOG_ROOT/${lane}.deadline"
      exit 75
    fi
    log="$LOG_ROOT/${condition}_seed${seed}.log"
    start="$now"
    "$PYTHON_BIN" "$ROOT/scripts/duet/run_pvb_bba_guidance_tuning.py" \
      --config "$CONFIG" --condition "$condition" --seed "$seed" \
      --device cuda:0 > "$log" 2>&1
    code="$?"
    end="$(date +%s)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$condition" "$seed" "$start" "$end" "$((end - start))" "$code" \
      >> "$status_file"
  done
done

touch "$LOG_ROOT/${lane}.done"
