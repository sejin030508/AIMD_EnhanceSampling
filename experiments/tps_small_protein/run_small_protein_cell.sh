#!/usr/bin/env bash
set -u

molecule="${1:?usage: run_small_protein_cell.sh molecule stride method seed [preflight]}"
stride="${2:?}"
method="${3:?}"
seed="${4:?}"
mode="${5:-production}"

case "$molecule" in chignolin|trpcage|bba) ;; *) exit 64 ;; esac
case "$stride" in 16|128) ;; *) exit 64 ;; esac
case "$method" in frozen|duet) ;; *) exit 64 ;; esac
case "$seed" in 199|211|223) ;; *) exit 64 ;; esac

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# shellcheck source=/dev/null
. "$script_dir/small_protein_env.sh"

if [ "$mode" = preflight ]; then
    config="$SMALL_PROTEIN_CODE_ROOT/configs/${molecule}_preflight_stride16_t1.yaml"
    stage="_preflight_stride16_t1"
    stride=16
elif [ "$mode" = production ]; then
    config="$SMALL_PROTEIN_CODE_ROOT/configs/${molecule}_stride${stride}_t32.yaml"
    stage="stride${stride}_t32"
else
    echo "unknown mode: $mode" >&2
    exit 64
fi

run_dir="$SMALL_PROTEIN_OUTPUT_ROOT/$molecule/$stage/$method/seed_$seed"
log_root="$SMALL_PROTEIN_OUTPUT_ROOT/_launcher"
mkdir -p "$log_root"
log="$log_root/${molecule}_${stage}_${method}_seed${seed}.log"

if [ -f "$run_dir/small_protein_metrics.json" ]; then
    echo "skip evaluated $run_dir" | tee -a "$log_root/status.log"
    exit 0
fi
if [ ! -f "$run_dir/metrics.json" ]; then
    if [ -d "$run_dir" ]; then
        echo "refusing incomplete existing run directory: $run_dir" | tee -a "$log_root/status.log" >&2
        exit 2
    fi
    cd "$DUET_PROJECT_ROOT" || exit 2
    if ! CUDA_VISIBLE_DEVICES=0 "$CONFROVER_PYTHON" \
        "$SMALL_PROTEIN_CODE_ROOT/small_protein_runner.py" \
        --config "$config" --method "$method" --seed "$seed" --device cuda:0 \
        >"$log" 2>&1; then
        echo "sampling_failed $molecule stride=$stride method=$method seed=$seed mode=$mode" \
            | tee -a "$log_root/status.log" >&2
        exit 1
    fi
fi

if ! "$SMALL_PROTEIN_EVAL_PYTHON" "$SMALL_PROTEIN_CODE_ROOT/evaluate_small_protein_run.py" \
    --run-dir "$run_dir" \
    --prepared-dir "$SMALL_PROTEIN_DATA_ROOT/prepared/$molecule" \
    >>"$log" 2>&1; then
    echo "evaluation_failed $molecule stride=$stride method=$method seed=$seed mode=$mode" \
        | tee -a "$log_root/status.log" >&2
    exit 1
fi
echo "complete $molecule stride=$stride method=$method seed=$seed mode=$mode" \
    | tee -a "$log_root/status.log"
