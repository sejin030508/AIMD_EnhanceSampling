#!/usr/bin/env bash
# Reorders an already-launched fixed matrix without changing any cell settings.
# It waits for the incumbent cell to finish, then runs all remaining DuET cells
# before any remaining Frozen cells.  Completed cells are skipped by the normal
# cell runner; incomplete directories are never overwritten.
set -euo pipefail

protein="${1:?usage: priority_scheduler protein duet|frozen incumbent_cell_pid incumbent_chain_pid}"
phase="${2:?}"
incumbent_cell_pid="${3:?}"
incumbent_chain_pid="${4:?}"

case "$protein" in chignolin|trpcage|bba) ;; *) exit 64 ;; esac
case "$phase" in duet|frozen) ;; *) exit 64 ;; esac

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. "$script_dir/small_protein_env.sh"

barrier="$SMALL_PROTEIN_OUTPUT_ROOT/_priority_barriers"
log_root="$SMALL_PROTEIN_OUTPUT_ROOT/_launcher"
mkdir -p "$barrier/duet" "$barrier/frozen" "$log_root"
log="$log_root/${protein}_priority_${phase}.log"

note() { printf '%s %s\n' "$(date -u '+%FT%TZ')" "$*" | tee -a "$log"; }

# A stopped legacy supervisor cannot reap its completed child, so `kill -0`
# remains true for that child's zombie PID.  Treat Z state as completed while
# retaining normal waiting behavior for a live sampler/evaluator.
incumbent_is_live() {
    state="$(ps -p "$incumbent_cell_pid" -o stat= 2>/dev/null | tr -d '[:space:]' || true)"
    [ -n "$state" ] && case "$state" in *Z*) return 1 ;; *) return 0 ;; esac
}

if [ "$phase" = duet ]; then
    note "waiting for incumbent cell pid=$incumbent_cell_pid"
    while incumbent_is_live; do sleep 30; done
    # The original supervisor is deliberately stopped by the controller. It
    # must not resume and schedule the old Frozen-first order after its child
    # exits.
    kill -TERM "$incumbent_chain_pid" 2>/dev/null || true
    note "incumbent cell finished; start fixed DuET-priority queue"
    for seed in 211 223; do
        for stride in 16 128; do
            "$script_dir/run_small_protein_cell.sh" "$protein" "$stride" duet "$seed" production \
                >>"$log" 2>&1 || note "FAILED duet stride=$stride seed=$seed; continuing fixed queue"
        done
    done
    touch "$barrier/duet/${protein}.done"
    note "all DuET cells resolved"
    exit 0
fi

note "waiting for all three DuET-priority queues"
while [ "$(find "$barrier/duet" -name '*.done' -type f | wc -l)" -lt 3 ]; do sleep 30; done
note "start remaining fixed Frozen queue"
for seed in 211 223; do
    for stride in 16 128; do
        "$script_dir/run_small_protein_cell.sh" "$protein" "$stride" frozen "$seed" production \
            >>"$log" 2>&1 || note "FAILED frozen stride=$stride seed=$seed; continuing fixed queue"
    done
done
touch "$barrier/frozen/${protein}.done"
note "all Frozen cells resolved"

if [ "$protein" = chignolin ]; then
    while [ "$(find "$barrier/frozen" -name '*.done' -type f | wc -l)" -lt 3 ]; do sleep 30; done
    "$SMALL_PROTEIN_EVAL_PYTHON" "$SMALL_PROTEIN_CODE_ROOT/summarize_small_protein_pilot.py" \
        --output-root "$SMALL_PROTEIN_OUTPUT_ROOT" --expected 24 >>"$log" 2>&1 || true
    touch "$barrier/final_summary.done"
fi
