#!/usr/bin/env bash
set -u

molecule="${1:?usage: run_small_protein_protein_chain.sh molecule}"
case "$molecule" in chignolin|trpcage|bba) ;; *) exit 64 ;; esac
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# shellcheck source=/dev/null
. "$script_dir/small_protein_env.sh"

barrier="$SMALL_PROTEIN_OUTPUT_ROOT/_barriers"
mkdir -p "$barrier/preflight" "$barrier/seed211" "$barrier/seed223"
chain_log="$SMALL_PROTEIN_OUTPUT_ROOT/_launcher/${molecule}_chain.log"
mkdir -p "$(dirname "$chain_log")"
: >"$chain_log"

run_cell() {
    if "$script_dir/run_small_protein_cell.sh" "$@" >>"$chain_log" 2>&1; then
        return 0
    fi
    echo "FAILED cell: $*" >>"$chain_log"
    return 1
}

ready=1
for method in frozen duet; do
    if ! run_cell "$molecule" 16 "$method" 199 preflight; then
        ready=0
    fi
done
printf '%s\n' "$ready" >"$barrier/preflight/$molecule.done"

# Every case marks its barrier even after an individual failure. This prevents
# one blocked molecule from preventing the other two from proceeding.
if [ "$ready" -eq 1 ]; then
    for stride in 16 128; do
        for method in frozen duet; do
            run_cell "$molecule" "$stride" "$method" 211 production || true
        done
    done
fi
printf '%s\n' "$ready" >"$barrier/seed211/$molecule.done"

while [ "$(find "$barrier/seed211" -name '*.done' -type f | wc -l)" -lt 3 ]; do
    sleep 30
done
if [ "$molecule" = chignolin ]; then
    "$SMALL_PROTEIN_EVAL_PYTHON" "$SMALL_PROTEIN_CODE_ROOT/summarize_small_protein_pilot.py" \
        --output-root "$SMALL_PROTEIN_OUTPUT_ROOT" --seed 211 --expected 12 \
        >>"$chain_log" 2>&1 || true
    touch "$barrier/seed211_summary.done"
else
    while [ ! -f "$barrier/seed211_summary.done" ]; do sleep 30; done
fi

if [ "$ready" -eq 1 ]; then
    for stride in 16 128; do
        for method in frozen duet; do
            run_cell "$molecule" "$stride" "$method" 223 production || true
        done
    done
fi
printf '%s\n' "$ready" >"$barrier/seed223/$molecule.done"

while [ "$(find "$barrier/seed223" -name '*.done' -type f | wc -l)" -lt 3 ]; do
    sleep 30
done
if [ "$molecule" = chignolin ]; then
    "$SMALL_PROTEIN_EVAL_PYTHON" "$SMALL_PROTEIN_CODE_ROOT/summarize_small_protein_pilot.py" \
        --output-root "$SMALL_PROTEIN_OUTPUT_ROOT" --expected 24 \
        >>"$chain_log" 2>&1 || true
    touch "$barrier/final_summary.done"
fi
