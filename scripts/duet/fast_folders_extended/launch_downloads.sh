#!/usr/bin/env bash
# Launch one resumable downloader per archive.
#
# The archives are fetched in parallel because a single stream to pub.htmd.org
# runs at roughly 1 MB/s, which would not finish 34.5 GB before the scheduled
# power cut.  Each archive has its own process and its own output file, and
# every one resumes from the bytes already on disk, so an interrupted run is
# restarted simply by re-running this script.
set -euo pipefail

ROOT="${1:-/workspace/sejin/AI_MD_NSMC_phase_b_recovery}"
PYTHON="${PYTHON:-/workspace/sejin/.conda/envs/tps-dps-eval/bin/python}"
DESTINATION="$ROOT/data/fast_folders_reference"
LOGS="$DESTINATION/_logs"

mkdir -p "$LOGS"
cd "$ROOT"

for name in bba homeodomain proteinb; do
    if pgrep -f "fetch_reference_md.py --destination $DESTINATION --names $name" > /dev/null; then
        echo "$name: already running"
        continue
    fi
    setsid nohup "$PYTHON" scripts/fast_folders_extended/fetch_reference_md.py \
        --destination "$DESTINATION" --names "$name" \
        >> "$LOGS/download_$name.log" 2>&1 < /dev/null &
    echo "$name: launched"
done

sleep 5
echo "--- running ---"
pgrep -f "fetch_reference_md.py --destination" | tr '\n' ' '
echo
