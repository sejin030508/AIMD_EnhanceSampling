#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "${DUET_PYTHON}" -m confmh.duet.cli run-grid \
  --config "${DUET_PROJECT_ROOT}/configs/duet/6j56_km_grid.yaml" "$@"

