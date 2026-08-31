#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

exec "${DUET_PYTHON}" -m confmh.duet.reporting \
  --root "${DUET_PROJECT_ROOT}/outputs/duet_md" \
  --output "${DUET_PROJECT_ROOT}/outputs/duet_md/reports" "$@"
