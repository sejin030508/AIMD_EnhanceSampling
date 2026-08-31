#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "${DUET_PYTHON}" -m confmh.duet.cli preflight \
  --config "${DUET_PROJECT_ROOT}/configs/duet/confrover_sde_preflight.yaml" "$@"

