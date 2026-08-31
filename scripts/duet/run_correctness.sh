#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "${DUET_PYTHON}" -m confmh.duet.cli run \
  --config "${DUET_PROJECT_ROOT}/configs/duet/exact_toy.yaml" "$@"

