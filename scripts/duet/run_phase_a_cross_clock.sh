#!/usr/bin/env bash
set -euo pipefail

DUET_PROJECT_ROOT="${DUET_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DUET_PROJECT_ROOT

python -m confmh.duet.cli run \
  --config "${DUET_PROJECT_ROOT}/configs/duet/phase_a_cross_clock.yaml" \
  "$@"
