#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

GATE="${DUET_PROJECT_ROOT}/outputs/duet_md/phase_e_interp_full/two_clock_claim_gate.json"
if [[ ! -f "${GATE}" ]]; then
  echo "Phase F remains gated: missing ${GATE}" >&2
  exit 3
fi
exec "${DUET_PYTHON}" -m confmh.duet.cli run \
  --config "${DUET_PROJECT_ROOT}/configs/duet/proar_transfer.yaml" "$@"

