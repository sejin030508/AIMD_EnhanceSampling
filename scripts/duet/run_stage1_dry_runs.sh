#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

for config in \
  exact_toy.yaml \
  prepare_6j56_programs.yaml \
  confrover_sde_preflight.yaml \
  6j56_factorial.yaml \
  6j56_km_grid.yaml
do
  "${DUET_PYTHON}" -m confmh.duet.cli validate \
    --config "${DUET_PROJECT_ROOT}/configs/duet/${config}" --dry-run
done

# These phases are intentionally gated on assets that are not present in the
# upstream workspace.  Exit code 2 means the validator reported those assets;
# any import, schema, or runtime failure still fails this script.
for config in \
  confrover_interp_pilot.yaml \
  confrover_interp_full.yaml \
  proar_transfer.yaml
do
  set +e
  "${DUET_PYTHON}" -m confmh.duet.cli validate \
    --config "${DUET_PROJECT_ROOT}/configs/duet/${config}" --dry-run
  status=$?
  set -e
  if [[ ${status} -ne 0 && ${status} -ne 2 ]]; then
    exit "${status}"
  fi
done
