#!/usr/bin/env python
"""Let phase_b_runner build a TICA-coordinate potential.

Dispatch is on a new program.potential key rather than program.type,
because config.py validates type against {terminal, windowed, ordered}
and a new value there would be rejected.

The RMSD potential stays the default.  A config whose program.type is
"tica_terminal" gets TicaEndpointPotential instead, with d0 taken from
the starting structure's own TICA distance rather than the manifest's
Angstrom d0.  The RMSD metric is left in place so every observable,
validity check and diagnostic keeps reporting what it reported before.
"""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = '''    potential = PocketEndpointPotential(
        metric,
        d0_a,
        coefficient=reward_coefficient,
        log_floor=reward_log_floor,
    )
'''

REPLACEMENT = '''    program_potential = str(run_cfg["program"].get("potential", "rmsd"))
    if program_potential == "tica":
        from confmh.duet.tica_potential import TicaEndpointMetric, TicaEndpointPotential

        tica_metric = TicaEndpointMetric(manifest)
        # d0 for this program is the starting structure's own TICA distance,
        # so the potential is scaled by the gap it actually has to close.
        tica_d0 = tica_metric.distance_a(initial_history[0])
        if tica_d0 <= 0.0:
            raise RuntimeError(f"tica_endpoint_not_separated: d0={tica_d0:.6f}")
        potential = TicaEndpointPotential(
            tica_metric,
            tica_d0,
            coefficient=reward_coefficient,
            log_floor=reward_log_floor,
        )
    else:
        potential = PocketEndpointPotential(
            metric,
            d0_a,
            coefficient=reward_coefficient,
            log_floor=reward_log_floor,
        )
'''


def main() -> int:
    path = Path(sys.argv[1])
    source = path.read_text(encoding="utf-8")
    if "program_potential" in source:
        print("already patched")
        return 0
    if source.count(ANCHOR) != 1:
        raise SystemExit(f"anchor appears {source.count(ANCHOR)} times")
    backup = path.with_name(path.name + ".tica.orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    path.write_text(source.replace(ANCHOR, REPLACEMENT), encoding="utf-8")
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
