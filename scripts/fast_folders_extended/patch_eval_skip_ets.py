#!/usr/bin/env python
"""Allow the ETS sampled-frame energy diagnostic to be skipped.

SampledFrameEnergy runs an OpenMM minimizeEnergy() for every frame of
every valid target-reaching path.  Until now no run produced such a path
(every cell recorded sampled_frame_ets_evaluated_path_count = 0), so the
branch was never exercised.  A TICA-coordinate reward does produce them,
and the diagnostic then dominates the cell's wall clock.

It feeds only sampled_frame_ets_* fields, which no comparison in this
project uses, so an opt-out leaves every reported metric unchanged and
keeps the new runs comparable with the existing baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = """    energy_rows = []
    failed_energy_evaluations = 0
    energy_evaluator_error = None
    if len(candidate_success):
"""

REPLACEMENT = """    energy_rows = []
    failed_energy_evaluations = 0
    energy_evaluator_error = None
    skip_ets = os.environ.get("SMALL_PROTEIN_SKIP_ETS", "") not in ("", "0")
    if skip_ets:
        energy_evaluator_error = "skipped_by_SMALL_PROTEIN_SKIP_ETS"
    if len(candidate_success) and not skip_ets:
"""


def main() -> int:
    path = Path(sys.argv[1])
    source = path.read_text(encoding="utf-8")
    if "SMALL_PROTEIN_SKIP_ETS" in source:
        print("already patched")
        return 0
    if source.count(ANCHOR) != 1:
        raise SystemExit(f"anchor appears {source.count(ANCHOR)} times")
    if "\nimport os\n" not in source:
        source = source.replace("\nimport json\n", "\nimport json\nimport os\n", 1)
    backup = path.with_name(path.name + ".ets.orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    path.write_text(source.replace(ANCHOR, REPLACEMENT), encoding="utf-8")
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
