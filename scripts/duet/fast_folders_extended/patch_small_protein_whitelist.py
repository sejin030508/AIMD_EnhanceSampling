#!/usr/bin/env python
"""Register the extended fast folders with the small-protein entry point.

``small_protein_runner.py`` replaces the Phase-B pocket module's
protein-specific hidden observables with an empty mapping, but only for a
hard-coded set of names.  Any other protein falls through to the pocket module,
which raises ``Unsupported Phase-B protein`` -- after the sampling has already
run, so the cost is paid and then discarded.

BBL, Protein B and Homeodomain have no pocket observables either, so they are
added to the same set.
"""
from __future__ import annotations

import sys
from pathlib import Path

OLD = '_SMALL_PROTEINS = {"chignolin", "trpcage", "bba"}'
NEW = (
    '_SMALL_PROTEINS = {\n'
    '    "chignolin", "trpcage", "bba",\n'
    '    # Extended fast folders; like the originals they carry no pocket\n'
    '    # observables, so the same empty mapping applies.\n'
    '    "bbl", "protein_b", "homeodomain",\n'
    '}'
)


def main() -> int:
    path = Path(sys.argv[1])
    source = path.read_text(encoding="utf-8")
    if '"bbl"' in source:
        print("already patched")
        return 0
    if source.count(OLD) != 1:
        raise SystemExit(f"anchor appears {source.count(OLD)} times")
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    path.write_text(source.replace(OLD, NEW), encoding="utf-8")
    print(f"patched {path} (backup {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
