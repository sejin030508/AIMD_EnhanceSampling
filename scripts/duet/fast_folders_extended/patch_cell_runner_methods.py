#!/usr/bin/env python
"""Let the cell runner dispatch complete_nested.

The runner whitelists only ``frozen`` and ``duet`` and exits 64 otherwise, so
the control arm that actually isolates DuET's claim -- selecting after the frame
is finished, rather than mid-diffusion -- could not be launched through it.
``complete_nested`` is already a supported method in OuterSMC; only this guard
was missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

OLD = 'case "$method" in frozen|duet) ;; *) exit 64 ;; esac'
NEW = 'case "$method" in frozen|duet|complete_nested) ;; *) exit 64 ;; esac'


def main() -> int:
    path = Path(sys.argv[1])
    source = path.read_text(encoding="utf-8")
    if "complete_nested" in source:
        print("already patched")
        return 0
    if source.count(OLD) != 1:
        raise SystemExit(f"anchor appears {source.count(OLD)} times")
    backup = path.with_name(path.name + ".methods.orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    path.write_text(source.replace(OLD, NEW), encoding="utf-8")
    print(f"patched {path} (backup {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
