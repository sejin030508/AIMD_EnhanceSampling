#!/usr/bin/env python
"""Open the cell runner's seed whitelist to the 20-seed sweep.

The runner accepts only 199, 211 and 223 and exits 64 otherwise, which caps
every arm at three replicates. Three is too few to separate the arms: the
checkpoint timings differ by less than the seed scatter, so the sweep cannot
say whether timing matters or whether the spread is noise.

The replacement is an arithmetic sequence with step 12 that contains the
original three, so existing cells stay valid and are simply the first three
members of the larger set.
"""
from __future__ import annotations

import sys
from pathlib import Path

SEEDS = [7 + 12 * i for i in range(20)]  # 7 .. 235, contains 199/211/223

OLD = 'case "$seed" in 199|211|223) ;; *) exit 64 ;; esac'
NEW = 'case "$seed" in ' + "|".join(str(s) for s in SEEDS) + ') ;; *) exit 64 ;; esac'


def main() -> int:
    path = Path(sys.argv[1])
    source = path.read_text(encoding="utf-8")
    if "|7|" in source or source.count(OLD) == 0 and "235" in source:
        print("already patched")
        return 0
    if source.count(OLD) != 1:
        raise SystemExit(f"anchor appears {source.count(OLD)} times")
    backup = path.with_name(path.name + ".seeds.orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    path.write_text(source.replace(OLD, NEW), encoding="utf-8")
    print(f"patched {path} -> {len(SEEDS)} seeds: {SEEDS[0]}..{SEEDS[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
