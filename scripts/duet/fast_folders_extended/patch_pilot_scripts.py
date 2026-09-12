#!/usr/bin/env python
"""Open the small-protein pilot scripts to the extended fast-folder set.

The pilot scripts hard-code the original three molecules and a single
``configs`` directory.  Rather than fork them, this makes the molecule list
and the config subdirectory selectable so the extended proteins reuse the
exact preparation, hashing and verification path the pilot used.  Backups are
written next to each file and re-running is a no-op.
"""
from __future__ import annotations

import argparse
from pathlib import Path

PREPARE_EDITS = [
    (
        'TEMPERATURE_K = {"chignolin": 300.0, "trpcage": 400.0, "bba": 400.0}',
        '''TEMPERATURE_K = {
    "chignolin": 300.0, "trpcage": 400.0, "bba": 400.0,
    # Extended set.  BBL keeps the official TPS-DPS default; the two DESRES
    # additions use the temperature of the Anton run they are derived from.
    "bbl": 300.0, "protein_b": 340.0, "homeodomain": 360.0,
}''',
    ),
    (
        "    args = parser.parse_args()",
        '''    parser.add_argument(
        "--molecules", nargs="+", default=["chignolin", "trpcage", "bba"]
    )
    args = parser.parse_args()''',
    ),
    (
        '    for molecule in ("chignolin", "trpcage", "bba"):',
        "    for molecule in args.molecules:",
    ),
]

VERIFY_EDITS = [
    (
        "    args = parser.parse_args()",
        '''    parser.add_argument(
        "--molecules", nargs="+", default=["chignolin", "trpcage", "bba"]
    )
    parser.add_argument("--configs-subdir", default="configs")
    args = parser.parse_args()''',
    ),
    (
        '    for molecule in ("chignolin", "trpcage", "bba"):',
        "    for molecule in args.molecules:",
    ),
    (
        '            config_path = args.code_root / "configs" / f"{molecule}_stride{stride}_t32.yaml"',
        "            config_path = (\n"
        "                args.code_root / args.configs_subdir\n"
        '                / f"{molecule}_stride{stride}_t32.yaml"\n'
        "            )",
    ),
]

CELL_EDITS = [
    (
        'case "$molecule" in chignolin|trpcage|bba) ;; *) exit 64 ;; esac',
        'case "$molecule" in chignolin|trpcage|bba|bbl|protein_b|homeodomain) ;;'
        " *) exit 64 ;; esac",
    ),
    (
        '    config="$SMALL_PROTEIN_CODE_ROOT/configs/${molecule}_preflight_stride16_t1.yaml"',
        '    config="$SMALL_PROTEIN_CODE_ROOT/${config_subdir}/'
        '${molecule}_preflight_stride16_t1.yaml"',
    ),
    (
        '    config="$SMALL_PROTEIN_CODE_ROOT/configs/${molecule}_stride${stride}_t32.yaml"',
        '    config="$SMALL_PROTEIN_CODE_ROOT/${config_subdir}/'
        '${molecule}_stride${stride}_t32.yaml"',
    ),
    (
        'mode="${5:-production}"',
        'mode="${5:-production}"\n'
        "# Extended proteins keep their configs in a sibling directory so the\n"
        "# original pilot's configs/ stays byte-identical.\n"
        'config_subdir="${SMALL_PROTEIN_CONFIG_SUBDIR:-configs}"',
    ),
]

SENTINELS = ("args.molecules", "SMALL_PROTEIN_CONFIG_SUBDIR")


def apply(path: Path, edits: list[tuple[str, str]]) -> str:
    source = path.read_text(encoding="utf-8")
    if any(sentinel in source for sentinel in SENTINELS):
        return f"already patched: {path.name}"
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    for old, new in edits:
        count = source.count(old)
        if count != 1:
            raise SystemExit(f"{path.name}: anchor appears {count}x: {old[:70]!r}")
        source = source.replace(old, new)
    path.write_text(source, encoding="utf-8")
    return f"patched: {path.name} (backup {backup.name})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    args = parser.parse_args()
    for name, edits in (
        ("prepare_small_protein_inputs.py", PREPARE_EDITS),
        ("verify_small_protein_setup.py", VERIFY_EDITS),
        ("run_small_protein_cell.sh", CELL_EDITS),
    ):
        path = args.pilot_root / name
        if not path.exists():
            print(f"missing: {path}")
            continue
        print(apply(path, edits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
