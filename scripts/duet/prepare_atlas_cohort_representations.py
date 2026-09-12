#!/usr/bin/env python3
"""Generate ConfRover representations for the frozen cohort on a GPU host."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    report = yaml.safe_load(args.report.read_text(encoding="utf-8"))
    cases = [row for row in report["cases"] if row["status"] == "task_frozen"]
    if len(cases) != int(report["target_frozen_count"]):
        raise RuntimeError("Cohort does not contain the target number of frozen tasks")

    environment = dict(os.environ)
    environment["DUET_PROJECT_ROOT"] = str(project_root)
    environment["DUET_ASSET_ROOT"] = str(asset_root)
    environment["PYTHONPATH"] = str(project_root / "src")
    helper = project_root / "scripts" / "duet" / "case_studies" / "prepare_confrover_repr.py"
    for row in cases:
        config = project_root / "configs" / "duet" / "protein_benchmark" / "route_v3_1" / f"{row['pdb_chain']}_main.yaml"
        command = [sys.executable, str(helper), "--config", str(config)]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, env=environment)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
