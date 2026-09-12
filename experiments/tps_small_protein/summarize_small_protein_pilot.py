#!/usr/bin/env python3
"""Create raw CSV/JSON and a compact report for the fixed 24-cell pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "protein", "stride_in_10ps", "method", "seed",
    "weighted_final_backbone_rmsd_a", "best_valid_final_backbone_rmsd_a",
    "heavy_rmsd_metric", "weighted_final_heavy_rmsd_a",
    "weighted_valid_thp", "returned_valid_target_hit_count",
    "population_size", "valid_anytime_hit_count",
    "path_diversity_normalized_tica_dtw_mean",
    "sampled_frame_ets_kj_mol_mean", "decoder_nfe", "wall_clock_s",
    "whole_path_valid_fraction",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/stride*_t32/*/seed_*/small_protein_metrics.json")):
        row = json.loads(path.read_text())
        if args.seed is not None and int(row["seed"]) != args.seed:
            continue
        row["run_directory"] = str(path.parent)
        rows.append(row)
    suffix = f"seed{args.seed}" if args.seed is not None else "all"
    csv_path = root / f"summary_{suffix}.csv"
    json_path = root / f"summary_{suffix}.json"
    report_path = root / (f"report_{suffix}.md" if args.seed is not None else "report.md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + ["run_directory"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS + ["run_directory"]})
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    missing = args.expected - len(rows)
    report_path.write_text(
        "# Small-protein transition recovery pilot\n\n"
        f"Scope: {suffix}. Completed evaluated production cells: "
        f"{len(rows)}/{args.expected}.\n\n"
        f"Missing or failed cells: {max(0, missing)}.\n\n"
        "This table is an exploratory two-seed ConfRover candidate evaluation. "
        "THP uses the fixed TPS-DPS TICA definition, while nominal ConfRover lag "
        "does not establish physical kinetics. Resampled paths and frames are not "
        "independent samples. Inspect `summary_*.csv`, per-run "
        "`small_protein_metrics.json`, and `small_protein_frame_diagnostics.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "scope": suffix,
        "complete": len(rows),
        "expected": args.expected,
        "csv": str(csv_path),
        "json": str(json_path),
        "report": str(report_path),
    }, indent=2))
    if len(rows) != args.expected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
