#!/usr/bin/env python
"""Collect the checkpoint-timing variation runs next to the existing baseline.

The (0.75, 0.90) arm is not re-run: it is the completed pilot's stride-128 DuET
cells, read from that pilot's summary so the comparison uses the same numbers
already reported rather than a fresh fit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "weighted_valid_thp",
    "weighted_final_backbone_rmsd_a",
    "best_valid_final_backbone_rmsd_a",
    "whole_path_valid_fraction",
    "returned_valid_target_hit_count",
    "valid_anytime_hit_count",
    "population_size",
]

TIMING_LABEL = {
    "cp2550": "0.25 + 0.50",
    "cp5075": "0.50 + 0.75",
    "cp7590": "0.75 + 0.90 (baseline)",
    "cp255075": "0.25 + 0.50 + 0.75",
}


def read_new(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("small_protein_metrics.json")):
        run = path.parent
        payload = json.loads(path.read_text(encoding="utf-8"))
        tag = run.parents[3].name
        rows.append(
            {
                "timing": tag,
                "protein": payload.get("protein", run.parents[2].name),
                "seed": payload.get("seed", run.name.replace("seed_", "")),
                **{field: payload.get(field) for field in FIELDS},
            }
        )
    return rows


def read_baseline(summary_csv: Path, proteins: set[str]) -> list[dict]:
    if not summary_csv.exists():
        return []
    rows = []
    with summary_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("method") == "duet"
                and row.get("stride_in_10ps") == "128"
                and row.get("protein") in proteins
            ):
                rows.append(
                    {
                        "timing": "cp7590",
                        "protein": row["protein"],
                        "seed": row["seed"],
                        **{field: row.get(field) or None for field in FIELDS},
                    }
                )
    return rows


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = read_new(args.root)
    proteins = {row["protein"] for row in rows}
    rows += read_baseline(args.baseline_summary, proteins)

    order = ["cp2550", "cp5075", "cp7590", "cp255075"]
    rows.sort(key=lambda r: (r["protein"], order.index(r["timing"]) if r["timing"] in order else 9, str(r["seed"])))

    header = f"{'protein':9s} {'timing':22s} {'seed':5s} {'wValidTHP':>10s} {'wBB(A)':>8s} {'bestBB':>7s} {'valid':>6s} {'hit':>5s} {'any':>4s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        thp = number(row["weighted_valid_thp"])
        wbb = number(row["weighted_final_backbone_rmsd_a"])
        best = number(row["best_valid_final_backbone_rmsd_a"])
        valid = number(row["whole_path_valid_fraction"])
        print(
            f"{row['protein']:9s} {TIMING_LABEL.get(row['timing'], row['timing']):22s} "
            f"{str(row['seed']):5s} "
            f"{thp if thp is None else round(thp, 3)!s:>10s} "
            f"{wbb if wbb is None else round(wbb, 2)!s:>8s} "
            f"{best if best is None else round(best, 2)!s:>7s} "
            f"{valid if valid is None else round(valid, 2)!s:>6s} "
            f"{str(row['returned_valid_target_hit_count']):>5s} "
            f"{str(row['valid_anytime_hit_count']):>4s}"
        )

    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
