#!/usr/bin/env python3
"""Summarize completed DuET protocol-v2 population-grid metric files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("method", choices=("complete_nested", "duet"))
    return parser.parse_args()


def optional_mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def main() -> None:
    args = parse_args()
    pattern = f"**/ordered/{args.method}/seed_*/metrics.json"
    metric_files = sorted(args.root.glob(pattern))
    rows = []
    for metric_file in metric_files:
        with metric_file.open() as handle:
            row = json.load(handle)
        row["metric_file"] = str(metric_file)
        rows.append(row)

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["outer_k"]), int(row["inner_m"]))].append(row)

    summaries = []
    for (outer_k, inner_m), group in sorted(grouped.items()):
        group.sort(key=lambda row: int(row["seed"]))
        successful_outputs = sum(
            float(row["joint_program_success_rate"]) * int(row["outer_k"])
            for row in group
        )
        total_outputs = sum(int(row["outer_k"]) for row in group)
        summaries.append(
            {
                "method": args.method,
                "outer_k": outer_k,
                "inner_m": inner_m,
                "n_seeds": len(group),
                "seeds": [int(row["seed"]) for row in group],
                "seed_success_rates": [
                    float(row["joint_program_success_rate"]) for row in group
                ],
                "seeds_with_success": sum(
                    float(row["joint_program_success_rate"]) > 0.0 for row in group
                ),
                "pooled_successful_outputs": successful_outputs,
                "pooled_outputs": total_outputs,
                "pooled_output_success_rate": successful_outputs / total_outputs,
                "mean_output_success_rate": optional_mean(
                    group, "joint_program_success_rate"
                ),
                "mean_structural_validity_rate": optional_mean(
                    group, "structural_validity_rate"
                ),
                "mean_path_diversity_nm": optional_mean(
                    group, "path_pairwise_ca_diversity_nm"
                ),
                "mean_held_out_path_distance": optional_mean(
                    group, "held_out_path_distance"
                ),
                "mean_endpoint_ca_rmsd_nm": optional_mean(
                    group, "held_out_endpoint_ca_rmsd_nm"
                ),
                "mean_surviving_initial_ancestors": optional_mean(
                    group, "surviving_initial_ancestors"
                ),
                "mean_inner_ess_1": optional_mean(group, "mean_inner_ess_1"),
                "mean_inner_ess_2": optional_mean(group, "mean_inner_ess_2"),
                "total_decoder_nfe": sum(int(row["decoder_nfe"]) for row in group),
                "total_wall_clock_s": sum(float(row["wall_clock_s"]) for row in group),
            }
        )

    print(json.dumps({"file_count": len(rows), "groups": summaries}, indent=2))


if __name__ == "__main__":
    main()
