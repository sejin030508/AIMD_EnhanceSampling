#!/usr/bin/env python3
"""Compute paired Complete-nested versus DuET differences from stable summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from confmh.duet.analysis import paired_bootstrap_ci


METRICS = (
    "pre_resampling_unique_success_rate",
    "pre_resampling_success_weight_mass",
    "post_resampling_success_rate",
    "held_out_path_distance",
    "held_out_endpoint_ca_rmsd_nm",
    "path_pairwise_ca_diversity_nm",
    "surviving_initial_ancestors",
)
LOWER_IS_BETTER = {
    "held_out_path_distance",
    "held_out_endpoint_ca_rmsd_nm",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs: dict[tuple[str, int, int, int], dict] = {}
    for path in args.inputs:
        with path.open() as handle:
            payload = json.load(handle)
        for row in payload["runs"]:
            method = str(row["method"])
            if method not in {"complete_nested", "duet"}:
                continue
            key = (
                method,
                int(row["outer_k"]),
                int(row["inner_m"]),
                int(row["seed"]),
            )
            runs[key] = row

    allocations = sorted({(key[1], key[2]) for key in runs})
    comparisons = []
    for outer_k, inner_m in allocations:
        seeds = sorted(
            {
                key[3]
                for key in runs
                if key[1:3] == (outer_k, inner_m)
                and ("complete_nested", outer_k, inner_m, key[3]) in runs
                and ("duet", outer_k, inner_m, key[3]) in runs
            }
        )
        if not seeds:
            continue
        metrics = {}
        for metric in METRICS:
            differences = np.asarray(
                [
                    float(runs[("duet", outer_k, inner_m, seed)][metric])
                    - float(runs[("complete_nested", outer_k, inner_m, seed)][metric])
                    for seed in seeds
                ],
                dtype=float,
            )
            low, high = paired_bootstrap_ci(
                differences, replicates=100_000, seed=20260903
            )
            tolerance = 1e-12
            favorable = (
                -differences if metric in LOWER_IS_BETTER else differences
            )
            metrics[metric] = {
                "duet_minus_complete_values": differences.tolist(),
                "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "bootstrap_95_interval": [low, high],
                "duet_favorable": int(np.sum(favorable > tolerance)),
                "ties": int(np.sum(np.abs(favorable) <= tolerance)),
                "complete_favorable": int(np.sum(favorable < -tolerance)),
            }
        comparisons.append(
            {
                "outer_k": outer_k,
                "inner_m": inner_m,
                "paired_seeds": seeds,
                "metrics": metrics,
            }
        )

    payload = {
        "difference_convention": (
            "DuET minus Complete nested; positive favors DuET for success/diversity/"
            "ancestors, while negative favors DuET for held-out distances."
        ),
        "comparisons": comparisons,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
