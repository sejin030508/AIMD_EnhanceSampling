#!/usr/bin/env python
"""Test whether whole-path validity separates proteins by geometry or by size.

Whole-path validity fails a 33-frame path if any single peptide bond in any
frame leaves the window.  A 47-residue protein offers 2.4x as many chances to
trip that as a 20-residue one, so a size gradient in path validity can appear
with no difference in per-bond quality at all.

This reports the per-bond violation rate alongside the per-path rate.  If the
two disagree, the gate is measuring chain length.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

CA, N, C = 1, 0, 2


def scan(path: Path, low: float, high: float) -> dict | None:
    population = path / "pre_final_population_atom37.npz"
    if not population.exists():
        return None
    with np.load(population) as data:
        key = "trajectories_atom37_a" if "trajectories_atom37_a" in data else data.files[0]
        trajectories = np.asarray(data[key])
    paths, frames = trajectories.shape[0], trajectories.shape[1]
    bonds_total = 0
    bonds_bad = 0
    paths_bad = 0
    worst = []
    for p in range(paths):
        path_bad = False
        for f in range(frames):
            atom37 = trajectories[p, f]
            peptide = np.linalg.norm(atom37[:-1, C, :] - atom37[1:, N, :], axis=-1)
            bad = (peptide < low) | (peptide > high)
            bonds_total += peptide.size
            bonds_bad += int(bad.sum())
            if bad.any():
                path_bad = True
                worst.append(float(peptide[bad].max()))
        paths_bad += int(path_bad)
    return {
        "paths": paths,
        "frames": frames,
        "residues": trajectories.shape[2],
        "bonds_measured": bonds_total,
        "bonds_violating": bonds_bad,
        "per_bond_violation_rate": bonds_bad / bonds_total,
        "paths_invalid": paths_bad,
        "per_path_invalid_rate": paths_bad / paths,
        "worst_violation_a": max(worst) if worst else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--low", type=float, default=1.0)
    parser.add_argument("--high", type=float, default=1.7)
    args = parser.parse_args()

    buckets: dict[str, list[dict]] = {}
    for run in sorted(glob.glob(str(args.root / "*/*/stride128_t32/*/seed_*"))):
        run_path = Path(run)
        protein = run_path.parents[2].name
        record = scan(run_path, args.low, args.high)
        if record:
            buckets.setdefault(protein, []).append(record)

    print(f"peptide C-N window [{args.low}, {args.high}] A\n")
    header = (
        f"{'protein':9s} {'res':>4s} {'cells':>6s} {'bonds':>9s} "
        f"{'bad':>6s} {'per-bond':>9s} {'per-path':>9s} {'worst':>7s}"
    )
    print(header)
    print("-" * len(header))
    for protein, records in buckets.items():
        bonds = sum(r["bonds_measured"] for r in records)
        bad = sum(r["bonds_violating"] for r in records)
        paths = sum(r["paths"] for r in records)
        invalid = sum(r["paths_invalid"] for r in records)
        worst = max((r["worst_violation_a"] or 0) for r in records)
        print(
            f"{protein:9s} {records[0]['residues']:4d} {len(records):6d} {bonds:9d} "
            f"{bad:6d} {bad / bonds:9.5f} {invalid / paths:9.3f} {worst:7.3f}"
        )
    (args.root / "gate_sensitivity.json").write_text(
        json.dumps(buckets, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
