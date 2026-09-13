#!/usr/bin/env python
"""Locate which geometry criterion fails, and when, along a trajectory.

Whole-path validity collapses to a single boolean, so a run that fails says
nothing about whether the break is gradual drift or one bad frame, nor which of
the three checks caught it.  This reports each criterion per frame so the
failure mode can be named rather than guessed at.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CA, N, C = 1, 0, 2


def per_frame(atom37: np.ndarray) -> dict:
    ca = atom37[:, CA, :]
    adjacent = np.linalg.norm(np.diff(ca, axis=0), axis=-1)
    peptide = np.linalg.norm(atom37[:-1, C, :] - atom37[1:, N, :], axis=-1)
    pair = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    nonneighbor = np.triu(np.ones(pair.shape, dtype=bool), k=2)
    return {
        "ca_adjacent_max": float(adjacent.max()),
        "ca_adjacent_over_5p5": int((adjacent > 5.501).sum()),
        "peptide_cn_min": float(peptide.min()),
        "peptide_cn_max": float(peptide.max()),
        "peptide_violations": int(((peptide < 1.0) | (peptide > 1.7)).sum()),
        "ca_clash_lt_1": int(((pair < 1.0) & nonneighbor).sum()),
        "radius_of_gyration": float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=3)
    args = parser.parse_args()

    with np.load(args.run_dir / "pre_final_population_atom37.npz") as data:
        key = "trajectories_atom37_a" if "trajectories_atom37_a" in data else data.files[0]
        trajectories = np.asarray(data[key])

    print(f"population {trajectories.shape[0]} paths x {trajectories.shape[1]} frames")
    summary = []
    for path in range(min(args.paths, trajectories.shape[0])):
        series = [per_frame(trajectories[path, f]) for f in range(trajectories.shape[1])]
        first_bad = next(
            (
                i
                for i, s in enumerate(series)
                if s["peptide_violations"] or s["ca_adjacent_over_5p5"] or s["ca_clash_lt_1"]
            ),
            None,
        )
        print(f"\npath {path}: first invalid frame = {first_bad}")
        print(
            f"  {'frame':>5s} {'pepMin':>7s} {'pepMax':>7s} {'pepBad':>6s} "
            f"{'caMax':>6s} {'ca>5.5':>6s} {'clash':>5s} {'Rg':>6s}"
        )
        for i in list(range(0, len(series), max(1, len(series) // 8))) + [len(series) - 1]:
            s = series[i]
            print(
                f"  {i:5d} {s['peptide_cn_min']:7.3f} {s['peptide_cn_max']:7.3f} "
                f"{s['peptide_violations']:6d} {s['ca_adjacent_max']:6.2f} "
                f"{s['ca_adjacent_over_5p5']:6d} {s['ca_clash_lt_1']:5d} "
                f"{s['radius_of_gyration']:6.2f}"
            )
        summary.append({"path": path, "first_invalid_frame": first_bad, "series": series})
    (args.run_dir / "validity_diagnosis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
