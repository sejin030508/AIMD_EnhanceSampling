#!/usr/bin/env python
"""Record native-contact Q for saved populations, as a sidecar.

Q is written next to each run rather than into ``small_protein_metrics.json``
so completed results stay byte-identical and the same numbers can be produced
retroactively for runs that finished before this existed.

It matters because the headline success metric is a hit in the first two TICA
coordinates, and the Bundle A audit found frames that pass that while sitting
11.5 A from the target.  Q measures how much of the folded contact map is
actually present, so a run that scores zero hits still leaves evidence of how
close it came.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Heavy-atom slots in the atom37 layout; hydrogens are not represented there.
CA_SLOT = 1


def heavy_coordinates(atom37: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Per-residue heavy-atom coordinates actually present in a frame."""
    return [atom37[i][mask[i]] for i in range(len(atom37))]


def native_contacts(
    atom37: np.ndarray, mask: np.ndarray, cutoff_a: float, min_separation: int
) -> tuple[np.ndarray, np.ndarray]:
    """Residue pairs in contact in the reference, and their reference distance.

    Contact is the minimum heavy-atom distance between two residues, which is
    the usual definition for Q and is insensitive to side-chain length in a way
    a C-alpha cutoff is not.
    """
    residues = heavy_coordinates(atom37, mask)
    count = len(residues)
    pairs, distances = [], []
    for i in range(count):
        if not len(residues[i]):
            continue
        for j in range(i + min_separation, count):
            if not len(residues[j]):
                continue
            d = float(
                np.min(
                    np.linalg.norm(
                        residues[i][:, None, :] - residues[j][None, :, :], axis=-1
                    )
                )
            )
            if d < cutoff_a:
                pairs.append((i, j))
                distances.append(d)
    return np.asarray(pairs, dtype=int), np.asarray(distances, dtype=float)


def fraction_formed(
    atom37: np.ndarray,
    mask: np.ndarray,
    pairs: np.ndarray,
    reference_distances: np.ndarray,
    tolerance: float,
) -> float:
    if not len(pairs):
        return float("nan")
    residues = heavy_coordinates(atom37, mask)
    formed = 0
    for (i, j), reference in zip(pairs, reference_distances):
        if not len(residues[i]) or not len(residues[j]):
            continue
        d = float(
            np.min(
                np.linalg.norm(
                    residues[i][:, None, :] - residues[j][None, :, :], axis=-1
                )
            )
        )
        if d <= tolerance * reference:
            formed += 1
    return formed / len(pairs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--cutoff-a", type=float, default=4.5)
    parser.add_argument("--min-separation", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.run_dir / "native_contact_q.json"
    if output.exists() and not args.overwrite:
        print(f"skip {output}")
        return 0

    payload = np.load(args.prepared_dir / "whole_backbone_reference_atom37.npz")
    target = np.asarray(payload["reference_atom37_a"])[0]
    target_mask = np.asarray(payload["reference_atom37_mask"])[0]
    pairs, reference_distances = native_contacts(
        target, target_mask, args.cutoff_a, args.min_separation
    )

    populations = sorted(args.run_dir.glob("pre_final_population_atom37.npz"))
    if not populations:
        print(f"no population in {args.run_dir}")
        return 2
    with np.load(populations[0]) as data:
        key = "trajectories_atom37_a" if "trajectories_atom37_a" in data else data.files[0]
        trajectories = np.asarray(data[key])
        mask_key = "trajectories_atom37_mask"
        masks = (
            np.asarray(data[mask_key])
            if mask_key in data
            else np.broadcast_to(target_mask, trajectories.shape[:-1]).copy()
        )

    # trajectories: (paths, frames, residues, 37, 3)
    per_path = []
    for path_index in range(trajectories.shape[0]):
        series = [
            fraction_formed(
                trajectories[path_index, frame],
                masks[path_index, frame]
                if masks.ndim == 4
                else target_mask,
                pairs,
                reference_distances,
                args.tolerance,
            )
            for frame in range(trajectories.shape[1])
        ]
        per_path.append(
            {
                "path_index": path_index,
                "q_final": series[-1],
                "q_max": float(np.nanmax(series)),
                "q_argmax_frame": int(np.nanargmax(series)),
                "q_series": series,
            }
        )

    record = {
        "definition": (
            f"fraction of reference residue pairs with min heavy-atom distance "
            f"< {args.cutoff_a} A and |i-j| >= {args.min_separation}, counted as "
            f"formed when within {args.tolerance}x the reference distance"
        ),
        "native_contact_count": int(len(pairs)),
        "q_final_mean": float(np.nanmean([p["q_final"] for p in per_path])),
        "q_final_max": float(np.nanmax([p["q_final"] for p in per_path])),
        "q_max_over_path_mean": float(np.nanmean([p["q_max"] for p in per_path])),
        "per_path": per_path,
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.run_dir.name}: contacts={len(pairs)} "
        f"q_final_mean={record['q_final_mean']:.3f} "
        f"q_final_max={record['q_final_max']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
