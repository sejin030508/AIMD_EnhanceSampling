#!/usr/bin/env python3
"""Evaluate the fixed PVB/BBA guidance matrix from saved weighted populations."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from confmh.duet.observables import ATOM37_INDEX
from confmh.duet.tica_potential import TicaEndpointMetric
from confmh.pca_cv import _kabsch_align


BACKBONE = [ATOM37_INDEX[name] for name in ("N", "CA", "C")]
CHIRAL = [ATOM37_INDEX[name] for name in ("N", "CA", "C", "CB")]
CHIRALITY_TOLERANCE_A3 = 0.5


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _aligned_backbone_rmsd_a(
    coordinates: np.ndarray,
    mask: np.ndarray,
    reference: np.ndarray,
    reference_mask: np.ndarray,
) -> float:
    common = mask[:, BACKBONE] & reference_mask[:, BACKBONE]
    moving = coordinates[:, BACKBONE][common]
    target = reference[:, BACKBONE][common]
    if len(moving) < 3:
        return float("nan")
    aligned = _kabsch_align(moving, target)
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=-1))))


def _heavy_native_contacts(
    reference: np.ndarray,
    mask: np.ndarray,
    cutoff_a: float = 4.5,
    min_separation: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    pairs: list[tuple[int, int]] = []
    distances: list[float] = []
    for left in range(len(reference)):
        left_xyz = reference[left][mask[left]]
        if not len(left_xyz):
            continue
        for right in range(left + min_separation, len(reference)):
            right_xyz = reference[right][mask[right]]
            if not len(right_xyz):
                continue
            distance = float(
                np.min(
                    np.linalg.norm(
                        left_xyz[:, None, :] - right_xyz[None, :, :], axis=-1
                    )
                )
            )
            if distance < cutoff_a:
                pairs.append((left, right))
                distances.append(distance)
    return np.asarray(pairs, dtype=int), np.asarray(distances, dtype=float)


def _native_contact_q(
    coordinates: np.ndarray,
    mask: np.ndarray,
    pairs: np.ndarray,
    reference_distances: np.ndarray,
    tolerance: float = 1.2,
) -> float:
    formed = 0
    for (left, right), reference in zip(pairs, reference_distances):
        left_xyz = coordinates[left][mask[left]]
        right_xyz = coordinates[right][mask[right]]
        if not len(left_xyz) or not len(right_xyz):
            continue
        distance = float(
            np.min(
                np.linalg.norm(
                    left_xyz[:, None, :] - right_xyz[None, :, :], axis=-1
                )
            )
        )
        formed += distance <= tolerance * reference
    return float(formed / len(pairs)) if len(pairs) else float("nan")


def _signed_ca_volume(coordinates: np.ndarray) -> np.ndarray:
    n, ca, c, cb = [coordinates[:, slot] for slot in CHIRAL]
    return np.einsum("ij,ij->i", n - ca, np.cross(c - ca, cb - ca))


def _chirality_inversions(
    coordinates: np.ndarray,
    mask: np.ndarray,
    reference_sign: np.ndarray,
    reference_evaluable: np.ndarray,
) -> int:
    present = np.all(mask[:, CHIRAL], axis=1)
    volumes = _signed_ca_volume(coordinates)
    evaluable = (
        present
        & reference_evaluable
        & np.isfinite(volumes)
        & (np.abs(volumes) >= CHIRALITY_TOLERANCE_A3)
    )
    signs = np.sign(volumes)
    return int(np.sum(evaluable & (signs != reference_sign)))


def _summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def evaluate_run(
    run_dir: Path,
    *,
    metric: TicaEndpointMetric,
    target: np.ndarray,
    target_mask: np.ndarray,
    contact_pairs: np.ndarray,
    contact_distances: np.ndarray,
    reference_sign: np.ndarray,
    reference_evaluable: np.ndarray,
    cutoff: float,
) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    weights = np.asarray(
        _load_json(run_dir / "pre_final_normalized_weights.json"), dtype=float
    )
    weights /= np.sum(weights)
    with np.load(run_dir / "pre_final_population_atom37.npz") as payload:
        trajectories = np.asarray(payload["trajectories_atom37_a"], dtype=float)
        masks = np.asarray(payload["trajectories_atom37_mask"], dtype=bool)
    if trajectories.shape[0] != len(weights):
        raise ValueError(f"population/weight mismatch in {run_dir}")

    valid = np.asarray(metrics["per_path_valid"], dtype=bool)
    d0 = float(metrics["reward_d0"])
    rows = []
    for index, (trajectory, trajectory_mask) in enumerate(zip(trajectories, masks)):
        final = trajectory[-1]
        final_mask = trajectory_mask[-1]
        tica_distance = float(
            np.linalg.norm(metric.project_atom37(final, final_mask) - metric.target)
        )
        chirality_by_frame = [
            _chirality_inversions(
                frame, frame_mask, reference_sign, reference_evaluable
            )
            for frame, frame_mask in zip(trajectory, trajectory_mask)
        ]
        hit = bool(tica_distance < cutoff)
        rows.append(
            {
                "path_index": index,
                "weight": float(weights[index]),
                "whole_path_valid": bool(valid[index]),
                "final_tica_distance": tica_distance,
                "final_tica_d_over_d0": tica_distance / d0,
                "final_tica_hit": hit,
                "valid_final_tica_hit": bool(valid[index] and hit),
                "final_backbone_rmsd_a": _aligned_backbone_rmsd_a(
                    final, final_mask, target, target_mask
                ),
                "final_native_contact_q": _native_contact_q(
                    final,
                    final_mask,
                    contact_pairs,
                    contact_distances,
                ),
                "final_chirality_inversion_count": chirality_by_frame[-1],
                "whole_path_has_chirality_inversion": bool(
                    any(count > 0 for count in chirality_by_frame)
                ),
                "max_frame_chirality_inversion_count": int(
                    max(chirality_by_frame, default=0)
                ),
            }
        )

    valid_hit = np.asarray([row["valid_final_tica_hit"] for row in rows], dtype=bool)
    tica_ratio = np.asarray([row["final_tica_d_over_d0"] for row in rows])
    backbone = np.asarray([row["final_backbone_rmsd_a"] for row in rows])
    q_values = np.asarray([row["final_native_contact_q"] for row in rows])
    inverted = np.asarray(
        [row["whole_path_has_chirality_inversion"] for row in rows], dtype=bool
    )

    records = _load_json(run_dir / "records.json")
    proposal_ratios: list[float] = []
    selected_ratios: list[float] = []
    checkpoint_ess: defaultdict[int, list[float]] = defaultdict(list)
    endpoint_ess: list[float] = []
    for record in records:
        proposal = record.get("guided_path_log_proposal_ratios")
        if proposal is not None:
            proposal_ratios.extend(map(float, proposal))
        selected = record.get("guided_selected_path_log_proposal_ratio")
        if selected is not None:
            selected_ratios.append(float(selected))
        for checkpoint, value in enumerate(record.get("inner_checkpoint_ess") or []):
            checkpoint_ess[checkpoint].append(float(value))
        endpoint = record.get("guided_endpoint_ess")
        if endpoint is not None:
            endpoint_ess.append(float(endpoint))

    output = {
        "condition": str(metrics["stage"]).rsplit("_", 1)[-1].upper(),
        "method": metrics["method"],
        "seed": int(metrics["seed"]),
        "population_size": len(rows),
        "tica_hit_cutoff": float(cutoff),
        "valid_final_tica_hit_count": int(np.sum(valid_hit)),
        "valid_final_tica_hit_fraction": float(np.mean(valid_hit)),
        "weighted_valid_final_tica_hit_mass": float(np.dot(weights, valid_hit)),
        "weighted_final_tica_d_over_d0": float(np.dot(weights, tica_ratio)),
        "weighted_final_backbone_rmsd_a": float(np.dot(weights, backbone)),
        "weighted_final_native_contact_q": float(np.dot(weights, q_values)),
        "whole_path_qc_pass_fraction": float(np.mean(valid)),
        "weighted_whole_path_qc_pass_mass": float(np.dot(weights, valid)),
        "whole_path_chirality_inversion_fraction": float(np.mean(inverted)),
        "weighted_chirality_clean_path_mass": float(np.dot(weights, ~inverted)),
        "final_outer_ess": float(1.0 / np.sum(weights**2)),
        "inner_checkpoint_ess": {
            str(index + 1): _summary(values)
            for index, values in sorted(checkpoint_ess.items())
        },
        "guided_endpoint_ess": _summary(endpoint_ess),
        "guided_path_log_proposal_ratio": _summary(proposal_ratios),
        "guided_selected_log_proposal_ratio": _summary(selected_ratios),
        "guidance_shift_magnitude_available": False,
        "guidance_shift_magnitude_note": (
            "The current sampler records exact proposal log-ratios but not "
            "the Cartesian mean-shift norm."
        ),
        "wall_clock_s": float(metrics["wall_clock_s"]),
        "decoder_nfe": int(metrics["decoder_nfe"]),
        "peak_gpu_memory_bytes": metrics["peak_gpu_memory_bytes"],
        "accounting": metrics["accounting"],
        "per_path": rows,
    }
    (run_dir / "guidance_tuning_evaluation.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=0.75)
    args = parser.parse_args()

    manifest = _load_json(args.prepared_dir / "manifest.json")
    metric = TicaEndpointMetric(manifest, dimensions=2)
    with np.load(args.prepared_dir / "whole_backbone_reference_atom37.npz") as ref:
        target = np.asarray(ref["reference_atom37_a"])[0]
        target_mask = np.asarray(ref["reference_atom37_mask"])[0]
    pairs, distances = _heavy_native_contacts(target, target_mask)
    reference_volume = _signed_ca_volume(target)
    reference_present = np.all(target_mask[:, CHIRAL], axis=1)
    reference_evaluable = (
        reference_present
        & np.isfinite(reference_volume)
        & (np.abs(reference_volume) >= CHIRALITY_TOLERANCE_A3)
    )
    reference_sign = np.sign(reference_volume)

    metrics_paths = sorted(args.output_root.glob("bba/*/*/seed_*/metrics.json"))
    results = [
        evaluate_run(
            path.parent,
            metric=metric,
            target=target,
            target_mask=target_mask,
            contact_pairs=pairs,
            contact_distances=distances,
            reference_sign=reference_sign,
            reference_evaluable=reference_evaluable,
            cutoff=args.cutoff,
        )
        for path in metrics_paths
    ]

    scalar_fields = [
        "valid_final_tica_hit_fraction",
        "weighted_valid_final_tica_hit_mass",
        "weighted_final_tica_d_over_d0",
        "weighted_final_backbone_rmsd_a",
        "weighted_final_native_contact_q",
        "whole_path_qc_pass_fraction",
        "whole_path_chirality_inversion_fraction",
        "final_outer_ess",
        "wall_clock_s",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["condition"]].append(result)
    aggregate = {
        condition: {
            "completed_runs": len(rows),
            **{
                field: _summary([float(row[field]) for row in rows])
                for field in scalar_fields
            },
        }
        for condition, rows in sorted(grouped.items())
    }
    summary = {
        "schema_version": 1,
        "completed_runs": len(results),
        "expected_runs": 21,
        "native_contact_count": int(len(pairs)),
        "chirality_tolerance_a3": CHIRALITY_TOLERANCE_A3,
        "condition_aggregate": aggregate,
        "runs": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "guidance_tuning_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = args.output_root / "guidance_tuning_runs.csv"
    fields = [
        "condition",
        "method",
        "seed",
        "population_size",
        "valid_final_tica_hit_count",
        *scalar_fields,
        "decoder_nfe",
        "peak_gpu_memory_bytes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"completed_runs": len(results), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()

