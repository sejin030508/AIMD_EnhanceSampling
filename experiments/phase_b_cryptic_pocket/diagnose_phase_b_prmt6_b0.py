#!/usr/bin/env python3
"""Geometry diagnosis for immutable PRMT6 B0 outputs.

This script is deliberately read-only with respect to the run directory.  It
creates a separately named diagnosis JSON/Markdown file so an existing strict
B0 failure cannot be overwritten or relabelled.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


CA_INDEX = 1  # AlphaFold/OpenFold atom37 CA index.


def _load_mapping(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["model_index_0based"]): row for row in rows}


def _distance(coords: np.ndarray, mask: np.ndarray, left: int, right: int) -> float | None:
    if not bool(mask[left, CA_INDEX]) or not bool(mask[right, CA_INDEX]):
        return None
    value = float(np.linalg.norm(coords[right, CA_INDEX] - coords[left, CA_INDEX]))
    return value if np.isfinite(value) else None


def _max_pair(coords: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    ca = np.asarray(coords[:, CA_INDEX, :], dtype=float)
    present = np.asarray(mask[:, CA_INDEX], dtype=bool)
    values = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
    values[~(present[1:] & present[:-1])] = np.nan
    if not np.any(np.isfinite(values)):
        raise RuntimeError("No finite adjacent CA pair is available")
    index = int(np.nanargmax(values))
    return index, float(values[index])


def _pair_label(mapping: dict[int, dict[str, Any]], left: int) -> dict[str, Any]:
    indices = (left, left + 1)
    rows = [mapping.get(index) for index in indices]
    return {
        "model_indices_0based": list(indices),
        "author_uniprot_residues": [int(row["uniprot_residue"]) if row else None for row in rows],
        "prepared_pdb_residues": [int(row["prepared_pdb_residue"]) if row else None for row in rows],
        "start_resnames": [row["start_resname"] if row else None for row in rows],
        "holo_resnames": [row["6w6d_resname"] if row else None for row in rows],
    }


def _artifact_check(
    pair: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    residues = pair["author_uniprot_residues"]
    missing_start = set(manifest.get("missing_start_backbone_uniprot", []))
    missing_target = {
        name: set(values)
        for name, values in manifest.get("missing_target_backbone_uniprot", {}).items()
    }
    missing_target_here = {
        name: sorted(set(residues).intersection(values))
        for name, values in missing_target.items()
        if set(residues).intersection(values)
    }
    mapping_complete = all(value is not None for value in residues)
    author_consecutive = mapping_complete and residues[1] == residues[0] + 1
    model_consecutive = pair["model_indices_0based"][1] == pair["model_indices_0based"][0] + 1
    near_chain_boundary = any(
        index in {0, int(manifest["model_length"]) - 1} for index in pair["model_indices_0based"]
    )
    indicators = {
        "mapping_complete": mapping_complete,
        "author_residues_consecutive": author_consecutive,
        "model_indices_consecutive": model_consecutive,
        "near_construct_boundary": near_chain_boundary,
        "missing_start_backbone_at_pair": sorted(set(residues).intersection(missing_start)),
        "missing_target_backbone_at_pair": missing_target_here,
    }
    artifact = (
        not mapping_complete
        or not author_consecutive
        or not model_consecutive
        or near_chain_boundary
        or bool(indicators["missing_start_backbone_at_pair"])
        or bool(missing_target_here)
    )
    return {"preprocessing_or_mapping_artifact_suspected": artifact, **indicators}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-readiness", action="store_true")
    args = parser.parse_args()

    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mapping = _load_mapping(args.mapping)
    with np.load(args.run_dir / "pre_final_population_atom37.npz") as payload:
        trajectories = np.asarray(payload["trajectories_atom37_a"])
        masks = np.asarray(payload["trajectories_atom37_mask"])
    with np.load(args.reference_npz, allow_pickle=True) as payload:
        start = np.asarray(payload["start_atom37_a"])
        start_mask = np.asarray(payload["start_atom37_mask"])
        names = [str(item) for item in payload["reference_names"].tolist()]
        holo = np.asarray(payload["reference_atom37_a"])
        holo_mask = np.asarray(payload["reference_atom37_mask"])

    validity = metrics["path_validity"]
    if trajectories.shape[:2] != (len(validity), len(validity[0])):
        raise RuntimeError("Saved population and validity records have incompatible shapes")
    path_maxima: list[float] = []
    invalid_events: list[dict[str, Any]] = []
    all_rows = [row for rows in validity for row in rows]
    for path_index, rows in enumerate(validity):
        maxima = [float(row["ca_adjacent_max_a"]) for row in rows]
        path_maxima.append(max(maxima))
        for frame_index, row in enumerate(rows):
            if bool(row["valid"]):
                continue
            left, observed = _max_pair(trajectories[path_index, frame_index], masks[path_index, frame_index])
            pair = _pair_label(mapping, left)
            previous = (
                _distance(trajectories[path_index, frame_index - 1], masks[path_index, frame_index - 1], left, left + 1)
                if frame_index > 0
                else None
            )
            pair["current_ca_distance_a"] = observed
            pair["previous_frame_ca_distance_a"] = previous
            pair["start_ca_distance_a"] = _distance(start, start_mask, left, left + 1)
            pair["holo_ca_distance_a"] = {
                name: _distance(holo[index], holo_mask[index], left, left + 1)
                for index, name in enumerate(names)
            }
            pair["mapping_artifact_check"] = _artifact_check(pair, manifest)
            invalid_events.append(
                {"path_index": path_index, "frame_index_including_start": frame_index, **pair}
            )

    pair_counts = Counter(
        tuple(event["model_indices_0based"]) for event in invalid_events
    )
    path_valid = [all(bool(row["valid"]) for row in rows) for rows in validity]
    nonfinite_rows = sum(int(row.get("nonfinite_coordinate_count", 0)) > 0 for row in all_rows)
    clash_rows = sum(int(row.get("ca_clash_count_lt_1a", 0)) > 0 for row in all_rows)
    generated_rows = [row for rows in validity for row in rows[1:]]
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_dir": str(args.run_dir),
        "run_identity": {
            key: metrics.get(key)
            for key in ("protein", "stage", "method", "seed", "outer_k", "inner_m", "horizon")
        },
        "valid_path_count": int(sum(path_valid)),
        "population_path_count": int(len(path_valid)),
        "valid_path_fraction": float(np.mean(path_valid)),
        "invalid_frame_count_including_start": int(len(invalid_events)),
        "frame_count_including_start": int(len(all_rows)),
        "invalid_frame_fraction_including_start": float(len(invalid_events) / len(all_rows)),
        "invalid_generated_frame_count": int(sum(not bool(row["valid"]) for row in generated_rows)),
        "generated_frame_count": int(len(generated_rows)),
        "invalid_generated_frame_fraction": float(
            sum(not bool(row["valid"]) for row in generated_rows) / len(generated_rows)
        ),
        "per_path_max_adjacent_ca_a": path_maxima,
        "invalid_residue_pair_frequency": [
            {"model_indices_0based": list(pair), "invalid_frame_count": count}
            for pair, count in sorted(pair_counts.items())
        ],
        "nan_frame_count": nonfinite_rows,
        "nan_frame_fraction": float(nonfinite_rows / len(all_rows)),
        "ca_clash_frame_count_lt_1a": clash_rows,
        "ca_clash_frame_fraction_lt_1a": float(clash_rows / len(all_rows)),
        "invalid_events": invalid_events,
    }
    if args.pilot_readiness:
        if len(path_valid) != 16:
            raise RuntimeError("Exploratory PRMT6 B0 must contain exactly 16 paths")
        repeated = [
            {"model_indices_0based": list(pair), "invalid_frame_count": count}
            for pair, count in sorted(pair_counts.items()) if count >= 2
        ]
        artifact = any(
            event["mapping_artifact_check"]["preprocessing_or_mapping_artifact_suspected"]
            for event in invalid_events
        )
        if nonfinite_rows or clash_rows or artifact:
            decision, reason = "hard_stop", "nonfinite_or_clash_or_preprocessing_artifact"
        elif repeated:
            decision, reason = "hold_b1", "repeated_same_site_break"
        elif sum(path_valid) >= 15:
            decision, reason = "proceed", "at_least_15_of_16_valid"
        elif sum(path_valid) >= 13:
            decision, reason = "exploratory_proceed", "13_to_14_of_16_valid"
        else:
            decision, reason = "hold_b1", "12_or_fewer_of_16_valid"
        report["exploratory_pilot_readiness"] = {
            "decision": decision,
            "reason": reason,
            "repeated_same_site_break_definition": "same exact adjacent model-index pair invalid in >=2 frames",
            "repeated_same_site_breaks": repeated,
            "strict_b0_result_preserved_as_fail": True,
        }
    _json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
