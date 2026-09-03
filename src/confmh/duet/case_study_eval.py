from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from confmh.duet.observables import (
    atom_pair_distance_a,
    circular_difference_deg,
    pseudo_dihedral_deg,
)
from confmh.duet.reference_eval import dtw_distance


def _first_true(values: np.ndarray) -> int | None:
    hits = np.flatnonzero(np.asarray(values, dtype=bool))
    return int(hits[0]) if len(hits) else None


def _normalized_trace(values: np.ndarray, points: int = 33) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.stack([np.interp(target, source, values[:, dim]) for dim in range(values.shape[1])], axis=1)


def _order_probability(rows: Sequence[Mapping[str, int | None]], left: str, right: str) -> float | None:
    comparable = [row for row in rows if row[left] is not None and row[right] is not None]
    if not comparable:
        return None
    return float(np.mean([int(row[left]) < int(row[right]) for row in comparable]))


@dataclass
class MechanismBlindEvaluator:
    case_id: str
    spec: dict[str, Any]
    reference: dict[str, np.ndarray]

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        reward_observables: Sequence[str],
    ) -> "MechanismBlindEvaluator":
        root = Path(str(cfg["project_root"])).expanduser().resolve()
        raw_path = Path(str(cfg["case_study"]["benchmark_spec"])).expanduser()
        path = raw_path if raw_path.is_absolute() else root / raw_path
        with path.open("r", encoding="utf-8") as handle:
            spec = yaml.safe_load(handle)
        reward_features = set(spec["feature_firewall"]["reward_features"])
        hidden_features = set(spec["feature_firewall"]["hidden_evaluation_features"])
        overlap = reward_features & hidden_features
        if overlap:
            raise ValueError(f"Reward leakage in benchmark spec: {sorted(overlap)}")
        unexpected = set(reward_observables) - reward_features
        if unexpected:
            raise ValueError(
                "Program observables are not declared reward features: "
                f"{sorted(unexpected)}"
            )
        reference_path = Path(str(spec["reference_features_npz"])).expanduser()
        if not reference_path.is_absolute():
            reference_path = root / reference_path
        with np.load(reference_path) as payload:
            reference = {key: np.asarray(payload[key]) for key in payload.files}
        return cls(str(spec["case_id"]), spec, reference)

    def evaluate(
        self,
        paths: Sequence[Sequence[Any]],
        *,
        endpoint_success: Sequence[bool],
        valid_flags: Sequence[bool],
    ) -> dict[str, Any]:
        if self.case_id != "abl1_dfg_flip":
            return {
                "status": "unavailable",
                "reason": f"No mechanism evaluator implemented for {self.case_id}",
            }
        return self._evaluate_abl1(paths, endpoint_success, valid_flags)

    def _abl1_features(self, path: Sequence[Any]) -> dict[str, np.ndarray]:
        mapping = self.spec["mapping"]["generated_residue_indices"]
        ala = int(mapping["Ala380"])
        asp = int(mapping["Asp381"])
        phe = int(mapping["Phe382"])
        val = int(mapping["Val299"])
        lys = int(mapping["Lys271"])
        glu = int(mapping["Glu286"])
        asp_atoms = [[ala, "CB"], [ala, "CA"], [asp, "CA"], [asp, "CG"]]
        phe_atoms = [[ala, "CB"], [ala, "CA"], [phe, "CA"], [phe, "CG"]]
        return {
            "asp_angle_deg": np.asarray([pseudo_dihedral_deg(frame, asp_atoms) for frame in path]),
            "phe_angle_deg": np.asarray([pseudo_dihedral_deg(frame, phe_atoms) for frame in path]),
            "asp_val_distance_a": np.asarray(
                [atom_pair_distance_a(frame, asp, "OD2", val, "O") for frame in path]
            ),
            "lys_glu_distance_a": np.asarray(
                [atom_pair_distance_a(frame, lys, "NZ", glu, "CD") for frame in path]
            ),
        }

    def _events(self, features: Mapping[str, np.ndarray]) -> dict[str, int | None]:
        endpoint = self.spec["endpoint_definition"]
        hidden = self.spec["hidden_event_definitions"]
        asp_error = np.abs(
            circular_difference_deg(features["asp_angle_deg"], endpoint["asp_center_degrees"])
        )
        phe_error = np.abs(
            circular_difference_deg(features["phe_angle_deg"], endpoint["phe_center_degrees"])
        )
        return {
            "asp_complete": _first_true(asp_error <= float(hidden["asp_completion_tolerance_degrees"])),
            "phe_complete": _first_true(phe_error <= float(hidden["phe_completion_tolerance_degrees"])),
            "dfg_inter_contact": _first_true(
                features["asp_val_distance_a"] <= float(hidden["asp_val_contact_cutoff_a"])
            ),
        }

    def _route_trace(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        endpoint = self.spec["endpoint_definition"]
        asp = np.unwrap(np.deg2rad(features["asp_angle_deg"]))
        phe = np.unwrap(np.deg2rad(features["phe_angle_deg"]))
        scaled = np.column_stack(
            [
                np.rad2deg(asp - asp[0]) / float(endpoint["asp_scale_degrees"]),
                np.rad2deg(phe - phe[0]) / float(endpoint["phe_scale_degrees"]),
                features["asp_val_distance_a"]
                / float(self.spec["hidden_event_definitions"]["asp_val_contact_cutoff_a"]),
                features["lys_glu_distance_a"]
                / float(self.spec["hidden_event_definitions"]["salt_bridge_reference_scale_a"]),
            ]
        )
        return _normalized_trace(scaled)

    def _evaluate_abl1(
        self,
        paths: Sequence[Sequence[Any]],
        endpoint_success: Sequence[bool],
        valid_flags: Sequence[bool],
    ) -> dict[str, Any]:
        selected = [
            index
            for index, (success, valid) in enumerate(zip(endpoint_success, valid_flags))
            if bool(success) and bool(valid)
        ]
        if not selected:
            return {
                "status": "suppressed",
                "reason": "no valid endpoint-successful trajectories",
                "evaluated_path_count": 0,
            }
        reference_names = ("concerted", "staggered")
        reference_traces = {
            name: np.asarray(self.reference[f"{name}_route_trace"], dtype=float)
            for name in reference_names
        }
        reference_events = [
            {
                key: (None if int(value) < 0 else int(value))
                for key, value in zip(
                    ("asp_complete", "phe_complete", "dfg_inter_contact"),
                    self.reference[f"{name}_event_frames"],
                )
            }
            for name in reference_names
        ]
        rows = []
        generated_events = []
        for index in selected:
            features = self._abl1_features(paths[index])
            events = self._events(features)
            trace = self._route_trace(features)
            distances = {
                name: dtw_distance(trace, reference_traces[name]) for name in reference_names
            }
            route = min(distances, key=distances.get)
            generated_events.append(events)
            rows.append(
                {
                    "particle_index": index,
                    "route": route,
                    "route_dtw": distances,
                    "event_frames": events,
                    "dfg_inter_visited": events["dfg_inter_contact"] is not None,
                    "lys_glu_distance_a": features["lys_glu_distance_a"].tolist(),
                    "asp_angle_deg": features["asp_angle_deg"].tolist(),
                    "phe_angle_deg": features["phe_angle_deg"].tolist(),
                    "asp_val_distance_a": features["asp_val_distance_a"].tolist(),
                }
            )
        event_names = ("asp_complete", "phe_complete", "dfg_inter_contact")
        order_rows = []
        errors = []
        for left_index, left in enumerate(event_names):
            for right in event_names[left_index + 1 :]:
                reference_probability = _order_probability(reference_events, left, right)
                generated_probability = _order_probability(generated_events, left, right)
                if reference_probability is not None and generated_probability is not None:
                    errors.append(abs(reference_probability - generated_probability))
                order_rows.append(
                    {
                        "left": left,
                        "right": right,
                        "reference_probability": reference_probability,
                        "generated_probability": generated_probability,
                    }
                )
        route_counts = {name: sum(row["route"] == name for row in rows) for name in reference_names}
        return {
            "status": "ok",
            "evaluated_path_count": len(rows),
            "reference_path_count": 2,
            "reference_weighting": "unweighted descriptive; no independent WE replicate split",
            "terminal_particle_weighting": "equal after final resampling",
            "dfg_inter_visit_rate": float(np.mean([row["dfg_inter_visited"] for row in rows])),
            "intermediate_recovery_rate": float(
                np.mean([row["dfg_inter_visited"] for row in rows])
            ),
            "mechanism_order_error": float(np.mean(errors)) if errors else None,
            "order_matrix": order_rows,
            "route_counts": route_counts,
            "route_fraction": {
                name: float(route_counts[name] / len(rows)) for name in reference_names
            },
            "route_jsd_to_reference": None,
            "route_jsd_suppression_reason": (
                "Only two selected WT reference paths are public in the archive; "
                "route fractions are descriptive, not a population estimate."
            ),
            "per_path": rows,
        }
