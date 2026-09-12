from __future__ import annotations

"""Fixed endpoint reward and evaluation helpers for the Phase-B pocket pilot.

This module is intentionally independent of the ordered-event benchmark.  The
only sampling potential is the fixed, core-aligned loop RMSD specified in the
Phase-B protocol.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from confmh.duet.observables import ATOM37_INDEX, atom37_coordinates_a
from confmh.duet.programs import ProgressState


BACKBONE_ATOMS = ("N", "CA", "C")


@dataclass(frozen=True)
class PocketReference:
    name: str
    atom37_a: np.ndarray
    atom37_mask: np.ndarray

    def __post_init__(self) -> None:
        coords = np.asarray(self.atom37_a)
        mask = np.asarray(self.atom37_mask)
        if coords.ndim != 3 or coords.shape[1:] != (37, 3):
            raise ValueError(f"{self.name}: expected atom37 coordinates, got {coords.shape}")
        if mask.shape != coords.shape[:2]:
            raise ValueError(f"{self.name}: atom mask shape mismatch")


def _atom_selection(indices: Sequence[int], atom_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    residues = np.repeat(np.asarray(indices, dtype=int), len(atom_names))
    atoms = np.tile(np.asarray([ATOM37_INDEX[name] for name in atom_names], dtype=int), len(indices))
    return residues, atoms


def _fit_transform(moving: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moving = np.asarray(moving, dtype=float)
    target = np.asarray(target, dtype=float)
    if moving.shape != target.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError(f"Alignment shape mismatch: {moving.shape} != {target.shape}")
    if len(moving) < 3:
        raise ValueError("At least three atoms are required for rigid alignment")
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (moving - moving_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.linalg.det(left @ right_t)])
    rotation = left @ correction @ right_t
    return rotation, moving_center, target_center


def _apply_transform(
    coordinates: np.ndarray,
    rotation: np.ndarray,
    moving_center: np.ndarray,
    target_center: np.ndarray,
) -> np.ndarray:
    return (np.asarray(coordinates, dtype=float) - moving_center) @ rotation + target_center


class PocketEndpointMetric:
    """Core-aligned N/CA/C loop RMSD, minimized over fixed holo references."""

    def __init__(
        self,
        *,
        references: Sequence[PocketReference],
        core_residue_indices: Sequence[int],
        loop_residue_indices: Sequence[int],
    ) -> None:
        if not references:
            raise ValueError("At least one holo reference is required")
        self.references = tuple(references)
        self.core_residue_indices = tuple(int(item) for item in core_residue_indices)
        self.loop_residue_indices = tuple(int(item) for item in loop_residue_indices)
        if not self.core_residue_indices or not self.loop_residue_indices:
            raise ValueError("Core and loop residue masks must be non-empty")
        lengths = {len(reference.atom37_a) for reference in self.references}
        if len(lengths) != 1:
            raise ValueError("All references must use the same model-index coordinate space")
        length = lengths.pop()
        selected = self.core_residue_indices + self.loop_residue_indices
        if min(selected) < 0 or max(selected) >= length:
            raise IndexError("Core/loop mask lies outside the reference coordinate space")
        core_r, core_a = _atom_selection(self.core_residue_indices, BACKBONE_ATOMS)
        loop_r, loop_a = _atom_selection(self.loop_residue_indices, BACKBONE_ATOMS)
        for reference in self.references:
            mask = np.asarray(reference.atom37_mask, dtype=bool)
            if not np.all(mask[core_r, core_a]):
                raise ValueError(f"{reference.name}: missing N/CA/C atom in fixed core")
            if not np.all(mask[loop_r, loop_a]):
                raise ValueError(f"{reference.name}: missing N/CA/C atom in reward loop")

    def _aligned_to_reference(
        self, frame: Any, reference: PocketReference
    ) -> tuple[np.ndarray, np.ndarray]:
        coords, mask = atom37_coordinates_a(frame)
        if coords.shape != reference.atom37_a.shape:
            raise ValueError(
                f"Generated/reference atom37 mismatch: {coords.shape} != {reference.atom37_a.shape}"
            )
        core_r, core_a = _atom_selection(self.core_residue_indices, BACKBONE_ATOMS)
        if not np.all(mask[core_r, core_a]):
            raise ValueError("Generated frame is missing N/CA/C atom in fixed core")
        moving_fit = coords[core_r, core_a]
        target_fit = np.asarray(reference.atom37_a, dtype=float)[core_r, core_a]
        transform = _fit_transform(moving_fit, target_fit)
        return _apply_transform(coords, *transform), mask

    def distances_a(self, frame: Any) -> dict[str, float]:
        loop_r, loop_a = _atom_selection(self.loop_residue_indices, BACKBONE_ATOMS)
        values: dict[str, float] = {}
        for reference in self.references:
            aligned, mask = self._aligned_to_reference(frame, reference)
            if not np.all(mask[loop_r, loop_a]):
                raise ValueError("Generated frame is missing N/CA/C atom in reward loop")
            residual = aligned[loop_r, loop_a] - np.asarray(reference.atom37_a)[loop_r, loop_a]
            values[reference.name] = float(np.sqrt(np.mean(np.sum(residual * residual, axis=-1))))
        return values

    def distance_a(self, frame: Any) -> float:
        return min(self.distances_a(frame).values())

    def core_rmsd_a(self, frame: Any, reference: PocketReference) -> float:
        aligned, mask = self._aligned_to_reference(frame, reference)
        residues, atoms = _atom_selection(self.core_residue_indices, BACKBONE_ATOMS)
        if not np.all(mask[residues, atoms]):
            raise ValueError("Generated frame is missing core backbone atoms")
        residual = aligned[residues, atoms] - np.asarray(reference.atom37_a)[residues, atoms]
        return float(np.sqrt(np.mean(np.sum(residual * residual, axis=-1))))


class PocketEndpointPotential:
    """Core-aligned endpoint potential with an optional numerical log floor."""

    def __init__(
        self,
        metric: PocketEndpointMetric,
        d0_a: float,
        *,
        coefficient: float = 4.0,
        log_floor: float | None = -30.0,
    ) -> None:
        self.metric = metric
        self.d0_a = float(d0_a)
        self.coefficient = float(coefficient)
        self.log_floor = None if log_floor is None else float(log_floor)
        if self.d0_a <= 0.0:
            raise ValueError("d0 must be positive")
        if self.coefficient <= 0.0:
            raise ValueError("Invalid fixed-potential coefficients")
        if self.log_floor is not None and self.log_floor >= 0.0:
            raise ValueError("log_floor must be negative or None")
        self.evaluations = 0
        self.clipped_evaluations = 0

    def values(self, frame: Any) -> dict[str, float]:
        distance = self.metric.distance_a(frame)
        ratio = distance / self.d0_a
        raw = -self.coefficient * ratio * ratio
        clipped = self.log_floor is not None and raw < self.log_floor
        return {
            "endpoint_distance_a": float(distance),
            "endpoint_distance_over_d0": float(ratio),
            "potential_was_clipped": float(clipped),
        }

    def log_psi(
        self,
        history: Sequence[Any],
        state: ProgressState,
        t: int,
        values: Mapping[str, float] | None = None,
    ) -> float:
        del state, t
        self.evaluations += 1
        if values is None:
            values = self.values(history[-1])
        raw = -self.coefficient * float(values["endpoint_distance_over_d0"]) ** 2
        if self.log_floor is not None and raw < self.log_floor:
            self.clipped_evaluations += 1
            return float(self.log_floor)
        return float(raw)

    def candidate_log_psi(
        self,
        parent_history: Sequence[Any],
        parent_state: ProgressState,
        frame: Any,
        t: int,
    ) -> tuple[float, ProgressState, dict[str, float]]:
        values = self.values(frame)
        history = list(parent_history) + [frame]
        return self.log_psi(history, parent_state, t, values), parent_state, values


def _available_points(
    frame: Any, residue_index: int, atom_names: Sequence[str], *, require_all: bool = True
) -> np.ndarray | None:
    coords, mask = atom37_coordinates_a(frame)
    atoms = [ATOM37_INDEX[name] for name in atom_names]
    available = mask[int(residue_index), atoms]
    if (require_all and not np.all(available)) or not np.any(available):
        return None
    points = coords[int(residue_index), np.asarray(atoms)[available]]
    return points if np.all(np.isfinite(points)) else None


def _minimum_distance(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    return float(np.min(np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)))


def _centroid_distance(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    return float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))


def ligand_space_observables(
    frame: Any,
    *,
    metric: PocketEndpointMetric,
    ligand_heavy_atom_coordinates_a: np.ndarray,
) -> dict[str, float]:
    """Descriptive, evaluation-only occupancy of the holo ligand volume.

    The protein is aligned to the first holo reference with the same fixed core
    used by the reward.  We retain raw nearest-distance summaries and counts at
    several declared cutoffs; none of these cutoffs changes sampling or defines
    a binary opening success criterion.
    """
    ligand = np.asarray(ligand_heavy_atom_coordinates_a, dtype=float)
    if ligand.ndim != 2 or ligand.shape[1] != 3 or len(ligand) == 0:
        raise ValueError("Expected non-empty holo ligand heavy-atom coordinates")
    aligned, mask = metric._aligned_to_reference(frame, metric.references[0])
    protein = np.asarray(aligned, dtype=float)[np.asarray(mask, dtype=bool)]
    if len(protein) == 0 or not np.all(np.isfinite(protein)):
        raise ValueError("Generated protein has no finite heavy-atom coordinates")
    pair_distances = np.linalg.norm(
        ligand[:, None, :] - protein[None, :, :], axis=-1
    )
    nearest = np.min(pair_distances, axis=1)
    result: dict[str, float] = {
        "ligand_space_min_heavy_distance_a": float(np.min(nearest)),
        "ligand_space_mean_nearest_heavy_distance_a": float(np.mean(nearest)),
        "ligand_space_median_nearest_heavy_distance_a": float(np.median(nearest)),
    }
    for cutoff in (1.5, 2.0, 2.5, 3.0, 4.0):
        label = str(cutoff).replace(".", "p")
        result[f"ligand_atoms_with_protein_within_{label}_a"] = float(
            np.sum(nearest <= cutoff)
        )
        result[f"protein_ligand_pairs_within_{label}_a"] = float(
            np.sum(pair_distances <= cutoff)
        )
    return result


def hidden_observables(
    frame: Any,
    *,
    protein: str,
    uniprot_to_model_index: Mapping[int, int],
    metric: PocketEndpointMetric,
) -> dict[str, float | None]:
    index = lambda number: int(uniprot_to_model_index[int(number)])
    if protein == "prmt5":
        asp = _available_points(frame, index(442), ("OD1", "OD2"))
        arg = _available_points(frame, index(604), ("NE", "NH1", "NH2"))
        phe = _available_points(frame, index(440), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"))
        val = _available_points(frame, index(503), ("CB", "CG1", "CG2"))
        return {
            "d442_o_r604_n_min_a": _minimum_distance(asp, arg),
            "f440_ring_v503_sidechain_centroid_a": _centroid_distance(phe, val),
        }
    if protein == "prmt6":
        his = _available_points(frame, index(163), ("CB", "CG", "ND1", "CD2", "CE1", "NE2"))
        met = _available_points(frame, index(373), ("CB", "CG", "SD", "CE"))
        leucine_atoms = ("CB", "CG", "CD1", "CD2")
        leucine = _available_points(frame, index(161), leucine_atoms)
        reference = metric.references[0]
        aligned, _ = metric._aligned_to_reference(frame, reference)
        ref_points = _available_points(reference, index(161), leucine_atoms)
        aligned_leucine = None
        if leucine is not None:
            atom_indices = [ATOM37_INDEX[name] for name in leucine_atoms]
            aligned_leucine = aligned[index(161), atom_indices]
        return {
            "h163_m373_sidechain_heavy_min_a": _minimum_distance(his, met),
            "l161_sidechain_centroid_to_holo_a": _centroid_distance(
                aligned_leucine, ref_points
            ),
        }
    if protein in {
        "smarca2", "pi3ka", "pi3ka_observed_mask",
        "chignolin", "trpcage", "bba",
    }:
        # Additional B4 proteins use the same local-loop sampling reward.
        # Ligand-space/contact observables are intentionally computed
        # post hoc from the saved populations once their evaluation-only
        # definitions are frozen; returning an empty mapping keeps those
        # observables out of the sampling path and preserves raw trajectories.
        return {}
    raise ValueError(f"Unsupported Phase-B protein: {protein}")


def evaluate_population(
    *,
    particles: Sequence[Any],
    normalized_weights: Sequence[float],
    adapter: Any,
    metric: PocketEndpointMetric,
    d0_a: float,
    protein: str,
    uniprot_to_model_index: Mapping[int, int],
    start_reference: PocketReference,
    ligand_heavy_atom_coordinates_a: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the terminal pre-resampling population and retain full curves."""

    weights = np.asarray(normalized_weights, dtype=float)
    if len(particles) == 0 or weights.shape != (len(particles),):
        raise ValueError("Population and final weights do not match")
    weights = weights / np.sum(weights)
    threshold = min(1.5, 0.5 * float(d0_a))
    distance_curves = []
    hidden_curves = []
    path_validity = []
    final_core_to_start = []
    final_core_to_holo = []
    open_like = []
    anytime_hit = []
    anytime_valid_hit = []
    for particle in particles:
        distance_curve = [metric.distance_a(frame) for frame in particle.history]
        distance_curves.append(distance_curve)
        frame_observables = []
        for frame in particle.history:
            observables = hidden_observables(
                    frame,
                    protein=protein,
                    uniprot_to_model_index=uniprot_to_model_index,
                    metric=metric,
                )
            if ligand_heavy_atom_coordinates_a is not None:
                observables.update(
                    ligand_space_observables(
                        frame,
                        metric=metric,
                        ligand_heavy_atom_coordinates_a=(
                            ligand_heavy_atom_coordinates_a
                        ),
                    )
                )
            frame_observables.append(observables)
        hidden_curves.append(frame_observables)
        validity = adapter.validate_frames(particle.history)
        path_validity.append(validity)
        valid_path = all(bool(row["valid"]) for row in validity)
        last_two = distance_curve[-2:]
        open_like.append(valid_path and len(last_two) == 2 and all(d <= threshold for d in last_two))
        hit = any(d <= threshold for d in distance_curve[1:])
        anytime_hit.append(hit)
        anytime_valid_hit.append(valid_path and hit)
        final = particle.history[-1]
        final_core_to_start.append(metric.core_rmsd_a(final, start_reference))
        final_core_to_holo.append(
            min(metric.core_rmsd_a(final, reference) for reference in metric.references)
        )

    distances = np.asarray(distance_curves, dtype=float)
    valid_flags = np.asarray(
        [all(bool(row["valid"]) for row in rows) for rows in path_validity], dtype=bool
    )
    final_distances = distances[:, -1]
    valid_final = final_distances[valid_flags]
    open_like_flags = np.asarray(open_like, dtype=bool)
    anytime_hit_flags = np.asarray(anytime_hit, dtype=bool)
    anytime_valid_hit_flags = np.asarray(anytime_valid_hit, dtype=bool)
    # Phase-B amendment (2026-09-08): the primary yield is a literal
    # trajectory fraction.  In particular, invalid paths remain in its
    # denominator.  Keep the SMC-weighted counterparts explicitly named so
    # neither estimator is silently substituted for the other.
    metrics = {
        "d0_a": float(d0_a),
        "open_like_threshold_a": float(threshold),
        "weighted_mean_final_d_over_d0": float(np.dot(weights, final_distances / d0_a)),
        "best_final_d_a_among_valid_paths": (
            float(np.min(valid_final)) if len(valid_final) else None
        ),
        "population_path_count": int(len(particles)),
        "valid_path_count": int(np.sum(valid_flags)),
        "valid_open_like_path_count": int(np.sum(open_like_flags)),
        "valid_path_fraction": float(np.mean(valid_flags)),
        "valid_open_like_fraction": float(np.mean(open_like_flags)),
        "anytime_hit_fraction": float(np.mean(anytime_hit_flags)),
        "anytime_valid_hit_fraction": float(np.mean(anytime_valid_hit_flags)),
        "weighted_valid_path_fraction": float(np.dot(weights, valid_flags.astype(float))),
        "weighted_valid_open_like_fraction": float(np.dot(weights, open_like_flags.astype(float))),
        "weighted_anytime_hit_fraction": float(np.dot(weights, anytime_hit_flags.astype(float))),
        "weighted_anytime_valid_hit_fraction": float(
            np.dot(weights, anytime_valid_hit_flags.astype(float))
        ),
        "final_core_rmsd_to_start_a_weighted_mean": float(
            np.dot(weights, np.asarray(final_core_to_start))
        ),
        "final_core_rmsd_to_holo_a_weighted_mean": float(
            np.dot(weights, np.asarray(final_core_to_holo))
        ),
        "pre_final_normalized_weights": weights.tolist(),
        "per_path_final_d_a": final_distances.tolist(),
        "per_path_valid": valid_flags.tolist(),
        "per_path_open_like": list(map(bool, open_like)),
        "per_path_anytime_hit": list(map(bool, anytime_hit)),
        "per_path_anytime_valid_hit": list(map(bool, anytime_valid_hit)),
        "path_validity": path_validity,
    }
    if ligand_heavy_atom_coordinates_a is not None:
        final_ligand_observables = [row[-1] for row in hidden_curves]
        metric_keys = sorted(final_ligand_observables[0])
        metrics.update(
            {
                "ligand_space_role": "evaluation_only_not_sampling_reward",
                "ligand_space_cutoffs_a": [1.5, 2.0, 2.5, 3.0, 4.0],
                "ligand_space_binary_success_rule": None,
                "ligand_space_start": ligand_space_observables(
                    start_reference,
                    metric=metric,
                    ligand_heavy_atom_coordinates_a=ligand_heavy_atom_coordinates_a,
                ),
                "ligand_space_holo_reference": ligand_space_observables(
                    metric.references[0],
                    metric=metric,
                    ligand_heavy_atom_coordinates_a=ligand_heavy_atom_coordinates_a,
                ),
                "per_path_final_ligand_space": final_ligand_observables,
                "weighted_mean_final_ligand_space": {
                    key: float(
                        np.dot(
                            weights,
                            np.asarray(
                                [item[key] for item in final_ligand_observables],
                                dtype=float,
                            ),
                        )
                    )
                    for key in metric_keys
                },
            }
        )
    curves = {
        "distance_a": distances.tolist(),
        "distance_over_d0": (distances / d0_a).tolist(),
        "hidden_observables": hidden_curves,
        "weighted_mean_distance_over_d0": np.average(
            distances / d0_a, axis=0, weights=weights
        ).tolist(),
        "best_distance_over_d0_among_whole_path_valid": (
            np.min(distances[valid_flags] / d0_a, axis=0).tolist()
            if np.any(valid_flags)
            else [None] * distances.shape[1]
        ),
    }
    if protein == "pi3ka_observed_mask":
        metrics.update(
            {
                "metric_name": "observed_loop_rmsd",
                "metric_interpretation": (
                    "Experimentally observed residues only; not full-loop "
                    "restoration or pocket-opening success"
                ),
                "observed_loop_weighted_mean_final_rmsd_a": float(
                    np.dot(weights, final_distances)
                ),
                "observed_loop_holo_like_fraction": float(
                    np.mean(open_like_flags)
                ),
                "observed_loop_anytime_valid_hit_fraction": float(
                    np.mean(anytime_valid_hit_flags)
                ),
            }
        )
        curves["observed_loop_rmsd_a"] = distances.tolist()
        curves["observed_loop_rmsd_over_d0"] = (distances / d0_a).tolist()
    return metrics, curves
