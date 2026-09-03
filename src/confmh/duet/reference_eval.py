from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from confmh.duet.analysis import jensen_shannon_divergence
from confmh.duet.config import resolve_config_path
from confmh.pca_cv import PCACV, _kabsch_align


def dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    table = np.full((len(left) + 1, len(right) + 1), np.inf, dtype=float)
    table[0, 0] = 0.0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            cost = float(np.linalg.norm(left[i - 1] - right[j - 1]))
            table[i, j] = cost + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    return float(table[-1, -1] / max(len(left), len(right)))


def kabsch_rmsd_nm(coords: np.ndarray, reference: np.ndarray) -> float:
    aligned = _kabsch_align(np.asarray(coords, dtype=float), np.asarray(reference, dtype=float))
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=-1))))


def _transition_windows(
    pc1: np.ndarray,
    start_interval: Sequence[float],
    target_interval: Sequence[float],
    horizon: int,
    maximum: int = 512,
) -> list[np.ndarray]:
    values = np.asarray(pc1, dtype=float)
    starts = np.flatnonzero((values >= start_interval[0]) & (values <= start_interval[1]))
    windows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for start in starts:
        targets = np.flatnonzero(
            (np.arange(len(values)) > start)
            & (values >= target_interval[0])
            & (values <= target_interval[1])
        )
        if not len(targets):
            continue
        end = int(targets[0])
        if end - int(start) < horizon:
            continue
        indices = tuple(np.rint(np.linspace(start, end, horizon + 1)).astype(int).tolist())
        if indices not in seen:
            windows.append(np.asarray(indices, dtype=int))
            seen.add(indices)
    if len(windows) > maximum:
        selected = np.rint(np.linspace(0, len(windows) - 1, maximum)).astype(int)
        windows = [windows[index] for index in selected]
    return windows


def _contact_map(ca_nm: np.ndarray, excluded: set[tuple[int, int]], threshold_nm: float) -> np.ndarray:
    pairs = []
    for i in range(len(ca_nm)):
        for j in range(i + 5, len(ca_nm)):
            if (i, j) not in excluded:
                pairs.append(np.linalg.norm(ca_nm[i] - ca_nm[j]) <= threshold_nm)
    return np.asarray(pairs, dtype=bool)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


@dataclass
class HeldOutReferenceEvaluator:
    pca: PCACV
    reference_ca_nm: np.ndarray
    reference_pc: np.ndarray
    windows: list[np.ndarray]
    excluded_contacts: set[tuple[int, int]]
    contact_threshold_nm: float = 0.8

    @classmethod
    def from_config(cls, cfg: dict[str, Any], catalog: dict[str, Any]) -> "HeldOutReferenceEvaluator":
        import mdtraj as md

        reference = cfg["reference"]
        topology_path = resolve_config_path(cfg, reference["topology"])
        trajectory_path = resolve_config_path(cfg, reference["held_out_trajectory"])
        pca = PCACV.load(resolve_config_path(cfg, reference["pca_model"]))
        topology = md.load(str(topology_path))
        protein = topology.topology.select(str(reference.get("protein_selection", "protein and chainid 0")))
        trajectory = md.load(
            str(trajectory_path),
            top=str(topology_path),
            atom_indices=protein,
            stride=int(reference.get("evaluation_stride", 10)),
        )
        maximum = reference.get("evaluation_max_frames")
        if maximum is not None:
            trajectory = trajectory[: int(maximum)]
        ca_indices = trajectory.topology.select("name CA")
        reference_ca = np.asarray(trajectory.xyz[:, ca_indices, :], dtype=float)
        reference_pc = pca.project_ca(reference_ca)
        selection = catalog.get("selection", {})
        start_interval = selection.get("start_interval")
        target_interval = selection.get("target_interval")
        if start_interval is None or target_interval is None:
            endpoint = catalog["tasks"]["endpoint"]["events"][0]
            start_score = float(pca.project_files(resolve_config_path(cfg, cfg["trajectory"]["initial_structure"]))[0, 0])
            width = float(reference.get("basin_half_width", 0.25))
            start_interval = [start_score - width, start_score + width]
            target_interval = endpoint["target_interval"]
        windows = _transition_windows(
            reference_pc[:, 0], start_interval, target_interval, int(cfg["trajectory"]["horizon"])
        )
        excluded = set()
        for observable in catalog.get("observables", {}).values():
            if observable.get("kind") in {"contact", "residue_distance"}:
                i, j = sorted((int(observable["residue_i"]), int(observable["residue_j"])))
                excluded.add((i, j))
        return cls(pca, reference_ca, reference_pc, windows, excluded)

    def evaluate(self, paths: Sequence[Sequence[Any]]) -> dict[str, Any]:
        if not self.windows:
            return {
                "status": "suppressed",
                "reason": "no held-out R3 start-to-target transition windows",
                "reference_transition_window_count": 0,
            }
        generated_ca = [np.stack([frame.ca_nm for frame in path]) for path in paths]
        generated_pc = [self.pca.project_ca(path) for path in generated_ca]
        reference_pc_paths = [self.reference_pc[indices] for indices in self.windows]
        reference_ca_paths = [self.reference_ca_nm[indices] for indices in self.windows]
        reference_interior = np.concatenate([path[1:-1, : min(3, path.shape[1])] for path in reference_pc_paths])
        if len(reference_interior) > 1:
            pairwise = np.linalg.norm(reference_interior[:, None] - reference_interior[None, :], axis=-1)
            np.fill_diagonal(pairwise, np.inf)
            coverage_radius = float(np.quantile(np.min(pairwise, axis=1), 0.95))
        else:
            coverage_radius = float("nan")
        per_path = []
        generated_contacts = []
        for ca_path, pc_path in zip(generated_ca, generated_pc):
            path_distances = [dtw_distance(pc_path[:, : min(3, pc_path.shape[1])], ref[:, : min(3, ref.shape[1])]) for ref in reference_pc_paths]
            closest = int(np.argmin(path_distances))
            endpoint_rmsd = kabsch_rmsd_nm(ca_path[-1], reference_ca_paths[closest][-1])
            start_rmsd = kabsch_rmsd_nm(ca_path[0], reference_ca_paths[closest][-1])
            smoothness = float(np.linalg.norm(np.diff(pc_path, n=2, axis=0), axis=-1).mean()) if len(pc_path) > 2 else 0.0
            interior = pc_path[1:-1, : min(3, pc_path.shape[1])]
            nearest = np.min(np.linalg.norm(interior[:, None] - reference_interior[None, :], axis=-1), axis=1) if len(interior) else np.asarray([])
            contact = _contact_map(ca_path[-1], self.excluded_contacts, self.contact_threshold_nm)
            ref_contact = _contact_map(reference_ca_paths[closest][-1], self.excluded_contacts, self.contact_threshold_nm)
            generated_contacts.append(contact)
            per_path.append(
                {
                    "held_out_path_distance": float(path_distances[closest]),
                    "held_out_endpoint_ca_rmsd_nm": endpoint_rmsd,
                    "ca_rmsd_progress": float(1.0 - endpoint_rmsd / start_rmsd) if start_rmsd > 0 else None,
                    "pca_progress": float(pc_path[-1, 0] - pc_path[0, 0]),
                    "path_smoothness": smoothness,
                    "reference_intermediate_coverage": float(np.mean(nearest <= coverage_radius)) if len(nearest) and np.isfinite(coverage_radius) else None,
                    "unspecified_contact_map_similarity": _jaccard(contact, ref_contact),
                    "closest_reference_window": closest,
                }
            )
        route_jsd = None
        sensitivity: dict[str, float] = {}
        suppression = None
        if len(self.windows) < 10 or len(generated_pc) < 4:
            suppression = "requires at least 10 held-out transition windows and 4 generated paths"
        else:
            reference_points = np.concatenate([path[1:-1, :2] for path in reference_pc_paths])
            generated_points = np.concatenate([path[1:-1, :2] for path in generated_pc])
            bounds = [
                [min(reference_points[:, dim].min(), generated_points[:, dim].min()), max(reference_points[:, dim].max(), generated_points[:, dim].max())]
                for dim in range(2)
            ]
            for bins in (6, 8, 10):
                ref_hist, _ = np.histogramdd(reference_points, bins=bins, range=bounds)
                gen_hist, _ = np.histogramdd(generated_points, bins=bins, range=bounds)
                value = jensen_shannon_divergence(ref_hist.ravel() + 1e-12, gen_hist.ravel() + 1e-12)
                sensitivity[str(bins)] = value
            route_jsd = sensitivity["8"]
        aggregate = {
            key: float(np.mean([row[key] for row in per_path if row[key] is not None]))
            if any(row[key] is not None for row in per_path)
            else None
            for key in (
                "held_out_path_distance",
                "held_out_endpoint_ca_rmsd_nm",
                "ca_rmsd_progress",
                "pca_progress",
                "path_smoothness",
                "reference_intermediate_coverage",
                "unspecified_contact_map_similarity",
            )
        }
        return {
            "status": "ok",
            "reference_transition_window_count": len(self.windows),
            "coverage_radius_pca": coverage_radius,
            **aggregate,
            "route_jsd": route_jsd,
            "route_jsd_binning_sensitivity": sensitivity,
            "route_jsd_suppression_reason": suppression,
            "per_path": per_path,
        }
