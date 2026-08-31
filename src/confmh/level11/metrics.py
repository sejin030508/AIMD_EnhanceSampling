from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from confmh.pca_cv import _kabsch_align


@dataclass(frozen=True)
class GeometryResult:
    valid: bool
    failure_reason: str
    ca_rmsd_nm: float
    ca_msd_nm2: float
    max_ca_displacement_nm: float
    min_nonbonded_ca_distance_nm: float
    min_adjacent_ca_distance_nm: float
    max_adjacent_ca_distance_nm: float


def ca_coordinates(frame) -> np.ndarray:
    indices = frame.topology.select("protein and name CA")
    if len(indices) == 0:
        raise ValueError("Trajectory frame contains no protein C-alpha atoms")
    return np.asarray(frame.xyz[0, indices, :], dtype=float)


def geometry_diagnostics(
    current_frame,
    proposal_frame,
    *,
    max_ca_displacement_nm: float = 3.0,
    min_nonbonded_ca_distance_nm: float = 0.20,
    adjacent_ca_distance_range_nm: tuple[float, float] = (0.25, 0.55),
) -> GeometryResult:
    current = ca_coordinates(current_frame)
    proposed = ca_coordinates(proposal_frame)
    if current.shape != proposed.shape:
        return GeometryResult(False, "ca_count_mismatch", *(float("nan"),) * 6)
    if not np.all(np.isfinite(proposed)):
        return GeometryResult(False, "non_finite_coordinates", *(float("nan"),) * 6)

    aligned = _kabsch_align(proposed, current)
    displacement = aligned - current
    squared = np.sum(displacement**2, axis=-1)
    rmsd = float(np.sqrt(np.mean(squared)))
    msd = float(np.mean(squared))
    max_displacement = float(np.sqrt(np.max(squared)))

    delta = proposed[:, None, :] - proposed[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    n = len(proposed)
    nonbonded = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > 1
    min_nonbonded = float(np.min(distances[nonbonded])) if np.any(nonbonded) else float("nan")
    adjacent = np.linalg.norm(np.diff(proposed, axis=0), axis=-1)
    min_adjacent = float(np.min(adjacent)) if len(adjacent) else float("nan")
    max_adjacent = float(np.max(adjacent)) if len(adjacent) else float("nan")

    failures = []
    if max_displacement > max_ca_displacement_nm:
        failures.append("extreme_ca_displacement")
    if min_nonbonded < min_nonbonded_ca_distance_nm:
        failures.append("ca_clash")
    lower, upper = adjacent_ca_distance_range_nm
    if min_adjacent < lower or max_adjacent > upper:
        failures.append("adjacent_ca_geometry")
    return GeometryResult(
        valid=not failures,
        failure_reason=";".join(failures),
        ca_rmsd_nm=rmsd,
        ca_msd_nm2=msd,
        max_ca_displacement_nm=max_displacement,
        min_nonbonded_ca_distance_nm=min_nonbonded,
        min_adjacent_ca_distance_nm=min_adjacent,
        max_adjacent_ca_distance_nm=max_adjacent,
    )


def structure_hash(frame, decimals: int = 3) -> str:
    coords = np.round(ca_coordinates(frame), decimals=decimals).astype(np.float32)
    return hashlib.sha256(coords.tobytes()).hexdigest()


def rejection_run_lengths(accepted: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in np.asarray(accepted, dtype=bool):
        if value:
            if current:
                runs.append(current)
            current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    return runs
