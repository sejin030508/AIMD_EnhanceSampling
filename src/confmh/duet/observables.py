from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from confmh.pca_cv import PCACV, _kabsch_align


# AlphaFold/OpenFold's fixed atom37 order.  Keeping the table local lets the
# lightweight metric tests run without importing the full ConfRover stack.
ATOM37_NAMES = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE",
    "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)
ATOM37_INDEX = {name: index for index, name in enumerate(ATOM37_NAMES)}


def ca_coordinates(frame: Any) -> np.ndarray:
    if isinstance(frame, Mapping):
        if "ca_nm" in frame:
            return np.asarray(frame["ca_nm"], dtype=float)
        if "ca" in frame:
            return np.asarray(frame["ca"], dtype=float)
    if hasattr(frame, "ca_nm"):
        return np.asarray(frame.ca_nm, dtype=float)
    array = np.asarray(frame, dtype=float)
    if array.ndim == 2 and array.shape[-1] == 3:
        return array
    raise TypeError("Frame must provide C-alpha coordinates with shape (residues, 3)")


def ca_rmsd_nm(frame: Any, reference_ca_nm: np.ndarray) -> float:
    coords = ca_coordinates(frame)
    reference = np.asarray(reference_ca_nm, dtype=float)
    if coords.shape != reference.shape:
        raise ValueError(f"RMSD shape mismatch: {coords.shape} != {reference.shape}")
    aligned = _kabsch_align(coords, reference)
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=-1))))


def residue_pair_distance_nm(frame: Any, residue_i: int, residue_j: int) -> float:
    coords = ca_coordinates(frame)
    return float(np.linalg.norm(coords[int(residue_i)] - coords[int(residue_j)]))


def atom37_coordinates_a(frame: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(frame, Mapping):
        coords = frame.get("atom37_a")
        mask = frame.get("atom37_mask")
    else:
        coords = getattr(frame, "atom37_a", None)
        mask = getattr(frame, "atom37_mask", None)
    if coords is None:
        raise TypeError("Frame must provide atom37_a coordinates")
    array = np.asarray(coords, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (37, 3):
        raise ValueError(f"atom37_a must have shape (residues, 37, 3), got {array.shape}")
    valid = np.ones(array.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != array.shape[:2]:
        raise ValueError(f"atom37_mask shape mismatch: {valid.shape} != {array.shape[:2]}")
    return array, valid


def atom_coordinate_a(frame: Any, residue_index: int, atom_name: str) -> np.ndarray:
    coords, mask = atom37_coordinates_a(frame)
    residue_index = int(residue_index)
    try:
        atom_index = ATOM37_INDEX[str(atom_name)]
    except KeyError as error:
        raise ValueError(f"Unknown atom37 name: {atom_name}") from error
    if residue_index < 0 or residue_index >= len(coords):
        raise IndexError(f"Residue index {residue_index} outside [0, {len(coords)})")
    if not mask[residue_index, atom_index]:
        raise ValueError(f"Missing atom {atom_name} at residue index {residue_index}")
    point = coords[residue_index, atom_index]
    if not np.all(np.isfinite(point)):
        raise ValueError(f"Non-finite atom {atom_name} at residue index {residue_index}")
    return point


def atom_pair_distance_a(
    frame: Any,
    left_residue: int,
    left_atom: str,
    right_residue: int,
    right_atom: str,
) -> float:
    left = atom_coordinate_a(frame, left_residue, left_atom)
    right = atom_coordinate_a(frame, right_residue, right_atom)
    return float(np.linalg.norm(left - right))


def pseudo_dihedral_deg(frame: Any, atoms: list[list[Any]] | tuple[tuple[Any, ...], ...]) -> float:
    if len(atoms) != 4:
        raise ValueError("A pseudo-dihedral requires exactly four [residue_index, atom_name] pairs")
    a, b, c, d = [atom_coordinate_a(frame, int(item[0]), str(item[1])) for item in atoms]
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    norm = float(np.linalg.norm(b1))
    if norm <= np.finfo(float).tiny:
        raise ValueError("Degenerate pseudo-dihedral: central bond has zero length")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    angle = np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
    return float(angle % 360.0)


def circular_difference_deg(value: float | np.ndarray, center: float | np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=float) - np.asarray(center, dtype=float) + 180.0) % 360.0 - 180.0


def composite_dihedral_distance(
    frame: Any,
    components: list[Mapping[str, Any]],
) -> float:
    residuals = []
    for component in components:
        angle = pseudo_dihedral_deg(frame, component["atoms"])
        scale = float(component["scale_degrees"])
        if scale <= 0:
            raise ValueError("scale_degrees must be positive")
        residuals.append(float(circular_difference_deg(angle, component["center_degrees"])) / scale)
    if not residuals:
        raise ValueError("composite_dihedral_distance requires at least one component")
    return float(np.sqrt(np.mean(np.square(residuals))))


def local_atom_rmsd_a(
    frame: Any,
    reference_atom37_a: np.ndarray,
    reference_mask: np.ndarray,
    fit_residue_indices: list[int],
    measure_atoms: list[list[Any]],
) -> float:
    coords, mask = atom37_coordinates_a(frame)
    reference = np.asarray(reference_atom37_a, dtype=float)
    ref_mask = np.asarray(reference_mask, dtype=bool)
    fit = np.asarray(fit_residue_indices, dtype=int)
    ca = ATOM37_INDEX["CA"]
    if np.any(~mask[fit, ca]) or np.any(~ref_mask[fit, ca]):
        raise ValueError("Missing C-alpha atom in local RMSD alignment selection")
    moving_fit = coords[fit, ca]
    target_fit = reference[fit, ca]
    moving_center = moving_fit.mean(axis=0)
    target_center = target_fit.mean(axis=0)
    covariance = (moving_fit - moving_center).T @ (target_fit - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.linalg.det(left @ right_t)])
    rotation = left @ correction @ right_t
    moving = []
    target = []
    for residue_index, atom_name in measure_atoms:
        atom_index = ATOM37_INDEX[str(atom_name)]
        residue_index = int(residue_index)
        if not mask[residue_index, atom_index] or not ref_mask[residue_index, atom_index]:
            raise ValueError(f"Missing local RMSD atom {atom_name} at residue {residue_index}")
        moving.append(coords[residue_index, atom_index])
        target.append(reference[residue_index, atom_index])
    aligned = (np.asarray(moving) - moving_center) @ rotation + target_center
    residual = aligned - np.asarray(target)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


@dataclass
class ObservableRegistry:
    functions: dict[str, Callable[[Any], float]]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], *, root: Path | None = None) -> "ObservableRegistry":
        functions: dict[str, Callable[[Any], float]] = {}
        for name, spec in cfg.items():
            kind = str(spec["kind"])
            if kind == "pc1":
                path = Path(str(spec["pca_model"])).expanduser()
                if root is not None and not path.is_absolute():
                    path = root / path
                model = PCACV.load(path)
                component = int(spec.get("component", 0))
                functions[name] = lambda frame, m=model, c=component: float(
                    m.project_ca(ca_coordinates(frame))[c]
                )
            elif kind == "ca_rmsd":
                reference = np.asarray(spec["reference_ca_nm"], dtype=float)
                functions[name] = lambda frame, ref=reference: ca_rmsd_nm(frame, ref)
            elif kind == "residue_distance":
                i, j = int(spec["residue_i"]), int(spec["residue_j"])
                functions[name] = lambda frame, a=i, b=j: residue_pair_distance_nm(frame, a, b)
            elif kind == "contact":
                i, j = int(spec["residue_i"]), int(spec["residue_j"])
                threshold = float(spec["threshold_nm"])
                functions[name] = lambda frame, a=i, b=j, cutoff=threshold: float(
                    residue_pair_distance_nm(frame, a, b) <= cutoff
                )
            elif kind == "atom_distance_a":
                left, right = spec["atoms"]
                functions[name] = lambda frame, a=left, b=right: atom_pair_distance_a(
                    frame, int(a[0]), str(a[1]), int(b[0]), str(b[1])
                )
            elif kind == "pseudo_dihedral_deg":
                atoms = list(spec["atoms"])
                functions[name] = lambda frame, selected=atoms: pseudo_dihedral_deg(
                    frame, selected
                )
            elif kind == "composite_dihedral_distance":
                components = list(spec["components"])
                functions[name] = lambda frame, selected=components: composite_dihedral_distance(
                    frame, selected
                )
            elif kind == "local_atom_rmsd_a":
                path = Path(str(spec["reference_npz"])).expanduser()
                if root is not None and not path.is_absolute():
                    path = root / path
                with np.load(path) as payload:
                    reference = np.asarray(payload[str(spec.get("coordinate_key", "atom37_a"))])
                    reference_mask = np.asarray(payload[str(spec.get("mask_key", "atom37_mask"))])
                fit = [int(item) for item in spec["fit_residue_indices"]]
                measure = list(spec["measure_atoms"])
                functions[name] = lambda frame, ref=reference, rm=reference_mask, f=fit, m=measure: (
                    local_atom_rmsd_a(frame, ref, rm, f, m)
                )
            else:
                raise ValueError(f"Unsupported observable kind: {kind}")
        return cls(functions)

    def evaluate(self, frame: Any, names: tuple[str, ...] | list[str]) -> dict[str, float]:
        return {name: float(self.functions[name](frame)) for name in names}
