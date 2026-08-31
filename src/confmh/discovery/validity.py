from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from confmh.discovery.structure import read_pdb, sequence, sequence_alignment_pairs


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    left, right = a - b, c - b
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    cosine = np.clip(np.dot(left, right) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _coarse_rama_allowed(phi: float, psi: float) -> bool:
    # Broad general-residue regions used only as a safety screen.  Publication
    # analysis should additionally run MolProbity on retained hits.
    beta = -180 <= phi <= -40 and (50 <= psi <= 180 or -180 <= psi <= -150)
    alpha_r = -160 <= phi <= -20 and -100 <= psi <= 60
    alpha_l = 20 <= phi <= 120 and -80 <= psi <= 100
    return bool(beta or alpha_r or alpha_l)


@dataclass
class ValidityResult:
    valid: bool
    paper_compliant_0_90: bool
    strengthened_valid: bool
    n_residues: int
    ca_valid_fraction: float
    peptide_cn_valid_fraction: float
    clash_free_fraction: float
    ca_adjacent_max_a: float
    peptide_cn_max_a: float
    backbone_clash_min_a: float
    bond_outlier_fraction: float
    angle_outlier_fraction: float
    rama_outlier_fraction_coarse: float
    secondary_structure_retention: float | None
    failed_checks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_THRESHOLDS: dict[str, float] = {
    "paper_compliance_tau": 0.90,
    "bond_outlier_fraction": 0.05,
    "angle_outlier_fraction": 0.10,
    "rama_outlier_fraction_coarse": 0.20,
}


def _secondary_structure_retention(sample: Path, reference: Path) -> float | None:
    try:
        import mdtraj as md
    except ImportError:
        return None
    try:
        sample_traj, reference_traj = md.load(str(sample)), md.load(str(reference))
        sample_dssp = md.compute_dssp(sample_traj, simplified=True)[0]
        reference_dssp = md.compute_dssp(reference_traj, simplified=True)[0]
    except Exception:
        return None
    sample_residues, reference_residues = read_pdb(sample), read_pdb(reference)
    pairs = sequence_alignment_pairs(sequence(sample_residues), sequence(reference_residues))
    pairs = [(i, j) for i, j in pairs if i < len(sample_dssp) and j < len(reference_dssp)]
    if not pairs:
        return None
    return float(np.mean([sample_dssp[i] == reference_dssp[j] for i, j in pairs]))


def evaluate_validity(
    pdb: str | Path,
    *,
    reference_pdb: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> ValidityResult:
    """Evaluate paper-compatible and strengthened backbone validity checks."""
    pdb = Path(pdb).expanduser().resolve()
    residues = read_pdb(pdb)
    active_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        active_thresholds.update({key: float(value) for key, value in thresholds.items()})

    ca_distances: list[float] = []
    cn_distances: list[float] = []
    bond_outliers: list[bool] = []
    angle_outliers: list[bool] = []
    rama_outliers: list[bool] = []
    backbone_atoms: list[tuple[int, str, np.ndarray]] = []
    for index, residue in enumerate(residues):
        for name in ("N", "CA", "C", "O"):
            if name in residue.atoms:
                backbone_atoms.append((index, name, residue.atoms[name]))
        if "N" in residue.atoms and "CA" in residue.atoms:
            distance = float(np.linalg.norm(residue.atoms["N"] - residue.atoms["CA"]))
            bond_outliers.append(not 1.25 <= distance <= 1.65)
        if "CA" in residue.atoms and "C" in residue.atoms:
            distance = float(np.linalg.norm(residue.atoms["CA"] - residue.atoms["C"]))
            bond_outliers.append(not 1.25 <= distance <= 1.70)
        if all(name in residue.atoms for name in ("N", "CA", "C")):
            value = _angle(residue.atoms["N"], residue.atoms["CA"], residue.atoms["C"])
            angle_outliers.append(not 95.0 <= value <= 130.0)
        if index + 1 < len(residues):
            following = residues[index + 1]
            if "CA" in residue.atoms and "CA" in following.atoms:
                ca_distances.append(float(np.linalg.norm(residue.atoms["CA"] - following.atoms["CA"])))
            if "C" in residue.atoms and "N" in following.atoms:
                distance = float(np.linalg.norm(residue.atoms["C"] - following.atoms["N"]))
                cn_distances.append(distance)
                bond_outliers.append(not 1.15 <= distance <= 1.55)
            if all(name in residue.atoms for name in ("CA", "C")) and "N" in following.atoms:
                value = _angle(residue.atoms["CA"], residue.atoms["C"], following.atoms["N"])
                angle_outliers.append(not 100.0 <= value <= 135.0)
            if "C" in residue.atoms and all(name in following.atoms for name in ("N", "CA")):
                value = _angle(residue.atoms["C"], following.atoms["N"], following.atoms["CA"])
                angle_outliers.append(not 100.0 <= value <= 135.0)
        if 0 < index < len(residues) - 1:
            previous, following = residues[index - 1], residues[index + 1]
            if all(
                atom in source.atoms
                for atom, source in (("C", previous), ("N", residue), ("CA", residue), ("C", residue))
            ) and "N" in following.atoms:
                phi = _dihedral(
                    previous.atoms["C"], residue.atoms["N"], residue.atoms["CA"], residue.atoms["C"]
                )
                psi = _dihedral(
                    residue.atoms["N"], residue.atoms["CA"], residue.atoms["C"], following.atoms["N"]
                )
                rama_outliers.append(not _coarse_rama_allowed(phi, psi))

    clash_distances = [
        float(np.linalg.norm(atom_i - atom_j))
        for (residue_i, _, atom_i), (residue_j, _, atom_j) in combinations(backbone_atoms, 2)
        if abs(residue_i - residue_j) > 1
    ]
    ca_max = max(ca_distances, default=float("inf"))
    cn_max = max(cn_distances, default=float("inf"))
    clash_min = min(clash_distances, default=0.0)
    ca_valid_fraction = float(np.mean(np.asarray(ca_distances) < 4.5)) if ca_distances else 0.0
    cn_valid_fraction = float(np.mean(np.asarray(cn_distances) < 2.0)) if cn_distances else 0.0
    clash_free_fraction = (
        float(np.mean(np.asarray(clash_distances) > 1.0)) if clash_distances else 0.0
    )
    bond_fraction = float(np.mean(bond_outliers)) if bond_outliers else 1.0
    angle_fraction = float(np.mean(angle_outliers)) if angle_outliers else 1.0
    rama_fraction = float(np.mean(rama_outliers)) if rama_outliers else 1.0

    strengthened_values = {
        "bond_outlier_fraction": bond_fraction,
        "angle_outlier_fraction": angle_fraction,
        "rama_outlier_fraction_coarse": rama_fraction,
    }
    tau = float(active_thresholds.pop("paper_compliance_tau"))
    paper_fractions = {
        "ca_valid_fraction": ca_valid_fraction,
        "peptide_cn_valid_fraction": cn_valid_fraction,
        "clash_free_fraction": clash_free_fraction,
    }
    failed = [key for key, value in paper_fractions.items() if value < tau]
    strengthened_failed = [
        key
        for key, threshold in active_thresholds.items()
        if key in strengthened_values and strengthened_values[key] > threshold
    ]
    paper_compliant = not failed
    strengthened_valid = paper_compliant and not strengthened_failed
    retention = (
        None
        if reference_pdb is None
        else _secondary_structure_retention(pdb, Path(reference_pdb).expanduser().resolve())
    )
    return ValidityResult(
        valid=paper_compliant,
        paper_compliant_0_90=paper_compliant,
        strengthened_valid=strengthened_valid,
        n_residues=len(residues),
        ca_valid_fraction=ca_valid_fraction,
        peptide_cn_valid_fraction=cn_valid_fraction,
        clash_free_fraction=clash_free_fraction,
        ca_adjacent_max_a=ca_max,
        peptide_cn_max_a=cn_max,
        backbone_clash_min_a=clash_min,
        bond_outlier_fraction=bond_fraction,
        angle_outlier_fraction=angle_fraction,
        rama_outlier_fraction_coarse=rama_fraction,
        secondary_structure_retention=retention,
        failed_checks=failed + strengthened_failed,
    )
