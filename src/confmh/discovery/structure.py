from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


@dataclass(frozen=True)
class Residue:
    chain: str
    resseq: int
    insertion_code: str
    name: str
    atoms: dict[str, np.ndarray]

    @property
    def code(self) -> str:
        return AA3_TO_1.get(self.name.upper(), "X")


def read_pdb(path: str | Path) -> list[Residue]:
    """Read the first PDB model and keep the primary polymer atom conformer."""
    path = Path(path)
    residues: dict[tuple[str, int, str], tuple[str, dict[str, np.ndarray]]] = {}
    saw_model = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            if saw_model:
                break
            saw_model = True
            continue
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  "):
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A"}:
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "A"
        try:
            resseq = int(line[22:26])
            xyz = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
        except ValueError:
            continue
        icode = line[26:27].strip()
        key = (chain, resseq, icode)
        if key not in residues:
            residues[key] = (resname, {})
        residues[key][1].setdefault(atom_name, xyz)
    result = [
        Residue(chain=k[0], resseq=k[1], insertion_code=k[2], name=v[0], atoms=v[1])
        for k, v in residues.items()
        if v[0] in AA3_TO_1
    ]
    if not result:
        raise ValueError(f"No polymer residues were read from {path}")
    return result


def sequence(residues: Iterable[Residue]) -> str:
    return "".join(residue.code for residue in residues)


def sequence_alignment_pairs(sequence_a: str, sequence_b: str) -> list[tuple[int, int]]:
    """Needleman-Wunsch residue mapping, returned as zero-based index pairs."""
    n, m = len(sequence_a), len(sequence_b)
    score = np.empty((n + 1, m + 1), dtype=np.int32)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    score[:, 0] = -2 * np.arange(n + 1)
    score[0, :] = -2 * np.arange(m + 1)
    trace[1:, 0] = 1
    trace[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = score[i - 1, j - 1] + (2 if sequence_a[i - 1] == sequence_b[j - 1] else -1)
            up = score[i - 1, j] - 2
            left = score[i, j - 1] - 2
            choice = int(np.argmax((diagonal, up, left)))
            score[i, j] = (diagonal, up, left)[choice]
            trace[i, j] = choice
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i or j:
        direction = int(trace[i, j])
        if i and j and direction == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and (j == 0 or direction == 1):
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def atom_coordinates(
    residues: list[Residue], atom_name: str, indices: Iterable[int] | None = None
) -> tuple[np.ndarray, list[int]]:
    selected = range(len(residues)) if indices is None else indices
    coordinates: list[np.ndarray] = []
    kept: list[int] = []
    for index in selected:
        atom = residues[index].atoms.get(atom_name)
        if atom is not None:
            coordinates.append(atom)
            kept.append(index)
    if not coordinates:
        raise ValueError(f"No {atom_name} atoms found")
    return np.stack(coordinates), kept


def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Kabsch inputs must both have shape (N, 3), got {mobile.shape}/{target.shape}")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt)) or 1.0
    rotation = u @ correction @ vt
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    rotation, translation = kabsch_transform(mobile, target)
    difference = mobile @ rotation + translation - target
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def aligned_local_rmsd(
    mobile_alignment: np.ndarray,
    target_alignment: np.ndarray,
    mobile_metric: np.ndarray,
    target_metric: np.ndarray,
) -> float:
    rotation, translation = kabsch_transform(mobile_alignment, target_alignment)
    difference = mobile_metric @ rotation + translation - target_metric
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def write_single_frame_pdb(frame, path: str | Path) -> None:
    """Save an mdtraj one-frame trajectory without exposing mdtraj at import time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.save_pdb(str(path))
