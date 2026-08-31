from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from confmh.discovery.structure import (
    Residue,
    atom_coordinates,
    kabsch_rmsd,
    read_pdb,
    sequence,
    sequence_alignment_pairs,
)


def _indices_from_ranges(residues: list[Residue], ranges: list[list[int | None]]) -> list[int]:
    result: list[int] = []
    for index, residue in enumerate(residues):
        for begin, end in ranges:
            lower = -10**9 if begin is None else int(begin)
            upper = 10**9 if end is None else int(end)
            if lower <= residue.resseq <= upper:
                result.append(index)
                break
    return result


def _secondary_structure_indices(pdb: Path) -> list[int]:
    try:
        import mdtraj as md
    except ImportError as exc:
        raise ImportError("secondary_structure mask requires mdtraj") from exc
    trajectory = md.load(str(pdb))
    labels = md.compute_dssp(trajectory, simplified=True)[0]
    return [index for index, label in enumerate(labels) if label in {"H", "E"}]


@dataclass
class CVEvaluation:
    control: float
    rmsd_start: float
    rmsd_target: float | None
    radius_gyration: float
    native_contact_fraction: float
    n_aligned_start: int
    n_aligned_target: int


class ControlCV:
    """Model-independent structural CV evaluated on one generated PDB.

    Target-blind modes never load the alternate-state PDB.  This separation is
    deliberate: the alternate state is only exposed to the post-hoc analyzer.
    """

    MODES = {"start_rmsd", "target_rmsd", "delta_rmsd", "radius_gyration", "native_contacts", "tica"}

    def __init__(
        self,
        *,
        start_pdb: str | Path,
        mode: str,
        target_pdb: str | Path | None = None,
        mask: str = "all_ca",
        residue_ranges: list[list[int | None]] | None = None,
        tica_model: str | Path | None = None,
        tica_component: int = 0,
    ):
        self.start_pdb = Path(start_pdb).expanduser().resolve()
        self.target_pdb = None if target_pdb is None else Path(target_pdb).expanduser().resolve()
        self.mode = str(mode).lower()
        if self.mode not in self.MODES:
            raise ValueError(f"Unsupported control CV {self.mode!r}; choose from {sorted(self.MODES)}")
        if self.mode in {"target_rmsd", "delta_rmsd"} and self.target_pdb is None:
            raise ValueError(f"{self.mode} requires target_pdb")
        if self.mode == "tica" and tica_model is None:
            raise ValueError("tica control requires tica_model")

        self.start = read_pdb(self.start_pdb)
        self.start_sequence = sequence(self.start)
        if mask == "all_ca":
            self.start_mask = list(range(len(self.start)))
        elif mask == "secondary_structure":
            self.start_mask = _secondary_structure_indices(self.start_pdb)
        elif mask == "residue_ranges":
            if not residue_ranges:
                raise ValueError("residue_ranges mask requires at least one range")
            self.start_mask = _indices_from_ranges(self.start, residue_ranges)
        else:
            raise ValueError("mask must be all_ca, secondary_structure, or residue_ranges")
        if len(self.start_mask) < 3:
            raise ValueError(f"Control CV mask contains only {len(self.start_mask)} residues")

        self.target = None if self.target_pdb is None else read_pdb(self.target_pdb)
        self.start_to_target = (
            {}
            if self.target is None
            else dict(sequence_alignment_pairs(self.start_sequence, sequence(self.target)))
        )
        self.start_contact_pairs = self._native_contact_pairs()
        self.tica_component = int(tica_component)
        self.tica: dict[str, np.ndarray] | None = None
        if tica_model is not None:
            loaded = np.load(Path(tica_model).expanduser().resolve())
            self.tica = {key: loaded[key] for key in loaded.files}

    def _native_contact_pairs(self) -> np.ndarray:
        ca, kept = atom_coordinates(self.start, "CA")
        ordinal = {residue_index: offset for offset, residue_index in enumerate(kept)}
        pairs = []
        for i in self.start_mask:
            if i not in ordinal:
                continue
            for j in self.start_mask:
                if j <= i + 2 or j not in ordinal:
                    continue
                distance = np.linalg.norm(ca[ordinal[i]] - ca[ordinal[j]])
                if distance <= 8.0:
                    pairs.append((i, j))
        return np.asarray(pairs, dtype=np.int32).reshape((-1, 2))

    @staticmethod
    def _mapped_ca(
        candidate: list[Residue],
        reference: list[Residue],
        pairs: list[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        valid = [
            (i, j)
            for i, j in pairs
            if "CA" in candidate[i].atoms and "CA" in reference[j].atoms
        ]
        if len(valid) < 3:
            raise ValueError(f"Only {len(valid)} aligned C-alpha atoms are available")
        return (
            np.stack([candidate[i].atoms["CA"] for i, _ in valid]),
            np.stack([reference[j].atoms["CA"] for _, j in valid]),
            valid,
        )

    def _tica_value(self, candidate: list[Residue], candidate_to_start: dict[int, int]) -> float:
        if self.tica is None:
            raise RuntimeError("TICA model was not loaded")
        start_to_candidate = {start: candidate for candidate, start in candidate_to_start.items()}
        pair_indices = np.asarray(self.tica["pair_indices"], dtype=np.int64)
        distances = []
        for start_i, start_j in pair_indices:
            if int(start_i) not in start_to_candidate or int(start_j) not in start_to_candidate:
                raise ValueError("Candidate does not cover all TICA residue pairs")
            ca_i = candidate[start_to_candidate[int(start_i)]].atoms["CA"]
            ca_j = candidate[start_to_candidate[int(start_j)]].atoms["CA"]
            distances.append(np.linalg.norm(ca_i - ca_j))
        feature = np.asarray(distances, dtype=np.float64)
        mean = np.asarray(self.tica["feature_mean"], dtype=np.float64)
        components = np.asarray(self.tica["tica_components"], dtype=np.float64)
        return float((feature - mean) @ components[:, self.tica_component])

    def evaluate(self, candidate_pdb: str | Path) -> CVEvaluation:
        candidate = read_pdb(candidate_pdb)
        pairs = sequence_alignment_pairs(sequence(candidate), self.start_sequence)
        masked_pairs = [(i, j) for i, j in pairs if j in set(self.start_mask)]
        candidate_ca, start_ca, valid_start_pairs = self._mapped_ca(candidate, self.start, masked_pairs)
        rmsd_start = kabsch_rmsd(candidate_ca, start_ca)
        candidate_to_start = dict(pairs)

        rmsd_target = None
        n_aligned_target = 0
        if self.target is not None:
            candidate_target_pairs = [
                (candidate_i, self.start_to_target[start_i])
                for candidate_i, start_i in pairs
                if start_i in self.start_to_target and start_i in set(self.start_mask)
            ]
            candidate_target_ca, target_ca, valid_target_pairs = self._mapped_ca(
                candidate, self.target, candidate_target_pairs
            )
            rmsd_target = kabsch_rmsd(candidate_target_ca, target_ca)
            n_aligned_target = len(valid_target_pairs)

        all_ca = np.stack([residue.atoms["CA"] for residue in candidate if "CA" in residue.atoms])
        radius_gyration = float(
            np.sqrt(np.mean(np.sum((all_ca - all_ca.mean(axis=0)) ** 2, axis=1)))
        )
        start_to_candidate = {start_i: candidate_i for candidate_i, start_i in pairs}
        retained = 0
        possible = 0
        for start_i, start_j in self.start_contact_pairs:
            if int(start_i) not in start_to_candidate or int(start_j) not in start_to_candidate:
                continue
            atom_i = candidate[start_to_candidate[int(start_i)]].atoms.get("CA")
            atom_j = candidate[start_to_candidate[int(start_j)]].atoms.get("CA")
            if atom_i is None or atom_j is None:
                continue
            possible += 1
            retained += int(np.linalg.norm(atom_i - atom_j) <= 8.0)
        q = float(retained / possible) if possible else float("nan")

        if self.mode == "start_rmsd":
            control = rmsd_start
        elif self.mode == "target_rmsd":
            control = float(rmsd_target)
        elif self.mode == "delta_rmsd":
            control = rmsd_start - float(rmsd_target)
        elif self.mode == "radius_gyration":
            control = radius_gyration
        elif self.mode == "native_contacts":
            control = q
        else:
            control = self._tica_value(candidate, candidate_to_start)
        return CVEvaluation(
            control=control,
            rmsd_start=rmsd_start,
            rmsd_target=rmsd_target,
            radius_gyration=radius_gyration,
            native_contact_fraction=q,
            n_aligned_start=len(valid_start_pairs),
            n_aligned_target=n_aligned_target,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "start_pdb": str(self.start_pdb),
            "target_loaded": self.target is not None,
            "target_pdb": None if self.target_pdb is None else str(self.target_pdb),
            "masked_residues": len(self.start_mask),
            "tica_component": self.tica_component if self.mode == "tica" else None,
        }
