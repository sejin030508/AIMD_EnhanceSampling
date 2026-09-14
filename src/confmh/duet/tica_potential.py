"""Terminal potential defined on the official TICA coordinate.

The RMSD potential asks how far a frame is from the target in Cartesian
backbone space.  This one asks how far it is in the two coordinates the
benchmark actually scores, so reward and success condition are the same
object.  Everything else - proper weighting, telescoping, the outer
update - is untouched; only the scalar log psi changes.

The projection reuses the official assets: the featurizer is built on
the official folded.pdb, the transform is the official tica_model.pkl,
and the target is the manifest's tica.target_first_two.  That makes the
reward numerically identical to the quantity the evaluator reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from confmh.duet.observables import ATOM37_INDEX, atom37_coordinates_a


def _topology_atom37_map(topology, residue_count: int):
    residues = list(topology.residues)
    if len(residues) != residue_count:
        raise ValueError(f"topology residue count {len(residues)} != {residue_count}")
    mapping = []
    for atom in topology.atoms:
        if atom.element is None or atom.element.symbol.upper() == "H":
            continue
        if atom.name not in ATOM37_INDEX:
            raise ValueError(f"unsupported official heavy atom {atom.residue}:{atom.name}")
        mapping.append((atom.index, atom.residue.index, ATOM37_INDEX[atom.name]))
    return mapping


class TicaEndpointMetric:
    """Distance to the folded target in the first two official TICA coordinates."""

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        import joblib
        import mdtraj as md
        import pyemma.coordinates as coor

        sources = {Path(row["path"]).name: Path(row["path"]) for row in manifest["official_source_files"]}
        folded = sources["folded.pdb"]
        self._md = md
        self._topology = md.load(str(folded)).topology
        self._feature = coor.featurizer(str(folded))
        self._feature.add_backbone_torsions(cossin=True)
        self._model = joblib.load(manifest["tica_model"])
        template = md.load(str(manifest["minimized_target_allatom"]))
        if template.n_atoms != self._topology.n_atoms:
            raise ValueError("minimized target and official folded topology differ")
        self._template_xyz_nm = template.xyz[0].copy()
        self._mapping = _topology_atom37_map(self._topology, int(manifest["model_length"]))
        self._target = np.asarray(manifest["tica"]["target_first_two"], dtype=float)
        # Backbone atoms must be present for a torsion to exist at all.
        self._backbone = [ATOM37_INDEX[name] for name in ("N", "CA", "C")]

    def project(self, frame: Any) -> np.ndarray:
        coords, mask = atom37_coordinates_a(frame)
        if not np.all(mask[:, self._backbone]):
            raise ValueError("Generated frame is missing an N/CA/C atom; TICA is undefined")
        xyz = self._template_xyz_nm.copy()
        for topology_atom, residue, atom37 in self._mapping:
            if mask[residue, atom37]:
                xyz[topology_atom] = coords[residue, atom37] / 10.0
        trajectory = self._md.Trajectory(xyz[None], self._topology)
        transformed = np.asarray(self._model.transform(self._feature.transform(trajectory)))
        return transformed[0, :2]

    def distance_a(self, frame: Any) -> float:
        """Named distance_a for interface parity; the unit is TICA, not Angstrom."""
        return float(np.linalg.norm(self.project(frame) - self._target))


class TicaEndpointPotential:
    """log psi = -coefficient * (tica_distance / d0)^2, same shape as the RMSD one."""

    def __init__(
        self,
        metric: TicaEndpointMetric,
        d0: float,
        *,
        coefficient: float = 16.0,
        log_floor: float | None = None,
    ) -> None:
        self.metric = metric
        self.d0_a = float(d0)
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
        # Key names match the RMSD potential so every downstream consumer
        # keeps working; the recorded numbers are TICA, not Angstrom.
        return {
            "endpoint_distance_a": float(distance),
            "endpoint_distance_over_d0": float(ratio),
            "potential_was_clipped": float(clipped),
        }

    def log_psi(
        self,
        history: Sequence[Any],
        state: Any,
        t: int,
        values: Mapping[str, float] | None = None,
    ) -> float:
        del state, t
        self.evaluations += 1
        if values is None:
            values = self.values(history[-1])
        ratio = float(values["endpoint_distance_over_d0"])
        raw = -self.coefficient * ratio * ratio
        if self.log_floor is not None and raw < self.log_floor:
            self.clipped_evaluations += 1
            return self.log_floor
        return raw

    def candidate_log_psi(
        self,
        parent_history: Sequence[Any],
        parent_state: Any,
        frame: Any,
        t: int,
    ) -> tuple[float, Any, dict[str, float]]:
        values = self.values(frame)
        history = list(parent_history) + [frame]
        return self.log_psi(history, parent_state, t, values), parent_state, values
