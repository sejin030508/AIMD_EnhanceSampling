"""Continuous endpoint potentials in frozen published TICA coordinates."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from confmh.duet.observables import ATOM37_INDEX, atom37_coordinates_a


CA_INDEX = ATOM37_INDEX["CA"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _topology_atom37_map(
    topology: Any, residue_count: int
) -> list[tuple[int, int, int]]:
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


class _PublishedCADistanceProjection:
    feature_kind = "all_upper_triangle_pairwise_ca_distances_a"
    distance_unit = "TICA coordinate"

    def __init__(
        self, manifest: Mapping[str, Any], projection_path: str | Path,
        dimensions: int,
    ) -> None:
        self.path = Path(projection_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"TICA projection asset not found: {self.path}")
        with np.load(self.path, allow_pickle=False) as payload:
            required = {
                "feature_mean", "components", "eigenvalues", "pair_i", "pair_j",
                "model_length", "source_artifact_sha256", "kinetic_map",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"TICA projection asset missing keys: {sorted(missing)}")
            self.mean = np.asarray(payload["feature_mean"], dtype=np.float64)
            components = np.asarray(payload["components"], dtype=np.float64)
            self.eigenvalues = np.asarray(payload["eigenvalues"], dtype=np.float64)
            self.pair_i = np.asarray(payload["pair_i"], dtype=np.int64)
            self.pair_j = np.asarray(payload["pair_j"], dtype=np.int64)
            self.model_length = int(np.asarray(payload["model_length"]).item())
            self.source_artifact_sha256 = str(
                np.asarray(payload["source_artifact_sha256"]).item()
            )
            self.kinetic_map = bool(np.asarray(payload["kinetic_map"]).item())
        expected_length = int(manifest["model_length"])
        expected_features = expected_length * (expected_length - 1) // 2
        if self.model_length != expected_length:
            raise ValueError(
                f"TICA model length {self.model_length} != manifest {expected_length}"
            )
        if self.mean.shape != (expected_features,):
            raise ValueError("TICA mean shape does not match CA-pair feature count")
        if components.ndim != 2 or components.shape[0] != expected_features:
            raise ValueError(f"Invalid TICA component shape: {components.shape}")
        if not 1 <= dimensions <= components.shape[1]:
            raise ValueError(
                f"Requested {dimensions} TICA dimensions, asset has {components.shape[1]}"
            )
        self.components = components[:, :dimensions]
        self.dimensions = int(dimensions)
        expected_i, expected_j = np.triu_indices(expected_length, 1)
        if not (
            np.array_equal(self.pair_i, expected_i)
            and np.array_equal(self.pair_j, expected_j)
        ):
            raise ValueError("TICA CA-pair ordering differs from numpy upper triangle")
        if not self.kinetic_map:
            raise ValueError("Published TICA projection must preserve kinetic-map scaling")
        source = Path(str(manifest["tica"]["artifact"])).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Published TICA source artifact not found: {source}")
        if _sha256(source) != self.source_artifact_sha256:
            raise ValueError("Frozen TICA transform/source hash mismatch")

    def project_atom37(self, coordinates_a: np.ndarray, mask: np.ndarray) -> np.ndarray:
        coords = np.asarray(coordinates_a, dtype=np.float64)
        available = np.asarray(mask, dtype=bool)
        if coords.shape != (self.model_length, 37, 3):
            raise ValueError("atom37 coordinate shape does not match TICA model")
        if available.shape != (self.model_length, 37):
            raise ValueError("atom37 mask shape does not match TICA model")
        if not np.all(available[:, CA_INDEX]):
            raise ValueError("Generated frame is missing a C-alpha atom")
        ca = coords[:, CA_INDEX]
        features = np.linalg.norm(ca[self.pair_i] - ca[self.pair_j], axis=-1)
        if not np.all(np.isfinite(features)):
            raise ValueError("Non-finite CA-distance feature")
        return (features - self.mean) @ self.components


class _OfficialTorsionProjection:
    feature_kind = "backbone_torsion_cossin"
    distance_unit = "TICA coordinate"

    def __init__(self, manifest: Mapping[str, Any], dimensions: int) -> None:
        import joblib
        import mdtraj as md
        import pyemma.coordinates as coor

        sources = {
            Path(row["path"]).name: Path(row["path"])
            for row in manifest["official_source_files"]
        }
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
        self._mapping = _topology_atom37_map(
            self._topology, int(manifest["model_length"])
        )
        self._backbone = [ATOM37_INDEX[name] for name in ("N", "CA", "C")]
        self.dimensions = int(dimensions)

    def project_atom37(self, coordinates_a: np.ndarray, mask: np.ndarray) -> np.ndarray:
        coords = np.asarray(coordinates_a, dtype=float)
        available = np.asarray(mask, dtype=bool)
        if not np.all(available[:, self._backbone]):
            raise ValueError("Generated frame is missing an N/CA/C atom")
        xyz = self._template_xyz_nm.copy()
        for topology_atom, residue, atom37 in self._mapping:
            if available[residue, atom37]:
                xyz[topology_atom] = coords[residue, atom37] / 10.0
        trajectory = self._md.Trajectory(xyz[None], self._topology)
        transformed = np.asarray(
            self._model.transform(self._feature.transform(trajectory))
        )
        if transformed.shape[1] < self.dimensions:
            raise ValueError("TICA transform has too few dimensions")
        return transformed[0, : self.dimensions]


class TicaEndpointMetric:
    """Distance to the folded endpoint in one fixed TICA projection."""

    def __init__(
        self, manifest: Mapping[str, Any], *,
        projection_path: str | Path | None = None, dimensions: int = 2,
    ) -> None:
        self.dimensions = int(dimensions)
        if self.dimensions < 1:
            raise ValueError("TICA dimensions must be positive")
        if projection_path is not None:
            self._projection = _PublishedCADistanceProjection(
                manifest, projection_path, self.dimensions
            )
            reference_path = Path(str(manifest["reference_npz"])).expanduser().resolve()
            with np.load(reference_path, allow_pickle=False) as payload:
                target_coordinates = np.asarray(payload["reference_atom37_a"])[0]
                target_mask = np.asarray(payload["reference_atom37_mask"])[0]
            self._target = self._projection.project_atom37(
                target_coordinates, target_mask
            )
        else:
            self._projection = _OfficialTorsionProjection(manifest, self.dimensions)
            target = np.asarray(manifest["tica"]["target_first_two"], dtype=float)
            if target.shape[0] < self.dimensions:
                raise ValueError("Manifest TICA target has too few dimensions")
            self._target = target[: self.dimensions]
        if self._target.shape != (self.dimensions,) or not np.all(
            np.isfinite(self._target)
        ):
            raise ValueError(f"Invalid folded TICA target: {self._target}")

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def feature_kind(self) -> str:
        return self._projection.feature_kind

    @property
    def distance_unit(self) -> str:
        return self._projection.distance_unit

    @property
    def projection_path(self) -> str | None:
        path = getattr(self._projection, "path", None)
        return None if path is None else str(path)

    def project_atom37(self, coordinates_a: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._projection.project_atom37(coordinates_a, mask)

    def project(self, frame: Any) -> np.ndarray:
        coordinates, mask = atom37_coordinates_a(frame)
        return self.project_atom37(coordinates, mask)

    def distance_a(self, frame: Any) -> float:
        return float(np.linalg.norm(self.project(frame) - self._target))


class TicaEndpointPotential:
    """Continuous reward ``log psi = -a * (TICA distance / d0)^2``."""

    def __init__(
        self, metric: TicaEndpointMetric, d0: float, *,
        coefficient: float = 16.0, log_floor: float | None = None,
    ) -> None:
        self.metric = metric
        self.d0 = float(d0)
        self.d0_a = self.d0
        self.coefficient = float(coefficient)
        self.log_floor = None if log_floor is None else float(log_floor)
        if self.d0 <= 0.0 or self.coefficient <= 0.0:
            raise ValueError("d0 and coefficient must be positive")
        if self.log_floor is not None and self.log_floor >= 0.0:
            raise ValueError("log_floor must be negative or None")
        self.evaluations = 0
        self.clipped_evaluations = 0

    def values(self, frame: Any) -> dict[str, float]:
        distance = self.metric.distance_a(frame)
        ratio = distance / self.d0
        raw = -self.coefficient * ratio * ratio
        clipped = self.log_floor is not None and raw < self.log_floor
        return {
            "reward_distance": float(distance),
            "reward_distance_over_d0": float(ratio),
            "endpoint_distance_a": float(distance),
            "endpoint_distance_over_d0": float(ratio),
            "potential_was_clipped": float(clipped),
        }

    def log_psi(
        self, history: Sequence[Any], state: Any, t: int,
        values: Mapping[str, float] | None = None,
    ) -> float:
        del state, t
        self.evaluations += 1
        if values is None:
            values = self.values(history[-1])
        ratio = float(values["reward_distance_over_d0"])
        raw = -self.coefficient * ratio * ratio
        if self.log_floor is not None and raw < self.log_floor:
            self.clipped_evaluations += 1
            return float(self.log_floor)
        return float(raw)

    def candidate_log_psi(
        self, parent_history: Sequence[Any], parent_state: Any,
        frame: Any, t: int,
    ) -> tuple[float, Any, dict[str, float]]:
        values = self.values(frame)
        history = list(parent_history) + [frame]
        return self.log_psi(history, parent_state, t, values), parent_state, values
