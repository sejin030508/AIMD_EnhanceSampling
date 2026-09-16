from __future__ import annotations

"""Differentiable copies of the frozen TICA endpoint potentials.

The production evaluator deliberately remains NumPy/MDTraj based.  This module
only mirrors its already-frozen feature map in Torch so a sampler can
differentiate the same scalar potential with respect to Cartesian coordinates.
"""

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def _check_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite {name}")


class TorchTicaEndpointPotential(nn.Module):
    """Frozen, differentiable TICA projection and endpoint log potential.

    Coordinates use the flattened heavy-atom order of the backend topology and
    Angstrom units.  ``kind`` is either ``ca_distance`` or ``torsion``.
    """

    def __init__(
        self,
        *,
        atom_count: int,
        kind: str,
        indices: np.ndarray,
        mean: np.ndarray,
        components: np.ndarray,
        target: np.ndarray,
        coefficient: float,
        d0: float,
        log_floor: float | None = None,
    ) -> None:
        super().__init__()
        if kind not in {"ca_distance", "torsion"}:
            raise ValueError(f"Unsupported differentiable TICA feature: {kind}")
        if atom_count < 1:
            raise ValueError("atom_count must be positive")
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("coefficient must be finite and positive")
        if not math.isfinite(d0) or d0 <= 0.0:
            raise ValueError("d0 must be finite and positive")
        if log_floor is not None and (not math.isfinite(log_floor) or log_floor >= 0.0):
            raise ValueError("log_floor must be finite and negative or None")

        index_array = np.asarray(indices, dtype=np.int64)
        expected_width = 2 if kind == "ca_distance" else 4
        if index_array.ndim != 2 or index_array.shape[1] != expected_width:
            raise ValueError(
                f"{kind} indices must have shape [features,{expected_width}]"
            )
        if np.any(index_array < 0) or np.any(index_array >= atom_count):
            raise ValueError("TICA atom indices fall outside the backend topology")

        mean_array = np.asarray(mean, dtype=np.float64)
        component_array = np.asarray(components, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        if mean_array.ndim != 1:
            raise ValueError("TICA mean must be one-dimensional")
        if component_array.ndim != 2 or component_array.shape[0] != len(mean_array):
            raise ValueError("TICA component/mean shape mismatch")
        if target_array.shape != (component_array.shape[1],):
            raise ValueError("TICA target/component shape mismatch")
        expected_features = len(index_array) if kind == "ca_distance" else 2 * len(index_array)
        if len(mean_array) != expected_features:
            raise ValueError(
                f"TICA feature width {len(mean_array)} != expected {expected_features}"
            )

        self.atom_count = int(atom_count)
        self.kind = str(kind)
        self.coefficient = float(coefficient)
        self.d0 = float(d0)
        self.log_floor = None if log_floor is None else float(log_floor)
        self.register_buffer("indices", torch.as_tensor(index_array, dtype=torch.long))
        self.register_buffer("mean", torch.as_tensor(mean_array, dtype=torch.float64))
        self.register_buffer(
            "components", torch.as_tensor(component_array, dtype=torch.float64)
        )
        self.register_buffer("target", torch.as_tensor(target_array, dtype=torch.float64))

    @property
    def dimensions(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def from_metric(
        cls,
        metric: Any,
        backend_topology: Any,
        *,
        coefficient: float,
        d0: float,
        log_floor: float | None = None,
    ) -> TorchTicaEndpointPotential:
        """Build from the exact frozen objects used by ``TicaEndpointMetric``.

        This intentionally validates private upstream objects instead of
        guessing torsion order or kinetic-map scaling.  Any unsupported asset
        layout fails closed.
        """

        projection = getattr(metric, "_projection", None)
        if projection is None:
            raise TypeError("TICA metric does not expose its frozen projection")
        atom_count = int(backend_topology.n_atoms)
        backend_by_key = {
            (atom.residue.index, atom.name): atom.index
            for atom in backend_topology.atoms
        }

        if all(hasattr(projection, name) for name in ("pair_i", "pair_j", "mean", "components")):
            ca_indices = []
            for residue in backend_topology.residues:
                matches = [atom.index for atom in residue.atoms if atom.name == "CA"]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one CA atom for residue {residue.index}, found {len(matches)}"
                    )
                ca_indices.append(matches[0])
            ca_indices = np.asarray(ca_indices, dtype=np.int64)
            pair_i = np.asarray(projection.pair_i, dtype=np.int64)
            pair_j = np.asarray(projection.pair_j, dtype=np.int64)
            indices = np.stack((ca_indices[pair_i], ca_indices[pair_j]), axis=-1)
            components = np.asarray(projection.components, dtype=np.float64)
            return cls(
                atom_count=atom_count,
                kind="ca_distance",
                indices=indices,
                mean=np.asarray(projection.mean, dtype=np.float64),
                components=components[:, : int(metric.dimensions)],
                target=np.asarray(metric.target, dtype=np.float64),
                coefficient=coefficient,
                d0=d0,
                log_floor=log_floor,
            )

        feature = getattr(projection, "_feature", None)
        estimator = getattr(projection, "_model", None)
        official_topology = getattr(projection, "_topology", None)
        if feature is None or estimator is None or official_topology is None:
            raise TypeError("Unsupported frozen TICA projection implementation")
        active = list(getattr(feature, "active_features", ()))
        if len(active) != 1 or not hasattr(active[0], "angle_indexes"):
            raise ValueError("Expected one explicit PyEMMA backbone-torsion feature")
        official_indices = np.asarray(active[0].angle_indexes, dtype=np.int64)
        mapped = np.empty_like(official_indices)
        official_atoms = list(official_topology.atoms)
        for row in range(official_indices.shape[0]):
            for column in range(4):
                atom = official_atoms[int(official_indices[row, column])]
                key = (atom.residue.index, atom.name)
                if key not in backend_by_key:
                    raise ValueError(f"Backend topology is missing TICA atom {key}")
                mapped[row, column] = backend_by_key[key]

        fitted = estimator.fetch_model()
        mean = np.asarray(fitted.mean_0, dtype=np.float64)
        components = np.asarray(
            fitted.instantaneous_coefficients[:, : int(metric.dimensions)],
            dtype=np.float64,
        )
        return cls(
            atom_count=atom_count,
            kind="torsion",
            indices=mapped,
            mean=mean,
            components=components,
            target=np.asarray(metric.target, dtype=np.float64),
            coefficient=coefficient,
            d0=d0,
            log_floor=log_floor,
        )

    def features(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 3 or coordinates.shape[1:] != (self.atom_count, 3):
            raise ValueError(
                f"coordinates must have shape [B,{self.atom_count},3], got "
                f"{tuple(coordinates.shape)}"
            )
        xyz = coordinates.to(torch.float64)
        selected = xyz[:, self.indices]
        if self.kind == "ca_distance":
            features = torch.linalg.vector_norm(
                selected[:, :, 0] - selected[:, :, 1], dim=-1
            )
        else:
            # This is the exact convention used by MDTraj/PyEMMA's backbone
            # torsion cossin feature, verified against the frozen evaluator.
            b0 = selected[:, :, 0] - selected[:, :, 1]
            b1 = selected[:, :, 2] - selected[:, :, 1]
            b2 = selected[:, :, 3] - selected[:, :, 2]
            norm_b1 = torch.linalg.vector_norm(b1, dim=-1, keepdim=True)
            if bool((norm_b1 <= 1.0e-12).any()):
                raise FloatingPointError("Undefined torsion: zero central bond")
            axis = b1 / norm_b1
            v = b0 - (b0 * axis).sum(-1, keepdim=True) * axis
            w = b2 - (b2 * axis).sum(-1, keepdim=True) * axis
            cosine_numerator = (v * w).sum(-1)
            sine_numerator = (torch.linalg.cross(axis, v, dim=-1) * w).sum(-1)
            normalizer = torch.sqrt(cosine_numerator.square() + sine_numerator.square())
            if bool((normalizer <= 1.0e-12).any()):
                raise FloatingPointError("Undefined torsion: collinear backbone atoms")
            features = torch.stack(
                (cosine_numerator / normalizer, sine_numerator / normalizer), dim=-1
            ).flatten(1)
        _check_finite(features, "TICA features")
        return features

    def project(self, coordinates: Tensor) -> Tensor:
        projected = (self.features(coordinates) - self.mean) @ self.components
        _check_finite(projected, "TICA coordinates")
        return projected

    def forward(self, coordinates: Tensor) -> Tensor:
        projected = self.project(coordinates)
        log_psi = -(
            self.coefficient / (self.d0 * self.d0)
        ) * (projected - self.target).square().sum(-1)
        if self.log_floor is not None:
            log_psi = torch.clamp_min(log_psi, self.log_floor)
        _check_finite(log_psi, "TICA log potential")
        return log_psi

    def forward_flat(self, flat_coordinates: Tensor, count: int) -> Tensor:
        if count < 1 or flat_coordinates.shape != (count * self.atom_count, 3):
            raise ValueError(
                f"flat coordinates must have shape {(count * self.atom_count, 3)}"
            )
        return self(flat_coordinates.reshape(count, self.atom_count, 3))
