from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from confmh.pca_cv import PCACV, _kabsch_align


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
            else:
                raise ValueError(f"Unsupported observable kind: {kind}")
        return cls(functions)

    def evaluate(self, frame: Any, names: tuple[str, ...] | list[str]) -> dict[str, float]:
        return {name: float(self.functions[name](frame)) for name in names}

