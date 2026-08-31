from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TorchPC1:
    """Differentiable equivalent of :class:`confmh.pca_cv.PCACV` for PC1."""

    reference_ca_nm: np.ndarray
    mean_flat: np.ndarray
    component: np.ndarray
    score_mean: float
    score_scale: float

    @classmethod
    def load(cls, path: str | Path) -> "TorchPC1":
        data = np.load(path)
        components = np.asarray(data["components"], dtype=np.float64)
        if components.ndim != 2 or len(components) < 1:
            raise ValueError("PCA asset has no PC1 component")
        score_scale = float(np.asarray(data["score_scale"])[0])
        if not np.isfinite(score_scale) or score_scale <= 0:
            raise ValueError("PCA PC1 score_scale must be finite and positive")
        return cls(
            reference_ca_nm=np.asarray(data["reference_ca_nm"], dtype=np.float64),
            mean_flat=np.asarray(data["mean_flat"], dtype=np.float64),
            component=components[0],
            score_mean=float(np.asarray(data["score_mean"])[0]),
            score_scale=score_scale,
        )

    def project(self, ca_nm):
        """Project ``(..., N, 3)`` coordinates to standardized PC1."""
        import torch

        if ca_nm.shape[-2:] != self.reference_ca_nm.shape:
            raise ValueError(
                f"Expected coordinates ending in {self.reference_ca_nm.shape}; "
                f"got {tuple(ca_nm.shape)}"
            )
        dtype = ca_nm.dtype
        device = ca_nm.device
        reference = torch.as_tensor(self.reference_ca_nm, device=device, dtype=dtype)
        mean_flat = torch.as_tensor(self.mean_flat, device=device, dtype=dtype)
        component = torch.as_tensor(self.component, device=device, dtype=dtype)

        centered = ca_nm - ca_nm.mean(dim=-2, keepdim=True)
        reference_centered = reference - reference.mean(dim=-2, keepdim=True)
        covariance = centered.transpose(-1, -2) @ reference_centered
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        rotation = u @ vh
        determinant = torch.linalg.det(rotation)
        correction = torch.ones(rotation.shape[:-2] + (3,), device=device, dtype=dtype)
        correction[..., -1] = torch.where(determinant < 0, -1.0, 1.0)
        rotation = (u * correction.unsqueeze(-2)) @ vh
        aligned = centered @ rotation + reference.mean(dim=-2, keepdim=True)
        flat = aligned.reshape(aligned.shape[:-2] + (-1,))
        raw = ((flat - mean_flat) * component).sum(dim=-1)
        return (raw - self.score_mean) / self.score_scale


def _ca_residue_rows(pdb_path: str | Path) -> list[dict[str, Any]]:
    import mdtraj as md

    frame = md.load(str(pdb_path))
    indices = frame.topology.select("protein and name CA")
    rows: list[dict[str, Any]] = []
    for order, atom_index in enumerate(indices):
        atom = frame.topology.atom(int(atom_index))
        residue = atom.residue
        code = getattr(residue, "code", None) or "X"
        rows.append(
            {
                "order": order,
                "chain_index": int(residue.chain.index),
                "resSeq": int(residue.resSeq),
                "name": str(residue.name),
                "code": str(code),
            }
        )
    if not rows:
        raise ValueError(f"No protein C-alpha atoms found in {pdb_path}")
    return rows


def validate_residue_order(
    *,
    reference_topology: str | Path,
    condition_pdb: str | Path,
    expected_seqres: str,
    expected_ca_count: int,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    reference_rows = _ca_residue_rows(reference_topology)
    condition_rows = _ca_residue_rows(condition_pdb)
    reference_sequence = "".join(row["code"] for row in reference_rows)
    condition_sequence = "".join(row["code"] for row in condition_rows)
    expected = expected_seqres.strip().upper()
    errors: list[str] = []
    if len(reference_rows) != expected_ca_count:
        errors.append(
            f"reference has {len(reference_rows)} C-alpha atoms; PCA expects {expected_ca_count}"
        )
    if len(condition_rows) != expected_ca_count:
        errors.append(
            f"condition has {len(condition_rows)} C-alpha atoms; PCA expects {expected_ca_count}"
        )
    if reference_sequence != expected:
        errors.append("reference C-alpha residue order does not match system.seqres")
    if condition_sequence != expected:
        errors.append("condition C-alpha residue order does not match system.seqres")
    if reference_sequence != condition_sequence:
        errors.append("reference and condition C-alpha residue order differ")
    payload = {
        "valid": not errors,
        "errors": errors,
        "expected_seqres": expected,
        "reference_sequence": reference_sequence,
        "condition_sequence": condition_sequence,
        "reference_residues": reference_rows,
        "condition_residues": condition_rows,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if errors:
        raise ValueError("Level 1.1 residue-order validation failed: " + "; ".join(errors))
    return payload
