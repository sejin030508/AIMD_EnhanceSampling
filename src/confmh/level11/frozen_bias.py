from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde


@dataclass
class HarmonicPotential:
    center: float
    kappa_kj_mol: float

    def energy(self, value: float | np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        return 0.5 * self.kappa_kj_mol * (value - self.center) ** 2

    def torch_energy(self, value):
        return 0.5 * self.kappa_kj_mol * (value - self.center).square()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, kind="level11_harmonic", center=self.center, kappa_kj_mol=self.kappa_kj_mol
        )
        return path


@dataclass
class FrozenGridBias:
    grid: np.ndarray
    bias_grid_kj_mol: np.ndarray
    bias_factor: float
    barrier_kj_mol: float

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=float)
        self.bias_grid_kj_mol = np.asarray(self.bias_grid_kj_mol, dtype=float)
        if self.grid.ndim != 1 or self.bias_grid_kj_mol.shape != self.grid.shape:
            raise ValueError("grid and bias_grid_kj_mol must be same-length 1D arrays")
        if len(self.grid) < 2 or not np.all(np.diff(self.grid) > 0):
            raise ValueError("grid must be strictly increasing")

    def energy(self, value: float | np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(value, dtype=float),
            self.grid,
            self.bias_grid_kj_mol,
            left=self.bias_grid_kj_mol[0],
            right=self.bias_grid_kj_mol[-1],
        )

    def torch_energy(self, value):
        import torch

        grid = torch.as_tensor(self.grid, dtype=value.dtype, device=value.device)
        bias = torch.as_tensor(self.bias_grid_kj_mol, dtype=value.dtype, device=value.device)
        clipped = value.clamp(min=grid[0], max=grid[-1])
        upper = torch.searchsorted(grid, clipped.contiguous(), right=True).clamp(1, len(self.grid) - 1)
        lower = upper - 1
        x0 = grid[lower]
        x1 = grid[upper]
        y0 = bias[lower]
        y1 = bias[upper]
        fraction = (clipped - x0) / (x1 - x0)
        return y0 + fraction * (y1 - y0)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            kind="level11_frozen_grid",
            grid=self.grid,
            bias_grid_kj_mol=self.bias_grid_kj_mol,
            bias_factor=self.bias_factor,
            barrier_kj_mol=self.barrier_kj_mol,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FrozenGridBias":
        data = np.load(path)
        return cls(
            grid=np.asarray(data["grid"]),
            bias_grid_kj_mol=np.asarray(data["bias_grid_kj_mol"]),
            bias_factor=float(data["bias_factor"]),
            barrier_kj_mol=float(data["barrier_kj_mol"]),
        )

    @classmethod
    def from_reference(
        cls,
        reference_cv: np.ndarray,
        *,
        kbt_kj_mol: float,
        bias_factor: float = 10.0,
        barrier_kj_mol: float = 15.0,
        n_grid: int = 401,
        margin: float = 0.5,
        bandwidth: float | None = None,
    ) -> "FrozenGridBias":
        values = np.asarray(reference_cv, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if len(values) < 20:
            raise ValueError("At least 20 finite reference CV values are required")
        if bias_factor <= 1:
            raise ValueError("bias_factor must be > 1")
        lower, upper = np.quantile(values, [0.005, 0.995])
        grid = np.linspace(lower - margin, upper + margin, int(n_grid))
        kde = gaussian_kde(values, bw_method=bandwidth)
        density = np.maximum(kde(grid), np.finfo(float).tiny)
        free_energy = -float(kbt_kj_mol) * np.log(density / density.max())
        scale = 1.0 - 1.0 / float(bias_factor)
        raw_bias = -scale * free_energy
        bias = np.clip(raw_bias, -float(barrier_kj_mol), 0.0)
        return cls(grid, bias, float(bias_factor), float(barrier_kj_mol))
