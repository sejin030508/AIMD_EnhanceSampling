from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.special import logsumexp


class Bias(Protocol):
    adaptive: bool

    def energy(self, value: float | np.ndarray) -> np.ndarray: ...
    def observe(self, value: float, energy_kj_mol: float, step: int) -> None: ...
    def maybe_update(self, step: int) -> bool: ...
    @property
    def frozen(self) -> bool: ...
    def save(self, path: str | Path) -> Path: ...


@dataclass
class ZeroBias:
    adaptive: bool = False

    def energy(self, value: float | np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(value, dtype=float))

    def observe(self, value: float, energy_kj_mol: float, step: int) -> None:
        del value, energy_kj_mol, step

    def maybe_update(self, step: int) -> bool:
        del step
        return False

    @property
    def frozen(self) -> bool:
        return True

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(path, kind="zero")
        return path


@dataclass
class HarmonicBias:
    center: float
    kappa_kj_mol: float
    adaptive: bool = False

    def energy(self, value: float | np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        return 0.5 * self.kappa_kj_mol * (value - self.center) ** 2

    def observe(self, value: float, energy_kj_mol: float, step: int) -> None:
        del value, energy_kj_mol, step

    def maybe_update(self, step: int) -> bool:
        del step
        return False

    @property
    def frozen(self) -> bool:
        return True

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(
            path,
            kind="harmonic",
            center=self.center,
            kappa_kj_mol=self.kappa_kj_mol,
        )
        return path


@dataclass
class OPES1D:
    """Small fixed-grid implementation of the OPES_METAD core relation.

    It uses reweighted fixed-bandwidth KDE and freezes after `adapt_steps`.
    This is intentionally compact; PLUMED remains the conventional-MD oracle.
    """

    grid_min: float
    grid_max: float
    n_grid: int
    bandwidth: float
    kbt_kj_mol: float
    bias_factor: float = 10.0
    barrier_kj_mol: float = 15.0
    update_interval: int = 100
    adapt_steps: int = 2000
    max_samples: int = 10000
    adaptive: bool = True
    _grid: np.ndarray = field(init=False)
    _bias_grid: np.ndarray = field(init=False)
    _samples: list[float] = field(default_factory=list)
    _sample_biases: list[float] = field(default_factory=list)
    _is_frozen: bool = False

    def __post_init__(self):
        if not self.grid_min < self.grid_max:
            raise ValueError("grid_min must be smaller than grid_max")
        if self.n_grid < 16 or self.bandwidth <= 0:
            raise ValueError("n_grid must be >=16 and bandwidth must be positive")
        if self.bias_factor <= 1:
            raise ValueError("bias_factor must be >1")
        self._grid = np.linspace(self.grid_min, self.grid_max, self.n_grid)
        self._bias_grid = np.zeros_like(self._grid)

    @property
    def frozen(self) -> bool:
        return self._is_frozen

    @property
    def grid(self) -> np.ndarray:
        return self._grid.copy()

    @property
    def bias_grid(self) -> np.ndarray:
        return self._bias_grid.copy()

    def energy(self, value: float | np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        return np.interp(value, self._grid, self._bias_grid, left=self._bias_grid[0], right=self._bias_grid[-1])

    def observe(self, value: float, energy_kj_mol: float, step: int) -> None:
        del step
        if self._is_frozen:
            return
        self._samples.append(float(value))
        self._sample_biases.append(float(energy_kj_mol))
        if len(self._samples) > self.max_samples:
            self._samples = self._samples[-self.max_samples :]
            self._sample_biases = self._sample_biases[-self.max_samples :]

    def _update(self) -> None:
        if len(self._samples) < max(20, self.update_interval // 2):
            return
        samples = np.asarray(self._samples, dtype=float)
        log_weights = np.asarray(self._sample_biases, dtype=float) / self.kbt_kj_mol
        delta = (self._grid[:, None] - samples[None, :]) / self.bandwidth
        log_kernel = -0.5 * delta**2 - np.log(self.bandwidth * np.sqrt(2.0 * np.pi))
        log_density = logsumexp(log_kernel + log_weights[None, :], axis=1) - logsumexp(log_weights)

        scale = 1.0 - 1.0 / self.bias_factor
        log_density -= np.max(log_density)
        log_epsilon = -self.barrier_kj_mol / (scale * self.kbt_kj_mol)
        regularized = np.logaddexp(log_density, log_epsilon)
        raw_bias = scale * self.kbt_kj_mol * regularized
        raw_bias -= np.min(raw_bias)
        self._bias_grid = np.clip(raw_bias, 0.0, self.barrier_kj_mol)

    def maybe_update(self, step: int) -> bool:
        if self._is_frozen:
            return False
        changed = False
        if step > 0 and step % self.update_interval == 0:
            self._update()
            changed = True
        if step >= self.adapt_steps:
            self._update()
            self._is_frozen = True
            changed = True
        return changed

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            kind="opes1d",
            grid=self._grid,
            bias_grid=self._bias_grid,
            samples=np.asarray(self._samples),
            sample_biases=np.asarray(self._sample_biases),
            bandwidth=self.bandwidth,
            kbt_kj_mol=self.kbt_kj_mol,
            bias_factor=self.bias_factor,
            barrier_kj_mol=self.barrier_kj_mol,
            update_interval=self.update_interval,
            adapt_steps=self.adapt_steps,
            frozen=self._is_frozen,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OPES1D":
        data = np.load(path)
        grid = np.asarray(data["grid"])
        obj = cls(
            grid_min=float(grid[0]),
            grid_max=float(grid[-1]),
            n_grid=len(grid),
            bandwidth=float(data["bandwidth"]),
            kbt_kj_mol=float(data["kbt_kj_mol"]),
            bias_factor=float(data["bias_factor"]),
            barrier_kj_mol=float(data["barrier_kj_mol"]),
            update_interval=int(data["update_interval"]),
            adapt_steps=int(data["adapt_steps"]),
        )
        obj._grid = grid
        obj._bias_grid = np.asarray(data["bias_grid"])
        obj._samples = np.asarray(data["samples"]).astype(float).tolist()
        obj._sample_biases = np.asarray(data["sample_biases"]).astype(float).tolist()
        obj._is_frozen = bool(data["frozen"])
        return obj


def load_bias(path: str | Path):
    data = np.load(path)
    kind = str(data["kind"])
    if kind == "zero":
        return ZeroBias()
    if kind == "harmonic":
        return HarmonicBias(float(data["center"]), float(data["kappa_kj_mol"]))
    if kind == "opes1d":
        return OPES1D.load(path)
    raise ValueError(f"Unknown saved bias kind: {kind}")
