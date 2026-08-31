from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(log_weights, dtype=float)
    maximum = float(np.max(values))
    shifted = np.exp(values - maximum)
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Non-finite or zero particle weight total")
    return shifted / total, maximum + float(np.log(total))


def logmeanexp(log_values: np.ndarray) -> float:
    _, log_total = normalize_log_weights(np.asarray(log_values, dtype=float))
    return float(log_total - np.log(len(log_values)))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights**2))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n = len(weights) if n is None else int(n)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right").astype(int)


def multinomial_resample(weights: np.ndarray, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    n = len(weights) if n is None else int(n)
    weights = np.asarray(weights, dtype=float)
    return rng.choice(len(weights), size=n, replace=True, p=weights / weights.sum()).astype(int)


def gather_state(value: Any, indices: np.ndarray) -> Any:
    """Recursively gather every batched component of a particle state."""
    if hasattr(value, "index_select") and value.__class__.__module__.startswith("torch"):
        import torch

        index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    if isinstance(value, np.ndarray):
        return value[np.asarray(indices, dtype=int)].copy()
    if is_dataclass(value):
        return value.__class__(**{field.name: gather_state(getattr(value, field.name), indices) for field in fields(value)})
    if isinstance(value, dict):
        return {key: gather_state(item, indices) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(gather_state(item, indices) for item in value)
    if isinstance(value, list):
        if len(value) and len(value) >= int(np.max(indices)) + 1:
            return [copy.deepcopy(value[int(index)]) for index in indices]
        return [gather_state(item, indices) for item in value]
    return copy.deepcopy(value)

