from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    index: int | None
    probabilities: np.ndarray
    scores: np.ndarray
    safety_fallback: bool


def select_candidate(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    controller: str,
    rng: np.random.Generator,
    center: float = 0.0,
    kappa: float = 1.0,
    temperature: float = 1.0,
) -> SelectionResult:
    """Select a candidate without a Metropolis accept/reject correction.

    ``raw`` samples uniformly from geometry-valid candidates. ``biased`` uses a
    softmin over a dimensionless harmonic control score.  This is a steering
    controller, not a Boltzmann or MH probability.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("values and valid must be one-dimensional arrays with equal shape")
    probabilities = np.zeros(len(values), dtype=np.float64)
    scores = 0.5 * float(kappa) * (values - float(center)) ** 2
    eligible = np.flatnonzero(valid & np.isfinite(values))
    if not len(eligible):
        return SelectionResult(None, probabilities, scores, True)
    if controller == "raw":
        probabilities[eligible] = 1.0 / len(eligible)
    elif controller == "biased":
        if temperature <= 0:
            raise ValueError("selection temperature must be positive")
        logits = -scores[eligible] / float(temperature)
        logits -= logits.max()
        weights = np.exp(np.clip(logits, -80.0, 0.0))
        probabilities[eligible] = weights / weights.sum()
    else:
        raise ValueError("controller must be 'raw' or 'biased'")
    index = int(rng.choice(len(values), p=probabilities))
    return SelectionResult(index, probabilities, scores, False)
