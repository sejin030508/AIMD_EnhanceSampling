from __future__ import annotations

import math

import numpy as np


def bias_only_log_alpha(delta_bias_kj_mol: float, kbt_kj_mol: float) -> float:
    """Return log min(1, exp[-beta * delta_bias])."""
    return min(0.0, -float(delta_bias_kj_mol) / float(kbt_kj_mol))


def accept(log_alpha: float, rng: np.random.Generator) -> bool:
    if log_alpha >= 0.0:
        return True
    return math.log(max(float(rng.random()), np.finfo(float).tiny)) < log_alpha


def acceptance_probability(log_alpha: float) -> float:
    return float(math.exp(min(0.0, float(log_alpha))))
