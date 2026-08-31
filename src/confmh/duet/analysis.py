from __future__ import annotations

from typing import Sequence

import numpy as np


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)
    left = np.sum(np.where(p > 0, p * np.log(p / midpoint), 0.0))
    right = np.sum(np.where(q > 0, q * np.log(q / midpoint), 0.0))
    return float(0.5 * (left + right))


def paired_bootstrap_ci(
    differences: Sequence[float], confidence: float = 0.95, replicates: int = 10000, seed: int = 0
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.mean(rng.choice(values, size=(int(replicates), len(values)), replace=True), axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def path_pairwise_diversity(paths: Sequence[np.ndarray]) -> float:
    if len(paths) < 2:
        return 0.0
    distances = []
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            a, b = np.asarray(paths[left], dtype=float), np.asarray(paths[right], dtype=float)
            distances.append(float(np.mean(np.linalg.norm(a - b, axis=-1))))
    return float(np.mean(distances))

