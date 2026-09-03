from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from confmh.duet.resampling import logmeanexp, normalize_log_weights


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including deterministic tie handling."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool = False) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 2:
        return None
    if ranked:
        left, right = _average_ranks(left), _average_ranks(right)
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_checkpoint_fidelity(
    checkpoint_log_psi: np.ndarray,
    child_log_psi: np.ndarray,
    *,
    checkpoint_observables: Mapping[str, np.ndarray] | None = None,
    child_observables: Mapping[str, np.ndarray] | None = None,
    child_advanced: np.ndarray | None = None,
    child_failed: np.ndarray | None = None,
    top_fraction: float = 0.25,
) -> dict[str, Any]:
    """Summarize how well predicted-clean checkpoint scores rank continuations.

    ``child_log_psi[i]`` contains independent diffusion continuations from
    checkpoint particle ``i``.  Their log-mean potential estimates the
    continuation-marginalized target for that checkpoint rather than a noisy
    single endpoint.
    """
    checkpoint = np.asarray(checkpoint_log_psi, dtype=float)
    children = np.asarray(child_log_psi, dtype=float)
    if checkpoint.ndim != 1 or children.ndim != 2:
        raise ValueError("checkpoint_log_psi must be 1-D and child_log_psi must be 2-D")
    if len(checkpoint) != children.shape[0] or children.shape[1] < 1:
        raise ValueError("Each checkpoint needs at least one child continuation")
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")

    conditional = np.asarray([logmeanexp(row) for row in children], dtype=float)
    checkpoint_weights, _ = normalize_log_weights(checkpoint)
    uniform_log_mean = logmeanexp(conditional)
    weighted_log_mean = logmeanexp(
        conditional + np.log(checkpoint_weights * len(checkpoint_weights))
    )
    oracle_log_mean = float(np.max(conditional))
    checkpoint_best = int(np.argmax(checkpoint))
    endpoint_best = int(np.argmax(conditional))
    checkpoint_order = np.argsort(-checkpoint, kind="mergesort")
    endpoint_best_rank = int(np.flatnonzero(checkpoint_order == endpoint_best)[0] + 1)
    top_count = max(1, int(np.ceil(len(checkpoint) * float(top_fraction))))
    top_checkpoint = set(np.argsort(-checkpoint, kind="mergesort")[:top_count].tolist())
    top_endpoint = set(np.argsort(-conditional, kind="mergesort")[:top_count].tolist())

    result: dict[str, Any] = {
        "checkpoint_count": int(len(checkpoint)),
        "continuations_per_checkpoint": int(children.shape[1]),
        "checkpoint_to_conditional_log_psi_pearson": _correlation(checkpoint, conditional),
        "checkpoint_to_conditional_log_psi_spearman": _correlation(
            checkpoint, conditional, ranked=True
        ),
        "conditional_log_mean_psi": conditional.tolist(),
        "mean_within_checkpoint_log_psi_std": float(np.mean(np.std(children, axis=1))),
        "between_checkpoint_conditional_log_psi_std": float(np.std(conditional)),
        "top_checkpoint_overlap_fraction": float(
            len(top_checkpoint & top_endpoint) / top_count
        ),
        "endpoint_best_rank_by_checkpoint": endpoint_best_rank,
        "checkpoint_best_index": checkpoint_best,
        "conditional_best_index": endpoint_best,
        "checkpoint_best_conditional_log_psi": float(conditional[checkpoint_best]),
        "uniform_log_expected_psi": float(uniform_log_mean),
        "checkpoint_weighted_log_expected_psi": float(weighted_log_mean),
        "oracle_log_expected_psi": oracle_log_mean,
        "checkpoint_selection_log_gain_vs_uniform": float(
            weighted_log_mean - uniform_log_mean
        ),
        "oracle_log_gain_vs_uniform": float(oracle_log_mean - uniform_log_mean),
    }

    if child_advanced is not None:
        advanced = np.asarray(child_advanced, dtype=bool)
        if advanced.shape != children.shape:
            raise ValueError("child_advanced must have the same shape as child_log_psi")
        probability = np.mean(advanced, axis=1)
        result.update(
            {
                "conditional_advance_probability": probability.tolist(),
                "checkpoint_to_advance_probability_spearman": _correlation(
                    checkpoint, probability, ranked=True
                ),
                "checkpoint_best_advance_probability": float(probability[checkpoint_best]),
                "uniform_advance_probability": float(np.mean(probability)),
                "checkpoint_weighted_advance_probability": float(
                    np.dot(checkpoint_weights, probability)
                ),
            }
        )
    if child_failed is not None:
        failed = np.asarray(child_failed, dtype=bool)
        if failed.shape != children.shape:
            raise ValueError("child_failed must have the same shape as child_log_psi")
        probability = np.mean(failed, axis=1)
        result.update(
            {
                "conditional_failure_probability": probability.tolist(),
                "uniform_failure_probability": float(np.mean(probability)),
                "checkpoint_weighted_failure_probability": float(
                    np.dot(checkpoint_weights, probability)
                ),
            }
        )

    observable_summary: dict[str, Any] = {}
    checkpoint_observables = checkpoint_observables or {}
    child_observables = child_observables or {}
    for name in sorted(set(checkpoint_observables) & set(child_observables)):
        predicted = np.asarray(checkpoint_observables[name], dtype=float)
        completed = np.asarray(child_observables[name], dtype=float)
        if predicted.shape != checkpoint.shape or completed.shape != children.shape:
            raise ValueError(f"Observable shape mismatch for {name}")
        completed_mean = np.mean(completed, axis=1)
        observable_summary[name] = {
            "checkpoint_to_child_mean_pearson": _correlation(predicted, completed_mean),
            "checkpoint_to_child_mean_spearman": _correlation(
                predicted, completed_mean, ranked=True
            ),
            "mean_absolute_prediction_error": float(
                np.mean(np.abs(predicted - completed_mean))
            ),
            "checkpoint_values": predicted.tolist(),
            "child_mean_values": completed_mean.tolist(),
        }
    result["observables"] = observable_summary
    return result
