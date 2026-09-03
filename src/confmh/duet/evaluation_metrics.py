from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from confmh.duet.resampling import effective_sample_size, normalize_log_weights


def reconstruct_outer_weight_history(
    records: Sequence[Mapping[str, Any]],
    *,
    outer_k: int,
    horizon: int,
    method: str,
    resampling_ess_fraction: float,
) -> list[dict[str, Any]]:
    """Reconstruct weights immediately before each outer resampling decision."""
    k = int(outer_k)
    horizon = int(horizon)
    if k < 1 or horizon < 1:
        raise ValueError("outer_k and horizon must be positive")
    if not 0.0 < float(resampling_ess_fraction) <= 1.0:
        raise ValueError("resampling_ess_fraction must be in (0, 1]")

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["t"])].append(record)
    log_weights = np.full(k, -np.log(k), dtype=float)
    weighted_methods = method not in {"frozen", "inner_only"}
    history: list[dict[str, Any]] = []
    for t in range(1, horizon + 1):
        rows = sorted(grouped[t], key=lambda row: int(row["particle"]))
        if len(rows) != k:
            raise ValueError(f"Expected {k} records at t={t}, found {len(rows)}")
        increments = np.asarray([float(row["log_increment"]) for row in rows])
        child_log_weights = log_weights + increments
        weights, _ = normalize_log_weights(child_log_weights)
        ess = effective_sample_size(weights)
        resampled = bool(
            weighted_methods
            and (ess <= float(resampling_ess_fraction) * k or t == horizon)
        )
        history.append(
            {
                "t": t,
                "weights": weights.tolist(),
                "ess": float(ess if weighted_methods else k),
                "resampled": resampled,
            }
        )
        if resampled:
            log_weights = np.full(k, -np.log(k), dtype=float)
        else:
            log_weights = np.log(weights)
    return history


def summarize_pre_resampling_population(
    records: Sequence[Mapping[str, Any]],
    *,
    outer_k: int,
    horizon: int,
    method: str,
    resampling_ess_fraction: float,
    required_success_stage: int,
    final_success_flags: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Compute stable success metrics before output-duplication resampling."""
    weight_history = reconstruct_outer_weight_history(
        records,
        outer_k=outer_k,
        horizon=horizon,
        method=method,
        resampling_ess_fraction=resampling_ess_fraction,
    )
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["t"])].append(record)

    curves: list[dict[str, Any]] = []
    for item in weight_history:
        t = int(item["t"])
        rows = sorted(grouped[t], key=lambda row: int(row["particle"]))
        weights = np.asarray(item["weights"], dtype=float)
        stages = np.asarray([int(row["progress_stage"]) for row in rows])
        failed = np.asarray([bool(row["progress_failed"]) for row in rows])
        stage_fraction = {}
        weighted_stage_mass = {}
        for stage in range(1, int(required_success_stage) + 1):
            reached = (stages >= stage) & ~failed
            stage_fraction[str(stage)] = float(np.mean(reached))
            weighted_stage_mass[str(stage)] = float(np.dot(weights, reached))
        curves.append(
            {
                **item,
                "stage_fraction": stage_fraction,
                "weighted_stage_mass": weighted_stage_mass,
                "failure_fraction": float(np.mean(failed)),
                "weighted_failure_mass": float(np.dot(weights, failed)),
            }
        )

    final_rows = sorted(grouped[int(horizon)], key=lambda row: int(row["particle"]))
    final_weights = np.asarray(weight_history[-1]["weights"], dtype=float)
    if final_success_flags is None:
        success = np.asarray(
            [
                int(row["progress_stage"]) >= int(required_success_stage)
                and not bool(row["progress_failed"])
                for row in final_rows
            ],
            dtype=bool,
        )
    else:
        success = np.asarray(final_success_flags, dtype=bool)
        if success.shape != (int(outer_k),):
            raise ValueError("final_success_flags must contain one flag per outer particle")
    success_mass = float(np.dot(final_weights, success))
    preterminal_resampling_steps = [
        int(item["t"])
        for item in weight_history[:-1]
        if bool(item["resampled"])
    ]
    return {
        "pre_resampling_success_flags": success.tolist(),
        "pre_resampling_unique_success_count": int(np.sum(success)),
        "pre_resampling_unique_success_rate": float(np.mean(success)),
        "pre_resampling_success_weight_mass": success_mass,
        "expected_success_copies_after_final_resampling": float(outer_k * success_mass),
        "pre_resampling_final_weights": final_weights.tolist(),
        "preterminal_outer_resampling_count": len(preterminal_resampling_steps),
        "preterminal_outer_resampling_steps": preterminal_resampling_steps,
        "event_progress_curves": curves,
    }
