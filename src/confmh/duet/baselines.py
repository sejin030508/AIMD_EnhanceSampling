from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from confmh.adapters.base_iterative_frame import IterativeFrameAdapter
from confmh.duet.programs import ProgressState
from confmh.duet.resampling import logmeanexp, normalize_log_weights, systematic_resample


@dataclass
class CompleteNestedResult:
    frame: Any
    progress: ProgressState
    values: dict[str, float]
    log_psi: float
    log_z_hat: float
    selected_index: int
    candidate_log_potentials: np.ndarray


def complete_frame_nested_step(
    *,
    adapter: IterativeFrameAdapter,
    history_state: Any,
    count: int,
    seeds: Sequence[int],
    checkpoint_progress: float,
    candidate_potential: Callable[[Any], tuple[float, ProgressState, dict[str, float]]],
    rng: np.random.Generator,
) -> CompleteNestedResult:
    frames = adapter.sample_complete_frames(
        history_state, count, seeds, checkpoint_progress=checkpoint_progress
    )
    evaluated = [candidate_potential(frame) for frame in frames]
    log_psi = np.asarray([item[0] for item in evaluated], dtype=float)
    weights, _ = normalize_log_weights(log_psi)
    selected = int(systematic_resample(weights, rng, n=1)[0])
    selected_log_psi, progress, values = evaluated[selected]
    return CompleteNestedResult(
        frame=frames[selected],
        progress=progress,
        values=values,
        log_psi=float(selected_log_psi),
        log_z_hat=logmeanexp(log_psi),
        selected_index=selected,
        candidate_log_potentials=log_psi,
    )


def select_best_completed_path(
    paths: Sequence[Sequence[Any]], final_log_rewards: Sequence[float]
) -> tuple[int, Sequence[Any]]:
    if not paths:
        raise ValueError("No completed paths supplied")
    index = int(np.argmax(np.asarray(final_log_rewards, dtype=float)))
    return index, paths[index]

