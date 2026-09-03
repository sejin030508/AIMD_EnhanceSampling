from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from confmh.adapters.base_iterative_frame import IterativeFrameAdapter
from confmh.duet.programs import ProgressState
from confmh.duet.resampling import (
    effective_sample_size,
    logmeanexp,
    normalize_log_weights,
    systematic_resample,
)


CandidatePotential = Callable[[Any], tuple[float, ProgressState, dict[str, float]]]


@dataclass
class InnerDiagnostics:
    log_mean_g1: float
    log_mean_g2: float
    log_z_hat: float
    first_stage_ess: float
    second_stage_ess: float
    checkpoint_ancestors: np.ndarray
    selected_index: int
    telescoping_max_abs_log_error: float
    checkpoint_log_potentials: np.ndarray
    endpoint_log_potentials: np.ndarray
    checkpoint_progresses: tuple[float, ...]
    checkpoint_ess: tuple[float, ...]
    checkpoint_ancestors_history: tuple[np.ndarray, ...]
    checkpoint_log_potentials_history: tuple[np.ndarray, ...]
    checkpoint_log_mean_increments: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_mean_g1": self.log_mean_g1,
            "log_mean_g2": self.log_mean_g2,
            "log_z_hat": self.log_z_hat,
            "first_stage_ess": self.first_stage_ess,
            "second_stage_ess": self.second_stage_ess,
            "checkpoint_ancestors": self.checkpoint_ancestors.tolist(),
            "selected_index": self.selected_index,
            "telescoping_max_abs_log_error": self.telescoping_max_abs_log_error,
            "checkpoint_log_potentials": self.checkpoint_log_potentials.tolist(),
            "endpoint_log_potentials": self.endpoint_log_potentials.tolist(),
            "checkpoint_progresses": list(self.checkpoint_progresses),
            "checkpoint_ess": list(self.checkpoint_ess),
            "checkpoint_ancestors_history": [
                item.tolist() for item in self.checkpoint_ancestors_history
            ],
            "checkpoint_log_potentials_history": [
                item.tolist() for item in self.checkpoint_log_potentials_history
            ],
            "checkpoint_log_mean_increments": list(
                self.checkpoint_log_mean_increments
            ),
        }


@dataclass
class InnerResult:
    frame: Any
    progress: ProgressState
    values: dict[str, float]
    log_psi: float
    log_z_hat: float
    diagnostics: InnerDiagnostics


def one_checkpoint_inner_step(
    *,
    adapter: IterativeFrameAdapter,
    history_state: Any,
    count: int,
    seeds: Sequence[int],
    continuation_seeds: Sequence[int],
    checkpoint_progress: float,
    candidate_potential: CandidatePotential,
    rng: np.random.Generator,
) -> InnerResult:
    """Run the one-checkpoint Feynman-Kac inner sampler exactly as specified."""
    return multi_checkpoint_inner_step(
        adapter=adapter,
        history_state=history_state,
        count=count,
        seeds=seeds,
        continuation_seed_families=[continuation_seeds],
        checkpoint_progresses=[checkpoint_progress],
        candidate_potential=candidate_potential,
        rng=rng,
    )


def multi_checkpoint_inner_step(
    *,
    adapter: IterativeFrameAdapter,
    history_state: Any,
    count: int,
    seeds: Sequence[int],
    continuation_seed_families: Sequence[Sequence[int]],
    checkpoint_progresses: Sequence[float],
    candidate_potential: CandidatePotential,
    rng: np.random.Generator,
) -> InnerResult:
    """Run a properly weighted Feynman--Kac sampler at one or more checkpoints.

    Each intermediate potential is evaluated on the predicted-clean frame.  Its
    incremental weight is the ratio to the previous checkpoint potential along
    the selected ancestry.  The endpoint ratio closes the telescope, so adding
    checkpoints does not change the final target or decoder-NFE budget.
    """
    if count < 1 or len(seeds) != count:
        raise ValueError("count and seeds must have the same positive length")
    progresses = tuple(float(item) for item in checkpoint_progresses)
    if not progresses:
        raise ValueError("At least one checkpoint progress is required")
    if any(not 0.0 < item < 1.0 for item in progresses):
        raise ValueError("Checkpoint progresses must lie in (0, 1)")
    if any(right <= left for left, right in zip(progresses, progresses[1:])):
        raise ValueError("Checkpoint progresses must be strictly increasing")
    if len(continuation_seed_families) != len(progresses):
        raise ValueError("Expected one continuation seed family per checkpoint")
    if any(len(family) != count for family in continuation_seed_families):
        raise ValueError("Every continuation seed family must match count")

    state = adapter.initialize_inner_particles(history_state, count, seeds)
    previous_log_potential: np.ndarray | None = None
    path_log_increment_sum = np.zeros(count, dtype=float)
    log_mean_increments: list[float] = []
    checkpoint_ess: list[float] = []
    ancestors_history: list[np.ndarray] = []
    potential_history: list[np.ndarray] = []

    for index, progress in enumerate(progresses):
        state = adapter.denoise_to_checkpoint(state, progress)
        predicted = adapter.predict_clean(state)
        if len(predicted) != count:
            raise RuntimeError("Adapter returned the wrong number of predicted-clean frames")
        checkpoint_eval = [candidate_potential(frame) for frame in predicted]
        log_potential = np.asarray([item[0] for item in checkpoint_eval], dtype=float)
        log_increment = (
            log_potential
            if previous_log_potential is None
            else log_potential - previous_log_potential
        )
        weights, _ = normalize_log_weights(log_increment)
        log_mean_increments.append(logmeanexp(log_increment))
        checkpoint_ess.append(effective_sample_size(weights))
        ancestors = systematic_resample(weights, rng, n=count)
        ancestors_history.append(ancestors)
        potential_history.append(log_potential)
        state = adapter.resample_particle_state(state, ancestors)
        path_log_increment_sum = (
            path_log_increment_sum[ancestors] + log_increment[ancestors]
        )
        previous_log_potential = log_potential[ancestors]
        if index < len(progresses) - 1:
            state = adapter.reseed_particle_state(
                state, continuation_seed_families[index]
            )

    state = adapter.denoise_to_end(state, continuation_seed_families[-1])
    frames = adapter.finalize_frames(state)
    if len(frames) != count:
        raise RuntimeError("Adapter returned the wrong number of clean endpoint frames")
    endpoint_eval = [candidate_potential(frame) for frame in frames]
    log_phi0 = np.asarray([item[0] for item in endpoint_eval], dtype=float)
    assert previous_log_potential is not None
    log_g2 = log_phi0 - previous_log_potential
    weights2, _ = normalize_log_weights(log_g2)
    log_mean_g2 = logmeanexp(log_g2)
    log_mean_g1 = float(sum(log_mean_increments))
    log_z_hat = float(log_mean_g1 + log_mean_g2)
    selected = int(systematic_resample(weights2, rng, n=1)[0])
    telescoping = path_log_increment_sum + log_g2 - log_phi0
    log_psi, progress, values = endpoint_eval[selected]
    diagnostics = InnerDiagnostics(
        log_mean_g1=log_mean_g1,
        log_mean_g2=log_mean_g2,
        log_z_hat=log_z_hat,
        first_stage_ess=checkpoint_ess[0],
        second_stage_ess=effective_sample_size(weights2),
        checkpoint_ancestors=ancestors_history[-1],
        selected_index=selected,
        telescoping_max_abs_log_error=float(np.max(np.abs(telescoping))),
        checkpoint_log_potentials=potential_history[-1],
        endpoint_log_potentials=log_phi0,
        checkpoint_progresses=progresses,
        checkpoint_ess=tuple(checkpoint_ess),
        checkpoint_ancestors_history=tuple(ancestors_history),
        checkpoint_log_potentials_history=tuple(potential_history),
        checkpoint_log_mean_increments=tuple(log_mean_increments),
    )
    return InnerResult(
        frame=frames[selected],
        progress=progress,
        values=values,
        log_psi=float(log_psi),
        log_z_hat=log_z_hat,
        diagnostics=diagnostics,
    )
