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
    if count < 1 or len(seeds) != count or len(continuation_seeds) != count:
        raise ValueError("count, seeds, and continuation_seeds must have the same positive length")
    state = adapter.initialize_inner_particles(history_state, count, seeds)
    state = adapter.denoise_to_checkpoint(state, checkpoint_progress)
    predicted = adapter.predict_clean(state)
    if len(predicted) != count:
        raise RuntimeError("Adapter returned the wrong number of predicted-clean frames")
    checkpoint_eval = [candidate_potential(frame) for frame in predicted]
    log_g1 = np.asarray([item[0] for item in checkpoint_eval], dtype=float)
    weights1, _ = normalize_log_weights(log_g1)
    log_mean_g1 = logmeanexp(log_g1)
    ancestors = systematic_resample(weights1, rng, n=count)

    resampled = adapter.resample_particle_state(state, ancestors)
    resampled = adapter.denoise_to_end(resampled, continuation_seeds)
    frames = adapter.finalize_frames(resampled)
    if len(frames) != count:
        raise RuntimeError("Adapter returned the wrong number of clean endpoint frames")
    endpoint_eval = [candidate_potential(frame) for frame in frames]
    log_phi0 = np.asarray([item[0] for item in endpoint_eval], dtype=float)
    log_g2 = log_phi0 - log_g1[ancestors]
    weights2, _ = normalize_log_weights(log_g2)
    log_mean_g2 = logmeanexp(log_g2)
    log_z_hat = float(log_mean_g1 + log_mean_g2)
    selected = int(systematic_resample(weights2, rng, n=1)[0])
    telescoping = log_g1[ancestors] + log_g2 - log_phi0
    log_psi, progress, values = endpoint_eval[selected]
    diagnostics = InnerDiagnostics(
        log_mean_g1=log_mean_g1,
        log_mean_g2=log_mean_g2,
        log_z_hat=log_z_hat,
        first_stage_ess=effective_sample_size(weights1),
        second_stage_ess=effective_sample_size(weights2),
        checkpoint_ancestors=ancestors,
        selected_index=selected,
        telescoping_max_abs_log_error=float(np.max(np.abs(telescoping))),
        checkpoint_log_potentials=log_g1,
        endpoint_log_potentials=log_phi0,
    )
    return InnerResult(
        frame=frames[selected],
        progress=progress,
        values=values,
        log_psi=float(log_psi),
        log_z_hat=log_z_hat,
        diagnostics=diagnostics,
    )

