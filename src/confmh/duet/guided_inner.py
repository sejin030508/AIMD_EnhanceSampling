from __future__ import annotations

"""Properly weighted guided inner bridge sampler."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from confmh.adapters.base_iterative_frame import IterativeFrameAdapter
from confmh.duet.guided_weights import GuidedInnerWeights
from confmh.duet.programs import ProgressState
from confmh.duet.resampling import systematic_resample

CandidatePotential = Callable[[Any], tuple[float, ProgressState, dict[str, float]]]


@dataclass
class GuidedInnerDiagnostics:
    log_z_hat: float
    selected_index: int
    checkpoint_progresses: tuple[float, ...]
    checkpoint_ess: tuple[float, ...]
    checkpoint_ancestors_history: tuple[np.ndarray, ...]
    checkpoint_log_potentials_history: tuple[np.ndarray, ...]
    checkpoint_log_normalizer_increments: tuple[float, ...]
    endpoint_log_potentials: np.ndarray
    endpoint_weights: np.ndarray
    endpoint_ess: float
    endpoint_log_normalizer_increment: float
    selected_path_log_proposal_ratio: float
    path_log_proposal_ratios: np.ndarray
    potential_telescoping_max_abs_log_error: float
    resampling_enabled: bool


@dataclass
class GuidedInnerResult:
    frame: Any
    progress: ProgressState
    values: dict[str, float]
    log_psi: float
    log_z_hat: float
    diagnostics: GuidedInnerDiagnostics


def guided_multi_checkpoint_inner_step(
    *,
    adapter: IterativeFrameAdapter,
    history_state: Any,
    count: int,
    seeds: Sequence[int],
    continuation_seed_families: Sequence[Sequence[int]],
    checkpoint_progresses: Sequence[float],
    candidate_potential: CandidatePotential,
    rng: np.random.Generator,
    resampling_enabled: bool,
) -> GuidedInnerResult:
    """Run PVB guided proposal, optional checkpoint resampling and correction."""

    required = (
        "guided_denoise_to_checkpoint",
        "guided_denoise_to_end",
        "guided_checkpoint_log_potential",
        "guided_endpoint_log_potential",
    )
    missing = [name for name in required if not callable(getattr(adapter, name, None))]
    if missing:
        raise TypeError(f"Adapter does not implement guided bridge methods: {missing}")
    if count < 1 or len(seeds) != count:
        raise ValueError("count and seeds must have the same positive length")
    progresses = tuple(float(item) for item in checkpoint_progresses)
    if not progresses or any(not 0.0 < item < 1.0 for item in progresses):
        raise ValueError("Guided checkpoints must lie in (0,1)")
    if any(right <= left for left, right in pairwise(progresses)):
        raise ValueError("Guided checkpoints must be strictly increasing")
    if len(continuation_seed_families) != len(progresses):
        raise ValueError("Expected one continuation seed family per checkpoint")
    if any(len(family) != count for family in continuation_seed_families):
        raise ValueError("Every continuation seed family must match count")

    state = adapter.initialize_inner_particles(history_state, count, seeds)
    weights = GuidedInnerWeights(count)
    checkpoint_ess: list[float] = []
    ancestor_history: list[np.ndarray] = []
    potential_history: list[np.ndarray] = []
    log_increment_history: list[float] = []

    for index, progress in enumerate(progresses):
        state, proposal_ratio = adapter.guided_denoise_to_checkpoint(state, progress)
        weights.add_proposal_ratio(proposal_ratio)
        log_phi = np.asarray(
            adapter.guided_checkpoint_log_potential(state), dtype=np.float64
        )
        observation = weights.observe(log_phi)
        checkpoint_ess.append(observation.ess)
        potential_history.append(log_phi.copy())
        log_increment_history.append(observation.log_normalizer_increment)
        if resampling_enabled:
            ancestors = systematic_resample(observation.weights, rng, n=count)
            state = adapter.resample_particle_state(state, ancestors)
            weights.resample(ancestors)
        else:
            ancestors = np.arange(count, dtype=np.int64)
        ancestor_history.append(ancestors.copy())
        if resampling_enabled and index < len(progresses) - 1:
            state = adapter.reseed_particle_state(
                state, continuation_seed_families[index]
            )

    state, proposal_ratio = adapter.guided_denoise_to_end(
        state,
        continuation_seed_families[-1] if resampling_enabled else None,
    )
    weights.add_proposal_ratio(proposal_ratio)
    frames = adapter.finalize_frames(state)
    if len(frames) != count:
        raise RuntimeError("Adapter returned the wrong number of endpoint frames")
    endpoint_log_phi = np.asarray(
        adapter.guided_endpoint_log_potential(state), dtype=np.float64
    )
    endpoint_observation = weights.observe(endpoint_log_phi)
    selected = int(systematic_resample(endpoint_observation.weights, rng, n=1)[0])
    evaluated = candidate_potential(frames[selected])
    selected_log_psi, progress_state, values = evaluated
    if not np.isclose(
        selected_log_psi,
        endpoint_log_phi[selected],
        rtol=0.0,
        atol=2.0e-5,
    ):
        raise RuntimeError(
            "Torch/production endpoint potential mismatch: "
            f"{endpoint_log_phi[selected]} != {selected_log_psi}"
        )
    telescoping_error = weights.potential_telescoping_error(endpoint_log_phi)
    diagnostics = GuidedInnerDiagnostics(
        log_z_hat=float(weights.log_z_hat),
        selected_index=selected,
        checkpoint_progresses=progresses,
        checkpoint_ess=tuple(checkpoint_ess),
        checkpoint_ancestors_history=tuple(ancestor_history),
        checkpoint_log_potentials_history=tuple(potential_history),
        checkpoint_log_normalizer_increments=tuple(log_increment_history),
        endpoint_log_potentials=endpoint_log_phi.copy(),
        endpoint_weights=endpoint_observation.weights.copy(),
        endpoint_ess=endpoint_observation.ess,
        endpoint_log_normalizer_increment=(
            endpoint_observation.log_normalizer_increment
        ),
        selected_path_log_proposal_ratio=float(
            weights.path_log_proposal_ratio[selected]
        ),
        path_log_proposal_ratios=weights.path_log_proposal_ratio.copy(),
        potential_telescoping_max_abs_log_error=telescoping_error,
        resampling_enabled=bool(resampling_enabled),
    )
    return GuidedInnerResult(
        frame=frames[selected],
        progress=progress_state,
        values=values,
        log_psi=float(selected_log_psi),
        log_z_hat=float(weights.log_z_hat),
        diagnostics=diagnostics,
    )
