from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from confmh.adapters.base_iterative_frame import IterativeFrameAdapter
from confmh.duet.baselines import complete_frame_nested_step
from confmh.duet.guided_inner import guided_multi_checkpoint_inner_step
from confmh.duet.inner_fkc import multi_checkpoint_inner_step
from confmh.duet.potentials import PrefixPotential
from confmh.duet.programs import ProgressState
from confmh.duet.resampling import effective_sample_size, normalize_log_weights, systematic_resample


METHODS = {
    "frozen",
    "outer_only",
    "inner_only",
    "naive_dual",
    "complete_nested",
    "duet",
    "guided_duet",
}


@dataclass
class OuterParticle:
    history: list[Any]
    progress: ProgressState
    log_weight: float
    lineage: list[int]
    seed_metadata: list[dict[str, Any]] = field(default_factory=list)
    history_state: Any = None
    last_log_psi: float = 0.0


@dataclass
class StepRecord:
    t: int
    particle: int
    parent: int
    method: str
    log_increment: float
    log_psi: float
    log_z_hat: float | None
    inner_ess_1: float | None
    inner_ess_2: float | None
    inner_checkpoint_ancestors: list[int] | None
    inner_checkpoint_progresses: list[float] | None
    inner_checkpoint_ess: list[float] | None
    inner_checkpoint_ancestors_history: list[list[int]] | None
    inner_checkpoint_log_potentials_history: list[list[float]] | None
    inner_endpoint_log_potentials: list[float] | None
    inner_selected_index: int | None
    telescoping_max_abs_log_error: float | None
    guided_path_log_proposal_ratios: list[float] | None
    guided_selected_path_log_proposal_ratio: float | None
    guided_endpoint_ess: float | None
    guided_inner_resampling_enabled: bool | None
    progress_stage: int
    progress_failed: bool
    progress_completed_frames: list[int]
    progress_current_streak: int
    values: dict[str, float]


@dataclass
class DuETRunResult:
    method: str
    particles: list[OuterParticle]
    records: list[StepRecord]
    outer_ess: list[float]
    ancestor_indices: list[list[int]]
    outer_resampled: list[bool]
    log_normalizer_estimate: float
    log_normalizer_increments: list[float]
    # The outer implementation force-resamples at the terminal step.  Keep the
    # weighted population just before that operation so endpoint metrics are
    # not distorted by multinomial duplicates.
    pre_final_particles: list[OuterParticle] = field(default_factory=list)
    pre_final_normalized_weights: list[float] = field(default_factory=list)
    snapshot_particles: dict[int, list[OuterParticle]] = field(default_factory=dict)
    snapshot_normalized_weights: dict[int, list[float]] = field(default_factory=dict)


def _seed_family(base_seed: int, t: int, parent: int, count: int, stream: int) -> list[int]:
    sequence = np.random.SeedSequence([int(base_seed), int(t), int(parent), int(stream)])
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(count)]


class OuterSMC:
    def __init__(
        self,
        *,
        adapter: IterativeFrameAdapter,
        potential: PrefixPotential,
        method: str,
        outer_k: int,
        inner_m: int,
        checkpoint_progress: float | Sequence[float],
        seed: int,
        outer_resampling_ess_fraction: float = 1.0,
        guided_inner_resampling: bool = True,
        snapshot_times: Sequence[int] = (),
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"Unsupported method: {method}")
        if method == "inner_only" and outer_k != 1:
            raise ValueError("inner_only requires K=1")
        if method == "outer_only" and inner_m != 1:
            raise ValueError("outer_only requires M=1")
        self.adapter = adapter
        self.potential = potential
        self.method = method
        self.outer_k = int(outer_k)
        self.inner_m = int(inner_m)
        if isinstance(checkpoint_progress, Sequence) and not isinstance(
            checkpoint_progress, (str, bytes)
        ):
            progresses = tuple(float(item) for item in checkpoint_progress)
        else:
            progresses = (float(checkpoint_progress),)
        if not progresses:
            raise ValueError("At least one inner checkpoint is required")
        if any(not 0.0 < item < 1.0 for item in progresses):
            raise ValueError("Inner checkpoint progresses must lie in (0, 1)")
        if any(right <= left for left, right in zip(progresses, progresses[1:])):
            raise ValueError("Inner checkpoint progresses must be strictly increasing")
        self.checkpoint_progresses = progresses
        self.checkpoint_progress = progresses[-1]
        self.outer_resampling_ess_fraction = float(outer_resampling_ess_fraction)
        if not 0.0 < self.outer_resampling_ess_fraction <= 1.0:
            raise ValueError("outer_resampling_ess_fraction must be in (0, 1]")
        self.seed = int(seed)
        self.guided_inner_resampling = bool(guided_inner_resampling)
        self.rng = np.random.default_rng(seed)
        self.snapshot_times = frozenset(int(item) for item in snapshot_times)
        if any(item < 1 for item in self.snapshot_times):
            raise ValueError("snapshot_times must contain positive physical steps")

    def run(self, initial_history: Sequence[Any], horizon: int) -> DuETRunResult:
        if not initial_history:
            raise ValueError("initial_history must contain at least one frame")
        initial_state = ProgressState()
        initial_log_psi = self.potential.log_psi(initial_history, initial_state, 0)
        particles = [
            OuterParticle(
                history=copy.deepcopy(list(initial_history)),
                progress=initial_state,
                log_weight=-np.log(self.outer_k),
                lineage=[index],
                last_log_psi=initial_log_psi,
            )
            for index in range(self.outer_k)
        ]
        records: list[StepRecord] = []
        outer_ess: list[float] = []
        ancestry: list[list[int]] = []
        outer_resampled: list[bool] = []
        log_normalizer = float(initial_log_psi)
        log_normalizer_increments: list[float] = []
        pre_final_particles: list[OuterParticle] = []
        pre_final_normalized_weights: list[float] = []
        snapshot_particles: dict[int, list[OuterParticle]] = {}
        snapshot_normalized_weights: dict[int, list[float]] = {}

        for t in range(1, int(horizon) + 1):
            children: list[OuterParticle] = []
            increments: list[float] = []
            for parent_index, parent in enumerate(particles):
                history_state = self.adapter.prepare_history(parent.history)

                def candidate(frame: Any):
                    return self.potential.candidate_log_psi(
                        parent.history, parent.progress, frame, t
                    )

                proposal_seeds = _seed_family(
                    self.seed, t, parent_index, self.inner_m, stream=11
                )
                continuation_seeds = _seed_family(
                    self.seed, t, parent_index, self.inner_m, stream=29
                )
                continuation_seed_families = [continuation_seeds]
                log_z_hat: float | None = None
                inner_ess_1: float | None = None
                inner_ess_2: float | None = None
                inner_checkpoint_ancestors: list[int] | None = None
                inner_checkpoint_progresses: list[float] | None = None
                inner_checkpoint_ess: list[float] | None = None
                inner_checkpoint_ancestors_history: list[list[int]] | None = None
                inner_checkpoint_log_potentials_history: list[list[float]] | None = None
                inner_endpoint_log_potentials: list[float] | None = None
                inner_selected_index: int | None = None
                telescoping_error: float | None = None
                guided_path_ratios: list[float] | None = None
                guided_selected_ratio: float | None = None
                guided_endpoint_ess: float | None = None
                guided_resampling: bool | None = None

                if self.method in {"duet", "guided_duet", "naive_dual", "inner_only"}:
                    continuation_seed_families = [
                        _seed_family(
                            self.seed,
                            t,
                            parent_index,
                            self.inner_m,
                            stream=29 + 18 * checkpoint_index,
                        )
                        for checkpoint_index in range(len(self.checkpoint_progresses))
                    ]
                    if self.method == "guided_duet":
                        result = guided_multi_checkpoint_inner_step(
                            adapter=self.adapter,
                            history_state=history_state,
                            count=self.inner_m,
                            seeds=proposal_seeds,
                            continuation_seed_families=continuation_seed_families,
                            checkpoint_progresses=self.checkpoint_progresses,
                            candidate_potential=candidate,
                            rng=self.rng,
                            resampling_enabled=self.guided_inner_resampling,
                        )
                    else:
                        result = multi_checkpoint_inner_step(
                            adapter=self.adapter,
                            history_state=history_state,
                            count=self.inner_m,
                            seeds=proposal_seeds,
                            continuation_seed_families=continuation_seed_families,
                            checkpoint_progresses=self.checkpoint_progresses,
                            candidate_potential=candidate,
                            rng=self.rng,
                        )
                    frame, progress, values = result.frame, result.progress, result.values
                    log_psi, log_z_hat = result.log_psi, result.log_z_hat
                    if self.method == "guided_duet":
                        inner_ess_1 = result.diagnostics.checkpoint_ess[0]
                        inner_ess_2 = result.diagnostics.endpoint_ess
                        inner_checkpoint_ancestors = (
                            result.diagnostics.checkpoint_ancestors_history[-1].tolist()
                        )
                        guided_path_ratios = (
                            result.diagnostics.path_log_proposal_ratios.tolist()
                        )
                        guided_selected_ratio = (
                            result.diagnostics.selected_path_log_proposal_ratio
                        )
                        guided_endpoint_ess = result.diagnostics.endpoint_ess
                        guided_resampling = result.diagnostics.resampling_enabled
                        inner_checkpoint_log_potentials_history = [
                            item.tolist()
                            for item in result.diagnostics.checkpoint_log_potentials_history
                        ]
                        inner_endpoint_log_potentials = (
                            result.diagnostics.endpoint_log_potentials.tolist()
                        )
                    else:
                        inner_ess_1 = result.diagnostics.first_stage_ess
                        inner_ess_2 = result.diagnostics.second_stage_ess
                        inner_checkpoint_ancestors = (
                            result.diagnostics.checkpoint_ancestors.tolist()
                        )
                        inner_checkpoint_log_potentials_history = [
                            item.tolist()
                            for item in result.diagnostics.checkpoint_log_potentials_history
                        ]
                        inner_endpoint_log_potentials = (
                            result.diagnostics.endpoint_log_potentials.tolist()
                        )
                    inner_checkpoint_progresses = list(
                        result.diagnostics.checkpoint_progresses
                    )
                    inner_checkpoint_ess = list(result.diagnostics.checkpoint_ess)
                    inner_checkpoint_ancestors_history = [
                        item.tolist()
                        for item in result.diagnostics.checkpoint_ancestors_history
                    ]
                    inner_selected_index = result.diagnostics.selected_index
                    telescoping_error = (
                        result.diagnostics.potential_telescoping_max_abs_log_error
                        if self.method == "guided_duet"
                        else result.diagnostics.telescoping_max_abs_log_error
                    )
                elif self.method == "complete_nested":
                    result = complete_frame_nested_step(
                        adapter=self.adapter,
                        history_state=history_state,
                        count=self.inner_m,
                        seeds=proposal_seeds,
                        checkpoint_progress=self.checkpoint_progress,
                        candidate_potential=candidate,
                        rng=self.rng,
                    )
                    frame, progress, values = result.frame, result.progress, result.values
                    log_psi, log_z_hat = result.log_psi, result.log_z_hat
                    inner_selected_index = result.selected_index
                else:
                    frames = self.adapter.sample_complete_frames(
                        history_state,
                        1,
                        proposal_seeds[:1],
                        checkpoint_progress=self.checkpoint_progress,
                    )
                    frame = frames[0]
                    log_psi, progress, values = candidate(frame)

                if self.method in {"duet", "guided_duet", "complete_nested"}:
                    assert log_z_hat is not None
                    increment = log_z_hat - parent.last_log_psi
                elif self.method in {"outer_only", "naive_dual"}:
                    increment = log_psi - parent.last_log_psi
                else:
                    increment = 0.0
                child = OuterParticle(
                    history=copy.deepcopy(parent.history) + [copy.deepcopy(frame)],
                    progress=progress,
                    log_weight=float(parent.log_weight + increment),
                    lineage=parent.lineage + [parent_index],
                    seed_metadata=parent.seed_metadata
                    + [
                        {
                            "t": t,
                            "proposal": proposal_seeds,
                            "continuation": continuation_seeds,
                            "continuation_families": continuation_seed_families,
                            "checkpoint_progresses": list(
                                self.checkpoint_progresses
                            ),
                        }
                    ],
                    last_log_psi=float(log_psi),
                )
                children.append(child)
                increments.append(float(increment))
                records.append(
                    StepRecord(
                        t=t,
                        particle=parent_index,
                        parent=parent_index,
                        method=self.method,
                        log_increment=float(increment),
                        log_psi=float(log_psi),
                        log_z_hat=log_z_hat,
                        inner_ess_1=inner_ess_1,
                        inner_ess_2=inner_ess_2,
                        inner_checkpoint_ancestors=inner_checkpoint_ancestors,
                        inner_checkpoint_progresses=inner_checkpoint_progresses,
                        inner_checkpoint_ess=inner_checkpoint_ess,
                        inner_checkpoint_ancestors_history=(
                            inner_checkpoint_ancestors_history
                        ),
                        inner_checkpoint_log_potentials_history=(
                            inner_checkpoint_log_potentials_history
                        ),
                        inner_endpoint_log_potentials=inner_endpoint_log_potentials,
                        inner_selected_index=inner_selected_index,
                        telescoping_max_abs_log_error=telescoping_error,
                        guided_path_log_proposal_ratios=guided_path_ratios,
                        guided_selected_path_log_proposal_ratio=guided_selected_ratio,
                        guided_endpoint_ess=guided_endpoint_ess,
                        guided_inner_resampling_enabled=guided_resampling,
                        progress_stage=progress.stage,
                        progress_failed=progress.failed,
                        progress_completed_frames=list(progress.completed_frames),
                        progress_current_streak=progress.current_streak,
                        values=values,
                    )
                )
                history_state = None
                release_transient = getattr(
                    self.adapter, "release_transient_memory", None
                )
                if callable(release_transient):
                    release_transient()

            if self.outer_k > 1 and self.method not in {"frozen", "inner_only"}:
                weights, log_weight_sum = normalize_log_weights(
                    np.asarray([p.log_weight for p in children])
                )
                if t in self.snapshot_times or t == int(horizon):
                    captured = copy.deepcopy(children)
                    captured_weights = weights.tolist()
                    if t in self.snapshot_times:
                        snapshot_particles[t] = captured
                        snapshot_normalized_weights[t] = captured_weights
                    if t == int(horizon):
                        pre_final_particles = captured
                        pre_final_normalized_weights = captured_weights
                log_normalizer += float(log_weight_sum)
                log_normalizer_increments.append(float(log_weight_sum))
                ess = effective_sample_size(weights)
                outer_ess.append(ess)
                should_resample = (
                    ess <= self.outer_resampling_ess_fraction * self.outer_k
                    or t == int(horizon)
                )
                if should_resample:
                    parent_indices = systematic_resample(weights, self.rng, n=self.outer_k)
                    ancestry.append(parent_indices.tolist())
                    particles = [copy.deepcopy(children[int(index)]) for index in parent_indices]
                    for particle in particles:
                        particle.log_weight = -np.log(self.outer_k)
                else:
                    ancestry.append(list(range(self.outer_k)))
                    particles = children
                    for particle in particles:
                        particle.log_weight -= float(log_weight_sum)
                outer_resampled.append(should_resample)
            else:
                if t in self.snapshot_times or t == int(horizon):
                    if self.method in {"frozen", "inner_only"}:
                        terminal_weights = np.full(
                            len(children), 1.0 / max(len(children), 1), dtype=float
                        )
                    else:
                        terminal_weights, _ = normalize_log_weights(
                            np.asarray([p.log_weight for p in children])
                        )
                    captured = copy.deepcopy(children)
                    captured_weights = terminal_weights.tolist()
                    if t in self.snapshot_times:
                        snapshot_particles[t] = captured
                        snapshot_normalized_weights[t] = captured_weights
                    if t == int(horizon):
                        pre_final_particles = captured
                        pre_final_normalized_weights = captured_weights
                if self.method not in {"frozen", "inner_only"}:
                    _, log_weight_sum = normalize_log_weights(
                        np.asarray([p.log_weight for p in children])
                    )
                    log_normalizer += float(log_weight_sum)
                    log_normalizer_increments.append(float(log_weight_sum))
                    for particle in children:
                        particle.log_weight -= float(log_weight_sum)
                outer_ess.append(float(self.outer_k))
                ancestry.append(list(range(self.outer_k)))
                outer_resampled.append(False)
                particles = children

        return DuETRunResult(
            method=self.method,
            particles=particles,
            records=records,
            outer_ess=outer_ess,
            ancestor_indices=ancestry,
            outer_resampled=outer_resampled,
            log_normalizer_estimate=log_normalizer,
            log_normalizer_increments=log_normalizer_increments,
            pre_final_particles=pre_final_particles,
            pre_final_normalized_weights=pre_final_normalized_weights,
            snapshot_particles=snapshot_particles,
            snapshot_normalized_weights=snapshot_normalized_weights,
        )
