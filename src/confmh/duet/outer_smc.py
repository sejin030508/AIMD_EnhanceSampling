from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from confmh.adapters.base_iterative_frame import IterativeFrameAdapter
from confmh.duet.baselines import complete_frame_nested_step
from confmh.duet.inner_fkc import one_checkpoint_inner_step
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
    inner_selected_index: int | None
    telescoping_max_abs_log_error: float | None
    progress_stage: int
    progress_failed: bool
    values: dict[str, float]


@dataclass
class DuETRunResult:
    method: str
    particles: list[OuterParticle]
    records: list[StepRecord]
    outer_ess: list[float]
    ancestor_indices: list[list[int]]
    log_normalizer_estimate: float
    log_normalizer_increments: list[float]


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
        checkpoint_progress: float,
        seed: int,
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
        self.checkpoint_progress = float(checkpoint_progress)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

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
        log_normalizer = float(initial_log_psi)
        log_normalizer_increments: list[float] = []

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
                log_z_hat: float | None = None
                inner_ess_1: float | None = None
                inner_ess_2: float | None = None
                inner_checkpoint_ancestors: list[int] | None = None
                inner_selected_index: int | None = None
                telescoping_error: float | None = None

                if self.method in {"duet", "naive_dual", "inner_only"}:
                    result = one_checkpoint_inner_step(
                        adapter=self.adapter,
                        history_state=history_state,
                        count=self.inner_m,
                        seeds=proposal_seeds,
                        continuation_seeds=continuation_seeds,
                        checkpoint_progress=self.checkpoint_progress,
                        candidate_potential=candidate,
                        rng=self.rng,
                    )
                    frame, progress, values = result.frame, result.progress, result.values
                    log_psi, log_z_hat = result.log_psi, result.log_z_hat
                    inner_ess_1 = result.diagnostics.first_stage_ess
                    inner_ess_2 = result.diagnostics.second_stage_ess
                    inner_checkpoint_ancestors = result.diagnostics.checkpoint_ancestors.tolist()
                    inner_selected_index = result.diagnostics.selected_index
                    telescoping_error = result.diagnostics.telescoping_max_abs_log_error
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

                if self.method in {"duet", "complete_nested"}:
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
                    + [{"t": t, "proposal": proposal_seeds, "continuation": continuation_seeds}],
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
                        inner_selected_index=inner_selected_index,
                        telescoping_max_abs_log_error=telescoping_error,
                        progress_stage=progress.stage,
                        progress_failed=progress.failed,
                        values=values,
                    )
                )

            if self.outer_k > 1 and self.method not in {"frozen", "inner_only"}:
                weights, log_weight_sum = normalize_log_weights(
                    np.asarray([p.log_weight for p in children])
                )
                log_normalizer += float(log_weight_sum)
                log_normalizer_increments.append(float(log_weight_sum))
                outer_ess.append(effective_sample_size(weights))
                parent_indices = systematic_resample(weights, self.rng, n=self.outer_k)
                ancestry.append(parent_indices.tolist())
                particles = [copy.deepcopy(children[int(index)]) for index in parent_indices]
                for particle in particles:
                    particle.log_weight = -np.log(self.outer_k)
            else:
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
                particles = children

        return DuETRunResult(
            method=self.method,
            particles=particles,
            records=records,
            outer_ess=outer_ess,
            ancestor_indices=ancestry,
            log_normalizer_estimate=log_normalizer,
            log_normalizer_increments=log_normalizer_increments,
        )
