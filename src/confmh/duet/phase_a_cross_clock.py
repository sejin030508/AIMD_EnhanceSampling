from __future__ import annotations

import json
import math
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr

from confmh.duet.config import resolve_config_path
from confmh.duet.records import initialize_run_directory, write_json
from confmh.duet.resampling import systematic_resample


STATE_NAMES = ("S", "L", "R", "C_L", "C_R", "O", "F")
S, L, R, C_L, C_R, O, F = range(7)

FIXED_METHOD_IDS = {
    "frozen": 1,
    "outer_only": 2,
    "inner_only": 3,
    "complete_nested": 4,
    "fixed_duet": 5,
    "duet_y1": 6,
    "adaptive": 20,
}


@dataclass(frozen=True)
class ProcessParameters:
    name: str
    instance_id: int
    s_l: float
    s_r: float
    commitment_probability: float
    commitment_beta2: float
    informative: bool
    split: str


@dataclass(frozen=True)
class BinaryTransition:
    one_state: int
    zero_state: int
    probability_one: float
    beta2: float


@dataclass(frozen=True)
class ReverseKernel:
    """Exact reverse kernel induced by a binary forward-corruption chain."""

    probability_x: np.ndarray
    probability_y3: np.ndarray
    probability_y2_given_y3: np.ndarray
    probability_y1_given_y2: np.ndarray
    probability_x_given_y1: np.ndarray
    joint_forward: np.ndarray

    @classmethod
    def build(
        cls,
        probability_one: float,
        *,
        beta1: float,
        beta2: float,
        beta3: float,
    ) -> ReverseKernel:
        probability_x = np.asarray([1.0 - probability_one, probability_one], dtype=np.float64)
        joint = np.zeros((2, 2, 2, 2), dtype=np.float64)
        for x in range(2):
            for y1 in range(2):
                for y2 in range(2):
                    for y3 in range(2):
                        joint[x, y1, y2, y3] = (
                            probability_x[x]
                            * _flip_probability(y1, x, beta1)
                            * _flip_probability(y2, y1, beta2)
                            * _flip_probability(y3, y2, beta3)
                        )
        probability_y3 = joint.sum(axis=(0, 1, 2))
        y2_y3 = joint.sum(axis=(0, 1))
        y1_y2 = joint.sum(axis=(0, 3))
        x_y1 = joint.sum(axis=(2, 3))
        return cls(
            probability_x=probability_x,
            probability_y3=probability_y3,
            probability_y2_given_y3=_conditional_rows(y2_y3.T).T,
            probability_y1_given_y2=_conditional_rows(y1_y2.T).T,
            probability_x_given_y1=_conditional_rows(x_y1.T).T,
            joint_forward=joint,
        )

    def phi(self, endpoint_potential: np.ndarray, stage: str) -> np.ndarray:
        endpoint_potential = np.asarray(endpoint_potential, dtype=np.float64)
        if stage == "Y1":
            joint = self.joint_forward.sum(axis=(2, 3))
        elif stage == "Y2":
            joint = self.joint_forward.sum(axis=(1, 3))
        elif stage == "Y3":
            joint = self.joint_forward.sum(axis=(1, 2))
        else:
            raise ValueError(f"Unknown diffusion stage: {stage}")
        marginal = joint.sum(axis=0)
        return (joint.T @ endpoint_potential) / marginal

    def chi(self, endpoint_potential: np.ndarray, stage: str) -> float:
        endpoint_potential = np.asarray(endpoint_potential, dtype=np.float64)
        denominator = _weighted_variance(endpoint_potential, self.probability_x)
        if denominator <= 0.0:
            return 0.0
        if stage == "Y1":
            marginal = self.joint_forward.sum(axis=(0, 2, 3))
        elif stage == "Y2":
            marginal = self.joint_forward.sum(axis=(0, 1, 3))
        elif stage == "Y3":
            marginal = self.joint_forward.sum(axis=(0, 1, 2))
        else:
            raise ValueError(f"Unknown diffusion stage: {stage}")
        return _weighted_variance(self.phi(endpoint_potential, stage), marginal) / denominator


@dataclass(frozen=True)
class ExactTarget:
    paths: tuple[tuple[int, ...], ...]
    base_probabilities: np.ndarray
    potentials: np.ndarray
    target_probabilities: np.ndarray
    normalizer: float
    base_success_probability: float
    target_success_probability: float
    route_l_mass: float
    route_r_mass: float


@dataclass(frozen=True)
class LocalProposal:
    next_state: int
    local_normalizer: float


@dataclass
class SamplerOutput:
    paths: list[tuple[int, ...]]
    normalizer: float | None
    schedule: list[tuple[int, int]]
    diagnostics: list[dict[str, float]]
    reverse_transition_evaluations: int


def _flip_probability(destination: int, source: int, beta: float) -> float:
    return 1.0 - float(beta) if int(destination) == int(source) else float(beta)


def _conditional_rows(joint_rows: np.ndarray) -> np.ndarray:
    result = np.asarray(joint_rows, dtype=np.float64).copy()
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise FloatingPointError("Cannot construct a conditional distribution with zero mass")
    return result / totals


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    mean = float(np.dot(weights, values))
    return float(np.dot(weights, (values - mean) ** 2))


def _sample_binary(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return int(rng.random() >= probabilities[0])


def potential(step: int, state: int, epsilon: float = 0.05) -> float:
    valid = {
        1: {L, R},
        2: {L, R},
        3: {C_L, C_R},
        4: {O},
        5: {O},
    }
    if step == 0:
        return 1.0
    return 1.0 if int(state) in valid[int(step)] else float(epsilon)


def transition_spec(
    process: ProcessParameters,
    parent: int,
    step: int,
    *,
    default_beta2: float = 0.20,
) -> BinaryTransition:
    parent, step = int(parent), int(step)
    if step == 1 and parent == S:
        return BinaryTransition(L, R, 0.5, default_beta2)
    if step == 2 and parent == L:
        return BinaryTransition(L, F, process.s_l, 0.5)
    if step == 2 and parent == R:
        return BinaryTransition(R, F, process.s_r, 0.5)
    if step == 3 and parent == L:
        return BinaryTransition(C_L, F, process.commitment_probability, process.commitment_beta2)
    if step == 3 and parent == R:
        return BinaryTransition(C_R, F, process.commitment_probability, process.commitment_beta2)
    if step == 4 and parent in {C_L, C_R}:
        return BinaryTransition(O, F, 0.9, default_beta2)
    if step == 5 and parent == O:
        return BinaryTransition(O, F, 0.9, default_beta2)
    # Failure and every off-schedule state are absorbing. Both binary labels map to F so
    # the three reverse-transition evaluations are still charged without changing q.
    return BinaryTransition(F, F, 0.5, default_beta2)


def reverse_kernel(
    transition: BinaryTransition,
    *,
    beta1: float = 0.05,
    beta3: float = 0.50,
) -> ReverseKernel:
    return ReverseKernel.build(
        transition.probability_one,
        beta1=beta1,
        beta2=transition.beta2,
        beta3=beta3,
    )


def endpoint_potentials(transition: BinaryTransition, step: int, epsilon: float) -> np.ndarray:
    # Binary X is ordered as [zero, one].
    return np.asarray(
        [
            potential(step, transition.zero_state, epsilon),
            potential(step, transition.one_state, epsilon),
        ],
        dtype=np.float64,
    )


def mapped_state(transition: BinaryTransition, x: int) -> int:
    return transition.one_state if int(x) == 1 else transition.zero_state


def sample_reverse_path(
    kernel: ReverseKernel,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    y3 = _sample_binary(kernel.probability_y3, rng)
    y2 = _sample_binary(kernel.probability_y2_given_y3[:, y3], rng)
    y1 = _sample_binary(kernel.probability_y1_given_y2[:, y2], rng)
    x = _sample_binary(kernel.probability_x_given_y1[:, y1], rng)
    return y3, y2, y1, x


def _sample_to_checkpoint(
    kernel: ReverseKernel,
    stage: str,
    rng: np.random.Generator,
) -> tuple[int, int, int | None]:
    y3 = _sample_binary(kernel.probability_y3, rng)
    y2 = _sample_binary(kernel.probability_y2_given_y3[:, y3], rng)
    if stage == "Y2":
        return y3, y2, None
    if stage == "Y1":
        y1 = _sample_binary(kernel.probability_y1_given_y2[:, y2], rng)
        return y3, y2, y1
    raise ValueError(f"Unsupported resampling checkpoint: {stage}")


def _continue_from_checkpoint(
    kernel: ReverseKernel,
    stage: str,
    y2: int,
    y1: int | None,
    rng: np.random.Generator,
) -> int:
    if stage == "Y2":
        y1 = _sample_binary(kernel.probability_y1_given_y2[:, int(y2)], rng)
    if y1 is None:
        raise RuntimeError("Y1 is required before sampling X")
    return _sample_binary(kernel.probability_x_given_y1[:, int(y1)], rng)


def base_local_proposal(
    process: ProcessParameters,
    parent: int,
    step: int,
    epsilon: float,
    rng: np.random.Generator,
) -> LocalProposal:
    transition = transition_spec(process, parent, step)
    _, _, _, x = sample_reverse_path(reverse_kernel(transition), rng)
    state = mapped_state(transition, x)
    return LocalProposal(state, potential(step, state, epsilon))


def complete_nested_local_proposal(
    process: ProcessParameters,
    parent: int,
    step: int,
    inner_m: int,
    epsilon: float,
    rng: np.random.Generator,
) -> LocalProposal:
    transition = transition_spec(process, parent, step)
    kernel = reverse_kernel(transition)
    states = []
    endpoint_weights = []
    for _ in range(int(inner_m)):
        _, _, _, x = sample_reverse_path(kernel, rng)
        state = mapped_state(transition, x)
        states.append(state)
        endpoint_weights.append(potential(step, state, epsilon))
    weights = np.asarray(endpoint_weights, dtype=np.float64)
    selected = int(systematic_resample(weights, rng, n=1)[0])
    return LocalProposal(states[selected], float(weights.mean()))


def duet_local_proposal(
    process: ProcessParameters,
    parent: int,
    step: int,
    inner_m: int,
    epsilon: float,
    checkpoint: str,
    rng: np.random.Generator,
) -> LocalProposal:
    transition = transition_spec(process, parent, step)
    kernel = reverse_kernel(transition)
    endpoint_values = endpoint_potentials(transition, step, epsilon)
    checkpoint_phi = kernel.phi(endpoint_values, checkpoint)
    partials: list[tuple[int, int | None]] = []
    first_weights = []
    for _ in range(int(inner_m)):
        _, y2, y1 = _sample_to_checkpoint(kernel, checkpoint, rng)
        partials.append((y2, y1))
        stage_value = y2 if checkpoint == "Y2" else int(y1)
        first_weights.append(float(checkpoint_phi[stage_value]))
    first_weights_array = np.asarray(first_weights, dtype=np.float64)
    ancestors = systematic_resample(first_weights_array, rng, n=int(inner_m))
    states = []
    correction_weights = []
    for ancestor in ancestors:
        y2, y1 = partials[int(ancestor)]
        x = _continue_from_checkpoint(kernel, checkpoint, y2, y1, rng)
        state = mapped_state(transition, x)
        states.append(state)
        correction_weights.append(
            potential(step, state, epsilon) / first_weights_array[int(ancestor)]
        )
    corrections = np.asarray(correction_weights, dtype=np.float64)
    selected = int(systematic_resample(corrections, rng, n=1)[0])
    local_normalizer = float(first_weights_array.mean() * corrections.mean())
    return LocalProposal(states[selected], local_normalizer)


def physical_transition_probabilities(
    process: ProcessParameters,
    parent: int,
    step: int,
) -> dict[int, float]:
    transition = transition_spec(process, parent, step)
    result: dict[int, float] = {}
    for state, probability_value in (
        (transition.zero_state, 1.0 - transition.probability_one),
        (transition.one_state, transition.probability_one),
    ):
        result[state] = result.get(state, 0.0) + float(probability_value)
    return result


def enumerate_target(process: ProcessParameters, epsilon: float = 0.05) -> ExactTarget:
    prefixes: list[tuple[tuple[int, ...], float]] = [((S,), 1.0)]
    for step in range(1, 6):
        expanded = []
        for path, probability_value in prefixes:
            for state, transition_probability in physical_transition_probabilities(
                process, path[-1], step
            ).items():
                expanded.append((path + (state,), probability_value * transition_probability))
        prefixes = expanded
    paths = tuple(item[0] for item in prefixes)
    base = np.asarray([item[1] for item in prefixes], dtype=np.float64)
    potentials = np.asarray([potential(5, path[-1], epsilon) for path in paths])
    unnormalized = base * potentials
    normalizer = float(unnormalized.sum())
    target = unnormalized / normalizer
    success = np.asarray([path[-2:] == (O, O) for path in paths], dtype=np.float64)
    route_l = np.asarray([path[1] == L for path in paths], dtype=np.float64)
    route_r = np.asarray([path[1] == R for path in paths], dtype=np.float64)
    return ExactTarget(
        paths=paths,
        base_probabilities=base,
        potentials=potentials,
        target_probabilities=target,
        normalizer=normalizer,
        base_success_probability=float(np.dot(base, success)),
        target_success_probability=float(np.dot(target, success)),
        route_l_mass=float(np.dot(target, route_l)),
        route_r_mass=float(np.dot(target, route_r)),
    )


def enumerate_inner_paths(
    process: ProcessParameters,
    parent: int,
    step: int,
) -> list[dict[str, Any]]:
    transition = transition_spec(process, parent, step)
    kernel = reverse_kernel(transition)
    rows = []
    for x in range(2):
        for y1 in range(2):
            for y2 in range(2):
                for y3 in range(2):
                    rows.append(
                        {
                            "Y3": y3,
                            "Y2": y2,
                            "Y1": y1,
                            "X": x,
                            "mapped_state": STATE_NAMES[mapped_state(transition, x)],
                            "probability": float(kernel.joint_forward[x, y1, y2, y3]),
                        }
                    )
    return rows


def inner_marginal_error(process: ProcessParameters, parent: int, step: int) -> float:
    transition = transition_spec(process, parent, step)
    kernel = reverse_kernel(transition)
    reverse_x = np.zeros(2, dtype=np.float64)
    for y3 in range(2):
        for y2 in range(2):
            for y1 in range(2):
                for x in range(2):
                    reverse_x[x] += (
                        kernel.probability_y3[y3]
                        * kernel.probability_y2_given_y3[y2, y3]
                        * kernel.probability_y1_given_y2[y1, y2]
                        * kernel.probability_x_given_y1[x, y1]
                    )
    return float(np.max(np.abs(reverse_x - kernel.probability_x)))


def repetition_seed_sequence(
    master_seed: int,
    instance_id: int,
    method_id: int,
    repetition_id: int,
) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [int(master_seed), int(instance_id), int(method_id), int(repetition_id)]
    )


def _run_fixed_sampler(
    process: ProcessParameters,
    method: str,
    outer_k: int,
    inner_m: int,
    epsilon: float,
    seed_sequence: np.random.SeedSequence,
    checkpoint: str = "Y2",
) -> SamplerOutput:
    step_streams = seed_sequence.spawn(5)
    paths: list[tuple[int, ...]] = [(S,)] * int(outer_k)
    normalizer = 1.0
    reverse_evaluations = 0
    schedule = []
    for step, stream in enumerate(step_streams, start=1):
        rng = np.random.default_rng(stream)
        schedule.append((int(outer_k), int(inner_m)))
        proposals: list[tuple[int, ...]] = []
        local_normalizers = []
        for path in paths:
            if method in {"frozen", "outer_only"}:
                proposal = base_local_proposal(process, path[-1], step, epsilon, rng)
            elif method == "complete_nested":
                proposal = complete_nested_local_proposal(
                    process, path[-1], step, inner_m, epsilon, rng
                )
            elif method in {"inner_only", "fixed_duet", "duet_y1"}:
                proposal = duet_local_proposal(
                    process,
                    path[-1],
                    step,
                    inner_m,
                    epsilon,
                    "Y1" if method == "duet_y1" else checkpoint,
                    rng,
                )
            else:
                raise ValueError(f"Unsupported fixed method: {method}")
            proposals.append(path + (proposal.next_state,))
            local_normalizers.append(proposal.local_normalizer)
        reverse_evaluations += 3 * int(outer_k) * int(inner_m)
        if method in {"frozen", "inner_only"}:
            paths = proposals
            continue
        incremental = np.asarray(
            [
                local_z / potential(step - 1, path[-1], epsilon)
                for path, local_z in zip(paths, local_normalizers)
            ],
            dtype=np.float64,
        )
        # The mean, not the sum, is the incremental global-normalizer estimate.
        normalizer *= mean_incremental_weight(incremental)
        ancestors = systematic_resample(incremental, rng, n=int(outer_k))
        paths = [proposals[int(index)] for index in ancestors]
    return SamplerOutput(
        paths=paths,
        normalizer=None if method in {"frozen", "inner_only"} else normalizer,
        schedule=schedule,
        diagnostics=[],
        reverse_transition_evaluations=reverse_evaluations,
    )


def compute_probe_diagnostics(phi: np.ndarray, psi: np.ndarray) -> dict[str, float]:
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if phi.shape != (4, 2) or psi.shape != (4, 2):
        raise ValueError("Probe diagnostics require four parent slots with two probes each")
    mean_phi = phi.mean(axis=1, keepdims=True)
    mean_psi = psi.mean(axis=1, keepdims=True)
    centered_phi = (phi - mean_phi).ravel()
    centered_psi = (psi - mean_psi).ravel()
    if np.var(centered_phi) == 0.0 or np.var(centered_psi) == 0.0:
        rho_squared = 0.0
    else:
        rho_squared = float(np.corrcoef(centered_phi, centered_psi)[0, 1] ** 2)
    within_slot = np.sum((psi - mean_psi) ** 2, axis=1)
    within = float(within_slot.mean())
    mean_variance = float(np.var(mean_psi[:, 0], ddof=1))
    outer_noise = float(np.mean(within_slot / 2.0))
    return {
        "rho_squared": rho_squared,
        "within_variation": within,
        "inner_need": rho_squared * within,
        "raw_parent_mean_variance": mean_variance,
        "outer_need": max(0.0, mean_variance - outer_noise),
    }


def choose_allocation(
    outer_need: float,
    inner_need: float,
    actions: Sequence[tuple[int, int]] = ((12, 2), (6, 4), (3, 8)),
) -> tuple[int, int]:
    risks = np.asarray(
        [float(outer_need) / k + float(inner_need) / m for k, m in actions],
        dtype=np.float64,
    )
    minimum = float(risks.min())
    tied = [action for action, risk in zip(actions, risks) if abs(float(risk) - minimum) <= 1e-15]
    if len(tied) > 1:
        balanced = (6, 4)
        return balanced if balanced in actions else tied[0]
    return tied[0]


def mean_incremental_weight(incremental_weights: Sequence[float]) -> float:
    values = np.asarray(incremental_weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Incremental weights must be a non-empty one-dimensional sequence")
    return float(values.mean())


def _probe_population(
    process: ProcessParameters,
    paths: Sequence[tuple[int, ...]],
    normalized_outer_weights: np.ndarray,
    step: int,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    parent_indices = systematic_resample(normalized_outer_weights, rng, n=4)
    phi = np.zeros((4, 2), dtype=np.float64)
    psi = np.zeros((4, 2), dtype=np.float64)
    for slot, parent_index in enumerate(parent_indices):
        parent = paths[int(parent_index)][-1]
        transition = transition_spec(process, parent, step)
        kernel = reverse_kernel(transition)
        endpoint_values = endpoint_potentials(transition, step, epsilon)
        phi_y2 = kernel.phi(endpoint_values, "Y2")
        for probe_index in range(2):
            _, y2, _, x = sample_reverse_path(kernel, rng)
            phi[slot, probe_index] = phi_y2[y2]
            psi[slot, probe_index] = endpoint_values[x]
    return compute_probe_diagnostics(phi, psi)


def _run_adaptive_sampler(
    process: ProcessParameters,
    epsilon: float,
    seed_sequence: np.random.SeedSequence,
    actions: Sequence[tuple[int, int]] = ((12, 2), (6, 4), (3, 8)),
) -> SamplerOutput:
    step_streams = seed_sequence.spawn(5)
    paths: list[tuple[int, ...]] = [(S,)]
    normalizer = 1.0
    schedule: list[tuple[int, int]] = []
    diagnostics = []
    reverse_evaluations = 0
    for step, step_stream in enumerate(step_streams, start=1):
        probe_stream, production_stream = step_stream.spawn(2)
        probe_rng = np.random.default_rng(probe_stream)
        production_rng = np.random.default_rng(production_stream)
        normalized_weights = np.full(len(paths), 1.0 / len(paths), dtype=np.float64)
        if step == 1:
            outer_k, inner_m = 32, 1
        else:
            diagnostic = _probe_population(
                process,
                paths,
                normalized_weights,
                step,
                epsilon,
                probe_rng,
            )
            outer_k, inner_m = choose_allocation(
                diagnostic["outer_need"], diagnostic["inner_need"], actions
            )
            diagnostics.append({"step": float(step), **diagnostic})
            reverse_evaluations += 3 * 8
        schedule.append((outer_k, inner_m))
        parent_indices = systematic_resample(
            normalized_weights, production_rng, n=int(outer_k)
        )
        selected_parents = [paths[int(index)] for index in parent_indices]
        proposals = []
        local_normalizers = []
        for parent_path in selected_parents:
            proposal = duet_local_proposal(
                process,
                parent_path[-1],
                step,
                inner_m,
                epsilon,
                "Y2",
                production_rng,
            )
            proposals.append(parent_path + (proposal.next_state,))
            local_normalizers.append(proposal.local_normalizer)
        reverse_evaluations += 3 * outer_k * inner_m
        incremental = np.asarray(
            [
                local_z / potential(step - 1, parent[-1], epsilon)
                for parent, local_z in zip(selected_parents, local_normalizers)
            ],
            dtype=np.float64,
        )
        normalizer *= mean_incremental_weight(incremental)
        output_indices = systematic_resample(incremental, production_rng, n=int(outer_k))
        paths = [proposals[int(index)] for index in output_indices]
    return SamplerOutput(
        paths=paths,
        normalizer=normalizer,
        schedule=schedule,
        diagnostics=diagnostics,
        reverse_transition_evaluations=reverse_evaluations,
    )


def _route_estimate(paths: Sequence[tuple[int, ...]]) -> tuple[float, float]:
    route_l = float(np.mean([path[1] == L for path in paths]))
    route_r = float(np.mean([path[1] == R for path in paths]))
    return route_l, route_r


def _success_estimate(paths: Sequence[tuple[int, ...]]) -> float:
    return float(np.mean([path[-2:] == (O, O) for path in paths]))


def _normalizer_summary(values: np.ndarray, exact_value: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    signed_bias = (mean - exact_value) / exact_value
    half_width = 1.96 * standard_error / exact_value
    return {
        "mean": mean,
        "signed_relative_bias": signed_bias,
        "absolute_relative_bias": abs(signed_bias),
        "relative_bias_95pct_ci": [signed_bias - half_width, signed_bias + half_width],
        "confidence_interval_contains_zero": (
            signed_bias - half_width <= 0.0 <= signed_bias + half_width
        ),
    }


def _target_tv(
    empirical_mass: dict[tuple[int, ...], float],
    exact: ExactTarget,
) -> float:
    exact_mass = {
        path: float(value)
        for path, value in zip(exact.paths, exact.target_probabilities)
    }
    support = set(exact_mass) | set(empirical_mass)
    return 0.5 * sum(
        abs(empirical_mass.get(path, 0.0) - exact_mass.get(path, 0.0))
        for path in support
    )


def evaluate_cell(
    process: ProcessParameters,
    *,
    method: str,
    method_id: int,
    repetitions: int,
    master_seed: int,
    epsilon: float,
    outer_k: int | None = None,
    inner_m: int | None = None,
    checkpoint: str = "Y2",
    adaptive_actions: Sequence[tuple[int, int]] = ((12, 2), (6, 4), (3, 8)),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    exact = enumerate_target(process, epsilon)
    success_estimates = np.zeros(repetitions, dtype=np.float64)
    route_l_estimates = np.zeros(repetitions, dtype=np.float64)
    route_r_estimates = np.zeros(repetitions, dtype=np.float64)
    normalizers = np.full(repetitions, np.nan, dtype=np.float64)
    reverse_cost = np.zeros(repetitions, dtype=np.int64)
    schedules = np.full((repetitions, 5, 2), -1, dtype=np.int16)
    diagnostic_values = np.full((repetitions, 4, 3), np.nan, dtype=np.float64)
    empirical_mass: dict[tuple[int, ...], float] = {}
    for repetition in range(repetitions):
        seed_sequence = repetition_seed_sequence(
            master_seed, process.instance_id, method_id, repetition
        )
        if method == "adaptive":
            output = _run_adaptive_sampler(process, epsilon, seed_sequence, adaptive_actions)
        else:
            if outer_k is None or inner_m is None:
                raise ValueError("Fixed methods require outer_k and inner_m")
            output = _run_fixed_sampler(
                process,
                method,
                outer_k,
                inner_m,
                epsilon,
                seed_sequence,
                checkpoint,
            )
        success_estimates[repetition] = _success_estimate(output.paths)
        route_l_estimates[repetition], route_r_estimates[repetition] = _route_estimate(output.paths)
        if output.normalizer is not None:
            normalizers[repetition] = output.normalizer
        reverse_cost[repetition] = output.reverse_transition_evaluations
        schedules[repetition] = np.asarray(output.schedule, dtype=np.int16)
        for index, diagnostic in enumerate(output.diagnostics):
            diagnostic_values[repetition, index] = (
                diagnostic["rho_squared"],
                diagnostic["inner_need"],
                diagnostic["outer_need"],
            )
        per_path_mass = 1.0 / (repetitions * len(output.paths))
        for path in output.paths:
            empirical_mass[path] = empirical_mass.get(path, 0.0) + per_path_mass
    success_rmse = float(
        np.sqrt(np.mean((success_estimates - exact.target_success_probability) ** 2))
    )
    route_error_squared = (
        (route_l_estimates - exact.route_l_mass) ** 2
        + (route_r_estimates - exact.route_r_mass) ** 2
    ) / 2.0
    route_rmse = float(np.sqrt(np.mean(route_error_squared)))
    proper = method in {"outer_only", "complete_nested", "fixed_duet", "duet_y1", "adaptive"}
    finite_normalizers = normalizers[np.isfinite(normalizers)]
    summary: dict[str, Any] = {
        "setting": process.name,
        "instance_id": process.instance_id,
        "method": method,
        "method_id": method_id,
        "outer_k": outer_k,
        "inner_m": inner_m,
        "checkpoint": (
            checkpoint
            if method in {"fixed_duet", "duet_y1", "adaptive", "inner_only"}
            else None
        ),
        "repetitions": repetitions,
        "mean_terminal_success_estimate": float(success_estimates.mean()),
        "terminal_success_rmse": success_rmse,
        "mean_route_l_estimate": float(route_l_estimates.mean()),
        "mean_route_r_estimate": float(route_r_estimates.mean()),
        "route_mass_rmse": route_rmse,
        "equal_weight_mean_rmse": 0.5 * (success_rmse + route_rmse),
        "target_tv": _target_tv(empirical_mass, exact) if proper else None,
        "normalizer": (
            _normalizer_summary(finite_normalizers, exact.normalizer)
            if proper and len(finite_normalizers)
            else None
        ),
        "reverse_transition_evaluations_per_repetition": sorted(
            set(int(item) for item in reverse_cost)
        ),
    }
    if method == "frozen":
        summary["uncontrolled_terminal_success_mean"] = float(success_estimates.mean())
        summary["uncontrolled_route_l_mean"] = float(route_l_estimates.mean())
        summary["uncontrolled_route_r_mean"] = float(route_r_estimates.mean())
    if method == "adaptive":
        flat = diagnostic_values[np.isfinite(diagnostic_values).all(axis=2)]
        summary["mean_probe_rho_squared"] = float(np.mean(flat[:, 0]))
        summary["mean_probe_inner_need"] = float(np.mean(flat[:, 1]))
        summary["mean_probe_outer_need"] = float(np.mean(flat[:, 2]))
        unique, counts = np.unique(schedules.reshape(-1, 2), axis=0, return_counts=True)
        summary["allocation_counts"] = {
            f"K{int(k)}_M{int(m)}": int(count)
            for (k, m), count in zip(unique, counts)
        }
    raw = {
        "terminal_success_estimate": success_estimates,
        "route_l_estimate": route_l_estimates,
        "route_r_estimate": route_r_estimates,
        "normalizer_estimate": normalizers,
        "reverse_transition_evaluations": reverse_cost,
        "schedule": schedules,
        "probe_diagnostics_rho2_inner_outer": diagnostic_values,
    }
    return summary, raw


def _cell_path(output: Path, phase: str, setting: str, method: str) -> Path:
    return output / "cells" / phase / setting / method


def _run_or_load_cell(
    output: Path,
    phase: str,
    process: ProcessParameters,
    method_label: str,
    *,
    resume: bool,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    directory = _cell_path(output, phase, process.name, method_label)
    summary_path = directory / "summary.json"
    raw_path = directory / "raw_repetitions.npz"
    if resume and summary_path.exists() and raw_path.exists():
        print(f"PHASE_A_RESUME phase={phase} setting={process.name} method={method_label}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"PHASE_A_START phase={phase} setting={process.name} method={method_label}")
    summary, raw = evaluate_cell(process, **kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **raw)
    # The summary is the completion marker and is written only after the raw archive.
    write_json(summary_path, summary)
    print(f"PHASE_A_DONE phase={phase} setting={process.name} method={method_label}")
    return summary


def canonical_processes() -> list[ProcessParameters]:
    result = []
    index = 0
    for demand, (s_l, s_r) in (
        ("low", (0.80, 0.80)),
        ("high", (0.98, 0.62)),
    ):
        for rarity, commitment in (("common", 0.20), ("rare", 0.02)):
            result.append(
                ProcessParameters(
                    name=f"{rarity}_{demand}_informative",
                    instance_id=index,
                    s_l=s_l,
                    s_r=s_r,
                    commitment_probability=commitment,
                    commitment_beta2=0.20,
                    informative=True,
                    split="canonical",
                )
            )
            index += 1
    for demand, (s_l, s_r) in (
        ("low", (0.80, 0.80)),
        ("high", (0.98, 0.62)),
    ):
        result.append(
            ProcessParameters(
                name=f"rare_{demand}_no_information",
                instance_id=index,
                s_l=s_l,
                s_r=s_r,
                commitment_probability=0.02,
                commitment_beta2=0.50,
                informative=False,
                split="no_information",
            )
        )
        index += 1
    return result


def generate_held_out_processes(master_seed: int, count: int = 20) -> list[ProcessParameters]:
    rng = np.random.default_rng(np.random.SeedSequence(int(master_seed)))
    result = []
    for index in range(int(count)):
        commitment = float(np.exp(rng.uniform(np.log(0.01), np.log(0.25))))
        d = float(rng.uniform(0.0, 0.18))
        beta2 = float(rng.uniform(0.1, 0.5))
        high_on_l = bool(rng.random() < 0.5)
        s_high, s_low = 0.8 + d, 0.8 - d
        result.append(
            ProcessParameters(
                name=f"held_out_{index:02d}",
                instance_id=100 + index,
                s_l=s_high if high_on_l else s_low,
                s_r=s_low if high_on_l else s_high,
                commitment_probability=commitment,
                commitment_beta2=beta2,
                informative=beta2 < 0.5,
                split="held_out",
            )
        )
    return result


def _exact_payload(process: ProcessParameters, epsilon: float) -> dict[str, Any]:
    exact = enumerate_target(process, epsilon)
    parents_by_step = {1: [S], 2: [L, R, F], 3: [L, R, F], 4: [C_L, C_R, F], 5: [O, F]}
    inner = {}
    maximum_error = 0.0
    for step, parents in parents_by_step.items():
        for parent in parents:
            key = f"t{step}_{STATE_NAMES[parent]}"
            inner[key] = enumerate_inner_paths(process, parent, step)
            maximum_error = max(maximum_error, inner_marginal_error(process, parent, step))
    return {
        "parameters": process.__dict__,
        "physical_paths": [
            {
                "states": [STATE_NAMES[state] for state in path],
                "base_probability": float(base),
                "psi5": float(psi),
                "target_probability": float(target),
            }
            for path, base, psi, target in zip(
                exact.paths,
                exact.base_probabilities,
                exact.potentials,
                exact.target_probabilities,
            )
        ],
        "normalizer": exact.normalizer,
        "base_terminal_success_probability": exact.base_success_probability,
        "target_terminal_success_probability": exact.target_success_probability,
        "route_l_target_mass": exact.route_l_mass,
        "route_r_target_mass": exact.route_r_mass,
        "maximum_inner_marginal_error": maximum_error,
        "inner_diffusion_paths": inner,
    }


def _validate_exact_spec(cfg: dict[str, Any]) -> None:
    phase = cfg.get("phase_a", {})
    expected = {
        "model.reverse_steps": (cfg["model"].get("reverse_steps"), 3),
        "trajectory.horizon": (cfg["trajectory"].get("horizon"), 5),
        "phase_a.epsilon": (phase.get("epsilon"), 0.05),
        "phase_a.beta1": (phase.get("beta1"), 0.05),
        "phase_a.beta2": (phase.get("beta2"), 0.20),
        "phase_a.beta3": (phase.get("beta3"), 0.50),
        "phase_a.repetitions": (phase.get("repetitions"), 5000),
        "phase_a.master_seed": (phase.get("master_seed"), 20260908),
        "phase_a.held_out_count": (phase.get("held_out_count"), 20),
        "phase_a.main_checkpoint": (phase.get("main_checkpoint"), "Y2"),
        "experiment.repetitions": (cfg["experiment"].get("repetitions"), 5000),
        "experiment.seed": (cfg["experiment"].get("seed"), 20260908),
        "experiment.seeds": (cfg["experiment"].get("seeds"), [20260908]),
    }
    # Keep the error list exhaustive so a single validation run reports every deviation.
    errors = []
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            errors.append(f"{label} must be {wanted!r}, got {actual!r}")
    if str(cfg["model"].get("dtype")) != "float64":
        errors.append("model.dtype must be 'float64'")
    fixed = phase.get("fixed_methods", {})
    required_fixed = {
        "frozen": [32, 1],
        "outer_only": [32, 1],
        "inner_only": [1, 32],
        "complete_nested": [8, 4],
        "fixed_duet": [8, 4],
    }
    if fixed != required_fixed:
        errors.append(f"phase_a.fixed_methods must equal {required_fixed!r}")
    if phase.get("static_allocations") != [[16, 2], [8, 4], [4, 8], [2, 16]]:
        errors.append("phase_a.static_allocations must be [[16,2],[8,4],[4,8],[2,16]]")
    adaptive = phase.get("adaptive", {})
    if adaptive.get("initial_allocation") != [32, 1]:
        errors.append("phase_a.adaptive.initial_allocation must be [32, 1]")
    if adaptive.get("probe_parent_slots") != 4 or adaptive.get("probes_per_slot") != 2:
        errors.append("adaptive probe layout must be four parent slots times two probes")
    if adaptive.get("actions") != [[12, 2], [6, 4], [3, 8]]:
        errors.append("phase_a.adaptive.actions must be [[12,2],[6,4],[3,8]]")
    if adaptive.get("tie_break") != [6, 4]:
        errors.append("phase_a.adaptive.tie_break must be [6,4]")
    diagnostic = phase.get("checkpoint_diagnostic", {})
    if diagnostic.get("representative_setting") != "rare_low_informative":
        errors.append(
            "phase_a.checkpoint_diagnostic.representative_setting must be "
            "'rare_low_informative'"
        )
    if diagnostic.get("stages") != ["Y3", "Y2", "Y1"]:
        errors.append("phase_a.checkpoint_diagnostic.stages must be [Y3,Y2,Y1]")
    if diagnostic.get("primary_metric") != "terminal_success_rmse":
        errors.append(
            "phase_a.checkpoint_diagnostic.primary_metric must be "
            "'terminal_success_rmse'"
        )
    held_out = phase.get("held_out", {})
    required_held_out = {
        "commitment_probability": "log_uniform_0.01_0.25",
        "route_heterogeneity": "uniform_0.0_0.18",
        "checkpoint_beta2": "uniform_0.1_0.5",
        "high_route_assignment_probability": 0.5,
    }
    if held_out != required_held_out:
        errors.append(f"phase_a.held_out must equal {required_held_out!r}")
    if errors:
        raise ValueError(
            "Phase A config does not match the frozen specification:\n- "
            + "\n- ".join(errors)
        )


def validate_phase_a_config(cfg: dict[str, Any]) -> dict[str, Any]:
    _validate_exact_spec(cfg)
    canonical = canonical_processes()
    held_out = generate_held_out_processes(20260908, 20)
    maximum_error = max(
        inner_marginal_error(process, parent, step)
        for process in canonical + held_out
        for step, parents in {
            1: [S],
            2: [L, R, F],
            3: [L, R, F],
            4: [C_L, C_R, F],
            5: [O, F],
        }.items()
        for parent in parents
    )
    if maximum_error >= 1e-12:
        raise AssertionError(f"Inner reverse marginal error {maximum_error} is not below 1e-12")
    no_information = next(item for item in canonical if item.name == "rare_low_no_information")
    transition = transition_spec(no_information, L, 3)
    kernel = reverse_kernel(transition)
    phi = kernel.phi(endpoint_potentials(transition, 3, 0.05), "Y2")
    if not np.allclose(phi, phi[0], atol=1e-12, rtol=0.0):
        raise AssertionError("No-information Y2 phi is not constant")
    return {
        "valid": True,
        "maximum_inner_marginal_error": maximum_error,
        "no_information_phi_y2": phi.tolist(),
        "planned_fixed_cells": 6 * 5,
        "planned_checkpoint_diagnostic_new_cells": 1,
        "checkpoint_diagnostic_reuses_fixed_cells": 2,
        "planned_development_allocation_new_cells": 4 * 3,
        "development_allocation_reuses_fixed_cells": 4,
        "planned_held_out_cells": 20 * 6,
        "independently_evaluated_cells": 163,
        "repetitions_per_cell": 5000,
        "total_repetitions": 163 * 5000,
        "total_reverse_transition_evaluations": 163 * 5000 * 480,
    }


def _save_exact_enumeration(
    output: Path,
    processes: Iterable[ProcessParameters],
    epsilon: float,
) -> dict[str, Any]:
    summaries = {}
    for process in processes:
        payload = _exact_payload(process, epsilon)
        write_json(output / "exact" / f"{process.name}.json", payload)
        summaries[process.name] = {
            key: payload[key]
            for key in (
                "normalizer",
                "base_terminal_success_probability",
                "target_terminal_success_probability",
                "route_l_target_mass",
                "route_r_target_mass",
                "maximum_inner_marginal_error",
            )
        }
    return summaries


def _method_id_for_allocation(index: int) -> int:
    return 100 + int(index)


def _correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    result = spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return {"spearman_rho": float(result.statistic), "p_value": float(result.pvalue)}


def run_phase_a_cross_clock(
    cfg: dict[str, Any],
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    validation = validate_phase_a_config(cfg)
    output = resolve_config_path(cfg, cfg["experiment"]["output_directory"])
    if resume and (output / "metrics.json").exists():
        return output
    initialize_run_directory(
        output,
        cfg,
        command=" ".join(shlex.quote(item) for item in sys.argv),
        overwrite=overwrite,
        resume=resume,
    )
    started = time.perf_counter()
    phase = cfg["phase_a"]
    epsilon = float(phase["epsilon"])
    repetitions = int(phase["repetitions"])
    master_seed = int(phase["master_seed"])
    static_allocations = [tuple(int(value) for value in row) for row in phase["static_allocations"]]
    adaptive_actions = [tuple(int(value) for value in row) for row in phase["adaptive"]["actions"]]
    canonical = canonical_processes()
    development = [item for item in canonical if item.split == "canonical"]
    held_out = generate_held_out_processes(master_seed, int(phase["held_out_count"]))
    write_json(output / "validation.json", validation)
    write_json(
        output / "held_out_parameters.json",
        [
            {
                **item.__dict__,
                "d": abs(item.s_l - item.s_r) / 2.0,
                "higher_survival_route": "L" if item.s_l > item.s_r else "R",
            }
            for item in held_out
        ],
    )
    exact = _save_exact_enumeration(output, canonical + held_out, epsilon)

    fixed_results = []
    for process in canonical:
        for method, allocation in phase["fixed_methods"].items():
            k, m = (int(value) for value in allocation)
            fixed_results.append(
                _run_or_load_cell(
                    output,
                    "fixed_methods",
                    process,
                    method,
                    resume=resume,
                    kwargs={
                        "method": method,
                        "method_id": FIXED_METHOD_IDS[method],
                        "repetitions": repetitions,
                        "master_seed": master_seed,
                        "epsilon": epsilon,
                        "outer_k": k,
                        "inner_m": m,
                        "checkpoint": "Y2",
                    },
                )
            )

    fixed_comparisons = []
    for process in canonical:
        complete = next(
            row
            for row in fixed_results
            if row["setting"] == process.name and row["method"] == "complete_nested"
        )
        duet = next(
            row
            for row in fixed_results
            if row["setting"] == process.name and row["method"] == "fixed_duet"
        )
        fixed_comparisons.append(
            {
                "setting": process.name,
                "informative": process.informative,
                "success_gain_pre": 1.0
                - duet["terminal_success_rmse"]
                / complete["terminal_success_rmse"],
                "route_gain_pre": 1.0
                - duet["route_mass_rmse"]
                / complete["route_mass_rmse"],
                "equal_weight_gain_pre": 1.0
                - duet["equal_weight_mean_rmse"]
                / complete["equal_weight_mean_rmse"],
            }
        )

    representative = next(item for item in canonical if item.name == "rare_low_informative")
    transition = transition_spec(representative, L, 3)
    kernel = reverse_kernel(transition)
    endpoint_values = endpoint_potentials(transition, 3, epsilon)
    stage_information = {
        stage: kernel.chi(endpoint_values, stage) for stage in ("Y3", "Y2", "Y1")
    }
    # Complete Nested and Y2 DuET are the same K=8, M=4 cells already evaluated
    # in A2. Reuse them rather than silently adding duplicate Monte Carlo work.
    stage_results = [
        next(
            row
            for row in fixed_results
            if row["setting"] == representative.name
            and row["method"] == "complete_nested"
        ),
        next(
            row
            for row in fixed_results
            if row["setting"] == representative.name and row["method"] == "fixed_duet"
        ),
        _run_or_load_cell(
            output,
            "checkpoint_diagnostic",
            representative,
            "duet_y1",
            resume=resume,
            kwargs={
                "method": "duet_y1",
                "method_id": FIXED_METHOD_IDS["duet_y1"],
                "repetitions": repetitions,
                "master_seed": master_seed,
                "epsilon": epsilon,
                "outer_k": 8,
                "inner_m": 4,
                "checkpoint": "Y1",
            },
        ),
    ]

    development_results = []
    for process in development:
        for allocation_index, (k, m) in enumerate(static_allocations):
            label = f"K{k}_M{m}"
            if (k, m) == (8, 4):
                reused = next(
                    row
                    for row in fixed_results
                    if row["setting"] == process.name
                    and row["method"] == "fixed_duet"
                )
                development_results.append({**reused, "allocation_label": label})
                continue
            development_results.append(
                _run_or_load_cell(
                    output,
                    "development_allocations",
                    process,
                    label,
                    resume=resume,
                    kwargs={
                        "method": "fixed_duet",
                        "method_id": _method_id_for_allocation(allocation_index),
                        "repetitions": repetitions,
                        "master_seed": master_seed,
                        "epsilon": epsilon,
                        "outer_k": k,
                        "inner_m": m,
                        "checkpoint": "Y2",
                    },
                )
            )
    dev_scores = {}
    for k, m in static_allocations:
        rows = [
            row
            for row in development_results
            if row["outer_k"] == k and row["inner_m"] == m
        ]
        dev_scores[f"K{k}_M{m}"] = float(
            np.mean([row["equal_weight_mean_rmse"] for row in rows])
        )
    global_label = min(dev_scores, key=dev_scores.get)
    global_index = [f"K{k}_M{m}" for k, m in static_allocations].index(global_label)
    global_allocation = static_allocations[global_index]

    held_out_results: dict[str, dict[str, dict[str, Any]]] = {}
    for process in held_out:
        rows: dict[str, dict[str, Any]] = {}
        for allocation_index, (k, m) in enumerate(static_allocations):
            label = f"K{k}_M{m}"
            rows[label] = _run_or_load_cell(
                output,
                "held_out",
                process,
                label,
                resume=resume,
                kwargs={
                    "method": "fixed_duet",
                    "method_id": _method_id_for_allocation(allocation_index),
                    "repetitions": repetitions,
                    "master_seed": master_seed,
                    "epsilon": epsilon,
                    "outer_k": k,
                    "inner_m": m,
                    "checkpoint": "Y2",
                },
            )
        rows["complete_nested_K8_M4"] = _run_or_load_cell(
            output,
            "held_out",
            process,
            "complete_nested_K8_M4",
            resume=resume,
            kwargs={
                "method": "complete_nested",
                "method_id": FIXED_METHOD_IDS["complete_nested"],
                "repetitions": repetitions,
                "master_seed": master_seed,
                "epsilon": epsilon,
                "outer_k": 8,
                "inner_m": 4,
            },
        )
        rows["adaptive"] = _run_or_load_cell(
            output,
            "held_out",
            process,
            "adaptive",
            resume=resume,
            kwargs={
                "method": "adaptive",
                "method_id": FIXED_METHOD_IDS["adaptive"],
                "repetitions": repetitions,
                "master_seed": master_seed,
                "epsilon": epsilon,
                "adaptive_actions": adaptive_actions,
            },
        )
        held_out_results[process.name] = rows

    held_out_summary = []
    rho_values, gain_values, route_gain_values, combined_gain_values = [], [], [], []
    outer_need_values, outer_gain_values = [], []
    outer_route_gain_values, outer_combined_gain_values = [], []
    for process in held_out:
        rows = held_out_results[process.name]
        adaptive = rows["adaptive"]
        global_static = rows[global_label]
        duet_k8_m4 = rows["K8_M4"]
        complete = rows["complete_nested_K8_M4"]
        oracle_label = min(
            (f"K{k}_M{m}" for k, m in static_allocations),
            key=lambda label: rows[label]["equal_weight_mean_rmse"],
        )
        success_oracle_label = min(
            (f"K{k}_M{m}" for k, m in static_allocations),
            key=lambda label: rows[label]["terminal_success_rmse"],
        )
        route_oracle_label = min(
            (f"K{k}_M{m}" for k, m in static_allocations),
            key=lambda label: rows[label]["route_mass_rmse"],
        )
        pre_gain = 1.0 - (
            duet_k8_m4["terminal_success_rmse"] / complete["terminal_success_rmse"]
        )
        pre_gain_route = 1.0 - (
            duet_k8_m4["route_mass_rmse"] / complete["route_mass_rmse"]
        )
        pre_gain_combined = 1.0 - (
            duet_k8_m4["equal_weight_mean_rmse"]
            / complete["equal_weight_mean_rmse"]
        )
        adaptive_score = adaptive["equal_weight_mean_rmse"]
        global_score = global_static["equal_weight_mean_rmse"]
        oracle_score = rows[oracle_label]["equal_weight_mean_rmse"]
        row = {
            "setting": process.name,
            "global_static": global_label,
            "hindsight_oracle": oracle_label,
            "adaptive_improvement": 1.0 - adaptive_score / global_score,
            "hindsight_oracle_regret": adaptive_score / oracle_score - 1.0,
            "pre_completion_success_rmse_gain": pre_gain,
            "pre_completion_route_rmse_gain": pre_gain_route,
            "pre_completion_equal_weight_rmse_gain": pre_gain_combined,
            "success_adaptive_improvement": 1.0
            - adaptive["terminal_success_rmse"]
            / global_static["terminal_success_rmse"],
            "route_adaptive_improvement": 1.0
            - adaptive["route_mass_rmse"]
            / global_static["route_mass_rmse"],
            "success_hindsight_oracle_regret": adaptive["terminal_success_rmse"]
            / rows[success_oracle_label]["terminal_success_rmse"]
            - 1.0,
            "route_hindsight_oracle_regret": adaptive["route_mass_rmse"]
            / rows[route_oracle_label]["route_mass_rmse"]
            - 1.0,
            "mean_probe_rho_squared": adaptive["mean_probe_rho_squared"],
            "mean_probe_outer_need": adaptive["mean_probe_outer_need"],
            "selected_schedule_counts": adaptive["allocation_counts"],
        }
        held_out_summary.append(row)
        rho_values.append(adaptive["mean_probe_rho_squared"])
        gain_values.append(pre_gain)
        route_gain_values.append(pre_gain_route)
        combined_gain_values.append(pre_gain_combined)
        outer_need_values.append(adaptive["mean_probe_outer_need"])
        outer_gain_values.append(
            rows["K2_M16"]["terminal_success_rmse"]
            - rows["K16_M2"]["terminal_success_rmse"]
        )
        outer_route_gain_values.append(
            rows["K2_M16"]["route_mass_rmse"]
            - rows["K16_M2"]["route_mass_rmse"]
        )
        outer_combined_gain_values.append(
            rows["K2_M16"]["equal_weight_mean_rmse"]
            - rows["K16_M2"]["equal_weight_mean_rmse"]
        )

    metrics = {
        "specification": "Phase A Exactly Enumerable Cross-Clock Diagnostic",
        "exact": exact,
        "fixed_methods": fixed_results,
        "fixed_duet_vs_complete": fixed_comparisons,
        "checkpoint_diagnostic": {
            "representative_setting": representative.name,
            "chi": stage_information,
            "information_strictly_increases": (
                stage_information["Y3"]
                < stage_information["Y2"]
                < stage_information["Y1"]
            ),
            "results": stage_results,
        },
        "development_allocation": {
            "candidate_scores": dev_scores,
            "selected_global_static": global_label,
            "results": development_results,
        },
        "held_out": {
            "process_count": len(held_out),
            "global_static": global_label,
            "per_process": held_out_summary,
            "mean_adaptive_improvement": float(
                np.mean([row["adaptive_improvement"] for row in held_out_summary])
            ),
            "mean_success_adaptive_improvement": float(
                np.mean(
                    [row["success_adaptive_improvement"] for row in held_out_summary]
                )
            ),
            "mean_route_adaptive_improvement": float(
                np.mean(
                    [row["route_adaptive_improvement"] for row in held_out_summary]
                )
            ),
            "mean_hindsight_oracle_regret": float(
                np.mean([row["hindsight_oracle_regret"] for row in held_out_summary])
            ),
        },
        "diagnostic_calibration": {
            "rho_squared_vs_pre_completion_gain": _correlation(rho_values, gain_values),
            "rho_squared_vs_pre_completion_route_gain": _correlation(
                rho_values, route_gain_values
            ),
            "rho_squared_vs_pre_completion_equal_weight_gain": _correlation(
                rho_values, combined_gain_values
            ),
            "outer_need_vs_smallK_minus_largeK_success_rmse": _correlation(
                outer_need_values, outer_gain_values
            ),
            "outer_need_vs_smallK_minus_largeK_route_rmse": _correlation(
                outer_need_values, outer_route_gain_values
            ),
            "outer_need_vs_smallK_minus_largeK_equal_weight_rmse": _correlation(
                outer_need_values, outer_combined_gain_values
            ),
        },
        "wall_clock_s": time.perf_counter() - started,
    }
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "nfe_accounting.json",
        {
            "backend": "exactly_enumerable_three_step_inner_process",
            "reverse_transition_evaluations_per_fixed_repetition": 5 * 3 * 32,
            "reverse_transition_evaluations_per_adaptive_repetition": 5 * 96,
            "independently_evaluated_cells": 163,
            "repetitions_per_cell": repetitions,
            "total_reverse_transition_evaluations": 163 * repetitions * 480,
            "neural_decoder_calls": 0,
        },
    )
    write_json(
        output / "wall_clock.json",
        {"seconds": metrics["wall_clock_s"], "peak_gpu_memory_bytes": 0},
    )
    write_json(
        output / "seed_manifest.json",
        {
            "master_seed": master_seed,
            "derivation": "SeedSequence([20260908, instance_id, method_id, repetition_id])",
            "probe_and_production": "separate spawned substreams",
            "fixed_method_ids": FIXED_METHOD_IDS,
            "static_allocation_method_ids": {
                f"K{k}_M{m}": _method_id_for_allocation(index)
                for index, (k, m) in enumerate(static_allocations)
            },
        },
    )
    return output
