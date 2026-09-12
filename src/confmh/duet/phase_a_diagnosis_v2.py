from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from confmh.duet.config import load_duet_config, resolve_config_path
from confmh.duet.phase_a_cross_clock import (
    C_L,
    C_R,
    F,
    L,
    O,
    R,
    S,
    ExactTarget,
    ProcessParameters,
    _continue_from_checkpoint,
    _normalizer_summary,
    _sample_to_checkpoint,
    _success_estimate,
    _target_tv,
    canonical_processes,
    choose_allocation,
    complete_nested_local_proposal,
    compute_probe_diagnostics,
    duet_local_proposal,
    endpoint_potentials,
    enumerate_target,
    generate_held_out_processes,
    mapped_state,
    mean_incremental_weight,
    potential,
    repetition_seed_sequence,
    reverse_kernel,
    sample_reverse_path,
    transition_spec,
)
from confmh.duet.records import write_json
from confmh.duet.resampling import systematic_resample


ACTIONS = ((12, 2), (6, 4), (3, 8))
METHOD_IDS = {
    "current_probe_allocator": 2000,
    "exact_stat_allocator": 2001,
    "matched_production_k12_m2": 2002,
    "global_static_k16_m2": 2003,
    "duet_y1": 2100,
    "temporal_static": 2200,
    "temporal_t1": 2201,
    "temporal_t2": 2202,
    "temporal_t3": 2203,
    "temporal_t4": 2204,
    "temporal_t5": 2205,
}
DIAGNOSTIC_FIELDS = (
    "rho_squared",
    "inner_need",
    "outer_need",
    "zero_phi_variance",
    "zero_psi_variance",
    "tie",
    "fallback",
    "exact_statistics",
)


@dataclass
class DiagnosticOutput:
    pre_paths: list[tuple[int, ...]]
    pre_weights: np.ndarray
    post_paths: list[tuple[int, ...]]
    normalizer: float
    schedule: list[tuple[int, int]]
    diagnostics: list[dict[str, float]]
    reverse_transition_evaluations: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_informative() -> list[ProcessParameters]:
    return [process for process in canonical_processes() if process.split == "canonical"]


def _route_values(paths: Sequence[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([path[1] == L for path in paths], dtype=np.float64),
        np.asarray([path[1] == R for path in paths], dtype=np.float64),
    )


def _success_values(paths: Sequence[tuple[int, ...]]) -> np.ndarray:
    return np.asarray([path[-2:] == (O, O) for path in paths], dtype=np.float64)


def _allocation_decision(outer_need: float, inner_need: float) -> tuple[tuple[int, int], bool]:
    risks = np.asarray(
        [float(outer_need) / k + float(inner_need) / m for k, m in ACTIONS],
        dtype=np.float64,
    )
    minimum = float(risks.min())
    tied = np.isclose(risks, minimum, atol=1e-15, rtol=0.0)
    return choose_allocation(outer_need, inner_need, ACTIONS), bool(tied.sum() > 1)


def _instrumented_probe(
    process: ProcessParameters,
    paths: Sequence[tuple[int, ...]],
    step: int,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    weights = np.full(len(paths), 1.0 / len(paths), dtype=np.float64)
    parent_indices = systematic_resample(weights, rng, n=4)
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
    result = compute_probe_diagnostics(phi, psi)
    centered_phi = phi - phi.mean(axis=1, keepdims=True)
    centered_psi = psi - psi.mean(axis=1, keepdims=True)
    action, tied = _allocation_decision(result["outer_need"], result["inner_need"])
    return {
        **result,
        "zero_phi_variance": float(np.var(centered_phi) == 0.0),
        "zero_psi_variance": float(np.var(centered_psi) == 0.0),
        "tie": float(tied),
        "fallback": 0.0,
        "exact_statistics": 0.0,
        "selected_k": float(action[0]),
        "selected_m": float(action[1]),
    }


def exact_population_diagnostics(
    process: ProcessParameters,
    paths: Sequence[tuple[int, ...]],
    step: int,
    epsilon: float,
) -> dict[str, float]:
    """Population-exact counterparts of the residualized eight-probe statistics."""

    if not paths:
        raise ValueError("Exact diagnostics require a non-empty outer population")
    parent_weight = 1.0 / len(paths)
    parent_means: list[float] = []
    residual_phi_variance = 0.0
    residual_psi_variance = 0.0
    residual_covariance = 0.0
    for path in paths:
        transition = transition_spec(process, path[-1], step)
        kernel = reverse_kernel(transition)
        endpoint_values = endpoint_potentials(transition, step, epsilon)
        joint_x_y2 = kernel.joint_forward.sum(axis=(1, 3))
        phi_y2 = kernel.phi(endpoint_values, "Y2")
        mean_psi = float(np.dot(kernel.probability_x, endpoint_values))
        probability_y2 = joint_x_y2.sum(axis=0)
        mean_phi = float(np.dot(probability_y2, phi_y2))
        parent_means.append(mean_psi)
        for x in range(2):
            for y2 in range(2):
                probability = parent_weight * float(joint_x_y2[x, y2])
                centered_phi = float(phi_y2[y2] - mean_phi)
                centered_psi = float(endpoint_values[x] - mean_psi)
                residual_phi_variance += probability * centered_phi**2
                residual_psi_variance += probability * centered_psi**2
                residual_covariance += probability * centered_phi * centered_psi
    if residual_phi_variance <= 0.0 or residual_psi_variance <= 0.0:
        rho_squared = 0.0
    else:
        rho_squared = residual_covariance**2 / (
            residual_phi_variance * residual_psi_variance
        )
    outer_need = float(np.var(np.asarray(parent_means, dtype=np.float64), ddof=0))
    inner_need = rho_squared * residual_psi_variance
    action, tied = _allocation_decision(outer_need, inner_need)
    return {
        "rho_squared": float(rho_squared),
        "within_variation": float(residual_psi_variance),
        "inner_need": float(inner_need),
        "raw_parent_mean_variance": outer_need,
        "outer_need": outer_need,
        "zero_phi_variance": float(residual_phi_variance <= 0.0),
        "zero_psi_variance": float(residual_psi_variance <= 0.0),
        "tie": float(tied),
        "fallback": 0.0,
        "exact_statistics": 1.0,
        "selected_k": float(action[0]),
        "selected_m": float(action[1]),
    }


def _run_adaptive_variant(
    process: ProcessParameters,
    epsilon: float,
    seed_sequence: np.random.SeedSequence,
    variant: str,
) -> DiagnosticOutput:
    if variant not in {"current_probe", "exact_stat", "matched"}:
        raise ValueError(f"Unknown adaptive diagnostic variant: {variant}")
    paths: list[tuple[int, ...]] = [(S,)]
    normalizer = 1.0
    schedule: list[tuple[int, int]] = []
    diagnostics: list[dict[str, float]] = []
    reverse_evaluations = 0
    pre_paths: list[tuple[int, ...]] | None = None
    pre_weights: np.ndarray | None = None
    for step, step_stream in enumerate(seed_sequence.spawn(5), start=1):
        probe_stream, production_stream = step_stream.spawn(2)
        probe_rng = np.random.default_rng(probe_stream)
        production_rng = np.random.default_rng(production_stream)
        if step == 1:
            outer_k, inner_m = 32, 1
        else:
            if variant == "current_probe":
                diagnostic = _instrumented_probe(
                    process, paths, step, epsilon, probe_rng
                )
                outer_k, inner_m = int(diagnostic["selected_k"]), int(
                    diagnostic["selected_m"]
                )
            elif variant == "exact_stat":
                diagnostic = exact_population_diagnostics(
                    process, paths, step, epsilon
                )
                outer_k, inner_m = int(diagnostic["selected_k"]), int(
                    diagnostic["selected_m"]
                )
            else:
                diagnostic = {
                    "rho_squared": math.nan,
                    "inner_need": math.nan,
                    "outer_need": math.nan,
                    "zero_phi_variance": math.nan,
                    "zero_psi_variance": math.nan,
                    "tie": 0.0,
                    "fallback": 0.0,
                    "exact_statistics": 0.0,
                    "selected_k": 12.0,
                    "selected_m": 2.0,
                }
                outer_k, inner_m = 12, 2
            diagnostic["step"] = float(step)
            diagnostics.append(diagnostic)
            # Eight full three-transition probe paths, whether used, ignored, or
            # replaced by exact oracle statistics.
            reverse_evaluations += 3 * 8
        schedule.append((outer_k, inner_m))
        parent_weights = np.full(len(paths), 1.0 / len(paths), dtype=np.float64)
        parent_indices = systematic_resample(
            parent_weights, production_rng, n=int(outer_k)
        )
        parents = [paths[int(index)] for index in parent_indices]
        proposals: list[tuple[int, ...]] = []
        local_normalizers = []
        for parent in parents:
            proposal = duet_local_proposal(
                process,
                parent[-1],
                step,
                inner_m,
                epsilon,
                "Y2",
                production_rng,
            )
            proposals.append(parent + (proposal.next_state,))
            local_normalizers.append(proposal.local_normalizer)
        reverse_evaluations += 3 * outer_k * inner_m
        incremental = np.asarray(
            [
                local_z / potential(step - 1, parent[-1], epsilon)
                for parent, local_z in zip(parents, local_normalizers)
            ],
            dtype=np.float64,
        )
        normalizer *= mean_incremental_weight(incremental)
        if step == 5:
            pre_paths = proposals
            pre_weights = incremental / incremental.sum()
        output_indices = systematic_resample(
            incremental, production_rng, n=int(outer_k)
        )
        paths = [proposals[int(index)] for index in output_indices]
    if pre_paths is None or pre_weights is None:
        raise RuntimeError("Final pre-resampling population was not captured")
    if reverse_evaluations != 480:
        raise AssertionError(f"Adaptive variant cost is {reverse_evaluations}, not 480")
    return DiagnosticOutput(
        pre_paths=pre_paths,
        pre_weights=pre_weights,
        post_paths=paths,
        normalizer=normalizer,
        schedule=schedule,
        diagnostics=diagnostics,
        reverse_transition_evaluations=reverse_evaluations,
    )


def _run_fixed_schedule(
    process: ProcessParameters,
    epsilon: float,
    seed_sequence: np.random.SeedSequence,
    inner_schedule: Sequence[int],
    *,
    method: str = "fixed_duet",
    checkpoint: str = "Y2",
    outer_k: int = 16,
) -> DiagnosticOutput:
    if len(inner_schedule) != 5:
        raise ValueError("A physical-time inner schedule must contain five entries")
    paths: list[tuple[int, ...]] = [(S,)] * int(outer_k)
    normalizer = 1.0
    reverse_evaluations = 0
    schedule: list[tuple[int, int]] = []
    pre_paths: list[tuple[int, ...]] | None = None
    pre_weights: np.ndarray | None = None
    for step, (inner_m, stream) in enumerate(
        zip(inner_schedule, seed_sequence.spawn(5)), start=1
    ):
        rng = np.random.default_rng(stream)
        schedule.append((int(outer_k), int(inner_m)))
        proposals: list[tuple[int, ...]] = []
        local_normalizers = []
        for path in paths:
            if method == "complete_nested":
                proposal = complete_nested_local_proposal(
                    process, path[-1], step, inner_m, epsilon, rng
                )
            elif method in {"fixed_duet", "duet_y1"}:
                proposal = duet_local_proposal(
                    process,
                    path[-1],
                    step,
                    inner_m,
                    epsilon,
                    checkpoint,
                    rng,
                )
            else:
                raise ValueError(f"Unsupported fixed-schedule method: {method}")
            proposals.append(path + (proposal.next_state,))
            local_normalizers.append(proposal.local_normalizer)
        reverse_evaluations += 3 * int(outer_k) * int(inner_m)
        incremental = np.asarray(
            [
                local_z / potential(step - 1, path[-1], epsilon)
                for path, local_z in zip(paths, local_normalizers)
            ],
            dtype=np.float64,
        )
        normalizer *= mean_incremental_weight(incremental)
        if step == 5:
            pre_paths = proposals
            pre_weights = incremental / incremental.sum()
        ancestors = systematic_resample(incremental, rng, n=int(outer_k))
        paths = [proposals[int(index)] for index in ancestors]
    if pre_paths is None or pre_weights is None:
        raise RuntimeError("Final pre-resampling population was not captured")
    return DiagnosticOutput(
        pre_paths=pre_paths,
        pre_weights=pre_weights,
        post_paths=paths,
        normalizer=normalizer,
        schedule=schedule,
        diagnostics=[],
        reverse_transition_evaluations=reverse_evaluations,
    )


def _estimates(output: DiagnosticOutput) -> tuple[float, float, float, float, float, float]:
    pre_success = float(np.dot(output.pre_weights, _success_values(output.pre_paths)))
    pre_l_values, pre_r_values = _route_values(output.pre_paths)
    pre_l = float(np.dot(output.pre_weights, pre_l_values))
    pre_r = float(np.dot(output.pre_weights, pre_r_values))
    post_success = _success_estimate(output.post_paths)
    post_l_values, post_r_values = _route_values(output.post_paths)
    return (
        pre_success,
        pre_l,
        pre_r,
        post_success,
        float(post_l_values.mean()),
        float(post_r_values.mean()),
    )


def _rmse(values: np.ndarray, target: float) -> float:
    return float(np.sqrt(np.mean((values - float(target)) ** 2)))


def _route_rmse(left: np.ndarray, right: np.ndarray, exact: ExactTarget) -> float:
    squared = ((left - exact.route_l_mass) ** 2 + (right - exact.route_r_mass) ** 2) / 2.0
    return float(np.sqrt(np.mean(squared)))


def evaluate_diagnostic_cell(
    process: ProcessParameters,
    *,
    label: str,
    method_id: int,
    repetitions: int,
    master_seed: int,
    sampler: Callable[[np.random.SeedSequence], DiagnosticOutput],
    epsilon: float = 0.05,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    exact = enumerate_target(process, epsilon)
    estimates = np.zeros((repetitions, 6), dtype=np.float64)
    normalizers = np.zeros(repetitions, dtype=np.float64)
    costs = np.zeros(repetitions, dtype=np.int64)
    schedules = np.zeros((repetitions, 5, 2), dtype=np.int16)
    diagnostics = np.full((repetitions, 4, len(DIAGNOSTIC_FIELDS)), np.nan)
    pre_mass: dict[tuple[int, ...], float] = {}
    post_mass: dict[tuple[int, ...], float] = {}
    for repetition in range(repetitions):
        output = sampler(
            repetition_seed_sequence(
                master_seed, process.instance_id, method_id, repetition
            )
        )
        estimates[repetition] = _estimates(output)
        normalizers[repetition] = output.normalizer
        costs[repetition] = output.reverse_transition_evaluations
        schedules[repetition] = np.asarray(output.schedule, dtype=np.int16)
        for index, diagnostic in enumerate(output.diagnostics):
            diagnostics[repetition, index] = [
                diagnostic.get(field, math.nan) for field in DIAGNOSTIC_FIELDS
            ]
        for path, weight in zip(output.pre_paths, output.pre_weights):
            pre_mass[path] = pre_mass.get(path, 0.0) + float(weight) / repetitions
        for path in output.post_paths:
            post_mass[path] = post_mass.get(path, 0.0) + 1.0 / (
                repetitions * len(output.post_paths)
            )
    pre_success, pre_l, pre_r, post_success, post_l, post_r = estimates.T
    pre_success_rmse = _rmse(pre_success, exact.target_success_probability)
    pre_route_rmse = _route_rmse(pre_l, pre_r, exact)
    post_success_rmse = _rmse(post_success, exact.target_success_probability)
    post_route_rmse = _route_rmse(post_l, post_r, exact)
    unique, counts = np.unique(schedules.reshape(-1, 2), axis=0, return_counts=True)
    summary = {
        "setting": process.name,
        "instance_id": process.instance_id,
        "label": label,
        "method_id": method_id,
        "repetitions": repetitions,
        "primary_population": "pre_outer_resampling_weighted",
        "pre": {
            "mean_terminal_success_estimate": float(pre_success.mean()),
            "terminal_success_rmse": pre_success_rmse,
            "mean_route_l_estimate": float(pre_l.mean()),
            "mean_route_r_estimate": float(pre_r.mean()),
            "route_mass_rmse": pre_route_rmse,
            "equal_weight_mean_rmse": 0.5 * (pre_success_rmse + pre_route_rmse),
            "target_tv": _target_tv(pre_mass, exact),
        },
        "post": {
            "mean_terminal_success_estimate": float(post_success.mean()),
            "terminal_success_rmse": post_success_rmse,
            "mean_route_l_estimate": float(post_l.mean()),
            "mean_route_r_estimate": float(post_r.mean()),
            "route_mass_rmse": post_route_rmse,
            "equal_weight_mean_rmse": 0.5 * (post_success_rmse + post_route_rmse),
            "target_tv": _target_tv(post_mass, exact),
        },
        "normalizer": _normalizer_summary(normalizers, exact.normalizer),
        "reverse_transition_evaluations_per_repetition": sorted(
            set(int(value) for value in costs)
        ),
        "allocation_counts": {
            f"K{int(k)}_M{int(m)}": int(count)
            for (k, m), count in zip(unique, counts)
        },
        "diagnostic_fields": list(DIAGNOSTIC_FIELDS),
    }
    raw = {
        "pre_terminal_success_estimate": pre_success,
        "pre_route_l_estimate": pre_l,
        "pre_route_r_estimate": pre_r,
        "post_terminal_success_estimate": post_success,
        "post_route_l_estimate": post_l,
        "post_route_r_estimate": post_r,
        "normalizer_estimate": normalizers,
        "reverse_transition_evaluations": costs,
        "schedule": schedules,
        "diagnostics": diagnostics,
        "diagnostic_fields": np.asarray(DIAGNOSTIC_FIELDS),
    }
    return summary, raw


def _run_cell(
    directory: Path,
    process: ProcessParameters,
    *,
    resume: bool,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    raw_path = directory / "raw_repetitions.npz"
    if resume and summary_path.exists() and raw_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic cell: {directory}")
    summary, raw = evaluate_diagnostic_cell(process, **kwargs)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(raw_path, **raw)
    write_json(summary_path, summary)
    return summary


def _frozen_cell_metadata(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    relative = path.parts[path.parts.index("cells") + 1 : -1]
    return {
        "phase": relative[0],
        "setting": summary["setting"],
        "method_dir": relative[2],
        "summary": summary,
        "raw_path": path.with_name("raw_repetitions.npz"),
    }


def _decomposition(values: np.ndarray, target: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    bias = float(values.mean() - target)
    variance = float(np.var(values, ddof=0))
    mse = float(np.mean((values - target) ** 2))
    return {
        "mean": float(values.mean()),
        "target": float(target),
        "bias": bias,
        "bias_squared": bias**2,
        "variance": variance,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "identity_residual": mse - bias**2 - variance,
        "variance_fraction_of_mse": variance / mse if mse > 0.0 else math.nan,
    }


def run_reanalysis(frozen_root: Path, output: Path) -> dict[str, Any]:
    destination = output / "reanalysis"
    destination.mkdir(parents=True, exist_ok=True)
    top_files = [
        frozen_root / name
        for name in (
            "metrics.json",
            "validation.json",
            "held_out_parameters.json",
            "resolved_config.yaml",
            "seed_manifest.json",
            "nfe_accounting.json",
        )
    ]
    frozen_hashes_before = {str(path.relative_to(frozen_root)): _sha256(path) for path in top_files}
    metrics = json.loads((frozen_root / "metrics.json").read_text(encoding="utf-8"))
    parameters = {
        row["name"]: row
        for row in json.loads(
            (frozen_root / "held_out_parameters.json").read_text(encoding="utf-8")
        )
    }
    exact = metrics["exact"]
    cells = [
        _frozen_cell_metadata(path)
        for path in sorted((frozen_root / "cells").glob("**/summary.json"))
    ]

    step_rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell["phase"] != "held_out" or cell["method_dir"] != "adaptive":
            continue
        raw = np.load(cell["raw_path"])
        schedules = np.asarray(raw["schedule"])
        diagnostic = np.asarray(raw["probe_diagnostics_rho2_inner_outer"])
        for index, step in enumerate(range(2, 6)):
            rho2, inner_need, outer_need = diagnostic[:, index, :].T
            selected = schedules[:, step - 1, :]
            risks = np.stack(
                [outer_need / k + inner_need / m for k, m in ACTIONS], axis=1
            )
            minima = risks.min(axis=1, keepdims=True)
            tied = np.isclose(risks, minima, atol=1e-15, rtol=0.0).sum(axis=1) > 1
            balanced = np.all(selected == np.asarray([6, 4]), axis=1)
            process = parameters[cell["setting"]]
            step_rows.append(
                {
                    "process": cell["setting"],
                    "commitment_probability": process["commitment_probability"],
                    "commitment_beta2": process["commitment_beta2"],
                    "route_heterogeneity_d": process["d"],
                    "physical_step": step,
                    "repetitions": len(rho2),
                    "mean_rho2": float(np.mean(rho2)),
                    "mean_inner_need": float(np.mean(inner_need)),
                    "mean_outer_need": float(np.mean(outer_need)),
                    "fraction_rho2_exact_zero": float(np.mean(rho2 == 0.0)),
                    "fraction_rho2_zero_due_to_zero_empirical_variance": None,
                    "zero_variance_attribution_available": False,
                    "fraction_inner_need_zero": float(np.mean(inner_need == 0.0)),
                    "fraction_outer_need_zero": float(np.mean(outer_need == 0.0)),
                    "fraction_selected_k12_m2": float(
                        np.mean(np.all(selected == [12, 2], axis=1))
                    ),
                    "fraction_selected_k6_m4": float(balanced.mean()),
                    "fraction_selected_k3_m8": float(
                        np.mean(np.all(selected == [3, 8], axis=1))
                    ),
                    "fraction_balanced_due_to_tie": float(np.mean(balanced & tied)),
                    "fraction_tie": float(tied.mean()),
                    "fraction_fallback": 0.0,
                    "note": (
                        "Frozen raw stores rho2/InnerNeed/OuterNeed but not centered phi/psi "
                        "variances, so the cause of rho2==0 is not identifiable without replay."
                    ),
                }
            )
    write_json(destination / "stepwise_adaptive_diagnostics.json", step_rows)
    _write_csv(destination / "stepwise_adaptive_diagnostics.csv", step_rows)

    decomposition_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for cell in cells:
        raw = np.load(cell["raw_path"])
        target = exact[cell["setting"]]
        arrays = {
            "terminal_success": (
                np.asarray(raw["terminal_success_estimate"]),
                target["target_terminal_success_probability"],
            ),
            "route_l": (
                np.asarray(raw["route_l_estimate"]), target["route_l_target_mass"]
            ),
            "route_r": (
                np.asarray(raw["route_r_estimate"]), target["route_r_target_mass"]
            ),
        }
        for quantity, (values, truth) in arrays.items():
            decomposition_rows.append(
                {
                    "phase": cell["phase"],
                    "process": cell["setting"],
                    "method": cell["method_dir"],
                    "quantity": quantity,
                    **_decomposition(values, float(truth)),
                }
            )
        normalizers = np.asarray(raw["normalizer_estimate"], dtype=np.float64)
        applicable = bool(np.isfinite(normalizers).all())
        for quantity, (values, truth) in arrays.items():
            if applicable:
                products = normalizers * (values - float(truth))
                residual = float(products.mean())
                standard_error = float(products.std(ddof=1) / math.sqrt(len(products)))
                exact_z = float(target["normalizer"])
                joint_rows.append(
                    {
                        "phase": cell["phase"],
                        "process": cell["setting"],
                        "method": cell["method_dir"],
                        "quantity": quantity,
                        "applicable": True,
                        "residual": residual,
                        "standard_error": standard_error,
                        "ci95_low": residual - 1.96 * standard_error,
                        "ci95_high": residual + 1.96 * standard_error,
                        "relative_to_exact_z": residual / exact_z,
                        "same_repetition_pairing_confirmed": True,
                    }
                )
            else:
                joint_rows.append(
                    {
                        "phase": cell["phase"],
                        "process": cell["setting"],
                        "method": cell["method_dir"],
                        "quantity": quantity,
                        "applicable": False,
                        "reason": "normalizer_estimate is NaN for this non-proper baseline",
                    }
                )
    write_json(destination / "bias_variance_decomposition.json", decomposition_rows)
    _write_csv(destination / "bias_variance_decomposition.csv", decomposition_rows)
    write_json(destination / "joint_proper_weighting.json", joint_rows)
    _write_csv(destination / "joint_proper_weighting.csv", joint_rows)

    adaptive_variance_rows = []
    by_key = {
        (row["process"], row["method"], row["quantity"]): row
        for row in decomposition_rows
        if row["phase"] == "held_out"
    }
    for process in sorted(parameters):
        for quantity in ("terminal_success", "route_l", "route_r"):
            adaptive = by_key[(process, "adaptive", quantity)]
            static = by_key[(process, "K16_M2", quantity)]
            adaptive_variance_rows.append(
                {
                    "process": process,
                    "quantity": quantity,
                    "adaptive_bias": adaptive["bias"],
                    "global_static_bias": static["bias"],
                    "adaptive_variance": adaptive["variance"],
                    "global_static_variance": static["variance"],
                    "variance_ratio_adaptive_over_static": adaptive["variance"]
                    / static["variance"],
                    "bias_squared_ratio_adaptive_over_static": adaptive["bias_squared"]
                    / static["bias_squared"]
                    if static["bias_squared"] > 0.0
                    else math.inf,
                }
            )
    write_json(destination / "adaptive_vs_static_bias_variance.json", adaptive_variance_rows)
    _write_csv(destination / "adaptive_vs_static_bias_variance.csv", adaptive_variance_rows)

    audit = {
        "frozen_primary_estimate_population": "post_outer_resampling_unweighted",
        "code_path": [
            "_run_fixed_sampler/_run_adaptive_sampler compute incremental weights",
            "systematic_resample is applied at every step including step 5",
            "evaluate_cell computes success/route/TV from output.paths after that resampling",
        ],
        "pre_outer_resampling_weighted_values_present_in_frozen_raw": False,
        "pre_outer_resampling_reconstruction_from_frozen_raw_possible": False,
        "reason": (
            "The frozen NPZ stores only scalar post-resampling estimates and does not "
            "store final proposals, incremental weights, or pre-resampling ancestry."
        ),
        "new_v2_cells_capture_both": True,
        "new_v2_primary_population": "pre_outer_resampling_weighted",
        "joint_weighting_applicability": (
            "Applicable when normalizer_estimate is finite: hhat and Zhat share the same "
            "repetition and the post-resampling empirical mean is an unbiased SMC test-function estimate."
        ),
    }
    write_json(destination / "pre_post_resampling_audit.json", audit)
    frozen_hashes_after = {str(path.relative_to(frozen_root)): _sha256(path) for path in top_files}
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("Frozen Phase A evidence changed during reanalysis")
    evidence = {
        "frozen_root": str(frozen_root),
        "read_only_verified": True,
        "top_level_sha256_before": frozen_hashes_before,
        "top_level_sha256_after": frozen_hashes_after,
        "summary_cells": len(cells),
        "raw_cells": len(list((frozen_root / "cells").glob("**/raw_repetitions.npz"))),
    }
    write_json(destination / "frozen_evidence_manifest.json", evidence)
    return {
        "stepwise_rows": len(step_rows),
        "decomposition_rows": len(decomposition_rows),
        "joint_rows": len(joint_rows),
        "evidence": evidence,
    }


def _cell_kwargs(
    process: ProcessParameters,
    *,
    label: str,
    method_id: int,
    repetitions: int,
    master_seed: int,
    sampler: Callable[[np.random.SeedSequence], DiagnosticOutput],
) -> dict[str, Any]:
    return {
        "label": label,
        "method_id": method_id,
        "repetitions": repetitions,
        "master_seed": master_seed,
        "sampler": sampler,
    }


def run_pilot(cfg: dict[str, Any], output: Path, *, resume: bool) -> dict[str, Any]:
    phase = cfg["phase_a_diagnosis_v2"]
    repetitions = int(phase["pilot_repetitions"])
    master_seed = int(phase["master_seed"])
    epsilon = float(phase["epsilon"])
    processes = _canonical_informative()
    results: dict[str, list[dict[str, Any]]] = {
        "exact_stat_allocator": [],
        "matched_production": [],
        "checkpoint_test": [],
        "temporal_budget_schedules": [],
    }
    for process in processes:
        exact_sampler = lambda seed, p=process: _run_adaptive_variant(
            p, epsilon, seed, "exact_stat"
        )
        results["exact_stat_allocator"].append(
            _run_cell(
                output / "exact_stat_allocator" / "pilot" / process.name / "exact_stat_allocator",
                process,
                resume=resume,
                kwargs=_cell_kwargs(
                    process,
                    label="exact_stat_allocator",
                    method_id=METHOD_IDS["exact_stat_allocator"],
                    repetitions=repetitions,
                    master_seed=master_seed,
                    sampler=exact_sampler,
                ),
            )
        )
        variants = {
            "current_probe_allocator": lambda seed, p=process: _run_adaptive_variant(
                p, epsilon, seed, "current_probe"
            ),
            "matched_production_k12_m2": lambda seed, p=process: _run_adaptive_variant(
                p, epsilon, seed, "matched"
            ),
            "global_static_k16_m2": lambda seed, p=process: _run_fixed_schedule(
                p, epsilon, seed, [2, 2, 2, 2, 2]
            ),
        }
        for label, sampler in variants.items():
            results["matched_production"].append(
                _run_cell(
                    output / "matched_production" / "pilot" / process.name / label,
                    process,
                    resume=resume,
                    kwargs=_cell_kwargs(
                        process,
                        label=label,
                        method_id=METHOD_IDS[label],
                        repetitions=repetitions,
                        master_seed=master_seed,
                        sampler=sampler,
                    ),
                )
            )
        if process.name != "rare_low_informative":
            y1_sampler = lambda seed, p=process: _run_fixed_schedule(
                p,
                epsilon,
                seed,
                [4, 4, 4, 4, 4],
                method="duet_y1",
                checkpoint="Y1",
                outer_k=8,
            )
            results["checkpoint_test"].append(
                _run_cell(
                    output / "checkpoint_test" / "pilot" / process.name / "duet_y1",
                    process,
                    resume=resume,
                    kwargs=_cell_kwargs(
                        process,
                        label="duet_y1",
                        method_id=METHOD_IDS["duet_y1"],
                        repetitions=repetitions,
                        master_seed=master_seed,
                        sampler=y1_sampler,
                    ),
                )
            )
        schedules = {"static_m2": [2, 2, 2, 2, 2]}
        for index in range(5):
            schedule = [1, 1, 1, 1, 1]
            schedule[index] = 6
            schedules[f"concentrated_t{index + 1}"] = schedule
        for index, (label, schedule) in enumerate(schedules.items()):
            sampler = lambda seed, p=process, s=tuple(schedule): _run_fixed_schedule(
                p, epsilon, seed, s
            )
            method_key = "temporal_static" if index == 0 else f"temporal_t{index}"
            results["temporal_budget_schedules"].append(
                _run_cell(
                    output / "temporal_budget_schedules" / "pilot" / process.name / label,
                    process,
                    resume=resume,
                    kwargs=_cell_kwargs(
                        process,
                        label=label,
                        method_id=METHOD_IDS[method_key],
                        repetitions=repetitions,
                        master_seed=master_seed,
                        sampler=sampler,
                    ),
                )
            )
    write_json(output / "report" / "pilot_metrics.json", results)
    return results


def _dispatch_confirmation_sampler(
    family: str,
    label: str,
    process: ProcessParameters,
    epsilon: float,
) -> tuple[int, Callable[[np.random.SeedSequence], DiagnosticOutput]]:
    if label == "exact_stat_allocator":
        return METHOD_IDS[label], lambda seed: _run_adaptive_variant(
            process, epsilon, seed, "exact_stat"
        )
    if label == "current_probe_allocator":
        return METHOD_IDS[label], lambda seed: _run_adaptive_variant(
            process, epsilon, seed, "current_probe"
        )
    if label == "matched_production_k12_m2":
        return METHOD_IDS[label], lambda seed: _run_adaptive_variant(
            process, epsilon, seed, "matched"
        )
    if label == "global_static_k16_m2":
        return METHOD_IDS[label], lambda seed: _run_fixed_schedule(
            process, epsilon, seed, [2, 2, 2, 2, 2]
        )
    if label == "duet_y1":
        return METHOD_IDS[label], lambda seed: _run_fixed_schedule(
            process,
            epsilon,
            seed,
            [4, 4, 4, 4, 4],
            method="duet_y1",
            checkpoint="Y1",
            outer_k=8,
        )
    if family == "temporal_budget_schedules":
        if label == "static_m2":
            schedule = [2, 2, 2, 2, 2]
            method_id = METHOD_IDS["temporal_static"]
        else:
            index = int(label.rsplit("t", 1)[1]) - 1
            schedule = [1, 1, 1, 1, 1]
            schedule[index] = 6
            method_id = METHOD_IDS[f"temporal_t{index + 1}"]
        return method_id, lambda seed: _run_fixed_schedule(
            process, epsilon, seed, schedule
        )
    raise ValueError(f"Unknown confirmation entry: {family}/{label}")


def run_confirmation(
    cfg: dict[str, Any], output: Path, plan_path: Path, *, resume: bool
) -> list[dict[str, Any]]:
    phase = cfg["phase_a_diagnosis_v2"]
    repetitions = int(phase["confirmation_repetitions"])
    master_seed = int(phase["master_seed"])
    epsilon = float(phase["epsilon"])
    processes = {process.name: process for process in _canonical_informative()}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    results = []
    for entry in plan["cells"]:
        process = processes[entry["setting"]]
        family = entry["family"]
        label = entry["label"]
        method_id, sampler = _dispatch_confirmation_sampler(
            family, label, process, epsilon
        )
        results.append(
            _run_cell(
                output / family / "confirmation" / process.name / label,
                process,
                resume=resume,
                kwargs=_cell_kwargs(
                    process,
                    label=label,
                    method_id=method_id,
                    repetitions=repetitions,
                    master_seed=master_seed,
                    sampler=sampler,
                ),
            )
        )
    write_json(output / "report" / "confirmation_metrics.json", results)
    shutil.copy2(plan_path, output / "report" / "confirmation_plan.json")
    return results


def validate_config(cfg: dict[str, Any]) -> None:
    phase = cfg.get("phase_a_diagnosis_v2", {})
    expected = {
        "epsilon": 0.05,
        "master_seed": 20260908,
        "pilot_repetitions": 1000,
        "confirmation_repetitions": 5000,
        "actions": [[12, 2], [6, 4], [3, 8]],
        "temporal_outer_k": 16,
        "temporal_total_inner_budget": 10,
    }
    errors = [
        f"{key} must be {wanted!r}, got {phase.get(key)!r}"
        for key, wanted in expected.items()
        if phase.get(key) != wanted
    ]
    output = resolve_config_path(cfg, cfg["experiment"]["output_directory"])
    frozen = resolve_config_path(cfg, phase["frozen_phase_a_root"])
    if output == frozen or frozen in output.parents or output in frozen.parents:
        errors.append("diagnosis output must not overlap frozen Phase A evidence")
    if cfg["model"].get("dtype") != "float64":
        errors.append("diagnosis must use float64")
    if errors:
        raise ValueError("Invalid Phase A diagnosis v2 config:\n- " + "\n- ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("reanalysis", "pilot", "confirmation"), required=True
    )
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_duet_config(args.config)
    validate_config(cfg)
    phase = cfg["phase_a_diagnosis_v2"]
    frozen = resolve_config_path(cfg, phase["frozen_phase_a_root"])
    output = resolve_config_path(cfg, cfg["experiment"]["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        "reanalysis",
        "exact_stat_allocator",
        "matched_production",
        "checkpoint_test",
        "temporal_budget_schedules",
        "report",
    ):
        (output / directory).mkdir(exist_ok=True)
    started = time.perf_counter()
    if args.mode == "reanalysis":
        result = run_reanalysis(frozen, output)
    elif args.mode == "pilot":
        result = run_pilot(cfg, output, resume=args.resume)
    else:
        if args.confirmation_plan is None:
            parser.error("--confirmation-plan is required for confirmation mode")
        result = run_confirmation(
            cfg, output, args.confirmation_plan.resolve(), resume=args.resume
        )
    write_json(
        output / "report" / f"{args.mode}_run.json",
        {
            "mode": args.mode,
            "wall_clock_s": time.perf_counter() - started,
            "result_count": len(result) if isinstance(result, list) else None,
            "frozen_root": str(frozen),
            "output_root": str(output),
            "frozen_metrics_sha256": _sha256(frozen / "metrics.json"),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
