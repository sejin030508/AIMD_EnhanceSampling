from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from confmh.duet.config import resolve_config_path
from confmh.duet.inner_fkc import one_checkpoint_inner_step
from confmh.duet.observables import ObservableRegistry
from confmh.duet.outer_smc import OuterSMC
from confmh.duet.potentials import PotentialCoefficients, PrefixPotential
from confmh.duet.programs import Event, ProgressState, TemporalProgram
from confmh.duet.records import initialize_run_directory, write_json
from confmh.duet.toy import FiniteStateAdapter, ExactPathTarget, enumerate_path_target, finite_kl, total_variation


def _programs(horizon: int) -> dict[str, TemporalProgram]:
    terminal = Event("terminal", "x", (5.0, 5.0), (horizon, horizon))
    return {
        "terminal": TemporalProgram("terminal", (terminal,)),
        "windowed": TemporalProgram(
            "windowed",
            (Event("intermediate", "x", (2.0, 2.0), (2, 3)),),
            terminal_event=terminal,
        ),
        "ordered": TemporalProgram(
            "ordered",
            (
                Event("route_A", "x", (1.0, 1.0), (1, 3)),
                Event("route_B", "x", (4.0, 4.0), (2, horizon - 1)),
            ),
            terminal_event=terminal,
            allow_same_frame=False,
        ),
    }


def _path_state(path: Sequence[int], potential: PrefixPotential) -> tuple[ProgressState, dict[str, float]]:
    state = ProgressState()
    values = potential.values(path[0])
    for t, frame in enumerate(path[1:], start=1):
        state, values = potential.advance(state, frame, t)
    return state, values


def _path_reward(path: Sequence[int], potential: PrefixPotential) -> float:
    state, values = _path_state(path, potential)
    return float(np.exp(potential.log_psi(path, state, len(path) - 1, values)))


def _route(path: Sequence[int]) -> bool:
    return 1 in path and 4 in path and path.index(1) < path.index(4)


def _exact_summary(exact: ExactPathTarget, horizon: int) -> dict[str, Any]:
    return {
        "normalizer": exact.normalizer,
        "route_mass": float(
            sum(probability for path, probability in zip(exact.paths, exact.target_probabilities) if _route(path))
        ),
        "intermediate_marginals": [exact.marginal(t).tolist() for t in range(horizon + 1)],
    }


def _parent_progress(history: Sequence[int], potential: PrefixPotential) -> ProgressState:
    state = ProgressState()
    for t, frame in enumerate(history[1:], start=1):
        state, _ = potential.advance(state, frame, t)
    return state


def _exact_local(
    adapter: FiniteStateAdapter,
    potential: PrefixPotential,
    history: Sequence[int],
    t: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    state = _parent_progress(history, potential)
    q = adapter.exact_q(history)
    phi = np.asarray(
        [np.exp(potential.candidate_log_psi(history, state, endpoint, t)[0]) for endpoint in range(6)],
        dtype=float,
    )
    return q, phi, float(np.dot(q, phi))


def _all_local_z_summary(
    adapter: FiniteStateAdapter,
    potential: PrefixPotential,
    horizon: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    prefixes: list[tuple[tuple[int, ...], float]] = [((0,), 1.0)]
    max_normalization_error = 0.0
    for t in range(1, horizon + 1):
        z_values, base_weights = [], []
        next_prefixes = []
        for history, probability in prefixes:
            q, phi, z_value = _exact_local(adapter, potential, history, t)
            ideal = q * phi / z_value
            max_normalization_error = max(max_normalization_error, abs(float(ideal.sum()) - 1.0))
            z_values.append(z_value)
            base_weights.append(probability)
            for endpoint, transition in enumerate(q):
                next_prefixes.append((history + (endpoint,), probability * float(transition)))
        weights = np.asarray(base_weights, dtype=float)
        weights /= weights.sum()
        values = np.asarray(z_values, dtype=float)
        result[str(t)] = {
            "prefix_count": len(values),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
            "base_weighted_mean": float(np.dot(weights, values)),
        }
        prefixes = next_prefixes
    result["ideal_local_proposal_max_normalization_error"] = max_normalization_error
    return result


def _proper_weighting_probe(
    program: TemporalProgram,
    coefficients: PotentialCoefficients,
    horizon: int,
    repetitions: int,
    inner_m: int,
    seed: int,
) -> dict[str, Any]:
    adapter = FiniteStateAdapter()
    registry = ObservableRegistry({"x": lambda frame: float(frame)})
    potential = PrefixPotential(program, registry, coefficients, horizon)
    history = [0]
    parent_state = _parent_progress(history, potential)
    q, phi, exact_z = _exact_local(adapter, potential, history, 1)
    exact_mass = q * phi
    estimated_mass = np.zeros(6, dtype=float)
    zhat = []
    for replicate in range(repetitions):
        base = seed + 104729 * replicate
        state = adapter.prepare_history(history)
        result = one_checkpoint_inner_step(
            adapter=adapter,
            history_state=state,
            count=inner_m,
            seeds=[base + index for index in range(inner_m)],
            continuation_seeds=[base + 10000 + index for index in range(inner_m)],
            checkpoint_progress=0.75,
            candidate_potential=lambda frame: potential.candidate_log_psi(history, parent_state, frame, 1),
            rng=np.random.default_rng(base + 20000),
        )
        weight = float(np.exp(result.log_z_hat))
        zhat.append(weight)
        estimated_mass[int(result.frame)] += weight
    estimated_mass /= repetitions
    zhat_array = np.asarray(zhat)
    residual = float(np.sum(np.abs(estimated_mass - exact_mass)) / exact_z)
    return {
        "inner_m": inner_m,
        "repetitions": repetitions,
        "exact_local_z": exact_z,
        "mean_z_hat": float(zhat_array.mean()),
        "normalizer_relative_bias": float((zhat_array.mean() - exact_z) / exact_z),
        "estimator_variance": float(zhat_array.var(ddof=1)),
        "proper_weighting_l1_relative_residual": residual,
        "exact_unnormalized_selected_mass": exact_mass.tolist(),
        "estimated_unnormalized_selected_mass": estimated_mass.tolist(),
    }


def _method_run(
    *,
    method: str,
    program: TemporalProgram,
    coefficients: PotentialCoefficients,
    exact: ExactPathTarget,
    horizon: int,
    repetitions: int,
    outer_k: int,
    inner_m: int,
    seed: int,
) -> dict[str, Any]:
    counts = np.zeros((horizon + 1, 6), dtype=float)
    route_hits = successes = samples = 0
    outer_ess, inner_ess_1, inner_ess_2, normalizers = [], [], [], []
    registry = ObservableRegistry({"x": lambda frame: float(frame)})
    for replicate in range(repetitions):
        adapter = FiniteStateAdapter()
        potential = PrefixPotential(program, registry, coefficients, horizon)
        run = OuterSMC(
            adapter=adapter,
            potential=potential,
            method=method,
            outer_k=outer_k,
            inner_m=inner_m,
            checkpoint_progress=0.75,
            seed=seed + replicate,
        ).run([0], horizon)
        if method not in {"frozen", "inner_only"}:
            normalizers.append(float(np.exp(run.log_normalizer_estimate)))
        for particle in run.particles:
            for t, state_value in enumerate(particle.history):
                counts[t, int(state_value)] += 1
            route_hits += int(_route(particle.history))
            state, values = _path_state(particle.history, potential)
            successes += int(program.successful(state, values, horizon))
            samples += 1
        outer_ess.extend(run.outer_ess)
        inner_ess_1.extend(record.inner_ess_1 for record in run.records if record.inner_ess_1 is not None)
        inner_ess_2.extend(record.inner_ess_2 for record in run.records if record.inner_ess_2 is not None)
    empirical = counts / counts.sum(axis=1, keepdims=True)
    exact_marginals = np.stack([exact.marginal(t) for t in range(horizon + 1)])
    tv = [total_variation(empirical[t], exact_marginals[t]) for t in range(horizon + 1)]
    kl = [finite_kl(empirical[t], exact_marginals[t]) for t in range(horizon + 1)]
    exact_route_mass = _exact_summary(exact, horizon)["route_mass"]
    normalizer_values = np.asarray(normalizers, dtype=float)
    return {
        "outer_k": outer_k,
        "inner_m": inner_m,
        "repetitions": repetitions,
        "samples": samples,
        "empirical_intermediate_marginals": empirical.tolist(),
        "intermediate_tv": tv,
        "intermediate_kl": kl,
        "mean_intermediate_tv": float(np.mean(tv[1:])),
        "final_tv_to_exact_target": tv[-1],
        "final_kl_to_exact_target": kl[-1],
        "route_probability": route_hits / samples,
        "route_probability_error": abs(route_hits / samples - exact_route_mass),
        "program_success": successes / samples,
        "normalizer_mean": float(normalizer_values.mean()) if len(normalizer_values) else None,
        "normalizer_relative_bias": (
            float((normalizer_values.mean() - exact.normalizer) / exact.normalizer)
            if len(normalizer_values)
            else None
        ),
        "normalizer_estimator_variance": float(normalizer_values.var(ddof=1)) if len(normalizer_values) > 1 else None,
        "mean_outer_ess": float(np.mean(outer_ess)),
        "mean_inner_ess_1": float(np.mean(inner_ess_1)) if inner_ess_1 else None,
        "mean_inner_ess_2": float(np.mean(inner_ess_2)) if inner_ess_2 else None,
    }


def run_exact_benchmark(cfg: dict[str, Any], *, resume: bool = False, overwrite: bool = False) -> Path:
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
    horizon = int(cfg["trajectory"]["horizon"])
    coefficients = PotentialCoefficients.from_config(cfg["program"])
    registry = ObservableRegistry({"x": lambda frame: float(frame)})
    programs = _programs(horizon)
    repetitions = int(cfg["experiment"].get("repetitions", 200))
    methods = cfg["experiment"]["methods"]
    seed = int(cfg["experiment"].get("seed", 0))
    payload: dict[str, Any] = {"horizon": horizon, "states": 6, "inner_latent_states": 3, "tasks": {}}
    exact_arrays: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for task_index, (task, program) in enumerate(programs.items()):
        exact_potential = PrefixPotential(program, registry, coefficients, horizon)
        exact = enumerate_path_target(
            FiniteStateAdapter(), 0, horizon, lambda path: _path_reward(path, exact_potential)
        )
        task_payload: dict[str, Any] = {
            "exact": _exact_summary(exact, horizon),
            "local_z": _all_local_z_summary(FiniteStateAdapter(), exact_potential, horizon),
            "methods": {},
        }
        for method in methods:
            settings = cfg["experiment"].get("method_settings", {}).get(method, {})
            task_payload["methods"][method] = _method_run(
                method=method,
                program=program,
                coefficients=coefficients,
                exact=exact,
                horizon=horizon,
                repetitions=repetitions,
                outer_k=int(settings.get("outer_k", cfg["particles"]["outer_k"])),
                inner_m=int(settings.get("inner_m", cfg["particles"]["inner_m"])),
                seed=seed + 1000000 * task_index,
            )
        payload["tasks"][task] = task_payload
        exact_arrays[f"{task}_paths"] = np.asarray(exact.paths, dtype=np.int8)
        exact_arrays[f"{task}_base_probabilities"] = exact.base_probabilities
        exact_arrays[f"{task}_rewards"] = exact.rewards
        exact_arrays[f"{task}_target_probabilities"] = exact.target_probabilities

    probe_cfg = cfg["experiment"].get("proper_weighting_probe", {})
    probe_repetitions = int(probe_cfg.get("repetitions", 3000))
    payload["proper_weighting"] = [
        _proper_weighting_probe(
            programs["ordered"], coefficients, horizon, probe_repetitions, inner_m, seed + inner_m * 100003
        )
        for inner_m in probe_cfg.get("inner_m_values", [1, 2, 4, 8, 16])
    ]

    convergence_cfg = cfg["experiment"].get("convergence", {})
    convergence_repetitions = int(convergence_cfg.get("repetitions", 100))
    ordered_potential = PrefixPotential(programs["ordered"], registry, coefficients, horizon)
    ordered_exact = enumerate_path_target(
        FiniteStateAdapter(), 0, horizon, lambda path: _path_reward(path, ordered_potential)
    )
    convergence = []
    for scale in convergence_cfg.get("scales", [2, 4, 8, 16]):
        for method in ("complete_nested", "duet", "naive_dual"):
            row = _method_run(
                method=method,
                program=programs["ordered"],
                coefficients=coefficients,
                exact=ordered_exact,
                horizon=horizon,
                repetitions=convergence_repetitions,
                outer_k=int(scale),
                inner_m=int(scale),
                seed=seed + 7000000 + int(scale) * 1000 + {"complete_nested": 1, "duet": 2, "naive_dual": 3}[method],
            )
            convergence.append({"method": method, "scale": int(scale), **row})
    payload["convergence"] = convergence

    maximum_probe_residual = max(row["proper_weighting_l1_relative_residual"] for row in payload["proper_weighting"])
    largest_scale = max(int(row["scale"]) for row in convergence)
    smallest_scale = min(int(row["scale"]) for row in convergence)
    at_largest = {row["method"]: row for row in convergence if int(row["scale"]) == largest_scale}
    at_smallest = {row["method"]: row for row in convergence if int(row["scale"]) == smallest_scale}
    checks = {
        "proper_weighting_residual_below_0.10": maximum_probe_residual < 0.10,
        "duet_normalizer_relative_bias_below_0.15": abs(payload["tasks"]["ordered"]["methods"]["duet"]["normalizer_relative_bias"]) < 0.15,
        "complete_nested_normalizer_relative_bias_below_0.15": abs(payload["tasks"]["ordered"]["methods"]["complete_nested"]["normalizer_relative_bias"]) < 0.15,
        "duet_tv_improves_with_scale": at_largest["duet"]["mean_intermediate_tv"] < at_smallest["duet"]["mean_intermediate_tv"],
        "complete_nested_tv_improves_with_scale": at_largest["complete_nested"]["mean_intermediate_tv"] < at_smallest["complete_nested"]["mean_intermediate_tv"],
        "naive_mismatch_is_non_degenerate": (
            abs(at_largest["naive_dual"]["normalizer_relative_bias"])
            > abs(at_largest["duet"]["normalizer_relative_bias"]) + 0.02
        ),
    }
    payload["gate"] = {"passed": bool(all(checks.values())), "checks": checks}
    payload["wall_clock_s"] = time.perf_counter() - started
    write_json(output / "metrics.json", payload)
    write_json(output / "nfe_accounting.json", {"backend": "fully_enumerable_toy", "neural_decoder_calls": 0})
    write_json(output / "wall_clock.json", {"seconds": payload["wall_clock_s"], "peak_gpu_memory_bytes": 0})
    write_json(output / "seed_manifest.json", {"base_seed": seed, "repetitions": repetitions})
    np.savez_compressed(output / "exact_target.npz", **exact_arrays)
    return output
