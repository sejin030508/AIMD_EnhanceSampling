from __future__ import annotations

import copy
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from confmh.adapters.confrover_duet import ConfRoverDuETAdapter, ConfRoverFrame
from confmh.duet.config import inner_checkpoint_progresses, project_root, resolve_config_path
from confmh.duet.case_study_eval import MechanismBlindEvaluator
from confmh.duet.analysis import path_pairwise_diversity
from confmh.duet.genealogy import genealogical_entropy, surviving_ancestor_count
from confmh.duet.observables import ObservableRegistry
from confmh.duet.outer_smc import OuterSMC
from confmh.duet.potentials import PotentialCoefficients, PrefixPotential
from confmh.duet.programs import TemporalProgram
from confmh.duet.records import initialize_run_directory, write_json
from confmh.duet.reference_eval import HeldOutReferenceEvaluator
from confmh.pca_cv import _kabsch_align


def _load_catalog(cfg: dict[str, Any]) -> dict[str, Any]:
    source = cfg["program"].get("catalog")
    if not source:
        return {"observables": cfg["program"]["observables"], "tasks": {"inline": cfg["program"]}}
    path = resolve_config_path(cfg, source)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _method_particles(cfg: dict[str, Any], method: str) -> tuple[int, int]:
    settings = cfg["experiment"].get("method_settings", {}).get(method, {})
    k = int(settings.get("outer_k", cfg["particles"]["outer_k"]))
    m = int(settings.get("inner_m", cfg["particles"]["inner_m"]))
    return k, m


def build_confrover_adapter(cfg: dict[str, Any]) -> ConfRoverDuETAdapter:
    model, trajectory = cfg["model"], cfg["trajectory"]
    geometry_gate = cfg.get("preflight", {}).get("gate", {})
    return ConfRoverDuETAdapter(
        repository_path=resolve_config_path(cfg, model["repository_path"]),
        checkpoint=resolve_config_path(cfg, model["checkpoint"]),
        initial_structure=resolve_config_path(cfg, trajectory["initial_structure"]),
        case_id=str(trajectory["case_id"]),
        seqres=str(trajectory["seqres"]),
        cache_dir=resolve_config_path(cfg, model["cache_dir"]),
        device=str(model.get("device", "cuda:0")),
        dtype=str(model.get("dtype", "float32")),
        stride_in_10ps=int(model.get("physical_lag_in_10ps", 256)),
        reverse_steps=int(model.get("reverse_steps", 200)),
        sampler_mode=str(model.get("sampler_mode", "sde")),
        kv_cache_type=str(model.get("kv_cache_type", "offloaded")),
        decoder_microbatch_size=(
            int(model["decoder_microbatch_size"])
            if model.get("decoder_microbatch_size") is not None
            else None
        ),
        ca_adjacent_quality_threshold_a=float(
            geometry_gate.get("ca_adjacent_quality_threshold_a", 4.5)
        ),
        ca_adjacent_hard_threshold_a=float(
            geometry_gate.get("ca_adjacent_hard_threshold_a", 5.5)
        ),
        ca_adjacent_hard_tolerance_a=float(
            geometry_gate.get("ca_adjacent_hard_tolerance_a", 1.0e-3)
        ),
    )


def _save_particles(output: Path, particles) -> None:
    histories = []
    for particle in particles:
        if all(isinstance(frame, ConfRoverFrame) for frame in particle.history):
            histories.append(np.stack([frame.atom37_a for frame in particle.history]))
    if histories:
        np.savez_compressed(output / "trajectories_atom37_a.npz", trajectories=np.stack(histories))


def _maximum_aligned_step_nm(ca_path: np.ndarray) -> float:
    path = np.asarray(ca_path, dtype=float)
    if len(path) < 2:
        return 0.0
    maxima = []
    for previous, current in zip(path[:-1], path[1:]):
        aligned = _kabsch_align(current, previous)
        maxima.append(float(np.max(np.linalg.norm(aligned - previous, axis=-1))))
    return max(maxima, default=0.0)


def run_experiment_config(
    cfg: dict[str, Any], *, resume: bool = False, overwrite: bool = False
) -> list[Path]:
    catalog = _load_catalog(cfg)
    tasks = cfg["experiment"].get("tasks", list(catalog["tasks"]))
    methods = cfg["experiment"]["methods"]
    seeds = [int(seed) for seed in cfg["experiment"].get("seeds", [0])]
    base_output = resolve_config_path(cfg, cfg["experiment"]["output_directory"])
    checkpoint_progresses = inner_checkpoint_progresses(cfg)
    reference_evaluator = None
    reference_evaluator_error = None
    if all(key in cfg.get("reference", {}) for key in ("topology", "pca_model", "held_out_trajectory")):
        try:
            reference_evaluator = HeldOutReferenceEvaluator.from_config(cfg, catalog)
        except Exception as error:  # Saved explicitly; held-out evaluation must never be silently omitted.
            reference_evaluator_error = f"{type(error).__name__}: {error}"
    mechanism_evaluator = None
    mechanism_evaluator_error = None
    if cfg.get("case_study", {}).get("benchmark_spec"):
        try:
            reward_observables = []
            for task_name in tasks:
                task = TemporalProgram.from_config(catalog["tasks"][task_name])
                reward_observables.extend(task.all_observables())
            mechanism_evaluator = MechanismBlindEvaluator.from_config(
                cfg, tuple(dict.fromkeys(reward_observables))
            )
        except Exception as error:
            mechanism_evaluator_error = f"{type(error).__name__}: {error}"
    completed: list[Path] = []
    for task_name in tasks:
        program_cfg = dict(catalog["tasks"][task_name])
        program_cfg.update({
            key: cfg["program"][key]
            for key in (
                "lambda_program", "distance_weight", "remaining_event_weight", "deadline_weight",
                "failure_penalty", "terminal_failure_penalty", "failure_guidance_weight",
                "distance_scale", "potential_floor",
            )
            if key in cfg["program"]
        })
        program = TemporalProgram.from_config(program_cfg)
        observables = ObservableRegistry.from_config(catalog["observables"], root=project_root(cfg))
        coefficients = PotentialCoefficients.from_config(program_cfg)
        for method in methods:
            k, m = _method_particles(cfg, method)
            run_method = "frozen" if method == "best_of_budget" else method
            for seed in seeds:
                output = base_output / task_name / method / f"seed_{seed}"
                if resume and (output / "metrics.json").exists():
                    completed.append(output)
                    continue
                initialize_run_directory(
                    output,
                    cfg,
                    command=" ".join(shlex.quote(item) for item in sys.argv),
                    overwrite=overwrite,
                    resume=resume,
                )
                adapter = build_confrover_adapter(cfg)
                adapter.load_model()
                initial_history = adapter.initial_history()
                potential = PrefixPotential(
                    program, observables, coefficients, int(cfg["trajectory"]["horizon"])
                )
                started = time.perf_counter()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                except ImportError:
                    torch = None
                result = OuterSMC(
                    adapter=adapter,
                    potential=potential,
                    method=run_method,
                    outer_k=k,
                    inner_m=m,
                    checkpoint_progress=checkpoint_progresses,
                    outer_resampling_ess_fraction=float(
                        cfg["particles"].get("outer_resampling_ess_fraction", 1.0)
                    ),
                    seed=seed,
                ).run(initial_history, int(cfg["trajectory"]["horizon"]))
                wall = time.perf_counter() - started
                adapter.accounting.reward_evaluations = potential.evaluations
                final_rewards = np.asarray([particle.last_log_psi for particle in result.particles])
                selected = int(np.argmax(final_rewards)) if method == "best_of_budget" else None
                active_indices = [selected] if selected is not None else list(range(len(result.particles)))
                active_particles = [result.particles[index] for index in active_indices]
                final_values = [
                    observables.evaluate(particle.history[-1], program.all_observables())
                    for particle in active_particles
                ]
                success = [
                    bool(program.successful(particle.progress, values, int(cfg["trajectory"]["horizon"])))
                    for particle, values in zip(active_particles, final_values)
                ]
                failed = [bool(particle.progress.failed) for particle in active_particles]
                unresolved = [not bad and not good for bad, good in zip(failed, success)]
                if program.kind == "ordered":
                    event_order = [
                        bool(
                            particle.progress.stage >= 2
                            and len(particle.progress.completed_frames) >= 2
                            and particle.progress.completed_frames[0]
                            < particle.progress.completed_frames[1]
                        )
                        for particle in active_particles
                    ]
                else:
                    event_order = []
                validation = adapter.validate_frames(
                    [particle.history[-1] for particle in active_particles]
                )
                valid_flags = [bool(item["valid"]) for item in validation]
                path_validation = [
                    adapter.validate_frames(particle.history[1:])
                    for particle in active_particles
                ]
                path_valid_flags = [
                    bool(rows) and all(bool(row["valid"]) for row in rows)
                    for rows in path_validation
                ]
                mechanism_metrics = (
                    mechanism_evaluator.evaluate(
                        [particle.history for particle in active_particles],
                        endpoint_success=success,
                        valid_flags=path_valid_flags,
                    )
                    if mechanism_evaluator is not None
                    else {
                        "status": "unavailable",
                        "reason": mechanism_evaluator_error
                        or "mechanism-blind case study not configured",
                    }
                )
                ca_paths = [
                    np.stack([frame.ca_nm for frame in particle.history])
                    for particle in active_particles
                    if all(isinstance(frame, ConfRoverFrame) for frame in particle.history)
                ]
                aligned_step_maxima = [
                    _maximum_aligned_step_nm(path) for path in ca_paths
                ]
                reference_metrics = (
                    reference_evaluator.evaluate([particle.history for particle in active_particles])
                    if reference_evaluator is not None
                    else {
                        "status": "unavailable",
                        "reason": reference_evaluator_error or "held-out reference not configured",
                    }
                )
                peak_memory = None
                if torch is not None and torch.cuda.is_available():
                    peak_memory = int(torch.cuda.max_memory_allocated())
                decoder_nfe = adapter.count_decoder_evaluations()
                gpu_hours = wall / 3600.0
                success_rate = float(np.mean(success)) if success else 0.0
                metrics = {
                    "method": method,
                    "task": task_name,
                    "seed": seed,
                    "outer_k": k,
                    "inner_m": m,
                    "inner_checkpoint_progresses": list(
                        result.records[0].inner_checkpoint_progresses
                        or checkpoint_progresses
                    ),
                    "decoder_nfe": decoder_nfe,
                    "accounting": adapter.accounting.to_dict(),
                    "wall_clock_s": wall,
                    "peak_gpu_memory_bytes": peak_memory,
                    "outer_ess": result.outer_ess,
                    "outer_resampled": result.outer_resampled,
                    "log_normalizer_estimate": result.log_normalizer_estimate,
                    "log_normalizer_increments": result.log_normalizer_increments,
                    "selected_best_of_budget": selected,
                    "final_log_rewards": final_rewards.tolist(),
                    "evaluated_particle_indices": active_indices,
                    "final_observables": final_values,
                    "joint_program_success": success,
                    "joint_program_success_rate": success_rate,
                    "event_order_accuracy": float(np.mean(event_order)) if event_order else None,
                    "failure_rate": float(np.mean(failed)) if failed else 0.0,
                    "unresolved_rate": float(np.mean(unresolved)) if unresolved else 0.0,
                    "structural_validity_rate": float(np.mean(valid_flags)) if valid_flags else 0.0,
                    "path_structural_validity_rate": (
                        float(np.mean(path_valid_flags)) if path_valid_flags else 0.0
                    ),
                    "valid_endpoint_success_rate": (
                        float(
                            np.mean(
                                np.asarray(success, dtype=bool)
                                & np.asarray(path_valid_flags, dtype=bool)
                            )
                        )
                        if success
                        else 0.0
                    ),
                    "maximum_kabsch_aligned_frame_step_nm": (
                        max(aligned_step_maxima, default=None)
                    ),
                    "per_path_maximum_kabsch_aligned_frame_step_nm": aligned_step_maxima,
                    "mechanism_order_error": mechanism_metrics.get("mechanism_order_error"),
                    "intermediate_recovery_rate": mechanism_metrics.get(
                        "intermediate_recovery_rate"
                    ),
                    "success_per_decoder_nfe": success_rate / decoder_nfe if decoder_nfe else None,
                    "success_per_million_decoder_nfe": (
                        success_rate * 1.0e6 / decoder_nfe if decoder_nfe else None
                    ),
                    "successful_outputs_per_gpu_hour": (
                        float(np.sum(success)) / gpu_hours if gpu_hours > 0 else None
                    ),
                    "path_pairwise_ca_diversity_nm": (
                        path_pairwise_diversity(ca_paths) if ca_paths else None
                    ),
                    **{
                        key: reference_metrics.get(key)
                        for key in (
                            "held_out_path_distance",
                            "held_out_endpoint_ca_rmsd_nm",
                            "ca_rmsd_progress",
                            "pca_progress",
                            "path_smoothness",
                            "reference_intermediate_coverage",
                            "unspecified_contact_map_similarity",
                            "route_jsd",
                        )
                    },
                    "surviving_initial_ancestors": surviving_ancestor_count(
                        [particle.lineage for particle in active_particles]
                    ),
                    "initial_ancestor_entropy": genealogical_entropy(
                        [particle.lineage for particle in active_particles]
                    ),
                    "mean_inner_ess_1": float(np.mean([
                        record.inner_ess_1 for record in result.records
                        if record.inner_ess_1 is not None
                    ])) if any(record.inner_ess_1 is not None for record in result.records) else None,
                    "mean_inner_ess_2": float(np.mean([
                        record.inner_ess_2 for record in result.records
                        if record.inner_ess_2 is not None
                    ])) if any(record.inner_ess_2 is not None for record in result.records) else None,
                    "mean_inner_checkpoint_ess": (
                        np.mean(
                            np.asarray(
                                [
                                    record.inner_checkpoint_ess
                                    for record in result.records
                                    if record.inner_checkpoint_ess is not None
                                ],
                                dtype=float,
                            ),
                            axis=0,
                        ).tolist()
                        if any(
                            record.inner_checkpoint_ess is not None
                            for record in result.records
                        )
                        else None
                    ),
                    "max_telescoping_abs_log_error": max(
                        [record.telescoping_max_abs_log_error for record in result.records
                         if record.telescoping_max_abs_log_error is not None],
                        default=None,
                    ),
                    "validity": validation,
                    "path_validity": path_validation,
                    "scientific_claim_boundary": (
                        "bias-conditioned transition paths under the frozen surrogate prior; "
                        "not exact U+b dynamics or kinetics"
                    ),
                }
                write_json(output / "metrics.json", metrics)
                write_json(output / "records.json", [record.__dict__ for record in result.records])
                write_json(output / "outer_ancestry.json", result.ancestor_indices)
                write_json(output / "held_out_reference_metrics.json", reference_metrics)
                write_json(output / "mechanism_metrics.json", mechanism_metrics)
                write_json(output / "seed_manifest.json", [particle.seed_metadata for particle in result.particles])
                write_json(output / "nfe_accounting.json", adapter.accounting.to_dict())
                write_json(output / "wall_clock.json", {"seconds": wall, "peak_gpu_memory_bytes": peak_memory})
                _save_particles(output, result.particles)
                completed.append(output)
    return completed


def run_allocation_grid(
    cfg: dict[str, Any], *, resume: bool = False, overwrite: bool = False
) -> list[Path]:
    completed: list[Path] = []
    original_output = cfg["experiment"]["output_directory"]
    for allocation in cfg["experiment"]["allocations"]:
        active = copy.deepcopy(cfg)
        k, m = int(allocation["outer_k"]), int(allocation["inner_m"])
        budget = int(cfg["experiment"].get("decoder_population_budget", k * m))
        if k * m != budget:
            raise ValueError(f"Allocation ({k}, {m}) violates fixed budget {budget}")
        active["particles"]["outer_k"] = k
        active["particles"]["inner_m"] = m
        active["experiment"]["output_directory"] = str(
            Path(original_output) / f"K{k}_M{m}"
        )
        completed.extend(
            run_experiment_config(active, resume=resume, overwrite=overwrite)
        )
    return completed
