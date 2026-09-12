from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from confmh.adapters.confrover_duet import ConfRoverFrame
from confmh.duet.config import (
    inner_checkpoint_progresses,
    load_duet_config,
    resolve_config_path,
)
from confmh.duet.outer_smc import OuterSMC
from confmh.duet.phase_b_pockets import (
    PocketEndpointMetric,
    PocketEndpointPotential,
    PocketReference,
    evaluate_population,
    hidden_observables,
)
from confmh.duet.records import initialize_run_directory, write_json
from confmh.duet.runner import build_confrover_adapter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(cfg: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    path = resolve_config_path(cfg, cfg["pocket"]["manifest"])
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "ready":
        raise RuntimeError(
            f"{manifest.get('protein')}: preparation status is {manifest.get('status')}"
        )
    return manifest, path


def _load_metric(
    cfg: dict[str, Any], manifest: dict[str, Any]
) -> tuple[PocketEndpointMetric, PocketReference]:
    path = resolve_config_path(cfg, cfg["pocket"]["reference_npz"])
    with np.load(path) as payload:
        start = PocketReference(
            "start",
            np.asarray(payload["start_atom37_a"]),
            np.asarray(payload["start_atom37_mask"]),
        )
        names = [str(item) for item in payload["reference_names"].tolist()]
        reference_coords = np.asarray(payload["reference_atom37_a"])
        reference_masks = np.asarray(payload["reference_atom37_mask"])
        references = [
            PocketReference(name, coords, mask)
            for name, coords, mask in zip(names, reference_coords, reference_masks)
        ]
        core = np.asarray(payload["core_residue_indices"], dtype=int).tolist()
        loop = np.asarray(payload["loop_residue_indices"], dtype=int).tolist()
    if core != [int(item) for item in manifest["core_model_indices_0based"]]:
        raise RuntimeError("Reference NPZ and manifest core masks disagree")
    if loop != [int(item) for item in manifest["loop_model_indices_0based"]]:
        raise RuntimeError("Reference NPZ and manifest loop masks disagree")
    return (
        PocketEndpointMetric(
            references=references,
            core_residue_indices=core,
            loop_residue_indices=loop,
        ),
        start,
    )


def _method_particles(cfg: dict[str, Any], method: str) -> tuple[int, int]:
    settings = cfg["experiment"]["method_settings"][method]
    return int(settings["outer_k"]), int(settings["inner_m"])


def _save_population(path: Path, particles: list[Any]) -> None:
    histories = []
    masks = []
    aatypes = []
    for particle in particles:
        if not all(isinstance(frame, ConfRoverFrame) for frame in particle.history):
            raise TypeError("Phase-B output population contains a non-ConfRover frame")
        histories.append(np.stack([frame.atom37_a for frame in particle.history]))
        masks.append(np.stack([frame.atom37_mask for frame in particle.history]))
        aatypes.append(np.stack([frame.aatype for frame in particle.history]))
    np.savez_compressed(
        path,
        trajectories_atom37_a=np.stack(histories),
        trajectories_atom37_mask=np.stack(masks),
        trajectories_aatype=np.stack(aatypes),
    )


def run_one(
    cfg: dict[str, Any], *, method: str, seed: int, overwrite: bool = False
) -> Path:
    if method not in cfg["experiment"]["methods"]:
        raise ValueError(f"Method {method} is not enabled by this config")
    manifest, manifest_path = _load_manifest(cfg)
    metric, start_reference = _load_metric(cfg, manifest)
    ligand_heavy_atom_coordinates_a: np.ndarray | None = None
    ligand_metadata = manifest.get("ligand")
    if isinstance(ligand_metadata, dict) and ligand_metadata.get("heavy_atom_npz"):
        ligand_path = resolve_config_path(cfg, ligand_metadata["heavy_atom_npz"])
        with np.load(ligand_path) as ligand_payload:
            ligand_heavy_atom_coordinates_a = np.asarray(
                ligand_payload["coordinates_a"], dtype=float
            )
    d0_a = float(manifest["d0_a"])
    if d0_a <= 1.0:
        raise RuntimeError(f"endpoint_not_separated: d0={d0_a:.6f} A")

    k, m = _method_particles(cfg, method)
    if k * m != int(cfg["experiment"]["decoder_population_budget"]):
        raise RuntimeError(f"Unequal decoder budget for {method}: K*M={k*m}")
    base_output = resolve_config_path(cfg, cfg["experiment"]["output_directory"])
    stage = str(cfg["experiment"]["stage"])
    protein = str(manifest["protein"])
    output = base_output / protein / stage / method / f"seed_{seed}"
    run_cfg = copy.deepcopy(cfg)
    run_cfg["experiment"]["methods"] = [method]
    run_cfg["experiment"]["seeds"] = [int(seed)]
    checkpoint_progresses = inner_checkpoint_progresses(run_cfg)
    reward_coefficient = float(run_cfg["program"].get("reward_coefficient", 4.0))
    configured_log_floor = run_cfg["program"].get("reward_log_floor", -30.0)
    reward_log_floor = (
        None if configured_log_floor is None else float(configured_log_floor)
    )
    analysis_transitions = tuple(
        int(item) for item in run_cfg["experiment"].get("analysis_transitions", [])
    )
    horizon = int(run_cfg["trajectory"]["horizon"])
    if any(item < 1 or item > horizon for item in analysis_transitions):
        raise ValueError("analysis_transitions must lie within the physical horizon")
    if len(set(analysis_transitions)) != len(analysis_transitions):
        raise ValueError("analysis_transitions must be unique")
    initialize_run_directory(
        output,
        run_cfg,
        command=" ".join(shlex.quote(item) for item in sys.argv),
        overwrite=overwrite,
        resume=False,
    )

    adapter = build_confrover_adapter(run_cfg)
    adapter.load_model()
    initial_history = adapter.initial_history()
    initial_d0 = metric.distance_a(initial_history[0])
    if not np.isclose(initial_d0, d0_a, rtol=0.0, atol=2.0e-3):
        raise RuntimeError(f"Prepared/runtime d0 mismatch: {d0_a} != {initial_d0}")
    potential = PocketEndpointPotential(
        metric,
        d0_a,
        coefficient=reward_coefficient,
        log_floor=reward_log_floor,
    )

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    started = time.perf_counter()
    result = OuterSMC(
        adapter=adapter,
        potential=potential,
        method=method,
        outer_k=k,
        inner_m=m,
        checkpoint_progress=checkpoint_progresses,
        outer_resampling_ess_fraction=0.5,
        seed=int(seed),
        snapshot_times=analysis_transitions,
    ).run(initial_history, horizon)
    wall = time.perf_counter() - started
    reverse_steps = int(run_cfg["model"]["reverse_steps"])
    checkpoint_reverse_update_steps = [
        int(round(reverse_steps * progress)) for progress in checkpoint_progresses
    ]
    checkpoint_audit: dict[str, Any] = {
        "checkpoint_progress_definition": "completed reverse-update fraction",
        "configured_progresses": list(checkpoint_progresses),
        "resolved_reverse_update_steps": checkpoint_reverse_update_steps,
        "reverse_schedule_steps": reverse_steps,
        "record_count": len(result.records),
        "records": [],
    }
    if method == "duet":
        max_telescoping_error = 0.0
        for record in result.records:
            if record.inner_checkpoint_progresses != list(checkpoint_progresses):
                raise RuntimeError(
                    f"Missing/misordered inner interventions at t={record.t}, "
                    f"particle={record.particle}: {record.inner_checkpoint_progresses}"
                )
            if record.inner_checkpoint_ancestors_history is None or len(
                record.inner_checkpoint_ancestors_history
            ) != len(checkpoint_progresses):
                raise RuntimeError(
                    f"Missing checkpoint ancestry at t={record.t}, "
                    f"particle={record.particle}"
                )
            if record.inner_checkpoint_ess is None or len(
                record.inner_checkpoint_ess
            ) != len(checkpoint_progresses):
                raise RuntimeError(
                    f"Missing checkpoint ESS at t={record.t}, particle={record.particle}"
                )
            telescoping_error = float(record.telescoping_max_abs_log_error or 0.0)
            max_telescoping_error = max(max_telescoping_error, telescoping_error)
            checkpoint_audit["records"].append(
                {
                    "t": record.t,
                    "particle": record.particle,
                    "checkpoint_progresses": record.inner_checkpoint_progresses,
                    "resolved_reverse_update_steps": checkpoint_reverse_update_steps,
                    "checkpoint_ess": record.inner_checkpoint_ess,
                    "checkpoint_ancestors_history": (
                        record.inner_checkpoint_ancestors_history
                    ),
                    "telescoping_max_abs_log_error": telescoping_error,
                }
            )
        checkpoint_audit["max_telescoping_abs_log_error"] = max_telescoping_error
        checkpoint_audit["telescoping_tolerance"] = 1.0e-10
        checkpoint_audit["passed"] = max_telescoping_error <= 1.0e-10
        if not checkpoint_audit["passed"]:
            raise RuntimeError(
                "Multi-checkpoint telescoping audit failed: "
                f"{max_telescoping_error:.3e} > 1e-10"
            )
    if reward_log_floor is None and potential.clipped_evaluations != 0:
        raise RuntimeError(
            "No-floor Phase-B4 run unexpectedly reported clipped potential evaluations"
        )
    pre_particles = result.pre_final_particles
    pre_weights = result.pre_final_normalized_weights
    if len(pre_particles) != k or len(pre_weights) != k:
        raise RuntimeError("Terminal pre-resampling population was not captured")

    construct_first = int(manifest["construct_uniprot_residues_inclusive"][0])
    uniprot_to_model = {
        number: number - construct_first
        for number in range(
            construct_first,
            int(manifest["construct_uniprot_residues_inclusive"][1]) + 1,
        )
    }
    endpoint_metrics, curves = evaluate_population(
        particles=pre_particles,
        normalized_weights=pre_weights,
        adapter=adapter,
        metric=metric,
        d0_a=d0_a,
        protein=protein,
        uniprot_to_model_index=uniprot_to_model,
        start_reference=start_reference,
        ligand_heavy_atom_coordinates_a=ligand_heavy_atom_coordinates_a,
    )
    intermediate_metrics: dict[str, dict[str, Any]] = {}
    for transition in analysis_transitions:
        snapshot = result.snapshot_particles.get(transition)
        snapshot_weights = result.snapshot_normalized_weights.get(transition)
        if snapshot is None or snapshot_weights is None:
            raise RuntimeError(f"Missing requested pre-resampling snapshot t={transition}")
        snapshot_metrics, snapshot_curves = evaluate_population(
            particles=snapshot,
            normalized_weights=snapshot_weights,
            adapter=adapter,
            metric=metric,
            d0_a=d0_a,
            protein=protein,
            uniprot_to_model_index=uniprot_to_model,
            start_reference=start_reference,
            ligand_heavy_atom_coordinates_a=ligand_heavy_atom_coordinates_a,
        )
        final_ratios = np.asarray(snapshot_metrics["per_path_final_d_a"]) / d0_a
        valid_flags = np.asarray(snapshot_metrics["per_path_valid"], dtype=bool)
        valid_ratios = final_ratios[valid_flags]
        intermediate_metrics[str(transition)] = {
            **snapshot_metrics,
            "transition": transition,
            "valid_final_d_over_d0_min": (
                float(np.min(valid_ratios)) if len(valid_ratios) else None
            ),
            "valid_final_d_over_d0_median": (
                float(np.median(valid_ratios)) if len(valid_ratios) else None
            ),
            "valid_final_d_over_d0_q25": (
                float(np.quantile(valid_ratios, 0.25)) if len(valid_ratios) else None
            ),
        }
        write_json(output / f"population_curves_t{transition}.json", snapshot_curves)
        write_json(
            output / f"pre_resampling_normalized_weights_t{transition}.json",
            snapshot_weights,
        )
        _save_population(
            output / f"pre_resampling_population_t{transition}_atom37.npz",
            snapshot,
        )
    peak_memory = None
    if torch is not None and torch.cuda.is_available():
        peak_memory = int(torch.cuda.max_memory_allocated())
    adapter.accounting.reward_evaluations = potential.evaluations
    checkpoint_path = resolve_config_path(run_cfg, run_cfg["model"]["checkpoint"])
    metrics = {
        "status": "complete",
        "phase": "B",
        "stage": stage,
        "protein": protein,
        "method": method,
        "seed": int(seed),
        "outer_k": k,
        "inner_m": m,
        "horizon": horizon,
        "total_frames_including_start": horizon + 1,
        "stride_in_10ps": int(run_cfg["model"]["physical_lag_in_10ps"]),
        "reverse_steps": reverse_steps,
        "sampler_mode": str(run_cfg["model"]["sampler_mode"]),
        # The singular key is retained for readers of the earlier B1 schema;
        # the list is authoritative for multi-checkpoint DuET runs.
        "inner_checkpoint_completed_reverse_fraction": float(
            checkpoint_progresses[-1]
        ),
        "inner_checkpoint_completed_reverse_fractions": list(
            checkpoint_progresses
        ),
        "inner_steering_checkpoint_progresses": (
            list(checkpoint_progresses) if method == "duet" else []
        ),
        "inner_steering_reverse_update_steps": (
            checkpoint_reverse_update_steps if method == "duet" else []
        ),
        "outer_resampling": "systematic",
        "outer_resampling_ess_fraction": 0.5,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "decoder_nfe": adapter.count_decoder_evaluations(),
        "accounting": adapter.accounting.to_dict(),
        "wall_clock_s": float(wall),
        "peak_gpu_memory_bytes": peak_memory,
        "outer_ess": result.outer_ess,
        "outer_resampled": result.outer_resampled,
        "log_normalizer_estimate": result.log_normalizer_estimate,
        "log_normalizer_increments": result.log_normalizer_increments,
        "potential_evaluations": potential.evaluations,
        "potential_clipped_evaluations": potential.clipped_evaluations,
        "potential_clipping_fraction": (
            potential.clipped_evaluations / potential.evaluations
            if potential.evaluations
            else 0.0
        ),
        "reward_coefficient": reward_coefficient,
        "reward_log_floor": reward_log_floor,
        "reward_definition": (
            f"-{reward_coefficient:g}*(d/d0)^2"
            if reward_log_floor is None
            else f"max({reward_log_floor:g}, -{reward_coefficient:g}*(d/d0)^2)"
        ),
        "analysis_transitions": list(analysis_transitions),
        "checkpoint_phi": "same reward evaluated on predicted-clean atom37 coordinates",
        "scientific_claim_boundary": (
            "exploratory candidate performance under a frozen surrogate prior; "
            "not physical opening probability or kinetics"
        ),
        **endpoint_metrics,
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "checkpoint_audit.json", checkpoint_audit)
    write_json(output / "intermediate_metrics.json", intermediate_metrics)
    write_json(output / "population_curves.json", curves)
    write_json(output / "records.json", [record.__dict__ for record in result.records])
    write_json(output / "outer_ancestry.json", result.ancestor_indices)
    write_json(output / "pre_final_normalized_weights.json", pre_weights)
    _save_population(output / "pre_final_population_atom37.npz", pre_particles)
    _save_population(output / "post_resampling_population_atom37.npz", result.particles)
    return output


def write_summary(output_root: Path) -> tuple[Path, Path]:
    rows = []
    for path in sorted(output_root.glob("*/b1/*/seed_*/metrics.json")):
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "protein": metrics["protein"],
                "method": metrics["method"],
                "seed": metrics["seed"],
                "final_d_over_d0": metrics["weighted_mean_final_d_over_d0"],
                "valid_open_like_fraction": metrics["valid_open_like_fraction"],
                "valid_path_fraction": metrics["valid_path_fraction"],
                "decoder_nfe": metrics["decoder_nfe"],
                "wall_time_s": metrics["wall_clock_s"],
            }
        )
    summary = output_root / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "protein", "method", "seed", "final_d_over_d0",
            "valid_open_like_fraction", "valid_path_fraction", "decoder_nfe", "wall_time_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    expected = 16
    report = output_root / "report.md"
    report.write_text(
        "# Phase B cryptic-pocket pilot\n\n"
        f"Completed B1 runs: {len(rows)}/{expected}.\n\n"
        "This is an exploratory two-seed candidate comparison. Resampled duplicates and "
        "trajectory frames are not treated as independent samples. See `summary.csv` and "
        "each run's `metrics.json`/`population_curves.json` for the four-method interpretation.\n",
        encoding="utf-8",
    )
    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--stage")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--outer-k", type=int)
    parser.add_argument("--inner-m", type=int)
    parser.add_argument("--decoder-microbatch-size", type=int)
    parser.add_argument("--reward-coefficient", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    cfg = load_duet_config(args.config)
    if args.summarize:
        output_root = resolve_config_path(cfg, cfg["experiment"]["output_directory"])
        paths = write_summary(output_root)
        print("\n".join(map(str, paths)))
        return
    if args.method is None or args.seed is None:
        parser.error("--method and --seed are required for a run")
    if args.device:
        cfg["model"]["device"] = args.device
    if args.decoder_microbatch_size is not None:
        cfg["model"]["decoder_microbatch_size"] = args.decoder_microbatch_size
    if args.reward_coefficient is not None:
        if args.reward_coefficient <= 0.0:
            parser.error("--reward-coefficient must be positive")
        cfg["program"]["reward_coefficient"] = args.reward_coefficient
    if args.stage:
        cfg["experiment"]["stage"] = args.stage
    if args.horizon is not None:
        cfg["trajectory"]["horizon"] = args.horizon
    if args.outer_k is not None or args.inner_m is not None:
        settings = cfg["experiment"]["method_settings"][args.method]
        settings["outer_k"] = int(args.outer_k or settings["outer_k"])
        settings["inner_m"] = int(args.inner_m or settings["inner_m"])
        cfg["experiment"]["decoder_population_budget"] = (
            settings["outer_k"] * settings["inner_m"]
        )
    output = run_one(cfg, method=args.method, seed=args.seed, overwrite=args.overwrite)
    print(output)


if __name__ == "__main__":
    main()
