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
    guidance_update_steps,
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
from confmh.duet.runner import build_adapter


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
    status = str(manifest.get("status") or "")
    if status != "ready" and not status.startswith("ready_"):
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
    horizon = int(run_cfg["trajectory"]["horizon"])
    analysis_transitions = tuple(
        int(item) for item in run_cfg["experiment"].get("analysis_transitions", [])
    )
    if any(item < 1 or item > horizon for item in analysis_transitions):
        raise ValueError("analysis_transitions must lie within the physical horizon")
    initialize_run_directory(
        output,
        run_cfg,
        command=" ".join(shlex.quote(item) for item in sys.argv),
        overwrite=overwrite,
        resume=False,
    )

    adapter = build_adapter(run_cfg)
    adapter.load_model()
    initial_history = adapter.initial_history()
    initial_d0 = metric.distance_a(initial_history[0])
    if not np.isclose(initial_d0, d0_a, rtol=0.0, atol=2.0e-3):
        raise RuntimeError(f"Prepared/runtime d0 mismatch: {d0_a} != {initial_d0}")
    program_potential = str(run_cfg["program"].get("potential", "rmsd"))
    tica_metric = None
    tica_d0 = None
    if program_potential == "tica":
        from confmh.duet.tica_potential import TicaEndpointMetric, TicaEndpointPotential

        tica_cfg = run_cfg["program"].get("tica", {})
        projection_value = tica_cfg.get("projection_npz")
        projection_path = (
            resolve_config_path(run_cfg, projection_value)
            if projection_value is not None else None
        )
        tica_metric = TicaEndpointMetric(
            manifest,
            projection_path=projection_path,
            dimensions=int(tica_cfg.get("dimensions", 2)),
        )
        tica_d0 = tica_metric.distance_a(initial_history[0])
        if tica_d0 <= 0.0:
            raise RuntimeError(f"tica_endpoint_not_separated: d0={tica_d0:.6f}")
        expected_target = tica_cfg.get("expected_folded_target")
        if expected_target is not None and not np.allclose(
            tica_metric.target, np.asarray(expected_target, dtype=float),
            rtol=0.0, atol=1.0e-5,
        ):
            raise RuntimeError("Prepared/runtime folded TICA target mismatch")
        expected_d0 = tica_cfg.get("expected_start_to_target_distance")
        if expected_d0 is not None and not np.isclose(
            tica_d0, float(expected_d0), rtol=0.0, atol=1.0e-5,
        ):
            raise RuntimeError("Prepared/runtime TICA d0 mismatch")
        potential = TicaEndpointPotential(
            tica_metric, tica_d0, coefficient=reward_coefficient,
            log_floor=reward_log_floor,
        )
        reward_metadata = {
            "reward_type": "tica_endpoint_distance",
            "reward_distance_unit": tica_metric.distance_unit,
            "reward_d0": float(tica_d0),
            "reward_tica_dimensions": int(tica_metric.dimensions),
            "reward_tica_feature": tica_metric.feature_kind,
            "reward_tica_folded_target": tica_metric.target.tolist(),
            "reward_tica_projection_path": tica_metric.projection_path,
        }
    else:
        potential = PocketEndpointPotential(
            metric, d0_a, coefficient=reward_coefficient,
            log_floor=reward_log_floor,
        )
        reward_metadata = {
            "reward_type": "aligned_backbone_rmsd",
            "reward_distance_unit": "Angstrom",
            "reward_d0": float(d0_a),
        }

    guidance_metadata: dict[str, Any] | None = None
    if method == "guided_duet":
        if tica_metric is None or tica_d0 is None:
            raise RuntimeError("guided_duet requires a TICA endpoint potential")
        from confmh.duet.torch_tica import TorchTicaEndpointPotential

        torch_potential = TorchTicaEndpointPotential.from_metric(
            tica_metric,
            adapter.topology,
            coefficient=reward_coefficient,
            d0=tica_d0,
            log_floor=reward_log_floor,
        )
        adapter.configure_guidance(
            torch_potential,
            strength=float(run_cfg["guidance"]["strength"]),
            update_steps=guidance_update_steps(run_cfg),
        )
        guidance_metadata = {
            **adapter.guidance_metadata,
            "schedule": str(run_cfg["guidance"].get("schedule", "all_stochastic")),
            "inner_resampling": bool(
                run_cfg["guidance"].get("inner_resampling", True)
            ),
            "proposal_correction": "exact_discrete_gaussian_log_p_base_over_q_guided",
        }

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
        outer_resampling_ess_fraction=float(
            run_cfg["particles"].get("outer_resampling_ess_fraction", 0.5)
        ),
        guided_inner_resampling=bool(
            run_cfg.get("guidance", {}).get("inner_resampling", True)
        ),
        seed=int(seed),
        snapshot_times=analysis_transitions,
    ).run(initial_history, horizon)
    wall = time.perf_counter() - started
    checkpoint_audit: dict[str, Any] = {
        "checkpoint_progress_definition": "completed reverse-update fraction",
        "configured_progresses": list(checkpoint_progresses),
        "record_count": len(result.records),
        "records": [],
    }
    if method in {"duet", "guided_duet"}:
        maximum_telescoping_error = 0.0
        all_ratios_finite = True
        for record in result.records:
            if record.inner_checkpoint_progresses != list(checkpoint_progresses):
                raise RuntimeError("Missing or misordered inner interventions")
            if record.inner_checkpoint_ancestors_history is None or len(
                record.inner_checkpoint_ancestors_history
            ) != len(checkpoint_progresses):
                raise RuntimeError("Missing checkpoint ancestry")
            error = float(record.telescoping_max_abs_log_error or 0.0)
            maximum_telescoping_error = max(maximum_telescoping_error, error)
            if record.guided_path_log_proposal_ratios is not None:
                all_ratios_finite = all_ratios_finite and bool(
                    np.all(np.isfinite(record.guided_path_log_proposal_ratios))
                )
            checkpoint_audit["records"].append(
                {
                    "t": record.t,
                    "particle": record.particle,
                    "checkpoint_progresses": record.inner_checkpoint_progresses,
                    "checkpoint_ess": record.inner_checkpoint_ess,
                    "checkpoint_ancestors_history": (
                        record.inner_checkpoint_ancestors_history
                    ),
                    "potential_telescoping_max_abs_log_error": error,
                    "guided_path_log_proposal_ratios": (
                        record.guided_path_log_proposal_ratios
                    ),
                }
            )
        checkpoint_audit.update(
            {
                "max_potential_telescoping_abs_log_error": maximum_telescoping_error,
                "proposal_log_ratios_all_finite": all_ratios_finite,
                "passed": maximum_telescoping_error <= 1.0e-10 and all_ratios_finite,
            }
        )
        if not checkpoint_audit["passed"]:
            raise RuntimeError("Inner checkpoint/guided-weight audit failed")
    pre_particles = result.pre_final_particles
    pre_weights = result.pre_final_normalized_weights
    if len(pre_particles) != k or len(pre_weights) != k:
        raise RuntimeError("Terminal pre-resampling population was not captured")

    construct = manifest.get("construct_uniprot_residues_inclusive")
    if construct is None:
        construct = [1, int(manifest["model_length"])]
    construct_first = int(construct[0])
    uniprot_to_model = {
        number: number - construct_first
        for number in range(
            construct_first,
            int(construct[1]) + 1,
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
        )
        intermediate_metrics[str(transition)] = snapshot_metrics
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
        "reverse_steps": int(run_cfg["model"]["reverse_steps"]),
        "sampler_mode": str(run_cfg["model"]["sampler_mode"]),
        "inner_checkpoint_completed_reverse_fraction": checkpoint_progresses[-1],
        "inner_checkpoint_completed_reverse_fractions": list(checkpoint_progresses),
        "outer_resampling": "systematic",
        "outer_resampling_ess_fraction": float(
            run_cfg["particles"].get("outer_resampling_ess_fraction", 0.5)
        ),
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
            if reward_log_floor is None else
            f"max({reward_log_floor:g}, -{reward_coefficient:g}*(d/d0)^2)"
        ),
        "analysis_transitions": list(analysis_transitions),
        "guidance": guidance_metadata,
        **reward_metadata,
        "checkpoint_phi": "same reward evaluated on predicted-clean atom37 coordinates",
        "scientific_claim_boundary": (
            "exploratory candidate performance under a frozen surrogate prior; "
            "not physical opening probability or kinetics"
        ),
        **endpoint_metrics,
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "checkpoint_audit.json", checkpoint_audit)
    write_json(output / "population_curves.json", curves)
    write_json(output / "intermediate_metrics.json", intermediate_metrics)
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
