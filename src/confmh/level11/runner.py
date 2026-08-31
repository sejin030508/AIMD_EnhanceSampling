from __future__ import annotations

import copy
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from confmh.analysis import _timeseries_ess
from confmh.config import resolve_path, save_config
from confmh.level11.config import GuidanceConfig, validate_level11_config
from confmh.level11.frozen_bias import FrozenGridBias, HarmonicPotential
from confmh.level11.kernel import GuidedConfRoverKernel
from confmh.level11.metrics import geometry_diagnostics, structure_hash
from confmh.level11.torch_pc1 import TorchPC1, validate_residue_order
from confmh.mh import accept, acceptance_probability, bias_only_log_alpha
from confmh.pca_cv import PCACV
from confmh.utils import kbt_kj_mol, seed_everything, write_json


def _potential_from_config(cfg: dict[str, Any], *, create_frozen: bool = False):
    experiment = cfg["experiment"]
    mode = str(experiment["mode"]).lower()
    if mode in {"umbrella", "pilot"}:
        center = experiment.get("center")
        if center is None:
            reference = np.load(resolve_path(cfg, cfg["reference"]["reference_cv"]))["cv"][:, 0]
            center = float(np.quantile(reference, 0.9))
        return HarmonicPotential(
            center=float(center),
            kappa_kj_mol=float(experiment["kappa_kj_mol"]),
        )
    if mode == "frozen_opes":
        path = resolve_path(cfg, experiment["frozen_bias_path"])
        if not path.exists():
            if not create_frozen:
                raise FileNotFoundError(
                    f"Frozen OPES bias is missing: {path}. Run confmh level11-prepare-opes first."
                )
            prepare_frozen_opes(cfg)
        return FrozenGridBias.load(path)
    raise ValueError(f"Unsupported Level 1.1 experiment.mode={mode!r}")


def prepare_frozen_opes(cfg: dict[str, Any]) -> Path:
    validate_level11_config(cfg)
    experiment = cfg["experiment"]
    if str(experiment["mode"]).lower() != "frozen_opes":
        raise ValueError("level11-prepare-opes requires experiment.mode=frozen_opes")
    reference = np.load(resolve_path(cfg, cfg["reference"]["reference_cv"]))["cv"][:, 0]
    kbt = kbt_kj_mol(float(cfg["system"].get("temperature_k", 300.0)))
    bias = FrozenGridBias.from_reference(
        reference,
        kbt_kj_mol=kbt,
        bias_factor=float(experiment.get("bias_factor", 10.0)),
        barrier_kj_mol=float(experiment.get("barrier_kj_mol", 15.0)),
        n_grid=int(experiment.get("n_grid", 401)),
        margin=float(experiment.get("grid_margin", 0.5)),
        bandwidth=(
            None if experiment.get("bandwidth") is None else float(experiment["bandwidth"])
        ),
    )
    output = resolve_path(cfg, experiment["frozen_bias_path"])
    return bias.save(output)


def _build_kernel(cfg: dict[str, Any], potential, guidance: GuidanceConfig, seed: int):
    model = cfg["model"]
    temperature = float(cfg["system"].get("temperature_k", 300.0))
    return GuidedConfRoverKernel(
        pca_model=resolve_path(cfg, cfg["reference"]["pca_model"]),
        potential=potential,
        guidance=guidance,
        beta=1.0 / kbt_kj_mol(temperature),
        repo=resolve_path(cfg, model.get("repo", "external/ConfRover")),
        model_name=str(model.get("name", "ConfRover-base-20M-v1.0")),
        device=str(model.get("device", "cuda:0")),
        case_id=str(cfg["system"]["case_id"]),
        seqres=str(cfg["system"]["seqres"]),
        stride_in_10ps=int(model.get("stride_in_10ps", 256)),
        diffusion_steps=int(model.get("diffusion_steps", 200)),
        seed=int(seed),
        cache_dir=resolve_path(cfg, model.get("cache_dir", "data/confrover_cache")),
        ckpt_dir=(
            None if model.get("ckpt_dir") is None else resolve_path(cfg, model["ckpt_dir"])
        ),
        kv_cache_type=str(model.get("kv_cache_type", "offloaded")),
        use_deepspeed_evo_attention=bool(model.get("use_deepspeed_evo_attention", False)),
    )


def _geometry_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    values = cfg.get("level11", {}).get("geometry", {})
    return {
        "max_ca_displacement_nm": float(values.get("max_ca_displacement_nm", 3.0)),
        "min_nonbonded_ca_distance_nm": float(
            values.get("min_nonbonded_ca_distance_nm", 0.20)
        ),
        "adjacent_ca_distance_range_nm": tuple(
            float(x) for x in values.get("adjacent_ca_distance_range_nm", [0.25, 0.55])
        ),
    }


def _empty_records() -> dict[str, list[Any]]:
    keys = [
        "step",
        "phase",
        "method",
        "level11_enabled",
        "guidance_clip_ratio",
        "guidance_active_fraction",
        "proposal_seed",
        "cv",
        "proposal_cv",
        "bias_kj_mol",
        "proposal_bias_kj_mol",
        "pc1_current",
        "pc1_proposed",
        "pc1_accepted",
        "bias_current_kj_mol",
        "bias_proposed_kj_mol",
        "bias_accepted_kj_mol",
        "delta_bias_kj_mol",
        "base_translation_score_rms_mean",
        "base_translation_score_rms_max",
        "raw_guidance_rms_mean",
        "clipped_guidance_rms_mean",
        "guidance_to_score_ratio_mean",
        "guidance_to_score_ratio_max",
        "guided_step_count",
        "total_denoising_steps",
        "acceptance_probability",
        "log_alpha",
        "accepted",
        "ca_rmsd_proposed_nm",
        "ca_rmsd_accepted_nm",
        "pc1_squared_jump_proposed",
        "pc1_squared_jump_accepted",
        "ca_esjd_proposed_nm2",
        "ca_esjd_accepted_nm2",
        "proposal_walltime_s",
        "guidance_walltime_s",
        "total_walltime_s",
        "geometry_valid",
        "failure_reason",
        "structure_hash",
        "nominal_time_ns",
    ]
    return {key: [] for key in keys}


def _load_records(path: Path) -> dict[str, list[Any]]:
    records = _empty_records()
    if not path.exists():
        return records
    data = np.load(path, allow_pickle=False)
    for key in records:
        if key in data:
            records[key] = np.asarray(data[key]).tolist()
    return records


def _save_checkpoint(
    output_dir: Path,
    records: dict[str, list[Any]],
    frames: list[np.ndarray],
    rng,
    topology,
) -> None:
    arrays = {key: np.asarray(values) for key, values in records.items()}
    np.savez_compressed(output_dir / "chain.npz", **arrays)
    np.savez_compressed(output_dir / "chain_coordinates.npz", xyz_nm=np.asarray(frames))
    write_json(output_dir / "rng_state.json", rng.bit_generator.state)
    if frames:
        import mdtraj as md

        md.Trajectory(np.asarray(frames, dtype=np.float32), topology).save_xtc(
            str(output_dir / "chain.xtc")
        )


def _restore_rng(path: Path, seed: int):
    rng = seed_everything(seed)
    if path.exists():
        rng.bit_generator.state = json.loads(path.read_text(encoding="utf-8"))
    return rng


def run_chain(
    cfg: dict[str, Any], *, guided: bool, output_dir: str | Path | None = None
) -> Path:
    """Run one resumable Level 1.1 umbrella or frozen-OPES chain."""
    import mdtraj as md
    from tqdm.auto import tqdm

    guidance = validate_level11_config(cfg)
    potential = _potential_from_config(cfg)
    run = cfg.get("run", {})
    seed = int(run.get("seed", 20260827))
    if output_dir is None:
        output_dir = run.get("output_dir", "outputs/level11/run")
    output_dir = resolve_path(cfg, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("run", {})["output_dir"] = str(output_dir)
    save_config(cfg, output_dir / "resolved_config.yaml")
    potential.save(output_dir / "bias.npz")

    current_condition = output_dir / "current.pdb"
    start_pdb = resolve_path(cfg, cfg["system"]["start_pdb"])
    chain_path = output_dir / "chain.npz"
    records = _load_records(chain_path)
    completed = len(records["step"])
    if completed == 0:
        shutil.copy2(start_pdb, current_condition)
        frames: list[np.ndarray] = []
    else:
        if not current_condition.exists():
            raise FileNotFoundError(f"Cannot resume without {current_condition}")
        coordinate_path = output_dir / "chain_coordinates.npz"
        if not coordinate_path.exists():
            raise FileNotFoundError(f"Cannot resume without {coordinate_path}")
        frames = np.load(coordinate_path)["xyz_nm"].tolist()

    rng = _restore_rng(output_dir / "rng_state.json", seed)
    pca = PCACV.load(resolve_path(cfg, cfg["reference"]["pca_model"]))
    kernel = _build_kernel(cfg, potential, guidance, seed)
    kernel.set_guidance(enabled=guided)

    stopping = cfg.get("level11", {}).get("stopping", {})
    burn_in = int(stopping.get("burn_in", 50))
    min_production = int(stopping.get("min_production", 300))
    max_production = int(stopping.get("max_production", 800))
    block_size = int(stopping.get("block_size", 100))
    target_ess = float(stopping.get("target_ess", 50.0))
    if not 0 <= burn_in or not 0 < min_production <= max_production or block_size <= 0:
        raise ValueError("Invalid level11.stopping configuration")
    max_total = burn_in + max_production
    cleanup = bool(run.get("cleanup_proposals", True))
    checkpoint_interval = int(run.get("checkpoint_interval", 25))
    reject_invalid = bool(cfg.get("level11", {}).get("geometry", {}).get("reject_invalid", True))
    proposal_root = output_dir / "proposals"
    proposal_root.mkdir(exist_ok=True)
    topology = None
    if (output_dir / "topology.pdb").exists():
        topology = md.load(str(output_dir / "topology.pdb")).topology

    started_run = time.perf_counter()
    try:
        for step in tqdm(range(completed, max_total), desc=f"level11 {'guided' if guided else 'unsteered'}"):
            production_completed = max(0, step - burn_in)
            if production_completed >= min_production and production_completed % block_size == 0:
                production_values = np.asarray(records["pc1_accepted"][burn_in:], dtype=float)
                if _timeseries_ess(production_values) >= target_ess:
                    break

            proposal_dir = proposal_root / f"step_{step:08d}"
            proposal_seed = seed + step * 1009
            proposal_started = time.perf_counter()
            proposals = kernel.propose_seeded(
                condition_pdb=current_condition,
                output_dir=proposal_dir,
                proposal_seed=proposal_seed,
            )
            proposal_walltime = time.perf_counter() - proposal_started
            trajectory = proposals[0].load()
            if len(trajectory) < 2:
                raise RuntimeError("ConfRover proposal must contain condition and candidate frames")
            current_frame, proposal_frame = trajectory[0], trajectory[-1]
            current_pc1 = float(pca.project_trajectory(current_frame)[0, 0])
            proposed_pc1 = float(pca.project_trajectory(proposal_frame)[0, 0])
            current_bias = float(np.asarray(potential.energy(current_pc1)))
            proposed_bias = float(np.asarray(potential.energy(proposed_pc1)))
            delta_bias = proposed_bias - current_bias
            geometry = geometry_diagnostics(
                current_frame, proposal_frame, **_geometry_kwargs(cfg)
            )
            log_alpha = bias_only_log_alpha(delta_bias, kbt_kj_mol(float(cfg["system"].get("temperature_k", 300.0))))
            accepted = accept(log_alpha, rng)
            failure_reason = geometry.failure_reason
            if reject_invalid and not geometry.valid:
                accepted = False
                log_alpha = float("-inf")
                failure_reason = geometry.failure_reason or "invalid_geometry"
            selected = proposal_frame if accepted else current_frame
            selected.save_pdb(str(current_condition))
            selected_pc1 = proposed_pc1 if accepted else current_pc1
            selected_bias = proposed_bias if accepted else current_bias
            if topology is None:
                current_frame.save_pdb(str(output_dir / "topology.pdb"))
                topology = current_frame.topology
                frames.append(np.asarray(current_frame.xyz[0], dtype=np.float32))
            frames.append(np.asarray(selected.xyz[0], dtype=np.float32))

            diagnostics = kernel.last_diagnostics
            proposed_pc1_jump = (proposed_pc1 - current_pc1) ** 2
            values = {
                "step": step,
                "phase": "burn_in" if step < burn_in else "production",
                "method": "level11_guided" if guided else "level1_matched_unsteered",
                "level11_enabled": guided,
                "guidance_clip_ratio": guidance.clip_ratio if guided else 0.0,
                "guidance_active_fraction": guidance.active_fraction,
                "proposal_seed": proposal_seed,
                "cv": selected_pc1,
                "proposal_cv": proposed_pc1,
                "bias_kj_mol": selected_bias,
                "proposal_bias_kj_mol": proposed_bias,
                "pc1_current": current_pc1,
                "pc1_proposed": proposed_pc1,
                "pc1_accepted": selected_pc1,
                "bias_current_kj_mol": current_bias,
                "bias_proposed_kj_mol": proposed_bias,
                "bias_accepted_kj_mol": selected_bias,
                "delta_bias_kj_mol": delta_bias,
                "base_translation_score_rms_mean": diagnostics.get("base_translation_score_rms_mean", 0.0),
                "base_translation_score_rms_max": diagnostics.get("base_translation_score_rms_max", 0.0),
                "raw_guidance_rms_mean": diagnostics.get("raw_guidance_rms_mean", 0.0),
                "clipped_guidance_rms_mean": diagnostics.get("clipped_guidance_rms_mean", 0.0),
                "guidance_to_score_ratio_mean": diagnostics.get("guidance_to_score_ratio_mean", 0.0),
                "guidance_to_score_ratio_max": diagnostics.get("guidance_to_score_ratio_max", 0.0),
                "guided_step_count": diagnostics.get("guided_step_count", 0),
                "total_denoising_steps": diagnostics.get("total_denoising_steps", kernel.diffusion_steps),
                "acceptance_probability": acceptance_probability(log_alpha),
                "log_alpha": log_alpha,
                "accepted": accepted,
                "ca_rmsd_proposed_nm": geometry.ca_rmsd_nm,
                "ca_rmsd_accepted_nm": geometry.ca_rmsd_nm if accepted else 0.0,
                "pc1_squared_jump_proposed": proposed_pc1_jump,
                "pc1_squared_jump_accepted": proposed_pc1_jump if accepted else 0.0,
                "ca_esjd_proposed_nm2": geometry.ca_msd_nm2,
                "ca_esjd_accepted_nm2": geometry.ca_msd_nm2 if accepted else 0.0,
                "proposal_walltime_s": proposal_walltime,
                "guidance_walltime_s": diagnostics.get("guidance_walltime_s", 0.0),
                "total_walltime_s": proposal_walltime,
                "geometry_valid": geometry.valid,
                "failure_reason": failure_reason,
                "structure_hash": structure_hash(selected),
                "nominal_time_ns": (step + 1) * kernel.stride_in_10ps * 0.01,
            }
            for key, value in values.items():
                records[key].append(value)
            if (step + 1) % checkpoint_interval == 0:
                _save_checkpoint(output_dir, records, frames, rng, topology)
            if cleanup:
                shutil.rmtree(proposal_dir, ignore_errors=True)
    finally:
        if topology is not None:
            _save_checkpoint(output_dir, records, frames, rng, topology)

    production = np.asarray(records["pc1_accepted"][burn_in:], dtype=float)
    metadata = {
        "method": "level11_guided" if guided else "level1_matched_unsteered",
        "mode": str(cfg["experiment"]["mode"]),
        "temperature_k": float(cfg["system"].get("temperature_k", 300.0)),
        "kbt_kj_mol": kbt_kj_mol(float(cfg["system"].get("temperature_k", 300.0))),
        "steps": len(records["step"]),
        "accepted": int(np.sum(np.asarray(records["accepted"], dtype=bool))),
        "acceptance_rate": float(np.mean(records["accepted"])) if records["accepted"] else 0.0,
        "completed_steps": len(records["step"]),
        "burn_in": burn_in,
        "production_samples": len(production),
        "production_ess": _timeseries_ess(production),
        "target_ess": target_ess,
        "stopped_by_ess": bool(len(production) >= min_production and _timeseries_ess(production) >= target_ess),
        "walltime_this_invocation_s": time.perf_counter() - started_run,
        "physical_time_warning": "One proposal is not physical MD time.",
        "exactness_warning": (
            "Approximate Level 1.1 MVP: clean-endpoint steering changes the proposal, while "
            "acceptance remains bias-only and omits q_g/q_0."
        ),
    }
    write_json(output_dir / "metadata.json", metadata)
    return output_dir


def _pilot_measurement(
    pca: PCACV,
    potential,
    start_pdb: Path,
    proposal,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    import mdtraj as md

    trajectory = proposal.load()
    current_frame, candidate = trajectory[0], trajectory[-1]
    current_pc1 = float(pca.project_trajectory(current_frame)[0, 0])
    proposed_pc1 = float(pca.project_trajectory(candidate)[0, 0])
    current_bias = float(np.asarray(potential.energy(current_pc1)))
    proposed_bias = float(np.asarray(potential.energy(proposed_pc1)))
    geometry = geometry_diagnostics(current_frame, candidate, **_geometry_kwargs(cfg))
    return {
        "start_pdb": str(start_pdb),
        "pc1_current": current_pc1,
        "pc1_proposed": proposed_pc1,
        "bias_current_kj_mol": current_bias,
        "bias_proposed_kj_mol": proposed_bias,
        "delta_bias_kj_mol": proposed_bias - current_bias,
        "target_distance": abs(proposed_pc1 - float(potential.center)),
        "expected_acceptance_probability": acceptance_probability(
            bias_only_log_alpha(
                proposed_bias - current_bias,
                kbt_kj_mol(float(cfg["system"].get("temperature_k", 300.0))),
            )
        ),
        "ca_rmsd_nm": geometry.ca_rmsd_nm,
        "ca_msd_nm2": geometry.ca_msd_nm2,
        "geometry_valid": geometry.valid,
        "failure_reason": geometry.failure_reason,
    }


def _write_pilot_summary(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    ratios = sorted({float(row["clip_ratio"]) for row in rows})
    summary: dict[str, Any] = {"ratios": {}, "selection": {}}
    for ratio in ratios:
        group = [row for row in rows if float(row["clip_ratio"]) == ratio]
        delta_improvement = np.asarray(
            [row["guided_delta_bias_kj_mol"] - row["base_delta_bias_kj_mol"] for row in group]
        )
        distance_improvement = np.asarray(
            [row["guided_target_distance"] - row["base_target_distance"] for row in group]
        )
        base_jump = np.asarray([row["base_ca_rmsd_nm"] for row in group])
        guided_jump = np.asarray([row["guided_ca_rmsd_nm"] for row in group])
        jump_ratio = float(np.median(guided_jump) / max(np.median(base_jump), 1e-12))
        failure_delta = float(
            np.mean([not row["guided_geometry_valid"] for row in group])
            - np.mean([not row["base_geometry_valid"] for row in group])
        )
        summary["ratios"][str(ratio)] = {
            "pairs": len(group),
            "median_delta_bias_change_kj_mol": float(np.median(delta_improvement)),
            "median_target_distance_change": float(np.median(distance_improvement)),
            "median_ca_rmsd_ratio": jump_ratio,
            "geometry_failure_rate_change": failure_delta,
            "mean_runtime_ratio": float(
                np.mean([row["guided_walltime_s"] for row in group])
                / max(np.mean([row["base_walltime_s"] for row in group]), 1e-12)
            ),
        }
    passing = [
        ratio
        for ratio in ratios
        if summary["ratios"][str(ratio)]["median_delta_bias_change_kj_mol"] < 0
        and summary["ratios"][str(ratio)]["median_target_distance_change"] < 0
        and summary["ratios"][str(ratio)]["median_ca_rmsd_ratio"] >= 0.8
        and summary["ratios"][str(ratio)]["geometry_failure_rate_change"] <= 0.025
    ]
    selected = min((ratio for ratio in passing if ratio > 0), default=0.25)
    summary["selection"] = {
        "passing_ratios": passing,
        "selected_clip_ratio": selected,
        "provisional": not any(ratio > 0 for ratio in passing),
        "rule": "smallest positive ratio passing bias, distance, jump, and geometry gates",
    }
    write_json(output_dir / "summary.json", summary)

    labels = [str(ratio) for ratio in ratios]
    plots = [
        ("delta_bias.png", "guided - base median Δbias (kJ/mol)", "median_delta_bias_change_kj_mol"),
        ("target_distance.png", "guided - base median target distance", "median_target_distance_change"),
        ("jump_size.png", "guided/base median Cα RMSD", "median_ca_rmsd_ratio"),
        ("timing.png", "guided/base mean wall time", "mean_runtime_ratio"),
    ]
    for filename, ylabel, key in plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, [summary["ratios"][label][key] for label in labels])
        ax.axhline(0 if "change" in key else 1, color="black", linewidth=0.8)
        ax.set_xlabel("clip ratio")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    lines = [
        "# Level 1.1 paired proposal pilot",
        "",
        "This is an engineering selection report, not evidence of exact MH.",
        "",
        f"Selected clip ratio: `{selected}`"
        + (" (provisional default)" if summary["selection"]["provisional"] else ""),
        "",
        "| ratio | pairs | Δbias change | target-distance change | RMSD ratio | failure Δ | runtime ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        item = summary["ratios"][label]
        lines.append(
            f"| {label} | {item['pairs']} | {item['median_delta_bias_change_kj_mol']:.4g} | "
            f"{item['median_target_distance_change']:.4g} | {item['median_ca_rmsd_ratio']:.3f} | "
            f"{item['geometry_failure_rate_change']:.3f} | {item['mean_runtime_ratio']:.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_pilot(cfg: dict[str, Any]) -> Path:
    guidance = validate_level11_config(cfg)
    if str(cfg["experiment"]["mode"]).lower() not in {"pilot", "umbrella"}:
        raise ValueError("The pilot requires a harmonic umbrella potential")
    potential = _potential_from_config(cfg)
    run = cfg.get("run", {})
    seed = int(run.get("seed", 20260827))
    output_dir = resolve_path(cfg, run.get("output_dir", "outputs/level11/pilot"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")
    pca = PCACV.load(resolve_path(cfg, cfg["reference"]["pca_model"]))
    kernel = _build_kernel(cfg, potential, guidance, seed)

    pilot = cfg.get("level11", {}).get("pilot", {})
    seed_dir = resolve_path(cfg, pilot.get("seed_dir", "data/reference/6j56_A/seed_frames"))
    start_paths = sorted(seed_dir.glob("seed_*.pdb"))
    n_starts = int(pilot.get("n_starting_structures", 20))
    if len(start_paths) < n_starts:
        raise FileNotFoundError(f"Need {n_starts} seed structures in {seed_dir}; found {len(start_paths)}")
    start_paths = start_paths[:n_starts]
    noise_seeds = [int(x) for x in pilot.get("noise_seeds", [11, 29, 47, 71])]
    ratios = [float(x) for x in pilot.get("clip_ratios", [0.0, 0.1, 0.25, 0.5])]
    proposal_root = output_dir / "proposals"
    proposal_root.mkdir(exist_ok=True)
    jsonl_path = output_dir / "paired_proposals.jsonl"
    rows: list[dict[str, Any]] = []
    if jsonl_path.exists():
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line]
    completed_keys = {(str(row["pair_id"]), float(row["clip_ratio"])) for row in rows}

    for start_index, start_path in enumerate(start_paths):
        for noise_index, noise_seed in enumerate(noise_seeds):
            proposal_seed = seed + start_index * 100_003 + noise_seed
            pair_id = f"start_{start_index:03d}_seed_{noise_index:02d}"
            missing_ratios = [ratio for ratio in ratios if (pair_id, ratio) not in completed_keys]
            if not missing_ratios:
                continue
            base_dir = proposal_root / pair_id / "base"
            kernel.set_guidance(enabled=False)
            started = time.perf_counter()
            base_proposal = kernel.propose_seeded(
                condition_pdb=start_path,
                output_dir=base_dir,
                proposal_seed=proposal_seed,
            )[0]
            base_walltime = time.perf_counter() - started
            base = _pilot_measurement(pca, potential, start_path, base_proposal, cfg)
            for ratio in missing_ratios:
                guided_dir = proposal_root / pair_id / f"clip_{ratio:g}"
                kernel.set_guidance(enabled=True, clip_ratio=ratio)
                started = time.perf_counter()
                guided_proposal = kernel.propose_seeded(
                    condition_pdb=start_path,
                    output_dir=guided_dir,
                    proposal_seed=proposal_seed,
                )[0]
                guided_walltime = time.perf_counter() - started
                guided = _pilot_measurement(pca, potential, start_path, guided_proposal, cfg)
                row = {
                    "pair_id": pair_id,
                    "start_index": start_index,
                    "start_pdb": str(start_path),
                    "proposal_seed": proposal_seed,
                    "clip_ratio": ratio,
                    **{f"base_{key}": value for key, value in base.items() if key != "start_pdb"},
                    **{f"guided_{key}": value for key, value in guided.items() if key != "start_pdb"},
                    "base_walltime_s": base_walltime,
                    "guided_walltime_s": guided_walltime,
                    **{f"diagnostic_{key}": value for key, value in kernel.last_diagnostics.items()},
                }
                rows.append(row)
                completed_keys.add((pair_id, ratio))
                if bool(run.get("cleanup_proposals", True)):
                    shutil.rmtree(guided_dir, ignore_errors=True)
            if bool(run.get("cleanup_proposals", True)):
                shutil.rmtree(base_dir, ignore_errors=True)

            fieldnames = list(rows[0])
            with (output_dir / "paired_proposals.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

    _write_pilot_summary(output_dir, rows)
    return output_dir


def validate_runtime(cfg: dict[str, Any], *, output_dir: str | Path | None = None) -> Path:
    """Dependency-light Stage-0 validation; does not generate a proposal."""
    import mdtraj as md
    import torch

    guidance = validate_level11_config(cfg)
    if output_dir is None:
        output_dir = cfg.get("level11", {}).get("validation_output", "outputs/level11/validation")
    output_dir = resolve_path(cfg, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pca_path = resolve_path(cfg, cfg["reference"]["pca_model"])
    pca = PCACV.load(pca_path)
    torch_pc1 = TorchPC1.load(pca_path)
    start_pdb = resolve_path(cfg, cfg["system"]["start_pdb"])
    reference_topology = resolve_path(cfg, cfg["reference"]["topology_pdb"])
    residue_manifest = output_dir / "residue_order_manifest.json"
    validate_residue_order(
        reference_topology=reference_topology,
        condition_pdb=start_pdb,
        expected_seqres=str(cfg["system"]["seqres"]),
        expected_ca_count=pca.reference_ca_nm.shape[0],
        output_path=residue_manifest,
    )
    frame = md.load(str(start_pdb))
    ca_indices = frame.topology.select("protein and name CA")
    ca = np.asarray(frame.xyz[:, ca_indices, :], dtype=np.float32)
    numpy_value = float(pca.project_ca(ca)[0, 0])
    coords = torch.tensor(ca, dtype=torch.float32, requires_grad=True)
    torch_value = torch_pc1.project(coords)
    torch_value.sum().backward()
    difference = abs(float(torch_value.detach().cpu()[0]) - numpy_value)
    gradient_finite = bool(torch.isfinite(coords.grad).all())
    checks = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(cfg["model"].get("device", "cuda:0")),
        "numpy_pc1": numpy_value,
        "torch_pc1": float(torch_value.detach().cpu()[0]),
        "pc1_absolute_difference": difference,
        "pc1_tolerance": 1.0e-5,
        "pc1_matches": difference < 1.0e-5,
        "gradient_finite": gradient_finite,
        "active_steps": int(round(int(cfg["model"].get("diffusion_steps", 200)) * guidance.active_fraction)),
        "residue_order_manifest": str(residue_manifest),
    }
    if str(cfg["experiment"]["mode"]).lower() == "frozen_opes":
        bias = _potential_from_config(cfg, create_frozen=True)
        probes = np.linspace(bias.grid[0] - 0.2, bias.grid[-1] + 0.2, 100)
        torch_probes = torch.tensor(probes, dtype=torch.float64, requires_grad=True)
        torch_energies = bias.torch_energy(torch_probes).detach().numpy()
        interpolation_error = float(np.max(np.abs(torch_energies - bias.energy(probes))))
        checks["frozen_bias_interpolation_max_error"] = interpolation_error
        checks["frozen_bias_interpolation_matches"] = interpolation_error < 1.0e-10
    checks["passed"] = bool(
        checks["pc1_matches"]
        and checks["gradient_finite"]
        and checks.get("frozen_bias_interpolation_matches", True)
        and checks["cuda_available"]
    )
    write_json(output_dir / "validation.json", checks)
    if not checks["passed"]:
        raise RuntimeError(f"Level 1.1 runtime validation failed; inspect {output_dir / 'validation.json'}")
    return output_dir


def validate_model_smoke(cfg: dict[str, Any], *, output_dir: str | Path | None = None) -> Path:
    """Run three one-step proposals to verify the real ConfRover sampler hook."""
    import mdtraj as md

    guidance = validate_level11_config(cfg)
    potential = _potential_from_config(cfg, create_frozen=True)
    if output_dir is None:
        output_dir = "outputs/level11/validation/model_smoke"
    output_dir = resolve_path(cfg, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("run", {}).get("seed", 20260827)) + 991
    start_pdb = resolve_path(cfg, cfg["system"]["start_pdb"])
    pca = PCACV.load(resolve_path(cfg, cfg["reference"]["pca_model"]))
    kernel = _build_kernel(cfg, potential, guidance, seed)

    upstream = kernel.propose_upstream_seeded(
        condition_pdb=start_pdb,
        output_dir=output_dir / "upstream",
        proposal_seed=seed,
    )[0]
    kernel.set_guidance(enabled=True, clip_ratio=0.0)
    zero = kernel.propose_seeded(
        condition_pdb=start_pdb,
        output_dir=output_dir / "guided_zero",
        proposal_seed=seed,
    )[0]
    zero_diagnostics = dict(kernel.last_diagnostics)
    kernel.set_guidance(enabled=True, clip_ratio=guidance.clip_ratio)
    guided = kernel.propose_seeded(
        condition_pdb=start_pdb,
        output_dir=output_dir / "guided",
        proposal_seed=seed,
    )[0]
    guided_diagnostics = dict(kernel.last_diagnostics)

    upstream_traj = upstream.load()
    zero_traj = zero.load()
    guided_traj = guided.load()
    upstream_candidate = upstream_traj[-1]
    zero_candidate = zero_traj[-1]
    guided_candidate = guided_traj[-1]
    max_zero_difference = float(
        np.max(np.abs(upstream_candidate.xyz.astype(float) - zero_candidate.xyz.astype(float)))
    )
    upstream_pc1 = float(pca.project_trajectory(upstream_candidate)[0, 0])
    zero_pc1 = float(pca.project_trajectory(zero_candidate)[0, 0])
    guided_pc1 = float(pca.project_trajectory(guided_candidate)[0, 0])
    upstream_geometry = geometry_diagnostics(
        upstream_traj[0], upstream_candidate, **_geometry_kwargs(cfg)
    )
    guided_geometry = geometry_diagnostics(
        guided_traj[0], guided_candidate, **_geometry_kwargs(cfg)
    )
    expected_max_guided_steps = max(
        1, int(round(int(cfg["model"].get("diffusion_steps", 200)) * guidance.active_fraction))
    )
    checks = {
        "proposal_seed": seed,
        "upstream_vs_zero_max_coordinate_difference_nm": max_zero_difference,
        "zero_equivalence_tolerance_nm": 1.0e-6,
        "zero_equivalent": max_zero_difference <= 1.0e-6,
        "upstream_pc1": upstream_pc1,
        "zero_pc1": zero_pc1,
        "guided_pc1": guided_pc1,
        "upstream_bias_kj_mol": float(np.asarray(potential.energy(upstream_pc1))),
        "guided_bias_kj_mol": float(np.asarray(potential.energy(guided_pc1))),
        "upstream_geometry_valid": upstream_geometry.valid,
        "guided_geometry_valid": guided_geometry.valid,
        "guided_geometry_failure_reason": guided_geometry.failure_reason,
        "zero_diagnostics": zero_diagnostics,
        "guided_diagnostics": guided_diagnostics,
        "expected_max_guided_steps": expected_max_guided_steps,
        "guided_step_count_valid": 0
        < int(guided_diagnostics["guided_step_count"])
        <= expected_max_guided_steps,
        "guidance_ratio_within_clip": float(
            guided_diagnostics["guidance_to_score_ratio_max"]
        )
        <= guidance.clip_ratio + 1e-5,
    }
    checks["passed"] = bool(
        checks["zero_equivalent"]
        and checks["upstream_geometry_valid"]
        and checks["guided_geometry_valid"]
        and checks["guided_step_count_valid"]
        and checks["guidance_ratio_within_clip"]
    )
    write_json(output_dir / "model_smoke.json", checks)
    if not checks["passed"]:
        raise RuntimeError(f"Level 1.1 model smoke failed; inspect {output_dir / 'model_smoke.json'}")
    return output_dir
