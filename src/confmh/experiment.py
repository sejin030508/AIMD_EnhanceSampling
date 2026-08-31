from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from confmh.bias import HarmonicBias, OPES1D, ZeroBias
from confmh.config import resolve_path, save_config
from confmh.kernel import ConfRoverKernel
from confmh.mh import accept, acceptance_probability, bias_only_log_alpha
from confmh.pca_cv import PCACV
from confmh.utils import kbt_kj_mol, seed_everything, write_json


def _build_kernel(cfg: dict[str, Any], seed: int):
    model = cfg["model"]
    backend = str(model.get("backend", "confrover")).lower()
    if backend == "proar":
        from confmh.proar_kernel import ProARKernel

        return ProARKernel(
            repo=resolve_path(cfg, model.get("repo", "external/ProAR")),
            device=str(model.get("device", "cuda:0")),
            case_id=str(cfg["system"]["case_id"]),
            input_data_dir=resolve_path(cfg, model.get("input_data_dir", "data/proar")),
            forecaster_checkpoint=resolve_path(cfg, model["forecaster_checkpoint"]),
            interpolator_checkpoint=resolve_path(cfg, model["interpolator_checkpoint"]),
            interpolator_config=resolve_path(cfg, model["interpolator_config"]),
            cache_dir=resolve_path(cfg, model.get("cache_dir", "data/proar_cache")),
            horizon=int(model.get("horizon", 6)),
            sampling_type=str(model.get("sampling_type", "naive")),
            refine_intermediate_predictions=bool(
                model.get("refine_intermediate_predictions", True)
            ),
            seed=seed,
        )
    if backend != "confrover":
        raise ValueError(f"Unsupported model.backend={backend!r}")
    return ConfRoverKernel(
        repo=resolve_path(cfg, model.get("repo", "external/ConfRover")),
        model_name=str(model.get("name", "ConfRover-base-20M-v1.0")),
        device=str(model.get("device", "cuda:0")),
        case_id=str(cfg["system"]["case_id"]),
        seqres=str(cfg["system"]["seqres"]),
        stride_in_10ps=int(model.get("stride_in_10ps", 256)),
        diffusion_steps=int(model.get("diffusion_steps", 200)),
        seed=seed,
        cache_dir=resolve_path(cfg, model.get("cache_dir", "data/confrover_cache")),
        ckpt_dir=(
            None
            if model.get("ckpt_dir") is None
            else resolve_path(cfg, model["ckpt_dir"])
        ),
        kv_cache_type=str(model.get("kv_cache_type", "offloaded")),
        use_deepspeed_evo_attention=bool(model.get("use_deepspeed_evo_attention", False)),
    )


def _build_bias(cfg: dict[str, Any], kbt: float):
    experiment = cfg["experiment"]
    mode = str(experiment["mode"]).lower()
    if mode == "base":
        return ZeroBias()
    if mode == "umbrella":
        return HarmonicBias(
            center=float(experiment["center"]),
            kappa_kj_mol=float(experiment["kappa_kj_mol"]),
        )
    if mode == "opes":
        reference = np.load(resolve_path(cfg, cfg["reference"]["reference_cv"]))["cv"][:, 0]
        lower = float(experiment.get("grid_min", np.quantile(reference, 0.005)))
        upper = float(experiment.get("grid_max", np.quantile(reference, 0.995)))
        margin = float(experiment.get("grid_margin", 0.5))
        return OPES1D(
            grid_min=lower - margin,
            grid_max=upper + margin,
            n_grid=int(experiment.get("n_grid", 401)),
            bandwidth=float(experiment.get("bandwidth", 0.2)),
            kbt_kj_mol=kbt,
            bias_factor=float(experiment.get("bias_factor", 10.0)),
            barrier_kj_mol=float(experiment.get("barrier_kj_mol", 15.0)),
            update_interval=int(experiment.get("update_interval", 100)),
            adapt_steps=int(experiment.get("adapt_steps", 2000)),
            max_samples=int(experiment.get("max_samples", 10000)),
        )
    raise ValueError(f"Unsupported experiment.mode={mode!r}")


def _save_records(output_dir: Path, records: dict[str, list[Any]]) -> None:
    arrays = {key: np.asarray(values) for key, values in records.items()}
    np.savez_compressed(output_dir / "chain.npz", **arrays)


def run_experiment(cfg: dict[str, Any]) -> Path:
    """Run raw rollout or bias-only MH using only q_theta samples and b(x)."""
    import mdtraj as md

    run = cfg.get("run", {})
    seed = int(run.get("seed", 20260827))
    rng = seed_everything(seed)
    output_dir = resolve_path(cfg, run.get("output_dir", "outputs/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")

    mode = str(cfg["experiment"]["mode"]).lower()
    temperature = float(cfg["system"].get("temperature_k", 300.0))
    kbt = kbt_kj_mol(temperature)
    pca = PCACV.load(resolve_path(cfg, cfg["reference"]["pca_model"]))
    bias = _build_bias(cfg, kbt)
    kernel = _build_kernel(cfg, seed)

    start_pdb = resolve_path(cfg, cfg["system"]["start_pdb"])
    if not start_pdb.exists():
        raise FileNotFoundError(start_pdb)
    current_condition = output_dir / "current.pdb"
    shutil.copy2(start_pdb, current_condition)

    if mode == "opes":
        total_steps = int(cfg["experiment"].get("adapt_steps", 2000)) + int(
            cfg["experiment"].get("production_steps", 3000)
        )
    else:
        total_steps = int(run.get("steps", 3000))
    cleanup = bool(run.get("cleanup_proposals", True))
    checkpoint_interval = int(run.get("checkpoint_interval", 100))

    proposal_root = output_dir / "proposals"
    proposal_root.mkdir(exist_ok=True)
    records: dict[str, list[Any]] = {
        "step": [],
        "phase": [],
        "cv": [],
        "proposal_cv": [],
        "bias_kj_mol": [],
        "proposal_bias_kj_mol": [],
        "log_alpha": [],
        "acceptance_probability": [],
        "accepted": [],
        "proposal_walltime_s": [],
    }
    xtc_writer = None
    topology_path = output_dir / "topology.pdb"
    accepted_count = 0

    try:
        backend = str(cfg["model"].get("backend", "confrover"))
        proposal_horizon = int(getattr(kernel, "proposal_horizon", 1))
        frame_stride_10ps = int(getattr(kernel, "stride_in_10ps", 0))
        for step in tqdm(range(total_steps), desc=f"{backend} {mode}"):
            proposal_dir = proposal_root / f"step_{step:08d}"
            start_time = time.perf_counter()
            proposals = kernel.propose(
                condition_pdb=current_condition,
                output_dir=proposal_dir,
                step=step,
                n_replicates=1,
            )
            proposal_walltime = time.perf_counter() - start_time
            trajectory = proposals[0].load()
            if len(trajectory) < 2:
                raise RuntimeError("ConfRover forward proposal must contain the condition and one future frame")

            current_frame = trajectory[0]
            candidate_frame = trajectory[-1]
            current_cv = float(pca.project_trajectory(current_frame)[0, 0])
            candidate_cv = float(pca.project_trajectory(candidate_frame)[0, 0])
            current_bias = float(np.asarray(bias.energy(current_cv)))
            candidate_bias = float(np.asarray(bias.energy(candidate_cv)))

            if mode == "base":
                log_alpha = 0.0
                accepted = True
            else:
                log_alpha = bias_only_log_alpha(candidate_bias - current_bias, kbt)
                accepted = accept(log_alpha, rng)

            selected = candidate_frame if accepted else current_frame
            if accepted:
                accepted_count += 1
            selected.save_pdb(str(current_condition))
            selected_cv = candidate_cv if accepted else current_cv
            selected_bias = candidate_bias if accepted else current_bias

            if xtc_writer is None:
                current_frame.save_pdb(str(topology_path))
                xtc_writer = md.formats.XTCTrajectoryFile(str(output_dir / "chain.xtc"), mode="w")
                xtc_writer.write(current_frame.xyz.astype(np.float32))
            xtc_writer.write(selected.xyz.astype(np.float32))

            phase = "production"
            if mode == "opes" and step < int(cfg["experiment"].get("adapt_steps", 2000)):
                phase = "adapt"
            records["step"].append(step)
            records["phase"].append(phase)
            records["cv"].append(selected_cv)
            records["proposal_cv"].append(candidate_cv)
            records["bias_kj_mol"].append(selected_bias)
            records["proposal_bias_kj_mol"].append(candidate_bias)
            records["log_alpha"].append(log_alpha)
            records["acceptance_probability"].append(acceptance_probability(log_alpha))
            records["accepted"].append(accepted)
            records["proposal_walltime_s"].append(proposal_walltime)
            records.setdefault("nominal_time_ns", []).append(
                (step + 1) * proposal_horizon * frame_stride_10ps * 0.01
            )

            bias.observe(selected_cv, selected_bias, step + 1)
            bias.maybe_update(step + 1)
            if (step + 1) % checkpoint_interval == 0:
                _save_records(output_dir, records)
                bias.save(output_dir / "bias.npz")
            if cleanup:
                shutil.rmtree(proposal_dir, ignore_errors=True)
    finally:
        if xtc_writer is not None:
            xtc_writer.close()

    _save_records(output_dir, records)
    bias.save(output_dir / "bias.npz")
    metadata = {
        "mode": mode,
        "temperature_k": temperature,
        "kbt_kj_mol": kbt,
        "steps": total_steps,
        "accepted": accepted_count,
        "acceptance_rate": accepted_count / total_steps,
        "model_backend": str(cfg["model"].get("backend", "confrover")),
        "model_frame_stride_in_10ps": int(getattr(kernel, "stride_in_10ps", 0)),
        "proposal_horizon_frames": int(getattr(kernel, "proposal_horizon", 1)),
        "physical_time_warning": "One chain step is an MCMC proposal, not physical MD time.",
        "exactness_warning": (
            "Bias-only MH is exact only if the frozen proposal transition is reversible with "
            "respect to its base equilibrium distribution. This is not assumed without diagnostics."
        ),
    }
    write_json(output_dir / "metadata.json", metadata)
    return output_dir
