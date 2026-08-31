from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from confmh.config import resolve_path, save_config
from confmh.experiment import _build_kernel
from confmh.pca_cv import PCACV
from confmh.utils import kbt_kj_mol, seed_everything, write_json


def run_forward_baseline(cfg: dict[str, Any]) -> Path:
    """Run one official-style ConfRover trajectory with full temporal history."""
    run = cfg.get("run", {})
    seed = int(run.get("seed", 20260827))
    seed_everything(seed)
    output_dir = resolve_path(cfg, run.get("output_dir", "outputs/full_history"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")

    n_frames = int(run.get("n_frames", 40))
    if n_frames < 2:
        raise ValueError("run.n_frames must be at least 2")
    start_pdb = resolve_path(cfg, cfg["system"]["start_pdb"])
    if not start_pdb.exists():
        raise FileNotFoundError(start_pdb)

    pca = PCACV.load(resolve_path(cfg, cfg["reference"]["pca_model"]))
    kernel = _build_kernel(cfg, seed)
    raw_dir = output_dir / "raw"
    started = time.perf_counter()
    proposal = kernel.generate_forward(
        condition_pdb=start_pdb,
        output_dir=raw_dir,
        n_frames=n_frames,
        n_replicates=1,
        seed=seed,
    )[0]
    walltime = time.perf_counter() - started
    trajectory = proposal.load()
    if len(trajectory) != n_frames:
        raise RuntimeError(f"Expected {n_frames} frames, got {len(trajectory)}")

    keep_every = int(run.get("keep_every", 1))
    if keep_every < 1:
        raise ValueError("run.keep_every must be at least 1")
    frame_stride_ns = kernel.stride_in_10ps * 0.01
    scores = pca.project_trajectory(trajectory)[:, 0]
    # Exclude the conditioning frame so both baselines contain one record per transition.
    analysis_indices = np.arange(keep_every, n_frames, keep_every, dtype=int)
    values = scores[analysis_indices]
    steps = len(values)
    if steps < 1:
        raise ValueError(
            f"run.keep_every={keep_every} retains no generated frame from n_frames={n_frames}"
        )
    per_transition_walltime = walltime / steps
    np.savez_compressed(
        output_dir / "chain.npz",
        step=np.arange(steps, dtype=int),
        phase=np.asarray(["production"] * steps),
        cv=values,
        proposal_cv=values,
        bias_kj_mol=np.zeros(steps),
        proposal_bias_kj_mol=np.zeros(steps),
        log_alpha=np.zeros(steps),
        acceptance_probability=np.ones(steps),
        accepted=np.ones(steps, dtype=bool),
        proposal_walltime_s=np.full(steps, per_transition_walltime),
        nominal_time_ns=analysis_indices.astype(float) * frame_stride_ns,
    )
    analysis_trajectory = trajectory[np.concatenate(([0], analysis_indices))]
    analysis_trajectory[0].save_pdb(str(output_dir / "topology.pdb"))
    analysis_trajectory.save_xtc(str(output_dir / "chain.xtc"))

    temperature = float(cfg["system"].get("temperature_k", 300.0))
    write_json(
        output_dir / "metadata.json",
        {
            "mode": "base",
            "baseline_kind": "full_history",
            "temperature_k": temperature,
            "kbt_kj_mol": kbt_kj_mol(temperature),
            "seed": seed,
            "frames_including_condition": n_frames,
            "analysis_keep_every": keep_every,
            "steps": steps,
            "model_frame_stride_in_10ps": kernel.stride_in_10ps,
            "analysis_stride_in_10ps": kernel.stride_in_10ps * keep_every,
            "nominal_length_ns": float(analysis_indices[-1] * frame_stride_ns),
            "model_backend": str(cfg["model"].get("backend", "confrover")),
            "accepted": steps,
            "acceptance_rate": 1.0,
            "walltime_s": walltime,
            "physical_time_warning": (
                "The time axis is the model conditioning stride, not integrated MD time."
            ),
        },
    )
    if bool(run.get("cleanup_raw", False)):
        shutil.rmtree(raw_dir, ignore_errors=True)
    return output_dir
