from __future__ import annotations

import copy
import shlex
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from confmh.duet.config import resolve_config_path
from confmh.duet.observables import ObservableRegistry
from confmh.duet.outer_smc import OuterSMC
from confmh.duet.potentials import PotentialCoefficients, PrefixPotential
from confmh.duet.programs import TemporalProgram
from confmh.duet.records import initialize_run_directory, write_json
from confmh.duet.runner import build_confrover_adapter
from confmh.pca_cv import _kabsch_align


def _aligned_ca_history(history):
    reference = np.asarray(history[0].ca_nm, dtype=float)
    return [_kabsch_align(np.asarray(frame.ca_nm, dtype=float), reference) for frame in history]


def _trace_metrics(histories, adapter, pca_model=None):
    final_frames = [history[-1] for history in histories]
    validity = adapter.validate_frames(final_frames)
    aligned = [_aligned_ca_history(history) for history in histories]
    displacement = [
        float(np.linalg.norm(frames[-1] - frames[0], axis=-1).mean())
        for frames in aligned
    ]
    temporal = [
        float(
            np.mean(
                [
                    np.linalg.norm(right - left, axis=-1).mean()
                    for left, right in zip(frames[:-1], frames[1:])
                ]
            )
        )
        for frames in aligned
    ]
    raw_displacement = [
        float(np.linalg.norm(history[-1].ca_nm - history[0].ca_nm, axis=-1).mean())
        for history in histories
    ]
    raw_temporal = [
        float(np.mean([
            np.linalg.norm(right.ca_nm - left.ca_nm, axis=-1).mean()
            for left, right in zip(history[:-1], history[1:])
        ]))
        for history in histories
    ]
    pc1 = None
    if pca_model is not None:
        pc1 = [float(pca_model.project_ca(frame.ca_nm)[0]) for frame in final_frames]
    return {
        "replicates": len(histories),
        "nonfinite_rate": float(
            np.mean([item["nonfinite_coordinate_count"] > 0 for item in validity])
        ),
        "geometry_valid_rate": float(np.mean([item["valid"] for item in validity])),
        "clash_rate": float(np.mean([item["ca_clash_count_lt_1a"] > 0 for item in validity])),
        "ca_adjacent_max_a": [float(item["ca_adjacent_max_a"]) for item in validity],
        "ca_adjacent_fraction_ge_4_5a": [
            float(item["ca_adjacent_fraction_ge_4_5a"]) for item in validity
        ],
        "ca_endpoint_displacement_nm": displacement,
        "temporal_displacement_nm": temporal,
        "raw_ca_endpoint_displacement_nm": raw_displacement,
        "raw_temporal_displacement_nm": raw_temporal,
        "coordinate_alignment": "Kabsch CA alignment of every frame to frame 0",
        "endpoint_pc1": pc1,
        "validity": validity,
    }


def evaluate_preflight_gate(results: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the explicit Phase-B engineering/scientific feasibility gate."""

    official = results["official_ode"]
    sde_names = [name for name in results if name != "official_ode"]
    official_mean_max = float(np.mean(official["ca_adjacent_max_a"]))
    max_ratio = float(gate_cfg.get("max_sde_to_ode_mean_ca_adjacent_ratio", 1.10))
    max_pc1_wasserstein = float(gate_cfg.get("max_pc1_wasserstein", 0.5))
    base_pc1 = np.asarray(results["unconditioned_sde"]["endpoint_pc1"], dtype=float)
    pc1_wasserstein = {
        name: float(wasserstein_distance(base_pc1, results[name]["endpoint_pc1"]))
        for name in ("sde_checkpoint_no_resampling", "sde_constant_resampling")
    }
    def adjacent_maxima(item):
        if "ca_adjacent_max_a" in item:
            return item["ca_adjacent_max_a"]
        return [row["ca_adjacent_max_a"] for row in item.get("validity", [])]

    mean_max = {
        name: float(np.mean(adjacent_maxima(results[name]))) for name in sde_names
    }
    ratios = {name: value / official_mean_max for name, value in mean_max.items()}
    checks = {
        "all_variants_nonfinite_rate_zero": all(
            float(item["nonfinite_rate"]) == 0.0 for item in results.values()
        ),
        "all_variants_clash_rate_zero": all(
            float(item["clash_rate"]) == 0.0 for item in results.values()
        ),
        "all_variants_hard_geometry_valid": all(
            float(item["geometry_valid_rate"]) == 1.0 for item in results.values()
        ),
        "sde_mean_max_ca_within_ode_ratio": all(value <= max_ratio for value in ratios.values()),
        "constant_potential_pc1_marginals_within_tolerance": all(
            value <= max_pc1_wasserstein for value in pc1_wasserstein.values()
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "ca_adjacent_quality_threshold_a": float(
                gate_cfg.get("ca_adjacent_quality_threshold_a", 4.5)
            ),
            "ca_adjacent_hard_threshold_a": float(
                gate_cfg.get("ca_adjacent_hard_threshold_a", 5.5)
            ),
            "max_sde_to_ode_mean_ca_adjacent_ratio": max_ratio,
            "max_pc1_wasserstein": max_pc1_wasserstein,
        },
        "observed": {
            "official_ode_mean_ca_adjacent_max_a": official_mean_max,
            "sde_mean_ca_adjacent_max_a": mean_max,
            "sde_to_ode_mean_ca_adjacent_ratios": ratios,
            "pc1_wasserstein_from_unconditioned_sde": pc1_wasserstein,
        },
        "rng_protocol": "NumPy and Torch seeded per particle and reverse step",
    }


def _official_ode_metrics(cfg, output, pca, replicates):
    """Run the unmodified public ``ConfRover.generate`` ODE path."""
    from confmh.kernel import ConfRoverKernel

    model, trajectory = cfg["model"], cfg["trajectory"]
    kernel = ConfRoverKernel(
        repo=resolve_config_path(cfg, model["repository_path"]),
        model_name=str(resolve_config_path(cfg, model["checkpoint"])),
        device=str(model.get("device", "cuda:0")),
        case_id=str(trajectory["case_id"]),
        seqres=str(trajectory["seqres"]),
        stride_in_10ps=int(model.get("physical_lag_in_10ps", 256)),
        diffusion_steps=int(model.get("reverse_steps", 200)),
        seed=int(cfg["experiment"]["seeds"][0]),
        cache_dir=resolve_config_path(cfg, model["cache_dir"]),
        ckpt_dir=resolve_config_path(cfg, model["checkpoint"]).parent,
        kv_cache_type=str(model.get("kv_cache_type", "offloaded")),
    )
    started = time.perf_counter()
    proposals = kernel.generate_forward(
        condition_pdb=resolve_config_path(cfg, trajectory["initial_structure"]),
        output_dir=output / "official_ode_raw",
        n_frames=int(trajectory["horizon"]) + 1,
        n_replicates=replicates,
        seed=int(cfg["experiment"]["seeds"][0]),
    )
    wall = time.perf_counter() - started
    trajectories = [proposal.load() for proposal in proposals]
    displacements, temporal, pc1 = [], [], []
    nonfinite, geometry, clash = [], [], []
    adjacent_max, adjacent_fraction = [], []
    gate_cfg = cfg.get("preflight", {}).get("gate", {})
    quality_threshold = float(gate_cfg.get("ca_adjacent_quality_threshold_a", 4.5))
    hard_threshold = float(gate_cfg.get("ca_adjacent_hard_threshold_a", 5.5))
    for trajectory_md in trajectories:
        ca_indices = trajectory_md.topology.select("name CA")
        coords = trajectory_md.xyz[:, ca_indices, :]
        displacements.append(float(np.linalg.norm(coords[-1] - coords[0], axis=-1).mean()))
        temporal.append(float(np.linalg.norm(np.diff(coords, axis=0), axis=-1).mean()))
        pc1.append(float(pca.project_trajectory(trajectory_md)[-1, 0]))
        nonfinite.append(bool(np.any(~np.isfinite(coords))))
        adjacent = np.linalg.norm(np.diff(coords[-1], axis=0), axis=-1) * 10.0
        adjacent_max.append(float(np.max(adjacent)))
        adjacent_fraction.append(float(np.mean(adjacent >= quality_threshold)))
        clashes = [
            np.linalg.norm(coords[-1, i] - coords[-1, j]) * 10.0 < 1.0
            for i in range(len(ca_indices))
            for j in range(i + 2, len(ca_indices))
        ]
        clash.append(bool(np.any(clashes)))
        geometry.append(
            bool(
                not nonfinite[-1]
                and not clash[-1]
                and np.all(adjacent < hard_threshold)
            )
        )
    return {
        "replicates": replicates,
        "nonfinite_rate": float(np.mean(nonfinite)),
        "geometry_valid_rate": float(np.mean(geometry)),
        "clash_rate": float(np.mean(clash)),
        "ca_adjacent_max_a": adjacent_max,
        "ca_adjacent_fraction_ge_4_5a": adjacent_fraction,
        "ca_endpoint_displacement_nm": displacements,
        "temporal_displacement_nm": temporal,
        "raw_ca_endpoint_displacement_nm": None,
        "raw_temporal_displacement_nm": None,
        "coordinate_alignment": "ConfRover writer center_coordinates + superpose to frame 0",
        "endpoint_pc1": pc1,
        "sampler_mode": "ode",
        "checkpoint_hook": False,
        "inner_m": 1,
        "wall_clock_s": wall,
        "nfe_accounting": {
            "reverse_decoder_evaluations": replicates
            * int(trajectory["horizon"])
            * int(model.get("reverse_steps", 200))
        },
    }


def run_confrover_preflight(
    cfg: dict[str, Any], *, resume: bool = False, overwrite: bool = False
):
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
    from confmh.pca_cv import PCACV

    pca = PCACV.load(resolve_config_path(cfg, cfg["reference"]["pca_model"]))
    variants = cfg["preflight"]["variants"]
    replicates = int(cfg["preflight"]["replicates"])
    results = {}
    for variant_name, variant in variants.items():
        if variant_name == "official_ode":
            results[variant_name] = _official_ode_metrics(
                cfg, output, pca, replicates
            )
            continue
        variant_cfg = copy.deepcopy(cfg)
        variant_cfg["model"]["sampler_mode"] = variant["sampler_mode"]
        adapter = build_confrover_adapter(variant_cfg)
        adapter.load_model()
        histories = []
        started = time.perf_counter()
        for replicate in range(replicates):
            history = adapter.initial_history()
            for t in range(1, int(cfg["trajectory"]["horizon"]) + 1):
                history_state = adapter.prepare_history(history)
                seed_base = int(cfg["experiment"]["seeds"][0]) + 100003 * replicate + 1009 * t
                if not variant["checkpoint_hook"]:
                    frame = adapter.sample_complete_frames_direct(history_state, 1, [seed_base])[0]
                else:
                    registry = ObservableRegistry({"constant": lambda frame: 0.0})
                    program = TemporalProgram.from_config(
                        {
                            "type": "terminal",
                            "events": [
                                {
                                    "name": "constant",
                                    "observable": "constant",
                                    "target_interval": [-1, 1],
                                    "physical_window": [
                                        int(cfg["trajectory"]["horizon"]),
                                        int(cfg["trajectory"]["horizon"]),
                                    ],
                                }
                            ],
                        }
                    )
                    potential = PrefixPotential(
                        program,
                        registry,
                        PotentialCoefficients(lambda_program=0.0, potential_floor=1e-30),
                        int(cfg["trajectory"]["horizon"]),
                    )
                    run = OuterSMC(
                        adapter=adapter,
                        potential=potential,
                        method="inner_only",
                        outer_k=1,
                        inner_m=int(variant["inner_m"]),
                        checkpoint_progress=float(cfg["particles"]["inner_checkpoint_progress"]),
                        seed=seed_base,
                    ).run(history, horizon=1)
                    frame = run.particles[0].history[-1]
                history.append(frame)
            histories.append(history)
        wall = time.perf_counter() - started
        results[variant_name] = {
            **_trace_metrics(histories, adapter, pca),
            "sampler_mode": variant["sampler_mode"],
            "checkpoint_hook": variant["checkpoint_hook"],
            "inner_m": variant["inner_m"],
            "wall_clock_s": wall,
            "nfe_accounting": adapter.accounting.to_dict(),
        }
    write_json(output / "metrics.json", {"variants": results})
    write_json(
        output / "nfe_accounting.json",
        {key: value["nfe_accounting"] for key, value in results.items()},
    )
    write_json(
        output / "wall_clock.json",
        {key: value["wall_clock_s"] for key, value in results.items()},
    )
    write_json(
        output / "seed_manifest.json",
        {"base_seed": cfg["experiment"]["seeds"][0], "replicates": replicates},
    )
    gate = evaluate_preflight_gate(results, cfg.get("preflight", {}).get("gate", {}))
    write_json(output / "gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"Phase B gate failed; inspect {output / 'gate.json'}")
    return output
