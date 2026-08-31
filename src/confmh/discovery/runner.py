from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from confmh.config import resolve_path, save_config
from confmh.discovery.cv import ControlCV
from confmh.discovery.selection import select_candidate
from confmh.discovery.structure import write_single_frame_pdb
from confmh.discovery.validity import evaluate_validity
from confmh.kernel import ConfRoverKernel


def _build_kernel(cfg: dict[str, Any], seed: int):
    model = cfg["model"]
    backend = str(model.get("backend", "confrover")).lower()
    if backend == "confrover":
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
            ckpt_dir=None if model.get("ckpt_dir") is None else resolve_path(cfg, model["ckpt_dir"]),
            kv_cache_type=str(model.get("kv_cache_type", "offloaded")),
            use_deepspeed_evo_attention=bool(model.get("use_deepspeed_evo_attention", False)),
        )
    if backend == "proar":
        from confmh.proar_kernel import ProARKernel

        return ProARKernel(
            repo=resolve_path(cfg, model.get("repo", "external/ProAR")),
            device=str(model.get("device", "cuda:0")),
            case_id=str(cfg["system"]["case_id"]),
            input_data_dir=resolve_path(cfg, model.get("input_data_dir", "data/proar_discovery")),
            forecaster_checkpoint=resolve_path(cfg, model["forecaster_checkpoint"]),
            interpolator_checkpoint=resolve_path(cfg, model["interpolator_checkpoint"]),
            interpolator_config=resolve_path(cfg, model["interpolator_config"]),
            cache_dir=resolve_path(cfg, model.get("cache_dir", "data/proar_discovery_cache")),
            horizon=int(model.get("horizon", 6)),
            sampling_type=str(model.get("sampling_type", "naive")),
            refine_intermediate_predictions=bool(model.get("refine_intermediate_predictions", True)),
            seed=seed,
        )
    raise ValueError(f"Unsupported model backend {backend!r}")


def _build_control_cv(cfg: dict[str, Any]) -> ControlCV:
    protocol = cfg["protocol"]
    cv_cfg = protocol["cv"]
    target_pdb = None
    if str(protocol["name"]) == "target_aware":
        target_pdb = resolve_path(cfg, cfg["benchmark"]["target_pdb"])
    elif str(protocol["name"]) != "target_blind":
        raise ValueError("protocol.name must be target_blind or target_aware")
    return ControlCV(
        start_pdb=resolve_path(cfg, cfg["system"]["start_pdb"]),
        target_pdb=target_pdb,
        mode=str(cv_cfg.get("mode", "start_rmsd")),
        mask=str(cv_cfg.get("mask", "all_ca")),
        residue_ranges=cv_cfg.get("residue_ranges"),
        tica_model=None if cv_cfg.get("tica_model") is None else resolve_path(cfg, cv_cfg["tica_model"]),
        tica_component=int(cv_cfg.get("tica_component", 0)),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_controlled_rollout(cfg: dict[str, Any]) -> Path:
    """Run equal-budget raw or bias-resampled autoregressive exploration.

    There is deliberately no MH acceptance step, proposal-density ratio, energy
    reweighting, or equilibrium reconstruction in this function.
    """
    run_cfg, controller_cfg = cfg["run"], cfg["controller"]
    seed = int(run_cfg.get("seed", 20260901))
    rng = np.random.default_rng(seed)
    output_dir = resolve_path(cfg, run_cfg.get("output_dir", "outputs/discovery/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")
    start_pdb = resolve_path(cfg, cfg["system"]["start_pdb"])
    if not start_pdb.exists():
        raise FileNotFoundError(start_pdb)

    centers = [float(value) for value in controller_cfg.get("centers", [0.0])]
    if not centers:
        raise ValueError("controller.centers cannot be empty")
    total_calls = int(run_cfg.get("total_model_calls", 128))
    candidates_per_step = int(run_cfg.get("candidates_per_step", 4))
    if total_calls < len(centers) or candidates_per_step < 1:
        raise ValueError("total_model_calls must cover every lane and candidates_per_step must be positive")
    base, extra = divmod(total_calls, len(centers))
    budgets = [base + int(index < extra) for index in range(len(centers))]

    lane_root = output_dir / "lanes"
    candidate_root = output_dir / "candidates"
    proposal_root = output_dir / "model_outputs"
    for directory in (lane_root, candidate_root, proposal_root):
        directory.mkdir(exist_ok=True)
    current_paths = []
    for lane_index in range(len(centers)):
        lane_dir = lane_root / f"lane_{lane_index:02d}"
        lane_dir.mkdir(exist_ok=True)
        current = lane_dir / "current.pdb"
        if not current.exists():
            shutil.copy2(start_pdb, current)
        current_paths.append(current)

    state_path, records_path = output_dir / "state.json", output_dir / "candidates.jsonl"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        used = [int(value) for value in state["used_calls_by_lane"]]
        batch_step = int(state["batch_step"])
        total_used = int(state["total_used"])
        if "rng_state" in state:
            rng.bit_generator.state = state["rng_state"]
    else:
        used, batch_step, total_used = [0] * len(centers), 0, 0
    kernel = _build_kernel(cfg, seed)
    control_cv = _build_control_cv(cfg)
    use_safety_filter = bool(cfg.get("validity", {}).get("safety_filter", False))
    cv_description = control_cv.describe()
    if str(cfg["protocol"]["name"]) == "target_blind" and cv_description["target_loaded"]:
        raise RuntimeError("Target-blind guard failed: alternate state was loaded by the runner")

    strict_valid_count = safety_valid_count = safety_fallback_count = 0
    run_started = time.perf_counter()
    while total_used < total_calls:
        progressed = False
        for lane_index, center in enumerate(centers):
            remaining = budgets[lane_index] - used[lane_index]
            if remaining <= 0:
                continue
            progressed = True
            n_candidates = min(candidates_per_step, remaining)
            proposal_dir = proposal_root / f"batch_{batch_step:08d}"
            proposal_started = time.perf_counter()
            proposals = kernel.propose(
                condition_pdb=current_paths[lane_index],
                output_dir=proposal_dir,
                step=batch_step,
                n_replicates=n_candidates,
            )
            proposal_walltime = time.perf_counter() - proposal_started
            if len(proposals) != n_candidates:
                raise RuntimeError(f"Requested {n_candidates} candidates, received {len(proposals)}")
            candidate_rows, values, selection_flags = [], [], []
            for candidate_index, proposal in enumerate(proposals):
                trajectory = proposal.load()
                if len(trajectory) < 2:
                    raise RuntimeError("Autoregressive proposal must include its condition and a future frame")
                global_index = total_used + candidate_index
                candidate_path = candidate_root / f"candidate_{global_index:06d}.pdb"
                write_single_frame_pdb(trajectory[-1], candidate_path)
                cv_result = control_cv.evaluate(candidate_path)
                validity = evaluate_validity(
                    candidate_path,
                    reference_pdb=resolve_path(cfg, cfg["validity"].get("reference_pdb", cfg["system"]["start_pdb"])),
                    thresholds=cfg["validity"].get("thresholds"),
                )
                safety_failed = set(validity.failed_checks) & {
                    "ca_valid_fraction", "peptide_cn_valid_fraction", "clash_free_fraction"
                }
                safety_valid = not safety_failed
                strict_valid_count += int(validity.valid)
                safety_valid_count += int(safety_valid)
                values.append(cv_result.control)
                selection_eligible = safety_valid if use_safety_filter else True
                selection_flags.append(selection_eligible)
                candidate_rows.append(
                    {
                        "model_call": global_index + 1,
                        "batch_step": batch_step,
                        "lane": lane_index,
                        "lane_center": center,
                        "lane_call": used[lane_index] + candidate_index + 1,
                        "candidate_index": candidate_index,
                        "candidate_pdb": str(candidate_path),
                        "control_cv": cv_result.control,
                        "rmsd_start_a": cv_result.rmsd_start,
                        "rmsd_target_a_during_generation": cv_result.rmsd_target,
                        "radius_gyration_a": cv_result.radius_gyration,
                        "native_contact_fraction": cv_result.native_contact_fraction,
                        "safety_valid": safety_valid,
                        "selection_eligible": selection_eligible,
                        "strict_valid": validity.valid,
                        "paper_compliant_0_90": validity.paper_compliant_0_90,
                        "strengthened_valid": validity.strengthened_valid,
                        "validity": validity.to_dict(),
                        "selected": False,
                        "selection_score": None,
                        "selection_probability": 0.0,
                        "proposal_batch_walltime_s": proposal_walltime,
                    }
                )
            selection = select_candidate(
                np.asarray(values),
                np.asarray(selection_flags),
                controller=str(controller_cfg["kind"]),
                rng=rng,
                center=center,
                kappa=float(controller_cfg.get("kappa", 1.0)),
                temperature=float(controller_cfg.get("selection_temperature", 1.0)),
            )
            for index, row in enumerate(candidate_rows):
                row["selection_score"] = float(selection.scores[index])
                row["selection_probability"] = float(selection.probabilities[index])
                row["selected"] = selection.index == index
                row["safety_fallback"] = selection.safety_fallback
            if selection.index is not None:
                shutil.copy2(candidate_rows[selection.index]["candidate_pdb"], current_paths[lane_index])
            else:
                safety_fallback_count += 1
            with records_path.open("a", encoding="utf-8") as handle:
                for row in candidate_rows:
                    handle.write(json.dumps(row) + "\n")
            used[lane_index] += n_candidates
            total_used += n_candidates
            batch_step += 1
            _write_json(
                state_path,
                {
                    "status": "running" if total_used < total_calls else "complete",
                    "used_calls_by_lane": used,
                    "budgets_by_lane": budgets,
                    "batch_step": batch_step,
                    "total_used": total_used,
                    "total_model_calls": total_calls,
                    "rng_state": rng.bit_generator.state,
                },
            )
            if bool(run_cfg.get("cleanup_model_outputs", True)):
                shutil.rmtree(proposal_dir, ignore_errors=True)
        if not progressed:
            break

    elapsed = time.perf_counter() - run_started
    summary = {
        "status": "complete",
        "benchmark": cfg.get("benchmark", {}).get("name"),
        "model_backend": cfg["model"].get("backend", "confrover"),
        "protocol": cfg["protocol"]["name"],
        "controller": controller_cfg["kind"],
        "control_cv": cv_description,
        "model_calls": total_used,
        "strict_valid_count_this_process": strict_valid_count,
        "safety_valid_count_this_process": safety_valid_count,
        "safety_fallback_batches_this_process": safety_fallback_count,
        "validity_used_during_selection": use_safety_filter,
        "walltime_s_this_process": elapsed,
        "mh_used": False,
        "equilibrium_claim": False,
        "physical_time_claim": False,
        "warning": "Selection probabilities are controller weights, not equilibrium weights.",
    }
    _write_json(output_dir / "run_summary.json", summary)
    return output_dir
