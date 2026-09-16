#!/usr/bin/env python3
"""One-transition real-checkpoint validation for Guided-PVB (no production run)."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from confmh.adapters.pvb_duet import PVBDuETAdapter
from confmh.duet.tica_potential import TicaEndpointMetric, TicaEndpointPotential
from confmh.duet.torch_tica import TorchTicaEndpointPotential


def _localized_manifest(path: Path, prepared: Path, official: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["tica_model"] = str(official / "tica_model.pkl")
    manifest["minimized_target_allatom"] = str(
        prepared / "folded_minimized_allatom.pdb"
    )
    manifest["reference_npz"] = str(
        prepared / "whole_backbone_reference_atom37.npz"
    )
    manifest["official_source_files"] = [
        {**row, "path": str(official / Path(row["path"]).name)}
        for row in manifest["official_source_files"]
    ]
    return manifest


def _complete_base(adapter, history, seeds, continuations, checkpoints):
    state = adapter.initialize_inner_particles(history, len(seeds), seeds)
    for index, checkpoint in enumerate(checkpoints):
        state = adapter.denoise_to_checkpoint(state, checkpoint)
        if index < len(checkpoints) - 1:
            state = adapter.reseed_particle_state(state, continuations[index])
    return adapter.denoise_to_end(state, continuations[-1])


def _complete_guided(adapter, history, seeds, continuations, checkpoints):
    state = adapter.initialize_inner_particles(history, len(seeds), seeds)
    total_ratio = np.zeros(len(seeds), dtype=np.float64)
    checkpoint_scores = []
    for index, checkpoint in enumerate(checkpoints):
        state, ratio = adapter.guided_denoise_to_checkpoint(state, checkpoint)
        total_ratio += ratio
        checkpoint_scores.append(adapter.guided_checkpoint_log_potential(state))
        if index < len(checkpoints) - 1:
            state = adapter.reseed_particle_state(state, continuations[index])
    state, ratio = adapter.guided_denoise_to_end(state, continuations[-1])
    total_ratio += ratio
    return state, total_ratio, checkpoint_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    manifest = _localized_manifest(
        args.prepared / "manifest.json", args.prepared, args.official
    )
    adapter = PVBDuETAdapter(
        repository_path=args.repository,
        checkpoint=args.checkpoint,
        initial_structure=args.prepared / "unfolded_minimized_allatom.pdb",
        device=args.device,
        sde_step=20,
    )
    adapter.load_model()
    history = adapter.prepare_history(adapter.initial_history())
    metric = TicaEndpointMetric(manifest, dimensions=2)
    initial_frame = adapter.initial_history()[0]
    d0 = metric.distance_a(initial_frame)
    production = TicaEndpointPotential(
        metric, d0, coefficient=16.0, log_floor=None
    )
    differentiable = TorchTicaEndpointPotential.from_metric(
        metric, adapter.topology, coefficient=16.0, d0=d0, log_floor=None
    ).to(args.device)
    differentiable.eval()
    differentiable.requires_grad_(False)
    coordinates = torch.as_tensor(
        history.coordinates_a, dtype=torch.float32, device=args.device
    )
    with torch.no_grad():
        torch_initial = float(differentiable.forward_flat(coordinates, 1)[0].cpu())
    production_initial = production.log_psi(
        [initial_frame], None, 0, production.values(initial_frame)
    )

    seeds = [701]
    continuations = [[1701], [2701]]
    checkpoints = [0.5, 0.75]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base_state = _complete_base(
        adapter, copy.deepcopy(history), seeds, continuations, checkpoints
    )
    base_endpoint = base_state.final_xt.detach().clone()
    base_repeat_state = _complete_base(
        adapter, copy.deepcopy(history), seeds, continuations, checkpoints
    )
    base_repeat_endpoint = base_repeat_state.final_xt.detach().clone()
    base_repeat_max_abs_a = float(
        torch.max(torch.abs(base_endpoint - base_repeat_endpoint)).cpu()
    )

    adapter.configure_guidance(differentiable, strength=0.0)
    zero_state, zero_ratio, _ = _complete_guided(
        adapter, copy.deepcopy(history), seeds, continuations, checkpoints
    )
    eta_zero_exact = bool(torch.equal(base_endpoint, zero_state.final_xt))
    eta_zero_max_abs_a = float(
        torch.max(torch.abs(base_endpoint - zero_state.final_xt)).cpu()
    )
    # PVB's CUDA scatter kernels are not bitwise deterministic across complete
    # repeated decodes.  The original PVB preflight measured an approximately
    # 1.4e-4 A repeatability floor, so full-checkpoint parity is judged against
    # both a conservative 2e-4 A ceiling and the observed base/base repeat.
    eta_zero_tolerance_a = max(2.0e-4, 2.0 * base_repeat_max_abs_a)

    before_backward = adapter.accounting.guidance_backward_evaluations
    adapter.configure_guidance(differentiable, strength=args.eta)
    guided_state, guided_ratio, checkpoint_scores = _complete_guided(
        adapter, copy.deepcopy(history), seeds, continuations, checkpoints
    )
    guided_endpoint_scores = adapter.guided_endpoint_log_potential(guided_state)
    elapsed = time.perf_counter() - started
    result = {
        "status": "pass",
        "device": args.device,
        "eta": float(args.eta),
        "tica_d0": float(d0),
        "torch_production_initial_log_potential_abs_error": float(
            abs(torch_initial - production_initial)
        ),
        "base_repeat_max_abs_coordinate_a": base_repeat_max_abs_a,
        "eta_zero_endpoint_bitwise_equal": eta_zero_exact,
        "eta_zero_max_abs_coordinate_a": eta_zero_max_abs_a,
        "eta_zero_backend_tolerance_a": eta_zero_tolerance_a,
        "eta_zero_proposal_ratio": zero_ratio.tolist(),
        "guided_proposal_ratio": guided_ratio.tolist(),
        "guided_checkpoint_log_potentials": [row.tolist() for row in checkpoint_scores],
        "guided_endpoint_log_potential": guided_endpoint_scores.tolist(),
        "guided_endpoint_finite": bool(torch.isfinite(guided_state.final_xt).all()),
        "guidance_backward_evaluations_delta": int(
            adapter.accounting.guidance_backward_evaluations - before_backward
        ),
        "accounting": adapter.accounting.to_dict(),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_clock_s": float(elapsed),
    }
    checks = {
        "torch_production_parity": (
            result["torch_production_initial_log_potential_abs_error"] <= 2.0e-5
        ),
        "eta_zero_within_backend_repeatability": (
            eta_zero_max_abs_a <= eta_zero_tolerance_a
        ),
        "eta_zero_ratio_zero": bool(np.array_equal(zero_ratio, np.zeros(1))),
        "guided_ratio_finite": bool(np.all(np.isfinite(guided_ratio))),
        "guided_score_finite": bool(np.all(np.isfinite(guided_endpoint_scores))),
        "backward_executed": result["guidance_backward_evaluations_delta"] > 0,
    }
    result["checks"] = checks
    if not all(checks.values()):
        result["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
