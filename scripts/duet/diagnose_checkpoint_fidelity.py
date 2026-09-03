from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from confmh.adapters.confrover_duet import ConfRoverFrame
from confmh.duet.checkpoint_diagnostics import summarize_checkpoint_fidelity
from confmh.duet.config import load_duet_config, project_root, resolve_config_path
from confmh.duet.observables import ObservableRegistry
from confmh.duet.potentials import PotentialCoefficients, PrefixPotential
from confmh.duet.programs import ProgressState, TemporalProgram
from confmh.duet.records import write_json
from confmh.duet.runner import build_confrover_adapter


POTENTIAL_KEYS = (
    "lambda_program",
    "distance_weight",
    "remaining_event_weight",
    "deadline_weight",
    "failure_penalty",
    "terminal_failure_penalty",
    "failure_guidance_weight",
    "distance_scale",
    "potential_floor",
)


def _load_program(cfg: dict[str, Any], task: str) -> tuple[TemporalProgram, ObservableRegistry, PrefixPotential]:
    catalog_path = resolve_config_path(cfg, cfg["program"]["catalog"])
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    program_cfg = dict(catalog["tasks"][task])
    program_cfg.update(
        {key: cfg["program"][key] for key in POTENTIAL_KEYS if key in cfg["program"]}
    )
    program = TemporalProgram.from_config(program_cfg)
    observables = ObservableRegistry.from_config(
        catalog["observables"], root=project_root(cfg)
    )
    potential = PrefixPotential(
        program,
        observables,
        PotentialCoefficients.from_config(program_cfg),
        int(cfg["trajectory"]["horizon"]),
    )
    return program, observables, potential


def _seed_vector(seed: int, parent_step: int, stream: int, count: int) -> list[int]:
    # Deliberately omit checkpoint progress: different progress values must see
    # paired initial particles and continuation seed families.  Otherwise seed
    # variation is confounded with the checkpoint-location comparison.
    sequence = np.random.SeedSequence([int(seed), int(parent_step), int(stream)])
    return sequence.generate_state(int(count), dtype=np.uint32).astype(np.int64).tolist()


def _evaluate_candidates(
    potential: PrefixPotential,
    history: list[ConfRoverFrame],
    parent_state: ProgressState,
    frames: list[ConfRoverFrame],
    candidate_step: int,
) -> list[tuple[float, ProgressState, dict[str, float]]]:
    return [
        potential.candidate_log_psi(history, parent_state, frame, candidate_step)
        for frame in frames
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure predicted-clean checkpoint fidelity using replicated continuations."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--task", default="ordered")
    parser.add_argument("--path-index", type=int, required=True)
    parser.add_argument("--parent-steps", type=int, nargs="+", required=True)
    parser.add_argument("--progresses", type=float, nargs="+", required=True)
    parser.add_argument("--checkpoint-count", type=int, default=8)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.checkpoint_count < 2 or args.continuations < 1:
        raise ValueError("checkpoint-count must be >=2 and continuations must be >=1")
    if any(not 0.0 < progress < 1.0 for progress in args.progresses):
        raise ValueError("Every checkpoint progress must be in (0, 1)")

    cfg = load_duet_config(args.config)
    trajectory_path = Path(args.trajectories).expanduser()
    if not trajectory_path.is_absolute():
        trajectory_path = project_root(cfg) / trajectory_path
    with np.load(trajectory_path.resolve(), allow_pickle=False) as archive:
        trajectories = np.asarray(archive["trajectories"])
    if not 0 <= args.path_index < len(trajectories):
        raise IndexError(f"path-index {args.path_index} outside [0, {len(trajectories)})")
    selected_path = trajectories[args.path_index]

    program, observables, potential = _load_program(cfg, args.task)
    adapter = build_confrover_adapter(cfg)
    adapter.load_model()
    template = adapter.initial_history()[0]
    frames = [
        ConfRoverFrame(
            atom37_a=np.asarray(atom37, dtype=np.float32),
            atom37_mask=np.asarray(template.atom37_mask, dtype=bool).copy(),
            aatype=np.asarray(template.aatype).copy(),
        )
        for atom37 in selected_path
    ]

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    names = program.all_observables()
    for parent_step in args.parent_steps:
        if not 0 <= parent_step < len(frames) - 1:
            raise IndexError(
                f"parent-step {parent_step} requires a following frame in [0, {len(frames)})"
            )
        history = frames[: parent_step + 1]
        parent_state = ProgressState()
        for physical_step in range(1, parent_step + 1):
            values = observables.evaluate(frames[physical_step], names)
            parent_state = program.update(parent_state, values, physical_step)
        candidate_step = parent_step + 1
        history_state = adapter.prepare_history(history)

        for progress in args.progresses:
            count = int(args.checkpoint_count)
            replicates = int(args.continuations)
            accounting_before = adapter.accounting.to_dict()
            proposal_seeds = _seed_vector(
                args.seed, parent_step, stream=0, count=count
            )
            state = adapter.initialize_inner_particles(history_state, count, proposal_seeds)
            state = adapter.denoise_to_checkpoint(state, progress)
            checkpoint_frames = adapter.predict_clean(state)
            checkpoint_eval = _evaluate_candidates(
                potential, history, parent_state, checkpoint_frames, candidate_step
            )

            ancestors = np.repeat(np.arange(count, dtype=int), replicates)
            state = adapter.resample_particle_state(state, ancestors)
            continuation_seeds = _seed_vector(
                args.seed,
                parent_step,
                stream=1,
                count=count * replicates,
            )
            state = adapter.denoise_to_end(state, continuation_seeds)
            child_frames = adapter.finalize_frames(state)
            child_eval = _evaluate_candidates(
                potential, history, parent_state, child_frames, candidate_step
            )

            checkpoint_log_psi = np.asarray([item[0] for item in checkpoint_eval])
            child_log_psi = np.asarray([item[0] for item in child_eval]).reshape(
                count, replicates
            )
            checkpoint_values = {
                name: np.asarray([item[2][name] for item in checkpoint_eval])
                for name in names
            }
            child_values = {
                name: np.asarray([item[2][name] for item in child_eval]).reshape(
                    count, replicates
                )
                for name in names
            }
            child_advanced = np.asarray(
                [item[1].stage > parent_state.stage for item in child_eval], dtype=bool
            ).reshape(count, replicates)
            child_failed = np.asarray(
                [item[1].failed for item in child_eval], dtype=bool
            ).reshape(count, replicates)
            child_stages = np.asarray(
                [item[1].stage for item in child_eval], dtype=int
            ).reshape(count, replicates)

            summary = summarize_checkpoint_fidelity(
                checkpoint_log_psi,
                child_log_psi,
                checkpoint_observables=checkpoint_values,
                child_observables=child_values,
                child_advanced=child_advanced,
                child_failed=child_failed,
            )
            accounting_after = adapter.accounting.to_dict()
            accounting_delta = {
                key: accounting_after[key] - accounting_before[key]
                for key in accounting_after
            }
            results.append(
                {
                    "parent_step": int(parent_step),
                    "candidate_step": int(candidate_step),
                    "parent_progress": {
                        "stage": int(parent_state.stage),
                        "failed": bool(parent_state.failed),
                        "completed_frames": list(parent_state.completed_frames),
                        "reason": parent_state.reason,
                    },
                    "checkpoint_progress": float(progress),
                    "checkpoint_log_psi": checkpoint_log_psi.tolist(),
                    "child_log_psi": child_log_psi.tolist(),
                    "checkpoint_observables": {
                        key: value.tolist() for key, value in checkpoint_values.items()
                    },
                    "child_observables": {
                        key: value.tolist() for key, value in child_values.items()
                    },
                    "child_stages": child_stages.tolist(),
                    "child_advanced": child_advanced.tolist(),
                    "child_failed": child_failed.tolist(),
                    "summary": summary,
                    "accounting_delta": accounting_delta,
                }
            )
            print(
                f"parent={parent_step} candidate={candidate_step} progress={progress:.3f} "
                f"spearman={summary['checkpoint_to_conditional_log_psi_spearman']} "
                f"gain={summary['checkpoint_selection_log_gain_vs_uniform']:.6f}",
                flush=True,
            )

    payload = {
        "diagnostic": "checkpoint_predicted_clean_to_continuation_marginal_fidelity",
        "config": str(Path(args.config).expanduser().resolve()),
        "trajectory_archive": str(trajectory_path.resolve()),
        "task": args.task,
        "path_index": int(args.path_index),
        "parent_steps": [int(item) for item in args.parent_steps],
        "checkpoint_progresses": [float(item) for item in args.progresses],
        "checkpoint_count": int(args.checkpoint_count),
        "continuations_per_checkpoint": int(args.continuations),
        "seed": int(args.seed),
        "wall_clock_s": float(time.perf_counter() - started),
        "accounting": adapter.accounting.to_dict(),
        "results": results,
        "interpretation_boundary": (
            "This measures one physical proposal step on a saved history. It diagnoses "
            "checkpoint ranking fidelity; it is not an end-to-end method comparison."
        ),
    }
    write_json(args.output, payload)
    print(Path(args.output).expanduser().resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
