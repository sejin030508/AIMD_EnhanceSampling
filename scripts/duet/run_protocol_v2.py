from __future__ import annotations

import argparse

from confmh.duet import runner as legacy_runner
from confmh.duet.config import load_duet_config
from confmh.duet.outer_smc import OuterSMC


class AdaptiveOuterSMC(OuterSMC):
    """V2-only default without mutating the frozen legacy experiment runner."""

    ess_fraction = 0.5

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("outer_resampling_ess_fraction", self.ess_fraction)
        super().__init__(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--outer-k", type=int)
    parser.add_argument("--inner-m", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--checkpoint-progress", type=float)
    parser.add_argument("--checkpoint-progresses", type=float, nargs="+")
    parser.add_argument("--outer-resampling-ess-fraction", type=float)
    parser.add_argument("--output-directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_progress is not None and args.checkpoint_progresses is not None:
        parser.error("--checkpoint-progress and --checkpoint-progresses are mutually exclusive")
    cfg = load_duet_config(args.config)
    if args.methods:
        cfg["experiment"]["methods"] = list(args.methods)
    if args.tasks:
        cfg["experiment"]["tasks"] = list(args.tasks)
        cfg["experiment"]["task_count"] = len(args.tasks)
    if args.seed is not None:
        cfg["experiment"]["seeds"] = [int(args.seed)]
    if (args.outer_k is None) != (args.inner_m is None):
        parser.error("--outer-k and --inner-m must be specified together")
    if args.outer_k is not None:
        if args.outer_k < 1 or args.inner_m < 1:
            parser.error("--outer-k and --inner-m must be positive")
        cfg["particles"]["outer_k"] = int(args.outer_k)
        cfg["particles"]["inner_m"] = int(args.inner_m)
        active_methods = args.methods or cfg["experiment"]["methods"]
        method_settings = cfg["experiment"].setdefault("method_settings", {})
        for method in active_methods:
            method_settings.setdefault(method, {})["outer_k"] = int(args.outer_k)
            method_settings[method]["inner_m"] = int(args.inner_m)
        cfg["experiment"]["decoder_population_budget"] = int(
            args.outer_k * args.inner_m
        )
        cfg["experiment"].setdefault("cost_budget", {})[
            "decoder_population_per_step"
        ] = int(args.outer_k * args.inner_m)
    if args.horizon is not None:
        if args.horizon < 1:
            parser.error("--horizon must be positive")
        cfg["trajectory"]["horizon"] = int(args.horizon)
    if args.checkpoint_progress is not None:
        if not 0.0 < args.checkpoint_progress < 1.0:
            parser.error("--checkpoint-progress must be in (0, 1)")
        cfg["particles"]["inner_checkpoint_progress"] = float(
            args.checkpoint_progress
        )
        cfg["particles"].pop("inner_checkpoint_progresses", None)
    if args.checkpoint_progresses is not None:
        progresses = [float(item) for item in args.checkpoint_progresses]
        if any(not 0.0 < item < 1.0 for item in progresses):
            parser.error("--checkpoint-progresses values must be in (0, 1)")
        if any(right <= left for left, right in zip(progresses, progresses[1:])):
            parser.error("--checkpoint-progresses must be strictly increasing")
        cfg["particles"]["inner_checkpoint_progresses"] = progresses
        cfg["particles"]["inner_checkpoint_progress"] = progresses[-1]
    if args.outer_resampling_ess_fraction is not None:
        if not 0.0 < args.outer_resampling_ess_fraction <= 1.0:
            parser.error("--outer-resampling-ess-fraction must be in (0, 1]")
        cfg["particles"]["outer_resampling_ess_fraction"] = float(
            args.outer_resampling_ess_fraction
        )
    if args.output_directory is not None:
        cfg["experiment"]["output_directory"] = str(args.output_directory)
    AdaptiveOuterSMC.ess_fraction = float(
        cfg["particles"].get("outer_resampling_ess_fraction", 0.5)
    )

    original = legacy_runner.OuterSMC
    legacy_runner.OuterSMC = AdaptiveOuterSMC
    try:
        outputs = legacy_runner.run_experiment_config(
            cfg, resume=args.resume, overwrite=args.overwrite
        )
    finally:
        legacy_runner.OuterSMC = original
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
