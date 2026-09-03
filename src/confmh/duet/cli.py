from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from confmh.duet.atlas_programs import prepare_atlas_programs
from confmh.duet.config import (
    estimate_decoder_nfe,
    inner_checkpoint_progresses,
    load_duet_config,
    missing_assets,
    resolve_config_path,
)
from confmh.duet.exact_benchmark import run_exact_benchmark
from confmh.duet.preflight import run_confrover_preflight
from confmh.duet.runner import run_allocation_grid, run_experiment_config


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="duet-md")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate", "Validate a DuET config and report assets/compute"),
        ("run", "Run a configured DuET experiment phase"),
        ("prepare-programs", "Construct R1/R2-only ATLAS temporal programs"),
        ("preflight", "Run the ConfRover ODE/SDE feasibility gate"),
        ("run-grid", "Run the fixed-compute K/M allocation grid"),
    ):
        child = sub.add_parser(name, help=help_text)
        _common(child)
    return parser


def _apply_overrides(cfg, args):
    if args.output_dir:
        if args.command == "prepare-programs":
            cfg.setdefault("output", {})["directory"] = args.output_dir
        else:
            cfg["experiment"]["output_directory"] = args.output_dir
    if args.seed is not None:
        cfg["experiment"]["seeds"] = [int(args.seed)]
    return cfg


def _summary(cfg):
    methods = cfg["experiment"].get("methods", [cfg["experiment"].get("method", "duet")])
    checkpoint_progresses = inner_checkpoint_progresses(cfg)
    return {
        "config": cfg["_config_path"],
        "model_repository": str(resolve_config_path(cfg, cfg["model"].get("repository_path", "."))),
        "model_checkpoint": str(resolve_config_path(cfg, cfg["model"].get("checkpoint", "."))),
        "tasks": cfg["experiment"].get("tasks", ["inline"]),
        "task_count": len(cfg["experiment"].get("tasks", ["inline"])),
        "methods": methods,
        "K": int(cfg["particles"]["outer_k"]),
        "M": int(cfg["particles"]["inner_m"]),
        "checkpoint_after_reverse_fraction": float(checkpoint_progresses[-1]),
        "checkpoint_after_reverse_fractions": [
            float(item) for item in checkpoint_progresses
        ],
        "estimated_decoder_nfe": {method: estimate_decoder_nfe(cfg, method) for method in methods},
        "output": str(resolve_config_path(cfg, cfg["experiment"].get("output_directory", "outputs/duet_md"))),
        "missing_assets": missing_assets(cfg),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _apply_overrides(load_duet_config(args.config), args)
    summary = _summary(cfg)
    print(json.dumps(summary, indent=2))
    if args.command == "prepare-programs":
        if args.dry_run or args.validate_only:
            return 0 if not summary["missing_assets"] else 2
        path = prepare_atlas_programs(cfg)
        print(path)
        return 0
    if args.command == "validate" or args.validate_only or args.dry_run:
        return 0 if not summary["missing_assets"] else 2
    if args.command == "preflight":
        outputs = [run_confrover_preflight(cfg, resume=args.resume, overwrite=args.overwrite)]
    elif args.command == "run-grid":
        outputs = run_allocation_grid(cfg, resume=args.resume, overwrite=args.overwrite)
    elif str(cfg["model"].get("backend", "confrover")) == "exact_toy":
        outputs = [run_exact_benchmark(cfg, resume=args.resume, overwrite=args.overwrite)]
    elif str(cfg["model"].get("backend", "confrover")) == "proar":
        raise RuntimeError(
            "Phase F is gated and the ProAR stochastic intermediate checkpoint has not been verified"
        )
    else:
        outputs = run_experiment_config(cfg, resume=args.resume, overwrite=args.overwrite)
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
