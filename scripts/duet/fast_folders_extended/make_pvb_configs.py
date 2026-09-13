#!/usr/bin/env python
"""Derive PVB small-protein configs from the existing ConfRover ones.

Copying rather than writing fresh keeps the reward, the pocket references, the
horizon and the particle counts byte-identical, so the emulator is the only axis
that differs from the completed pilot.

Only the model block is replaced.  PVB carries its own 100 ps ATLAS lag per
transition, so ``physical_lag_in_10ps`` has no meaning here and is dropped: the
two backends are matched on the number of autoregressive transitions, not on
nominal physical time.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--molecules", nargs="+", default=["trpcage", "bba"])
    parser.add_argument("--source-stride", type=int, default=128)
    parser.add_argument("--pvb-repo", required=True)
    parser.add_argument("--pvb-checkpoint", required=True)
    parser.add_argument("--sde-step", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[211])
    parser.add_argument("--subdir", default="configs_pvb")
    parser.add_argument("--outer-k", type=int, default=8)
    parser.add_argument("--inner-m", type=int, default=8)
    parser.add_argument("--frozen-k", type=int, default=64)
    parser.add_argument(
        "--checkpoints", nargs="+", type=float, default=[0.75, 0.90]
    )
    parser.add_argument(
        "--methods", nargs="+", default=["frozen", "complete_nested", "duet"]
    )
    args = parser.parse_args()

    destination = args.code_root / args.subdir
    destination.mkdir(parents=True, exist_ok=True)
    for molecule in args.molecules:
        # BBL was prepared later and its ConfRover config lives in
        # configs_extended/, so both locations are searched.
        candidates = [
            args.code_root / sub / f"{molecule}_stride{args.source_stride}_t32.yaml"
            for sub in ("configs", "configs_extended")
        ]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise SystemExit(
                f"no ConfRover config for {molecule}; looked in "
                + ", ".join(str(path) for path in candidates)
            )
        config = copy.deepcopy(yaml.safe_load(source.read_text(encoding="utf-8")))
        config["model"] = {
            "backend": "pvb",
            "repository_path": args.pvb_repo,
            "checkpoint": args.pvb_checkpoint,
            "device": "cuda:0",
            "sde_step": int(args.sde_step),
            # The runner records which solver step each checkpoint lands on, so
            # reverse_steps is PVB's bridge resolution rather than ConfRover's.
            "reverse_steps": int(args.sde_step),
            # PVB's ATLAS fine-tuning pairs are 100 ps apart, i.e. 10 units of
            # 10 ps.  Recording it truthfully is what keeps the two backends
            # from being read as the same physical horizon.
            "physical_lag_in_10ps": 10,
            "sampler_mode": "sde",
        }
        trajectory = config["trajectory"]
        trajectory["case_id"] = f"pvb_small_protein_{molecule}"
        # PVB removes hydrogens itself, so it takes the all-atom structure
        # rather than the ConfRover heavy-atom export.
        trajectory["initial_structure"] = (
            "${SMALL_PROTEIN_DATA_ROOT}/prepared/"
            f"{molecule}/unfolded_minimized_allatom.pdb"
        )
        # The stage must equal what the cell runner computes for itself
        # ("stride<N>_t32"); a different name here sends the sampler and the
        # evaluator to different directories and evaluation fails after the
        # sampling has already succeeded.  Isolation comes from a separate
        # SMALL_PROTEIN_OUTPUT_ROOT instead.
        config["experiment"]["stage"] = f"stride{args.source_stride}_t32"
        budget = args.outer_k * args.inner_m
        if args.frozen_k != budget:
            raise SystemExit(
                f"frozen K ({args.frozen_k}) must equal K*M ({budget}); the runner "
                "rejects methods whose population differs from the budget"
            )
        config["particles"]["outer_k"] = args.outer_k
        config["particles"]["inner_m"] = args.inner_m
        config["particles"]["inner_checkpoint_progresses"] = list(args.checkpoints)
        config["experiment"]["methods"] = list(args.methods)
        config["experiment"]["method_settings"] = {
            "frozen": {"outer_k": args.frozen_k, "inner_m": 1},
            "complete_nested": {"outer_k": args.outer_k, "inner_m": args.inner_m},
            "duet": {"outer_k": args.outer_k, "inner_m": args.inner_m},
        }
        config["experiment"]["decoder_population_budget"] = budget
        config["experiment"]["seeds"] = [int(seed) for seed in args.seeds]
        config["experiment"]["task_count"] = len(args.methods) * len(args.seeds)
        gate = config.setdefault("preflight", {}).setdefault("gate", {})
        gate.setdefault("ca_adjacent_quality_threshold_a", 4.5)
        gate.setdefault("ca_adjacent_hard_threshold_a", 5.5)
        # PVB produces intact peptide bonds, so the check Bundle A showed
        # ConfRover could not pass is enforced rather than merely reported.
        gate["enforce_peptide_bond"] = True

        # The cell runner derives its evaluation path from molecule and stride,
        # so the filename must keep the stride the runner will be told.
        target = destination / f"{molecule}_stride{args.source_stride}_t32.yaml"
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"{molecule}: {target}")
        print(
            f"    horizon={trajectory['horizon']} sde_step={args.sde_step} "
            f"checkpoints={config['particles']['inner_checkpoint_progresses']} "
            f"seeds={config['experiment']['seeds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
