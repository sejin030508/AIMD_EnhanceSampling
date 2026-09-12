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
    args = parser.parse_args()

    destination = args.code_root / args.subdir
    destination.mkdir(parents=True, exist_ok=True)
    for molecule in args.molecules:
        source = (
            args.code_root / "configs"
            / f"{molecule}_stride{args.source_stride}_t32.yaml"
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
        config["experiment"]["seeds"] = [int(seed) for seed in args.seeds]
        config["experiment"]["task_count"] = 2 * len(args.seeds)
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
