#!/usr/bin/env python
"""Generate checkpoint-timing variants of the small-protein DuET configs.

Only ``particles.inner_checkpoint_progresses`` changes; every other field is
copied from the existing stride-128 config so the variants differ from the
completed (0.75, 0.90) cells in exactly one axis.

Each timing gets its own config subdirectory because the cell runner derives a
config name from the molecule and stride alone, and its own output root because
the run stage name is derived the same way.  Keeping them separate means the
existing results are never written over.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

TIMINGS = {
    "cp2550": [0.25, 0.50],
    "cp5075": [0.50, 0.75],
    "cp7590": [0.75, 0.90],
    "cp255075": [0.25, 0.50, 0.75],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--molecules", nargs="+", default=["trpcage", "bba"])
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--timings", nargs="+", default=["cp2550", "cp5075", "cp255075"])
    parser.add_argument("--source-subdir", default="configs")
    args = parser.parse_args()

    written = []
    for tag in args.timings:
        progresses = TIMINGS[tag]
        destination = args.code_root / f"configs_{tag}"
        destination.mkdir(parents=True, exist_ok=True)
        for molecule in args.molecules:
            source = (
                args.code_root / args.source_subdir
                / f"{molecule}_stride{args.stride}_t32.yaml"
            )
            config = yaml.safe_load(source.read_text(encoding="utf-8"))
            config = copy.deepcopy(config)
            config["particles"]["inner_checkpoint_progresses"] = list(progresses)
            # Frozen ignores checkpoints entirely, so re-running it would only
            # duplicate the baseline already collected at (0.75, 0.90).
            config["experiment"]["methods"] = ["duet"]
            config["experiment"]["method_settings"] = {
                "duet": config["experiment"]["method_settings"]["duet"]
            }
            config["experiment"]["task_count"] = 2
            config["experiment"]["stage"] = f"stride{args.stride}_t32_{tag}"
            target = destination / f"{molecule}_stride{args.stride}_t32.yaml"
            target.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            written.append((tag, molecule, progresses, str(target)))

    for tag, molecule, progresses, path in written:
        steps = [int(round(200 * p)) for p in progresses]
        print(f"{tag:10s} {molecule:9s} progresses={progresses} reverse_steps={steps}")
        print(f"           {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
