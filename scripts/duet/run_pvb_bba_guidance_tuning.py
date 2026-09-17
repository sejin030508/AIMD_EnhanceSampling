#!/usr/bin/env python3
"""Run one cell of the fixed 2026-09-17 PVB/BBA guidance matrix."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from confmh.duet.config import (
    load_duet_config,
    resolve_config_path,
    validate_duet_config,
)
from confmh.duet.phase_b_runner import run_one


@dataclass(frozen=True)
class Condition:
    method: str
    strength: float | None = None
    first_one_based_update: int | None = None
    inner_resampling: bool | None = None


CONDITIONS = {
    "C0": Condition("complete_nested"),
    "D0": Condition("duet"),
    "G1": Condition("guided_duet", 0.25, 11, True),
    "G2": Condition("guided_duet", 1.0, 11, True),
    "G3": Condition("guided_duet", 4.0, 11, True),
    "G4": Condition("guided_duet", 1.0, 6, True),
    "G5": Condition("guided_duet", 1.0, 11, False),
}


def configure(base: dict, condition_id: str, seed: int, device: str) -> dict:
    condition = CONDITIONS[condition_id]
    method = condition.method
    base["model"]["device"] = device
    base["experiment"]["stage"] = (
        f"pvb_bba_guidance_tuning_20260917_{condition_id.lower()}"
    )
    base["experiment"]["methods"] = [method]
    base["experiment"]["method_settings"] = {
        method: {"outer_k": 8, "inner_m": 8}
    }
    base["experiment"]["seeds"] = [int(seed)]
    base["experiment"]["task_count"] = 1
    if method != "guided_duet":
        base.pop("guidance", None)
    else:
        # PVB exposes stochastic zero-based updates 0..18.  The protocol's
        # one-based 11..19 and 6..19 therefore map to 10..18 and 5..18.
        assert condition.first_one_based_update is not None
        first = condition.first_one_based_update - 1
        base["guidance"] = {
            "enabled": True,
            "strength": float(condition.strength),
            "schedule": "explicit",
            "update_steps": list(range(first, 19)),
            "inner_resampling": bool(condition.inner_resampling),
        }
    validate_duet_config(base)
    return base


def expected_output(cfg: dict, condition_id: str, seed: int) -> Path:
    condition = CONDITIONS[condition_id]
    root = resolve_config_path(cfg, cfg["experiment"]["output_directory"])
    return (
        root
        / "bba"
        / cfg["experiment"]["stage"]
        / condition.method
        / f"seed_{seed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = configure(
        load_duet_config(args.config), args.condition, args.seed, args.device
    )
    output = expected_output(cfg, args.condition, args.seed)
    metrics = output / "metrics.json"
    if metrics.is_file() and not args.overwrite:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "skipped_complete", "output": str(output)}))
            return
    completed = run_one(
        cfg,
        method=CONDITIONS[args.condition].method,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": "complete", "output": str(completed)}))


if __name__ == "__main__":
    main()

