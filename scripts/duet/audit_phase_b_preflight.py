#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def audit_protein(data_root: Path, output_root: Path, protein: str) -> dict[str, Any]:
    manifest_path = data_root / "prepared" / protein / "manifest.json"
    manifest = load(manifest_path)
    checks: dict[str, Any] = {
        "manifest_ready": bool(manifest and manifest.get("status") == "ready"),
        "d0_a": None if manifest is None else manifest.get("d0_a"),
    }
    run_specs = {
        "memory_gate_k1_m4_t1": ("duet", 17),
        "memory_gate_k4_m4_t2": ("duet", 17),
        "b0": ("frozen", 17),
    }
    metrics_by_stage = {}
    for stage, (method, seed) in run_specs.items():
        path = output_root / protein / stage / method / f"seed_{seed}" / "metrics.json"
        metrics = load(path)
        metrics_by_stage[stage] = None if metrics is None else {
            "path": str(path),
            "status": metrics.get("status"),
            "decoder_nfe": metrics.get("decoder_nfe"),
            "peak_gpu_memory_bytes": metrics.get("peak_gpu_memory_bytes"),
            "valid_path_fraction": metrics.get("valid_path_fraction"),
            "potential_clipping_fraction": metrics.get("potential_clipping_fraction"),
        }
        checks[f"{stage}_complete"] = bool(metrics and metrics.get("status") == "complete")
        checks[f"{stage}_has_valid_path"] = bool(
            metrics and float(metrics.get("valid_path_fraction", 0.0)) > 0.0
        )
    b0_path = output_root / protein / "b0" / "frozen" / "seed_17" / "metrics.json"
    b0 = load(b0_path)
    if b0:
        rows = [row for path in b0.get("path_validity", []) for row in path]
        checks["b0_no_nonfinite"] = all(
            int(row.get("nonfinite_coordinate_count", 0)) == 0 for row in rows
        )
        checks["b0_no_ca_clashes"] = all(
            int(row.get("ca_clash_count_lt_1a", 0)) == 0 for row in rows
        )
        checks["b0_no_invalid_frames"] = all(bool(row.get("valid")) for row in rows)
    else:
        checks.update(
            {
                "b0_no_nonfinite": False,
                "b0_no_ca_clashes": False,
                "b0_no_invalid_frames": False,
            }
        )
    ready = all(bool(value) for key, value in checks.items() if key != "d0_a")
    return {
        "protein": protein,
        "ready_for_b1": ready,
        "checks": checks,
        "runs": metrics_by_stage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        audit_protein(args.data_root, args.output_root, protein)
        for protein in ("prmt5", "prmt6")
    ]
    payload = {
        "phase": "B",
        "b1_started": False,
        "all_ready_for_b1": all(row["ready_for_b1"] for row in rows),
        "proteins": rows,
    }
    destination = args.output_root / "preflight_status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(destination)
    for row in rows:
        print(row["protein"], "ready" if row["ready_for_b1"] else "pending")


if __name__ == "__main__":
    main()
