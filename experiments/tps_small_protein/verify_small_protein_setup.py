#!/usr/bin/env python3
"""Read-only launch gate for the fixed small-protein pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


EXPECTED_COMMIT = "61fd65ad2e2f110d65c176a8c8f5c2fe8bdab034"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    rows = []
    for molecule in ("chignolin", "trpcage", "bba"):
        manifest_path = args.data_root / "prepared" / molecule / "manifest.json"
        if not manifest_path.exists():
            failures.append(f"{molecule}: missing manifest")
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "ready":
            failures.append(f"{molecule}: status={manifest.get('status')}")
        if manifest.get("official_repository_commit") != EXPECTED_COMMIT:
            failures.append(f"{molecule}: TPS-DPS commit mismatch")
        for name, expected in manifest.get("prepared_file_sha256", {}).items():
            path = args.data_root / "prepared" / molecule / name
            if not path.exists() or sha256(path) != expected:
                failures.append(f"{molecule}: prepared hash mismatch {name}")
        configs = []
        for stride in (16, 128):
            config_path = args.code_root / "configs" / f"{molecule}_stride{stride}_t32.yaml"
            config = yaml.safe_load(config_path.read_text())
            checks = {
                "lag": config["model"]["physical_lag_in_10ps"] == stride,
                "horizon": config["trajectory"]["horizon"] == 32,
                "reward": config["program"]["reward_coefficient"] == 16.0,
                "no_floor": config["program"]["reward_log_floor"] is None,
                "checkpoints": config["particles"]["inner_checkpoint_progresses"] == [0.75, 0.9],
                "frozen": config["experiment"]["method_settings"]["frozen"] == {"outer_k": 16, "inner_m": 1},
                "duet": config["experiment"]["method_settings"]["duet"] == {"outer_k": 4, "inner_m": 4},
                "seeds": config["experiment"]["seeds"] == [211, 223],
                "snapshots": config["experiment"]["analysis_transitions"] == [8, 16, 32],
            }
            if not all(checks.values()):
                failures.append(f"{molecule} stride={stride}: config checks={checks}")
            configs.append({"stride": stride, "checks": checks})
        rows.append({
            "protein": molecule,
            "length": manifest["model_length"],
            "d0_a": manifest["d0_a"],
            "tica_start_to_target": manifest["tica"]["start_to_target_distance"],
            "configs": configs,
        })
    report = {"ready": not failures, "failures": failures, "proteins": rows}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
