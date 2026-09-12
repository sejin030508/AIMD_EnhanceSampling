#!/usr/bin/env python3
"""Audit frozen route-v3.1 tasks without loading ConfRover or starting runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


EXPECTED_METHODS = [
    "frozen",
    "outer_only",
    "inner_only",
    "complete_nested",
    "duet",
]
EXPECTED_ALLOCATIONS = {
    "frozen": {"outer_k": 64, "inner_m": 1},
    "outer_only": {"outer_k": 64, "inner_m": 1},
    "inner_only": {"outer_k": 1, "inner_m": 64},
    "complete_nested": {"outer_k": 16, "inner_m": 4},
    "duet": {"outer_k": 16, "inner_m": 4},
}


def _expand(value: str, project_root: Path, asset_root: Path) -> Path:
    value = value.replace("${DUET_PROJECT_ROOT}", str(project_root))
    value = value.replace("${DUET_ASSET_ROOT}", str(asset_root))
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _event_semantics(catalog: dict[str, Any], horizon: int) -> list[str]:
    errors: list[str] = []
    if set(catalog.get("tasks", {})) != {"endpoint", "ordered"}:
        errors.append("catalog must contain exactly endpoint and ordered tasks")
        return errors
    endpoint = catalog["tasks"]["endpoint"]
    ordered = catalog["tasks"]["ordered"]
    for raw in endpoint.get("events", []):
        if raw.get("physical_window") != [1, horizon]:
            errors.append("endpoint target window is not the common deadline")
        if int(raw.get("persistence_frames", 1)) != 2:
            errors.append("endpoint target persistence is not 2")
    for raw in ordered.get("events", []) + [ordered.get("terminal_event", {})]:
        if raw.get("physical_window") != [1, horizon]:
            errors.append("ordered event window is not the common deadline")
        if int(raw.get("persistence_frames", 1)) != 2:
            errors.append("ordered event persistence is not 2")
    if bool(ordered.get("allow_same_frame", False)):
        errors.append("ordered task permits same-frame stage completion")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    report = yaml.safe_load(args.report.read_text(encoding="utf-8"))
    frozen = [row for row in report["cases"] if row["status"] == "task_frozen"]
    global_errors: list[str] = []
    if report.get("model_outcomes_inspected") is not False:
        global_errors.append("model outcomes were inspected before task freeze")
    if report.get("method_runs_started") is not False:
        global_errors.append("method runs started before audit")
    if len(frozen) != int(report["target_frozen_count"]):
        global_errors.append("frozen task count does not equal target")

    audited = []
    for row in frozen:
        errors: list[str] = []
        config_path = _expand(row["main_config"], project_root, asset_root)
        catalog_path = _expand(row["catalog"], project_root, asset_root)
        selection_path = _expand(row["selection_report"], project_root, asset_root)
        manifest_path = _expand(row["case_manifest"], project_root, asset_root)
        for label, path in (
            ("main_config", config_path),
            ("catalog", catalog_path),
            ("selection_report", selection_path),
            ("case_manifest", manifest_path),
        ):
            if not path.exists():
                errors.append(f"missing {label}: {path}")
        if errors:
            audited.append({"pdb_chain": row["pdb_chain"], "errors": errors})
            continue

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        horizon = int(cfg["trajectory"]["horizon"])
        if horizon != 24:
            errors.append(f"horizon is {horizon}, expected 24")
        if cfg["experiment"].get("methods") != EXPECTED_METHODS:
            errors.append("method list differs from frozen five-method suite")
        if cfg["experiment"].get("method_settings") != EXPECTED_ALLOCATIONS:
            errors.append("method K/M allocations differ from budget-64 plan")
        if int(cfg["experiment"].get("decoder_population_budget", 0)) != 64:
            errors.append("decoder population budget is not 64")
        if cfg["experiment"].get("tasks") != ["endpoint", "ordered"]:
            errors.append("task list differs from endpoint plus ordered")
        if float(cfg["particles"].get("inner_checkpoint_progress", 0)) != 0.95:
            errors.append("inner checkpoint is not 0.95")
        if float(cfg["particles"].get("outer_resampling_ess_fraction", 0)) != 0.5:
            errors.append("outer ESS threshold is not 0.5")
        errors.extend(_event_semantics(catalog, horizon))
        if not manifest.get("archive_sha256"):
            errors.append("archive SHA-256 is absent")

        referenced = [
            cfg["trajectory"]["initial_structure"],
            cfg["reference"]["topology"],
            *cfg["reference"]["design_trajectories"],
            *cfg["reference"]["held_out_trajectories"],
            cfg["reference"]["route_endpoint_structure"],
        ]
        missing_reference_assets = [
            str(_expand(item, project_root, asset_root))
            for item in referenced
            if not _expand(item, project_root, asset_root).exists()
        ]
        if missing_reference_assets:
            errors.extend(f"missing reference asset: {path}" for path in missing_reference_assets)
        audited.append(
            {
                "pdb_chain": row["pdb_chain"],
                "status": "pass" if not errors else "fail",
                "errors": errors,
                "nominal_compression": row["nominal_compression"],
                "route_status": row["route_status"],
                "held_out_status": row["H_diagnostic"]["status"],
            }
        )

    output = {
        "protocol_version": report["protocol_version"],
        "audit_scope": "static_prelaunch_no_model_run",
        "model_outcomes_inspected": False,
        "method_runs_started": False,
        "target_frozen_count": int(report["target_frozen_count"]),
        "audited_frozen_count": len(frozen),
        "static_audit_passed": not global_errors
        and all(row.get("status") == "pass" for row in audited),
        "global_errors": global_errors,
        "cases": audited,
        "pending_gpu_gates": [
            "ConfRover representation cache",
            "unsteered one-step calibration and reachability",
            "checkpoint continuation-predictivity",
            "generated whole-path structural validity",
        ],
    }
    output_path = args.output or args.report.with_name("cohort_prelaunch_audit.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(output, sort_keys=False))
    return 0 if output["static_audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
