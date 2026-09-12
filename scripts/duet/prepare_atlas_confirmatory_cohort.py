#!/usr/bin/env python3
"""Materialize the preregistered ATLAS route-v3.1 cohort without model runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import mdtraj as md
import yaml

from confmh.duet.atlas_programs import prepare_atlas_programs

from fetch_case_study_assets import extract_archive, parallel_download


ATLAS_DOWNLOAD = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/{pdb_chain}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_files(asset_root: Path, case: dict[str, Any]) -> dict[str, Path]:
    chain = str(case["pdb_chain"])
    root = asset_root / "data" / "atlas" / chain
    start = int(case["start_frame"])
    end = start + 8 * int(case["official_stride_in_10ps"])
    d1 = str(case["d1"])
    return {
        "root": root,
        "archive": asset_root / "data" / "atlas_archives" / f"{chain}_protein.zip",
        "topology": root / f"{chain}.pdb",
        "d1": root / f"{chain}_prod_{case['d1']}_fit.xtc",
        "d2": root / f"{chain}_prod_{case['d2']}_fit.xtc",
        "h": root / f"{chain}_prod_{case['h']}_fit.xtc",
        "start": root / f"{chain}_{d1}F{start}_start.pdb",
        "end": root / f"{chain}_{d1}F{end}_end.pdb",
        "case_manifest": root / f"{case['official_case']}_manifest.json",
    }


def _extract_case_structures(
    topology: Path,
    trajectory: Path,
    start_frame: int,
    end_frame: int,
    start_output: Path,
    end_output: Path,
) -> dict[str, Any]:
    with md.open(str(trajectory)) as handle:
        frame_count = len(handle)
    if not 0 <= start_frame < end_frame < frame_count:
        raise IndexError(
            f"Official frames {start_frame}:{end_frame} outside trajectory with "
            f"{frame_count} frames"
        )
    topology_frame = md.load(str(topology))
    protein_atoms = topology_frame.topology.select("protein and chainid 0")
    if not len(protein_atoms):
        raise RuntimeError("No atoms matched protein and chainid 0")
    protein_topology = topology_frame.atom_slice(protein_atoms).topology
    fasta = protein_topology.to_fasta()
    if len(fasta) != 1:
        raise RuntimeError(f"Expected one protein chain, found {len(fasta)}")
    for index, destination in ((start_frame, start_output), (end_frame, end_output)):
        frame = md.load_frame(
            str(trajectory), index, top=str(topology), atom_indices=protein_atoms
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.save_pdb(str(destination))
    return {
        "trajectory_frame_count": int(frame_count),
        "protein_atoms": int(len(protein_atoms)),
        "protein_residues": int(protein_topology.n_residues),
        "seqres": str(fasta[0]),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frame_indexing": "zero_based",
    }


def _base_config(
    cohort: dict[str, Any],
    case: dict[str, Any],
    files: dict[str, Path],
    seqres: str,
    project_root: Path,
    asset_root: Path,
) -> dict[str, Any]:
    horizon = int(cohort["generated_horizon"])
    frame_step_ps = float(cohort["reference_frame_step_ps"])
    start_frame = int(case["start_frame"])
    end_frame = start_frame + 8 * int(case["official_stride_in_10ps"])
    output = project_root / "data" / "duet" / "protein_benchmark" / "route_v3_1" / str(
        case["pdb_chain"]
    )
    return {
        "project_root": str(project_root),
        "model": {
            "backend": "confrover",
            "repository_path": str(asset_root / "external" / "ConfRover"),
            "checkpoint": str(
                asset_root
                / "data"
                / "confrover_cache"
                / "confrover_ckpts"
                / "confrover_base_20m_v1_0.pt"
            ),
            "cache_dir": str(asset_root / "data" / "confrover_cache"),
            "sampler_mode": "sde",
            # The builder replaces this bootstrap value using the frozen
            # first-passage/target-compression rule.
            "physical_lag_in_10ps": int(cohort["model_lag_bounds_in_10ps"][0]),
            "reverse_steps": 200,
            "device": "cuda:0",
            "dtype": "float32",
            "kv_cache_type": "offloaded",
        },
        "trajectory": {
            "case_id": str(case["pdb_chain"]),
            "seqres": seqres,
            "horizon": horizon,
            "initial_structure": str(files["start"]),
            "initial_history": None,
        },
        "particles": {
            "outer_k": 16,
            "inner_m": 4,
            "inner_checkpoint_progress": 0.95,
            "outer_resampling": "systematic",
            "outer_resampling_ess_fraction": 0.5,
            "inner_resampling": "systematic",
        },
        "program": {
            "type": "ordered",
            "failure_guidance_weight": 1.0,
            "potential_floor": 1.0e-30,
        },
        "reference": {
            "topology": str(files["topology"]),
            "design_trajectories": [str(files["d1"]), str(files["d2"])],
            "design_labels": ["D1", "D2"],
            "held_out_trajectories": [str(files["h"])],
            "held_out_labels": ["H"],
            "held_out_trajectory": str(files["h"]),
            "replicate_roles": {
                "D1": str(case["d1"]),
                "D2": str(case["d2"]),
                "H": str(case["h"]),
            },
            "initial_structure": str(files["start"]),
            "route_endpoint_structure": str(files["end"]),
            "protein_selection": "protein and chainid 0",
            "stride": 10,
            "evaluation_stride": 10,
            "max_frames_per_trajectory": None,
            "n_components": 3,
            "basin_half_width": 0.25,
            "target_basin_half_width": 0.50,
        },
        "benchmark_protocol": {
            "version": "route_v3_1",
            "transition_mode": "last_exit_route_geometry",
            "route_discovery_role": "D1",
            "route_discovery_trajectory_index": 0,
            "route_discovery_original_start_frame": start_frame,
            "route_discovery_original_end_frame": end_frame,
            "route_discovery_start_time_ps": start_frame * frame_step_ps,
            "route_discovery_end_time_ps": end_frame * frame_step_ps,
            "reference_route_analysis_frames": int(
                cohort["reference_route_analysis_frames"]
            ),
            "reference_start_dwell_ps": 200.0,
            "reference_target_dwell_ps": 200.0,
            "max_start_ca_rmsd_nm": 0.30,
            "lag_selection_mode": "reference_first_passage_target_compression",
            "target_nominal_compression": float(cohort["target_nominal_compression"]),
            "model_lag_bounds_in_10ps": list(cohort["model_lag_bounds_in_10ps"]),
            "nominal_compression_factor": list(
                cohort["allowed_nominal_compression"]
            ),
            "event_timing_mode": "ordered_by_common_deadline",
            "generated_event_persistence_frames": 2,
            "generated_target_persistence_frames": 2,
            "allow_same_frame_stage_completion": False,
            "terminal_mode": "first_stable_hit_by_deadline",
            "record_time_to_success": True,
            "held_out_role": "optional_secondary_fidelity",
            "nominal_time_only": True,
            "physical_kinetics_claim": False,
        },
        "trajectory_horizon": horizon,
        "contact_selection": {
            "min_sequence_separation": 5,
            "min_change_nm": 0.15,
            "min_initial_margin_nm": 0.15,
            "min_event_separation": 2,
            "threshold_statistic": "discovery_route_midpoint",
            "reference_crossing_persistence": 2,
            "min_event_support_count": 1,
            "min_route_support_count": 1,
        },
        "output": {"directory": str(output)},
        "experiment": {
            "methods": list(cohort["methods"]),
            "decoder_population_budget": int(cohort["decoder_population_budget"]),
            "method_settings": {
                "frozen": {"outer_k": 64, "inner_m": 1},
                "outer_only": {"outer_k": 64, "inner_m": 1},
                "inner_only": {"outer_k": 1, "inner_m": 64},
                "complete_nested": {"outer_k": 16, "inner_m": 4},
                "duet": {"outer_k": 16, "inner_m": 4},
            },
            "tasks": ["endpoint", "ordered"],
            "task_count": 2,
            "seeds": list(cohort["evaluation_seeds"]),
            "output_directory": str(
                project_root
                / "outputs"
                / "duet_md"
                / "protein_benchmark"
                / "route_v3_1"
                / str(case["pdb_chain"])
            ),
            "reference_split": "D1_D2_design_H_optional_secondary",
        },
    }


def _portable(value: Any, project_root: Path, asset_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _portable(item, project_root, asset_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item, project_root, asset_root) for item in value]
    if isinstance(value, str):
        for root, variable in (
            (project_root, "${DUET_PROJECT_ROOT}"),
            (asset_root, "${DUET_ASSET_ROOT}"),
        ):
            prefix = str(root)
            if value == prefix:
                return variable
            if value.startswith(prefix + "/"):
                return variable + value[len(prefix) :]
    return value


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _representation_cached(asset_root: Path, seqres: str) -> bool:
    index = asset_root / "data" / "confrover_cache" / "folding_repr" / "seqres_to_index.csv"
    if not index.exists():
        return False
    with index.open(newline="", encoding="utf-8") as handle:
        return any(row.get("seqres") == seqres for row in csv.DictReader(handle))


def prepare_case(
    cohort: dict[str, Any],
    case: dict[str, Any],
    *,
    project_root: Path,
    asset_root: Path,
    config_root: Path,
    download: bool,
    archive_workers: int,
    hash_archive: bool,
) -> dict[str, Any]:
    files = _case_files(asset_root, case)
    if download:
        parallel_download(
            ATLAS_DOWNLOAD.format(pdb_chain=case["pdb_chain"]),
            files["archive"],
            archive_workers,
            (
                int(case["archive_size_bytes"])
                if case.get("archive_size_bytes") is not None
                else None
            ),
        )
        expected = [
            files["topology"].name,
            files["d1"].name,
            files["d2"].name,
            files["h"].name,
        ]
        if not all((files["root"] / name).exists() for name in expected):
            extract_archive(files["archive"], files["root"], [])
    required = [files[key] for key in ("topology", "d1", "d2", "h")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ATLAS assets: {missing}")

    start_frame = int(case["start_frame"])
    end_frame = start_frame + 8 * int(case["official_stride_in_10ps"])
    structure = _extract_case_structures(
        files["topology"],
        files["d1"],
        start_frame,
        end_frame,
        files["start"],
        files["end"],
    )
    if int(structure["protein_residues"]) != int(case["residues"]):
        raise RuntimeError(
            f"Residue count mismatch: {structure['protein_residues']} != {case['residues']}"
        )
    case_manifest = {
        **case,
        **structure,
        "topology": str(files["topology"]),
        "D1_trajectory": str(files["d1"]),
        "D2_trajectory": str(files["d2"]),
        "H_trajectory": str(files["h"]),
        "start_structure": str(files["start"]),
        "endpoint_structure": str(files["end"]),
        "archive": str(files["archive"]),
        "archive_sha256": _sha256(files["archive"]) if hash_archive else None,
    }
    files["case_manifest"].write_text(
        json.dumps(case_manifest, indent=2) + "\n", encoding="utf-8"
    )

    cfg = _base_config(
        cohort,
        case,
        files,
        structure["seqres"],
        project_root,
        asset_root,
    )
    prepare_config = config_root / f"{case['pdb_chain']}_prepare.yaml"
    _write_yaml(prepare_config, _portable(cfg, project_root, asset_root))
    catalog_path = prepare_atlas_programs(cfg)
    selection_path = catalog_path.parent / "selection_report.yaml"
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))

    cfg["program"]["catalog"] = str(catalog_path)
    cfg["reference"]["pca_model"] = str(
        catalog_path.parent / "design_pca_d1_d2" / "pca_cv.npz"
    )
    main_config = config_root / f"{case['pdb_chain']}_main.yaml"
    _write_yaml(main_config, _portable(cfg, project_root, asset_root))

    selected = selection["selected_contacts"]
    challenge = selection["nominal_challenge"]
    return {
        "priority": int(case["priority"]),
        "screening_group": str(case.get("screening_group", "primary")),
        "reserve_order": case.get("reserve_order"),
        "official_case": str(case["official_case"]),
        "pdb_chain": str(case["pdb_chain"]),
        "status": "task_frozen",
        "residues": int(structure["protein_residues"]),
        "seqres": str(structure["seqres"]),
        "replicate_roles": {
            "D1": str(case["d1"]),
            "D2": str(case["d2"]),
            "H": str(case["h"]),
        },
        "D1": str(case["d1"]),
        "D2": str(case["d2"]),
        "H": str(case["h"]),
        "start_frame": start_frame,
        "endpoint_frame": end_frame,
        "reference_first_passage_ps": challenge["reference_first_passage_ps"],
        "reference_transition_path_ps": challenge["reference_transition_path_ps"],
        "physical_lag_in_10ps": challenge["selected_physical_lag_in_10ps"],
        "model_duration_ps": challenge["model_duration_ps"],
        "nominal_compression": challenge["nominal_compression"],
        "route_status": selection.get("route_status"),
        "route_support_count": selection.get("route_support_count"),
        "event_A": selected[0],
        "event_B": selected[1],
        "H_diagnostic": selection["held_out_diagnostic"],
        "confrover_representation_cached": _representation_cached(
            asset_root, str(structure["seqres"])
        ),
        "prepare_config": str(prepare_config),
        "main_config": str(main_config),
        "catalog": str(catalog_path),
        "selection_report": str(selection_path),
        "case_manifest": str(files["case_manifest"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--archive-workers", type=int, default=8)
    parser.add_argument("--skip-archive-hash", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed rows in the existing cohort preparation report.",
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    cohort = yaml.safe_load(args.cohort.read_text(encoding="utf-8"))
    config_root = (
        args.config_root.expanduser().resolve()
        if args.config_root
        else project_root / "configs" / "duet" / "protein_benchmark" / "route_v3_1"
    )
    selected = set(args.only)
    primary_cases = [dict(case, screening_group="primary") for case in cohort["cases"]]
    reserve_cases = [
        dict(case, screening_group="reserve") for case in cohort.get("reserves", [])
    ]
    target_frozen_count = int(cohort.get("target_frozen_count", len(primary_cases)))
    report_root = project_root / "data" / "duet" / "protein_benchmark" / "route_v3_1"
    existing_by_chain: dict[str, dict[str, Any]] = {}
    existing_report_path = report_root / "cohort_preparation_report.yaml"
    if args.resume and existing_report_path.exists():
        existing = yaml.safe_load(existing_report_path.read_text(encoding="utf-8"))
        existing_by_chain = {
            str(row["pdb_chain"]): row for row in existing.get("cases", [])
        }
    rows = []
    unevaluated_reserves = []
    for case in primary_cases + reserve_cases:
        if selected and case["pdb_chain"] not in selected:
            continue
        if (
            not selected
            and case["screening_group"] == "reserve"
            and sum(row["status"] == "task_frozen" for row in rows)
            >= target_frozen_count
        ):
            unevaluated_reserves.append(str(case["pdb_chain"]))
            continue
        cached = existing_by_chain.get(str(case["pdb_chain"]))
        transient_download_failure = bool(
            cached
            and cached.get("status") == "unresolved"
            and "HTTPError" in str(cached.get("reason", ""))
        )
        if cached is not None and not transient_download_failure:
            cached = dict(cached)
            cached.setdefault("screening_group", str(case["screening_group"]))
            cached.setdefault("reserve_order", case.get("reserve_order"))
            rows.append(cached)
            print(json.dumps(cached, indent=2, default=str))
            continue
        try:
            row = prepare_case(
                cohort,
                case,
                project_root=project_root,
                asset_root=asset_root,
                config_root=config_root,
                download=not args.no_download,
                archive_workers=int(args.archive_workers),
                hash_archive=not args.skip_archive_hash,
            )
        except Exception as error:
            row = {
                "priority": int(case["priority"]),
                "screening_group": str(case.get("screening_group", "primary")),
                "reserve_order": case.get("reserve_order"),
                "official_case": str(case["official_case"]),
                "pdb_chain": str(case["pdb_chain"]),
                "status": "unresolved",
                "reason": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
        print(json.dumps(row, indent=2, default=str))

    portable_rows = _portable(rows, project_root, asset_root)
    report = {
        "cohort": cohort["name"],
        "protocol_version": cohort["protocol_version"],
        "model_outcomes_inspected": False,
        "method_runs_started": False,
        "target_frozen_count": target_frozen_count,
        "task_frozen_count": sum(row["status"] == "task_frozen" for row in rows),
        "unresolved_count": sum(row["status"] != "task_frozen" for row in rows),
        "unevaluated_reserves": unevaluated_reserves,
        "cases": portable_rows,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    _write_yaml(report_root / "cohort_preparation_report.yaml", report)
    (report_root / "cohort_preparation_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0 if report["task_frozen_count"] >= target_frozen_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
