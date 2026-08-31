from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from confmh.discovery.structure import read_pdb, sequence


DEFAULT_CASES = {
    "domainmotion": ["P69441", "P0DP23", "P03047"],
    "crypticpocket": ["P69441", "P0DP23", "P26281"],
}


def _write_model_pdb(source: Path, destination: Path) -> None:
    """Write a first-model, single-chain PDB with contiguous 1..L residue IDs."""
    lines: list[str] = []
    residue_map: dict[tuple[str, str, str], int] = {}
    saw_model = False
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            if saw_model:
                break
            saw_model = True
            continue
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[16:17] not in {" ", "A"}:
            continue
        key = (line[21:22], line[22:26], line[26:27])
        if key not in residue_map:
            residue_map[key] = len(residue_map) + 1
        new_resseq = residue_map[key]
        lines.append(line[:16] + " " + line[17:21] + "A" + f"{new_resseq:4d}" + " " + line[27:])
    if not lines:
        raise ValueError(f"No model atoms found in {source}")
    destination.write_text("\n".join(lines + ["TER", "END"]) + "\n", encoding="utf-8")


def _reference_path(root: Path, benchmark: str, case: str, pdb_id: str) -> Path:
    directory = root / benchmark / "reference" / case
    candidates = [directory / f"{pdb_id}.pdb", directory / f"{pdb_id.upper()}.pdb", directory / f"{pdb_id.lower()}.pdb"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Reference {benchmark}/{case}/{pdb_id} was not found")


def prepare_bioemu_assets(
    *,
    bioemu_repo: str | Path,
    output_dir: str | Path,
    proar_data_dir: str | Path,
    include_reverse: bool = False,
) -> Path:
    """Copy a preregistered 3+3 BioEmu screen and create a portable manifest."""
    repo = Path(bioemu_repo).expanduser().resolve()
    asset_root = repo / "bioemu_benchmarks" / "assets" / "multiconf_benchmark_0.1"
    output_dir = Path(output_dir).expanduser().resolve()
    proar_data_dir = Path(proar_data_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    proar_data_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for benchmark, selected_cases in DEFAULT_CASES.items():
        references_csv = asset_root / benchmark / "references.csv"
        rows = {row["test_case"]: row for row in csv.DictReader(references_csv.open(encoding="utf-8"))}
        for case in selected_cases:
            if case not in rows:
                raise KeyError(f"Preregistered case {case} is absent from {references_csv}")
            row = rows[case]
            directions = [(row["pdbidchain_i"], row["pdbidchain_j"])]
            if include_reverse:
                directions.append((row["pdbidchain_j"], row["pdbidchain_i"]))
            for start_id, target_id in directions:
                source_start = _reference_path(asset_root, benchmark, case, start_id)
                source_target = _reference_path(asset_root, benchmark, case, target_id)
                case_id = f"{benchmark}_{case}_{start_id}_to_{target_id}".replace("-", "_")
                case_dir = output_dir / benchmark / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                evaluation_start_path = case_dir / "start.pdb"
                start_path, target_path = case_dir / "model_start.pdb", case_dir / "target.pdb"
                shutil.copy2(source_start, evaluation_start_path)
                _write_model_pdb(source_start, start_path)
                shutil.copy2(source_target, target_path)
                local_info_source = asset_root / benchmark / "local_residinfo" / f"{case}.json"
                local_info_path = None
                if local_info_source.exists():
                    local_info_path = case_dir / "local_residinfo.json"
                    shutil.copy2(local_info_source, local_info_path)
                proar_case = proar_data_dir / case_id
                proar_case.mkdir(parents=True, exist_ok=True)
                shutil.copy2(start_path, proar_case / "init.pdb")
                start_residues, target_residues = read_pdb(start_path), read_pdb(target_path)
                manifest.append(
                    {
                        "benchmark": benchmark,
                        "test_case": case,
                        "case_id": case_id,
                        "start_id": start_id,
                        "target_id": target_id,
                        "start_pdb": str(start_path),
                        "evaluation_start_pdb": str(evaluation_start_path),
                        "target_pdb": str(target_path),
                        "local_residinfo": None if local_info_path is None else str(local_info_path),
                        "seqres": sequence(start_residues),
                        "start_residues": len(start_residues),
                        "target_residues": len(target_residues),
                    }
                )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _model_config(backend: str) -> dict[str, Any]:
    if backend == "confrover":
        return {
            "backend": "confrover",
            "repo": "external/ConfRover",
            "name": "ConfRover-base-20M-v1.0",
            "device": "cuda:0",
            "stride_in_10ps": 256,
            "diffusion_steps": 200,
            "cache_dir": "data/confrover_cache",
            "kv_cache_type": "offloaded",
        }
    if backend == "proar":
        return {
            "backend": "proar",
            "repo": "external/ProAR",
            "device": "cuda:0",
            "input_data_dir": "data/proar_discovery",
            "forecaster_checkpoint": "data/proar/checkpoints/forecaster.ckpt",
            "interpolator_checkpoint": "data/proar/checkpoints/interpolator.ckpt",
            "interpolator_config": "data/proar/checkpoints/interpolator_config.yaml",
            "cache_dir": "data/proar_discovery_cache",
            "horizon": 6,
            "sampling_type": "naive",
            "refine_intermediate_predictions": True,
        }
    raise ValueError(f"Unsupported backend {backend}")


def make_bioemu_configs(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    seeds: list[int] | None = None,
    total_model_calls: int = 128,
    candidates_per_step: int = 4,
) -> list[Path]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = [20260901, 20260902, 20260903] if seeds is None else seeds
    written: list[Path] = []
    conditions = {
        "raw": {
            "protocol": "target_blind",
            "controller": "raw",
            "cv_mode": "start_rmsd",
            "centers": [0, 1, 2, 3, 4, 6, 8],
            "cv_mask": "all_ca",
        },
        "blind_guided": {
            "protocol": "target_blind",
            "controller": "biased",
            "cv_mode": "start_rmsd",
            "centers": [0, 1, 2, 3, 4, 6, 8],
            "cv_mask": "all_ca",
        },
        "blind_guided_ss": {
            "protocol": "target_blind",
            "controller": "biased",
            "cv_mode": "start_rmsd",
            "centers": [0, 1, 2, 3, 4, 6, 8],
            "cv_mask": "secondary_structure",
        },
        "aware_guided": {
            "protocol": "target_aware",
            "controller": "biased",
            "cv_mode": "target_rmsd",
            "centers": [8, 6, 4, 3, 2, 1, 0],
            "cv_mask": "all_ca",
        },
    }
    for row in rows:
        start_rel = str(Path(row["start_pdb"]).resolve().relative_to(project_root))
        evaluation_start_rel = str(
            Path(row["evaluation_start_pdb"]).resolve().relative_to(project_root)
        )
        target_rel = str(Path(row["target_pdb"]).resolve().relative_to(project_root))
        local_rel = (
            None
            if row["local_residinfo"] is None
            else str(Path(row["local_residinfo"]).resolve().relative_to(project_root))
        )
        for backend in ("confrover", "proar"):
            for condition_name, condition in conditions.items():
                for seed_index, seed in enumerate(seeds):
                    output_rel = Path("outputs/discovery/bioemu") / backend / condition_name / row["case_id"] / f"seed_{seed_index:02d}"
                    cfg = {
                        "project_root": ".",
                        "benchmark": {
                            "name": "bioemu_alternate_state",
                            "category": row["benchmark"],
                            "test_case": row["test_case"],
                            "start_id": row["start_id"],
                            "target_id": row["target_id"],
                            "target_pdb": target_rel,
                            "evaluation_start_pdb": evaluation_start_rel,
                            "local_residinfo": local_rel,
                        },
                        "system": {
                            "case_id": row["case_id"],
                            "seqres": row["seqres"],
                            "start_pdb": start_rel,
                        },
                        "model": _model_config(backend),
                        "protocol": {
                            "name": condition["protocol"],
                            "cv": {"mode": condition["cv_mode"], "mask": condition["cv_mask"]},
                        },
                        "controller": {
                            "kind": condition["controller"],
                            "centers": condition["centers"],
                            "kappa": 1.0,
                            "selection_temperature": 1.0,
                            "description": "dimensionless softmin controller; not MH or Boltzmann reweighting",
                        },
                        "validity": {
                            "safety_filter": False,
                            "evaluation_only": True,
                            "reference_pdb": start_rel,
                        },
                        "run": {
                            "seed": int(seed),
                            "total_model_calls": int(total_model_calls),
                            "candidates_per_step": int(candidates_per_step),
                            "checkpoint_interval": 1,
                            "cleanup_model_outputs": True,
                            "output_dir": str(output_rel),
                        },
                    }
                    path = output_dir / backend / condition_name / row["case_id"] / f"seed_{seed_index:02d}.yaml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cfg["project_root"] = os.path.relpath(project_root, path.parent)
                    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                    written.append(path)
    index = {
        "configs": [str(path) for path in written],
        "n_configs": len(written),
        "preregistered_cases": DEFAULT_CASES,
        "conditions": list(conditions),
        "models": ["confrover", "proar"],
        "seeds": seeds,
        "model_calls_per_config": total_model_calls,
        "mh_used": False,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return written
