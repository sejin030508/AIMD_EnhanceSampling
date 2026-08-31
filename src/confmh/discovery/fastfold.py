from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

from confmh.discovery.structure import read_pdb, sequence, sequence_alignment_pairs
from confmh.discovery.bioemu import _model_config


FASTFOLD_SYSTEMS = {
    "protein_g": {"pdb_id": "1MI0", "chain": "A", "residue_range": [5, 61], "class": "alpha_beta"},
    "ww_domain": {"pdb_id": "2F21", "chain": "A", "residue_range": [6, 39], "class": "beta"},
    "alpha3d": {"pdb_id": "2A3D", "chain": "A", "class": "alpha"},
}


def _pair_indices(n_residues: int, minimum_sequence_separation: int = 3) -> np.ndarray:
    return np.asarray(
        [
            (i, j)
            for i in range(n_residues)
            for j in range(i + minimum_sequence_separation, n_residues)
        ],
        dtype=np.int32,
    )


def _distance_features(xyz_a: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz_a[:, pairs[:, 0], :] - xyz_a[:, pairs[:, 1], :], axis=-1)


def prepare_fastfold_reference(
    *,
    topology_pdb: str | Path,
    trajectories: Iterable[str | Path],
    output_path: str | Path,
    stride: int = 1,
    lag_frames: int = 10,
    n_components: int = 3,
    n_clusters: int = 20,
    minimum_state_population: float = 0.01,
    radius_quantile: float = 0.95,
    temperature_k: float = 300.0,
    free_energy_threshold_kcal_mol: float = 4.0,
    grid_bins: int = 30,
    max_frames: int | None = None,
) -> Path:
    """Fit a reference-only TICA/state model from public long MD.

    The resulting states are labels for coverage, never a free-energy estimate
    from generated occupancy.
    """
    import mdtraj as md

    topology_pdb = Path(topology_pdb).expanduser().resolve()
    trajectories = [Path(path).expanduser().resolve() for path in trajectories]
    output_path = Path(output_path).expanduser().resolve()
    topology = md.load(str(topology_pdb))
    ca_indices = topology.top.select("protein and name CA")
    if len(ca_indices) < 10:
        raise ValueError(f"Only {len(ca_indices)} C-alpha atoms found in {topology_pdb}")
    pairs = _pair_indices(len(ca_indices))
    feature_blocks = []
    remaining = max_frames
    for trajectory_path in trajectories:
        trajectory = md.load(str(trajectory_path), top=str(topology_pdb), stride=int(stride))
        if remaining is not None:
            trajectory = trajectory[:remaining]
            remaining -= len(trajectory)
        ca_xyz_a = trajectory.xyz[:, ca_indices, :] * 10.0
        feature_blocks.append(_distance_features(ca_xyz_a, pairs))
        if remaining is not None and remaining <= 0:
            break
    if not feature_blocks:
        raise ValueError("No public reference trajectory frames were loaded")
    features = np.concatenate(feature_blocks, axis=0).astype(np.float64)
    if len(features) <= lag_frames + 2:
        raise ValueError(f"Need more than {lag_frames + 2} reference frames, got {len(features)}")
    mean = features.mean(axis=0)
    centered = features - mean
    x0, xt = centered[:-lag_frames], centered[lag_frames:]
    covariance = 0.5 * ((x0.T @ x0) + (xt.T @ xt)) / len(x0)
    time_covariance = 0.5 * ((x0.T @ xt) + (xt.T @ x0)) / len(x0)
    regularization = max(1e-8, 1e-6 * float(np.trace(covariance)) / covariance.shape[0])
    eigenvalues, eigenvectors = eigh(time_covariance, covariance + regularization * np.eye(covariance.shape[0]))
    order = np.argsort(np.abs(eigenvalues))[::-1][: int(n_components)]
    components = eigenvectors[:, order]
    projections = centered @ components
    kmeans = KMeans(n_clusters=int(n_clusters), n_init=20, random_state=20260901)
    labels = kmeans.fit_predict(projections)
    populations = np.bincount(labels, minlength=n_clusters).astype(np.float64) / len(labels)
    distances = np.linalg.norm(projections - kmeans.cluster_centers_[labels], axis=1)
    radii = np.asarray(
        [
            np.quantile(distances[labels == state], radius_quantile)
            if np.any(labels == state)
            else 0.0
            for state in range(n_clusters)
        ]
    )
    low_energy = populations >= float(minimum_state_population)
    if projections.shape[1] < 2:
        raise ValueError("Fast-fold coverage requires at least two TICA components")
    grid_range = [
        [float(np.quantile(projections[:, axis], 0.005)), float(np.quantile(projections[:, axis], 0.995))]
        for axis in range(2)
    ]
    grid_counts, grid_x_edges, grid_y_edges = np.histogram2d(
        projections[:, 0], projections[:, 1], bins=int(grid_bins), range=grid_range
    )
    kbt_kcal_mol = 0.00198720425864083 * float(temperature_k)
    with np.errstate(divide="ignore", invalid="ignore"):
        grid_free_energy = -kbt_kcal_mol * np.log(grid_counts / grid_counts.max())
    low_energy_grid = (grid_counts > 0) & (grid_free_energy <= float(free_energy_threshold_kcal_mol))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pair_indices=pairs,
        feature_mean=mean,
        tica_components=components,
        tica_eigenvalues=eigenvalues[order],
        reference_projections=projections,
        reference_labels=labels,
        state_centers=kmeans.cluster_centers_,
        state_radii=radii,
        state_populations=populations,
        low_energy_state_mask=low_energy,
        tica_grid_counts=grid_counts,
        tica_grid_x_edges=grid_x_edges,
        tica_grid_y_edges=grid_y_edges,
        grid_free_energy_kcal_mol=grid_free_energy,
        low_energy_grid_mask=low_energy_grid,
        free_energy_threshold_kcal_mol=np.asarray(free_energy_threshold_kcal_mol),
        temperature_k=np.asarray(temperature_k),
        lag_frames=np.asarray(lag_frames),
        stride=np.asarray(stride),
        radius_quantile=np.asarray(radius_quantile),
        minimum_state_population=np.asarray(minimum_state_population),
    )
    metadata = {
        "topology_pdb": str(topology_pdb),
        "trajectories": [str(path) for path in trajectories],
        "reference_frames": len(features),
        "n_residues": len(ca_indices),
        "n_features": features.shape[1],
        "n_tica_components": components.shape[1],
        "n_clusters": int(n_clusters),
        "n_low_energy_states": int(low_energy.sum()),
        "n_low_energy_tica_regions": int(low_energy_grid.sum()),
        "free_energy_threshold_kcal_mol": float(free_energy_threshold_kcal_mol),
        "generated_occupancy_used_for_free_energy": False,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path


def project_pdb_to_tica(pdb: str | Path, start_pdb: str | Path, model_path: str | Path) -> np.ndarray:
    model = np.load(Path(model_path).expanduser().resolve())
    candidate, start = read_pdb(pdb), read_pdb(start_pdb)
    mapping = {start_i: candidate_i for candidate_i, start_i in sequence_alignment_pairs(sequence(candidate), sequence(start))}
    pair_indices = np.asarray(model["pair_indices"], dtype=np.int64)
    distances = []
    for start_i, start_j in pair_indices:
        if int(start_i) not in mapping or int(start_j) not in mapping:
            raise ValueError("Generated PDB does not cover the reference TICA residue set")
        atom_i = candidate[mapping[int(start_i)]].atoms["CA"]
        atom_j = candidate[mapping[int(start_j)]].atoms["CA"]
        distances.append(np.linalg.norm(atom_i - atom_j))
    feature = np.asarray(distances)
    return (feature - model["feature_mean"]) @ model["tica_components"]


def analyze_fastfold_run(
    *,
    run_dir: str | Path,
    start_pdb: str | Path,
    reference_model: str | Path,
    raw_final_coverage: float | None = None,
) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    model = np.load(Path(reference_model).expanduser().resolve())
    centers = np.asarray(model["state_centers"])
    radii = np.asarray(model["state_radii"])
    low_energy_grid = np.asarray(model["low_energy_grid_mask"], dtype=bool)
    grid_x_edges = np.asarray(model["tica_grid_x_edges"])
    grid_y_edges = np.asarray(model["tica_grid_y_edges"])
    low_energy_region_ids = set(np.flatnonzero(low_energy_grid).tolist())
    rows = [json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines() if line.strip()]
    covered: set[int] = set()
    coverage_curve = []
    analyzed = []
    for row in rows:
        projection = project_pdb_to_tica(row["candidate_pdb"], start_pdb, reference_model)
        state_distances = np.linalg.norm(centers - projection[None, :], axis=1)
        state = int(np.argmin(state_distances))
        in_reference_region = bool(state_distances[state] <= radii[state])
        grid_x = int(np.searchsorted(grid_x_edges, projection[0], side="right") - 1)
        grid_y = int(np.searchsorted(grid_y_edges, projection[1], side="right") - 1)
        in_grid = 0 <= grid_x < low_energy_grid.shape[0] and 0 <= grid_y < low_energy_grid.shape[1]
        region = grid_x * low_energy_grid.shape[1] + grid_y if in_grid else -1
        valid_low_energy = bool(
            row["strict_valid"] and in_grid and low_energy_grid[grid_x, grid_y]
        )
        if valid_low_energy:
            covered.add(region)
        coverage = len(covered) / max(len(low_energy_region_ids), 1)
        coverage_curve.append(coverage)
        analyzed.append(
            {
                "model_call": row["model_call"],
                "state": state,
                "state_distance": float(state_distances[state]),
                "in_reference_region": in_reference_region,
                "valid_low_energy": valid_low_energy,
                "low_energy_region": region,
                "coverage": coverage,
                **{f"tica_{index + 1}": float(value) for index, value in enumerate(projection)},
            }
        )
    csv_path = run_dir / "fastfold_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analyzed[0]) if analyzed else ["model_call"])
        writer.writeheader()
        writer.writerows(analyzed)
    target = float(raw_final_coverage) if raw_final_coverage is not None else (coverage_curve[-1] if coverage_curve else 0.0)
    ttc = next((index + 1 for index, value in enumerate(coverage_curve) if value >= target), None)
    summary = {
        "model_calls": len(rows),
        "valid_low_energy_state_coverage": coverage_curve[-1] if coverage_curve else 0.0,
        "maximum_coverage": max(coverage_curve, default=0.0),
        "coverage_auc": float(np.mean(coverage_curve)) if coverage_curve else 0.0,
        "target_raw_final_coverage": target,
        "time_to_coverage_model_calls": ttc,
        "states_hit": sorted(covered),
        "n_reference_low_energy_states": len(low_energy_region_ids),
        "reference_state_definition": "reference-MD TICA grid regions with F <= 4 kcal/mol",
        "generated_occupancy_used_for_free_energy": False,
        "time_unit": "model_calls (not ns)",
        "mh_used": False,
    }
    path = run_dir / "fastfold_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def fastfold_data_status(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    systems = {}
    for name, metadata in FASTFOLD_SYSTEMS.items():
        system_dir = root / name
        topology = system_dir / "topology.pdb"
        trajectories = sorted(system_dir.glob("reference/*")) if (system_dir / "reference").exists() else []
        trajectories = [path for path in trajectories if path.suffix.lower() in {".xtc", ".dcd", ".trr", ".nc"}]
        systems[name] = {
            **metadata,
            "topology_pdb": str(topology),
            "topology_present": topology.exists(),
            "reference_trajectories": [str(path) for path in trajectories],
            "ready": topology.exists() and bool(trajectories),
        }
    return systems


def make_fastfold_configs(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    total_model_calls: int = 160,
    candidates_per_step: int = 4,
    seeds: list[int] | None = None,
) -> list[Path]:
    """Generate deployable and oracle-CV configurations for ready fast folders."""
    import yaml

    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    seeds = [20260911, 20260912, 20260913] if seeds is None else seeds
    written: list[Path] = []
    for system_name, metadata in FASTFOLD_SYSTEMS.items():
        system_dir = data_root / system_name
        start_pdb = system_dir / "topology.pdb"
        reference_model = system_dir / "reference_tica.npz"
        if not start_pdb.exists() or not reference_model.exists():
            continue
        residues = read_pdb(start_pdb)
        reference = np.load(reference_model)
        oracle_centers = np.quantile(reference["reference_projections"][:, 0], np.linspace(0.05, 0.95, 7)).tolist()
        conditions = {
            "raw": ("raw", "start_rmsd", [0, 1, 2, 3, 4, 6, 8]),
            "deployable_guided": ("biased", "start_rmsd", [0, 1, 2, 3, 4, 6, 8]),
            "oracle_tica_guided": ("biased", "tica", oracle_centers),
        }
        for backend in ("confrover", "proar"):
            model_cfg = _model_config(backend)
            if backend == "proar":
                model_cfg["input_data_dir"] = "data/proar_fastfold"
                model_cfg["cache_dir"] = "data/proar_fastfold_cache"
            for condition_name, (controller, cv_mode, centers) in conditions.items():
                for seed_index, seed in enumerate(seeds):
                    path = output_dir / backend / condition_name / system_name / f"seed_{seed_index:02d}.yaml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv_cfg: dict[str, Any] = {"mode": cv_mode, "mask": "all_ca"}
                    if cv_mode == "tica":
                        cv_cfg.update(
                            {
                                "tica_model": str(reference_model.relative_to(project_root)),
                                "tica_component": 0,
                            }
                        )
                    cfg = {
                        "project_root": os.path.relpath(project_root, path.parent),
                        "benchmark": {
                            "name": "desres_fastfold_coverage",
                            "system": system_name,
                            "structural_class": metadata["class"],
                            "reference_model": str(reference_model.relative_to(project_root)),
                            # Discovery runner requires this key but never reads it in target-blind mode.
                            "target_pdb": str(start_pdb.relative_to(project_root)),
                        },
                        "system": {
                            "case_id": f"fastfold_{system_name}",
                            "seqres": sequence(residues),
                            "start_pdb": str(start_pdb.relative_to(project_root)),
                        },
                        "model": model_cfg,
                        "protocol": {"name": "target_blind", "cv": cv_cfg},
                        "controller": {
                            "kind": controller,
                            "centers": [float(value) for value in centers],
                            "kappa": 1.0,
                            "selection_temperature": 1.0,
                            "description": "dimensionless controller; no MH and no generated-FES",
                        },
                        "validity": {
                            "safety_filter": False,
                            "evaluation_only": True,
                            "reference_pdb": str(start_pdb.relative_to(project_root)),
                        },
                        "run": {
                            "seed": int(seed),
                            "total_model_calls": int(total_model_calls),
                            "candidates_per_step": int(candidates_per_step),
                            "checkpoint_interval": 1,
                            "cleanup_model_outputs": True,
                            "output_dir": str(
                                Path("outputs/discovery/fastfold")
                                / backend
                                / condition_name
                                / system_name
                                / f"seed_{seed_index:02d}"
                            ),
                        },
                    }
                    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                    written.append(path)
    index = {
        "configs": [str(path) for path in written],
        "conditions": ["raw", "deployable_guided", "oracle_tica_guided"],
        "systems": list(FASTFOLD_SYSTEMS),
        "models": ["confrover", "proar"],
        "model_calls_per_config": total_model_calls,
        "mh_used": False,
        "generated_occupancy_used_for_free_energy": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return written
