#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import mdtraj as md
import numpy as np
import yaml


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASH": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HSD": "H", "HSE": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def atom_index(topology: md.Topology, residue_number: int, atom_name: str) -> int:
    matches = [
        atom.index
        for atom in topology.atoms
        if atom.residue.resSeq == residue_number and atom.name == atom_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {residue_number}:{atom_name} atom, found {len(matches)}"
        )
    return matches[0]


def circular_difference(values: np.ndarray, center: float) -> np.ndarray:
    return (np.asarray(values) - center + 180.0) % 360.0 - 180.0


def circular_center(values: np.ndarray) -> float:
    radians = np.deg2rad(values)
    return float(np.degrees(np.angle(np.mean(np.exp(1j * radians)))) % 360.0)


def first_true(values: np.ndarray) -> int:
    indices = np.flatnonzero(values)
    return int(indices[0]) if len(indices) else -1


def normalize_trace(values: np.ndarray, points: int = 33) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.stack(
        [np.interp(target, source, values[:, dimension]) for dimension in range(values.shape[1])],
        axis=1,
    )


def trajectory_features(trajectory: md.Trajectory) -> dict[str, np.ndarray]:
    topology = trajectory.topology
    asp_quad = np.asarray(
        [[
            atom_index(topology, 380, "CB"), atom_index(topology, 380, "CA"),
            atom_index(topology, 381, "CA"), atom_index(topology, 381, "CG"),
        ]]
    )
    phe_quad = np.asarray(
        [[
            atom_index(topology, 380, "CB"), atom_index(topology, 380, "CA"),
            atom_index(topology, 382, "CA"), atom_index(topology, 382, "CG"),
        ]]
    )
    distance_pairs = np.asarray(
        [
            [atom_index(topology, 381, "OD2"), atom_index(topology, 299, "O")],
            [atom_index(topology, 271, "NZ"), atom_index(topology, 286, "CD")],
        ]
    )
    return {
        "asp_angle_deg": np.degrees(md.compute_dihedrals(trajectory, asp_quad)[:, 0]) % 360.0,
        "phe_angle_deg": np.degrees(md.compute_dihedrals(trajectory, phe_quad)[:, 0]) % 360.0,
        "asp_val_distance_a": md.compute_distances(trajectory, distance_pairs[:, :][0:1])[:, 0]
        * 10.0,
        "lys_glu_distance_a": md.compute_distances(trajectory, distance_pairs[:, :][1:2])[:, 0]
        * 10.0,
    }


def route_trace(
    features: dict[str, np.ndarray],
    asp_scale: float,
    phe_scale: float,
    contact_cutoff: float,
    salt_scale: float,
) -> np.ndarray:
    asp = np.unwrap(np.deg2rad(features["asp_angle_deg"]))
    phe = np.unwrap(np.deg2rad(features["phe_angle_deg"]))
    values = np.column_stack(
        [
            np.rad2deg(asp - asp[0]) / asp_scale,
            np.rad2deg(phe - phe[0]) / phe_scale,
            features["asp_val_distance_a"] / contact_cutoff,
            features["lys_glu_distance_a"] / salt_scale,
        ]
    )
    return normalize_trace(values)


def write_heavy_start(trajectory: md.Trajectory, path: Path) -> None:
    # The source protonates Asp381 as residue name ASH.  MDTraj's ``protein``
    # selector excludes ASH, so use all heavy atoms; this topology contains only
    # the 287-residue kinase construct and no solvent/ligand residues.
    heavy = trajectory.atom_slice(trajectory.topology.select("not element H"))
    temporary = path.with_suffix(".raw.pdb")
    heavy[0].save_pdb(str(temporary))
    lines = []
    residue_numbers = {}
    for line in temporary.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM  "):
            continue
        residue_key = (line[21], line[22:26], line[26])
        if residue_key not in residue_numbers:
            residue_numbers[residue_key] = len(residue_numbers) + 1
        line = line[:22] + f"{residue_numbers[residue_key]:4d}" + line[26:]
        if line[17:20] == "ASH":
            line = line[:17] + "ASP" + line[20:]
        lines.append(line)
    if len(residue_numbers) != len(list(heavy.topology.residues)):
        raise ValueError(
            f"PDB rewrite lost residues: {len(residue_numbers)} != {heavy.topology.n_residues}"
        )
    lines.extend([f"TER   {len(lines) + 1:5d}", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.unlink()


def hash_spec(spec: dict) -> str:
    payload = yaml.safe_dump(spec, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-root",
        type=Path,
        default=Path("data/duet/case_studies/abl1_dfg_flip"),
    )
    args = parser.parse_args()
    case_root = args.case_root.resolve()
    raw_root = (
        case_root
        / "reference_raw/GMdSilva-abl1_dfgflip_westpa_rawdata-8ae0b5d/selected_trajs"
    )
    topology_path = raw_root / "abl1_wild_type_top.pdb"
    paths = {
        "concerted": md.load(str(raw_root / "wt_concerted_flip.trr"), top=str(topology_path)),
        "staggered": md.load(str(raw_root / "wt_staggered_flip.trr"), top=str(topology_path)),
    }
    topology = paths["concerted"].topology
    sequence = "".join(AA3_TO_1.get(residue.name, "X") for residue in topology.residues)
    if "X" in sequence or len(sequence) != 287:
        raise ValueError(f"Unexpected ABL1 source sequence ({len(sequence)} aa): {sequence}")
    residue_mapping = {}
    for label, number in {
        "Tyr253": 253, "Lys271": 271, "Glu286": 286, "Val299": 299,
        "Thr315": 315, "Ala380": 380, "Asp381": 381, "Phe382": 382,
    }.items():
        matches = [residue for residue in topology.residues if residue.resSeq == number]
        if len(matches) != 1:
            raise ValueError(f"Non-unique canonical mapping for {label}: {matches}")
        residue_mapping[label] = int(matches[0].index)

    features = {name: trajectory_features(path) for name, path in paths.items()}
    endpoint_tail_frames = 10
    endpoint_tail = np.concatenate(
        [
            np.column_stack(
                [
                    row["asp_angle_deg"][-endpoint_tail_frames:],
                    row["phe_angle_deg"][-endpoint_tail_frames:],
                ]
            )
            for row in features.values()
        ]
    )
    asp_center = circular_center(endpoint_tail[:, 0])
    phe_center = circular_center(endpoint_tail[:, 1])
    asp_scale = max(15.0, float(np.quantile(np.abs(circular_difference(endpoint_tail[:, 0], asp_center)), 0.95)))
    phe_scale = max(15.0, float(np.quantile(np.abs(circular_difference(endpoint_tail[:, 1], phe_center)), 0.95)))
    endpoint_distance = np.sqrt(
        0.5
        * (
            np.square(circular_difference(endpoint_tail[:, 0], asp_center) / asp_scale)
            + np.square(circular_difference(endpoint_tail[:, 1], phe_center) / phe_scale)
        )
    )
    endpoint_threshold = float(np.quantile(endpoint_distance, 0.95))
    asp_tolerance = asp_scale
    phe_tolerance = phe_scale
    contact_cutoff = 4.0
    salt_scale = 4.0
    event_frames = {}
    traces = {}
    for name, row in features.items():
        asp_done = np.abs(circular_difference(row["asp_angle_deg"], asp_center)) <= asp_tolerance
        phe_done = np.abs(circular_difference(row["phe_angle_deg"], phe_center)) <= phe_tolerance
        contact = row["asp_val_distance_a"] <= contact_cutoff
        event_frames[name] = np.asarray(
            [first_true(asp_done), first_true(phe_done), first_true(contact)], dtype=np.int64
        )
        traces[name] = route_trace(row, asp_scale, phe_scale, contact_cutoff, salt_scale)

    prepared = case_root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    write_heavy_start(paths["concerted"], prepared / "abl1_wt_dfg_in_start.pdb")
    np.savez_compressed(
        prepared / "reference_features.npz",
        concerted_route_trace=traces["concerted"],
        staggered_route_trace=traces["staggered"],
        concerted_event_frames=event_frames["concerted"],
        staggered_event_frames=event_frames["staggered"],
        concerted_asp_angle_deg=features["concerted"]["asp_angle_deg"],
        concerted_phe_angle_deg=features["concerted"]["phe_angle_deg"],
        staggered_asp_angle_deg=features["staggered"]["asp_angle_deg"],
        staggered_phe_angle_deg=features["staggered"]["phe_angle_deg"],
    )

    asp_atoms = [
        [residue_mapping["Ala380"], "CB"], [residue_mapping["Ala380"], "CA"],
        [residue_mapping["Asp381"], "CA"], [residue_mapping["Asp381"], "CG"],
    ]
    phe_atoms = [
        [residue_mapping["Ala380"], "CB"], [residue_mapping["Ala380"], "CA"],
        [residue_mapping["Phe382"], "CA"], [residue_mapping["Phe382"], "CG"],
    ]
    relative_case = "data/duet/case_studies/abl1_dfg_flip"
    spec = {
        "schema_version": 1,
        "case_id": "abl1_dfg_flip",
        "scientific_claim": "mechanism-blind path recovery under a frozen surrogate prior",
        "source": {
            "paper": "https://arxiv.org/abs/2405.14968",
            "data": "https://doi.org/10.5281/zenodo.11194787",
            "starting_structure": "PDB 6XR6; prepared WE topology is used for sequence compatibility",
            "topology": f"{relative_case}/reference_raw/GMdSilva-abl1_dfgflip_westpa_rawdata-8ae0b5d/selected_trajs/abl1_wild_type_top.pdb",
            "trajectories": [
                f"{relative_case}/reference_raw/GMdSilva-abl1_dfgflip_westpa_rawdata-8ae0b5d/selected_trajs/wt_concerted_flip.trr",
                f"{relative_case}/reference_raw/GMdSilva-abl1_dfgflip_westpa_rawdata-8ae0b5d/selected_trajs/wt_staggered_flip.trr",
            ],
        },
        "mapping": {
            "chain": 0,
            "sequence_length": len(sequence),
            "seqres": sequence,
            "canonical_residue_numbers": "WE topology resSeq is canonical ABL1 numbering",
            "generated_residue_indices": residue_mapping,
            "pdb_6xr6_offset": "canonical = PDB chain A residue number - 19",
        },
        "endpoint_definition": {
            "name": "dfg_out_endpoint_distance",
            "source_subset": "final 10 frames of each of the two selected WT transition paths",
            "asp_center_degrees": asp_center,
            "phe_center_degrees": phe_center,
            "asp_scale_degrees": asp_scale,
            "phe_scale_degrees": phe_scale,
            "success_distance_threshold": endpoint_threshold,
            "threshold_rule": "95th percentile of endpoint-tail composite distance; generated data unseen",
        },
        "hidden_event_definitions": {
            "asp_completion_tolerance_degrees": asp_tolerance,
            "phe_completion_tolerance_degrees": phe_tolerance,
            "asp_val_contact_cutoff_a": contact_cutoff,
            "salt_bridge_reference_scale_a": salt_scale,
            "route_classifier": "nearest DTW to concerted/staggered reference hidden-CV traces",
        },
        "feature_firewall": {
            "reward_features": ["dfg_out_endpoint_distance"],
            "hidden_evaluation_features": [
                "dfg_inter_contact", "asp_phe_completion_order", "route_family",
                "lys271_glu286_distance",
            ],
            "selection_rule": "hidden features are evaluation-only and never loaded by PrefixPotential",
        },
        "reference_split": {
            "endpoint_design": "endpoint-only tails from both selected WT paths",
            "mechanism_evaluation": "two public selected paths, unweighted descriptive",
            "limitation": (
                "The public archive contains one WT WESTPA HDF5 run and two selected paths, "
                "not independent WT replicate files; no route-population or kinetic claim is permitted."
            ),
        },
        "production": {
            "horizon": 24,
            "methods": ["frozen", "complete_nested", "duet"],
            "frozen_population": 64,
            "outer_k": 16,
            "inner_m": 4,
            "duet_checkpoints": [0.85, 0.95],
            "complete_nested_checkpoint": 0.95,
            "seeds": [20261001, 20261002, 20261003, 20261004, 20261005],
            "reverse_steps": 200,
            "sampler": "sde",
        },
        "reference_features_npz": f"{relative_case}/prepared/reference_features.npz",
        "claim_boundary": "No kinetics, MFPT, free energy, or unbiased-dynamics claim.",
    }
    spec["config_hash_sha256"] = hash_spec(spec)
    with (case_root / "benchmark_spec.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)

    catalog = {
        "observables": {
            "dfg_out_endpoint_distance": {
                "kind": "composite_dihedral_distance",
                "components": [
                    {"atoms": asp_atoms, "center_degrees": asp_center, "scale_degrees": asp_scale},
                    {"atoms": phe_atoms, "center_degrees": phe_center, "scale_degrees": phe_scale},
                ],
            }
        },
        "tasks": {
            "endpoint": {
                "type": "terminal",
                "events": [
                    {
                        "name": "dfg_out",
                        "observable": "dfg_out_endpoint_distance",
                        "target_interval": [0.0, endpoint_threshold],
                        "physical_window": [24, 24],
                    }
                ],
            }
        },
    }
    with (prepared / "endpoint_program.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(catalog, handle, sort_keys=False)

    report = f"""# ABL1 DFG-flip reference mapping

- Sampling sequence: `{sequence}` ({len(sequence)} residues)
- Sampling topology: canonical residue numbers 227–513; chain 0.
- PDB 6XR6/6XR7 chain A uses coordinates 248–534 and maps as canonical = PDB − 19.
- The two sequences overlap exactly for 285 residues; the WE construct adds canonical 227–228
  and omits canonical 514–515 relative to the NMR construct.
- Generated residue indices: `{residue_mapping}`.
- Endpoint reward uses only the final Asp381/Phe382 pseudo-dihedral state.
- DFG-inter contact, event order, salt-bridge profile, and route identity are evaluation-only.
- Endpoint centers: Asp={asp_center:.3f}°, Phe={phe_center:.3f}°.
- Endpoint scales: Asp={asp_scale:.3f}°, Phe={phe_scale:.3f}°; success distance ≤ {endpoint_threshold:.4f}.
- Reference route event frames: concerted={event_frames['concerted'].tolist()},
  staggered={event_frames['staggered'].tolist()} (Asp complete, Phe complete, DFG-inter contact).

The public archive does not expose independent WT replicate HDF5 files. Route comparisons are
therefore descriptive and unweighted; rate, free-energy, and route-population claims are blocked.
"""
    (case_root / "reference_mapping_report.md").write_text(report, encoding="utf-8")

    with (case_root / "run_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "method", "seed", "gpu", "status", "output_directory"],
        )
        writer.writeheader()
        for seed in spec["production"]["seeds"]:
            for method in spec["production"]["methods"]:
                writer.writerow(
                    {
                        "case": "abl1_dfg_flip",
                        "method": method,
                        "seed": seed,
                        "gpu": "assigned_at_launch",
                        "status": "planned",
                        "output_directory": (
                            f"outputs/duet_md/case_studies/abl1_dfg_flip/endpoint/{method}/seed_{seed}"
                        ),
                    }
                )

    project_root = case_root.parents[3]
    config_directory = project_root / "configs/duet/case_studies"
    config_directory.mkdir(parents=True, exist_ok=True)
    config = {
        "project_root": "${DUET_PROJECT_ROOT}",
        "model": {
            "backend": "confrover",
            "repository_path": "${DUET_ASSET_ROOT}/external/ConfRover",
            "checkpoint": (
                "${DUET_ASSET_ROOT}/data/confrover_cache/confrover_ckpts/"
                "confrover_base_20m_v1_0.pt"
            ),
            "cache_dir": "${DUET_PROJECT_ROOT}/data/confrover_cache",
            "sampler_mode": "sde",
            "physical_lag_in_10ps": 256,
            "reverse_steps": 200,
            "kv_cache_type": "offloaded",
            "device": "cuda:0",
            "dtype": "float32",
        },
        "trajectory": {
            "case_id": "abl1_dfg_flip",
            "seqres": sequence,
            "horizon": 24,
            "initial_structure": f"{relative_case}/prepared/abl1_wt_dfg_in_start.pdb",
            "initial_history": None,
        },
        "particles": {
            "outer_k": 16,
            "inner_m": 4,
            "inner_checkpoint_progresses": [0.85, 0.95],
            "inner_checkpoint_progress": 0.95,
            "outer_resampling": "systematic",
            "outer_resampling_ess_fraction": 0.5,
            "inner_resampling": "systematic",
        },
        "program": {
            "type": "terminal",
            "catalog": f"{relative_case}/prepared/endpoint_program.yaml",
            "lambda_program": 0.5,
            "distance_weight": 1.0,
            "remaining_event_weight": 0.0,
            "deadline_weight": 0.0,
            "failure_penalty": 4.0,
            "terminal_failure_penalty": 4.0,
            "failure_guidance_weight": 1.0,
            "distance_scale": 1.0,
            "potential_floor": 1.0e-30,
        },
        "case_study": {"benchmark_spec": f"{relative_case}/benchmark_spec.yaml"},
        "preflight": {
            "gate": {
                "ca_adjacent_quality_threshold_a": 4.5,
                "ca_adjacent_hard_threshold_a": 5.5,
                "ca_adjacent_hard_tolerance_a": 1.0e-3,
            }
        },
        "experiment": {
            "methods": ["frozen", "complete_nested", "duet"],
            "method_settings": {
                "frozen": {"outer_k": 64, "inner_m": 1},
                "complete_nested": {"outer_k": 16, "inner_m": 4},
                "duet": {"outer_k": 16, "inner_m": 4},
            },
            "tasks": ["endpoint"],
            "task_count": 1,
            "seeds": [20261001, 20261002, 20261003, 20261004, 20261005],
            "decoder_population_budget": 64,
            "cost_budget": {
                "decoder_population_per_step": 64,
                "reverse_steps": 200,
                "horizon": 24,
            },
            "output_directory": "outputs/duet_md/case_studies/abl1_dfg_flip",
            "reference_split": "endpoint tails for definition; two selected paths descriptive",
            "evaluation_features": [
                "valid_endpoint_success", "dfg_inter", "event_order", "route_family",
                "genealogy", "diversity", "continuity",
            ],
        },
    }
    with (config_directory / "abl1_dfg_flip_production.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(case_root / "benchmark_spec.yaml")


if __name__ == "__main__":
    main()
