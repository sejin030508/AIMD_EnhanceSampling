#!/usr/bin/env python3
"""Prepare the fixed TPS-DPS small-protein endpoints for ConfRover.

This script reproduces the zero-bias input minimization in TPS-DPS
``BaseDynamics`` once for unfolded and folded PDBs, validates topology and atom
mapping, and writes the whole-backbone endpoint reference used by DuET.
Generated frames are never passed through this preparation code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import joblib
import deeptime
import mdtraj as md
import numpy as np
import openmm as mm
from openmm import app, unit
from openmmtools.integrators import VVVRIntegrator
import pyemma.coordinates as coor
import pyemma
import yaml


ATOM37_NAMES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
ATOM37_INDEX = {name: index for index, name in enumerate(ATOM37_NAMES)}
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
BACKBONE = ("N", "CA", "C")
TEMPERATURE_K = {"chignolin": 300.0, "trpcage": 400.0, "bba": 400.0}
REQUIRED_DATA_FILES = (
    "unfolded.pdb", "folded.pdb", "tica_model.pkl", "pmf.npy", "xs.npy", "ys.npy"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = root / ".git" / ref
    if loose.exists():
        return loose.read_text(encoding="utf-8").strip()
    packed = root / ".git" / "packed-refs"
    for line in packed.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            commit, name = line.split(" ", 1)
            if name == ref:
                return commit
    raise RuntimeError(f"cannot resolve TPS-DPS HEAD ref {ref}")


def topology_signature(topology: app.Topology) -> list[tuple[int, str, str, str]]:
    rows = []
    for residue_index, residue in enumerate(topology.residues()):
        for atom in residue.atoms():
            rows.append((residue_index, residue.name, atom.name, atom.element.symbol))
    return rows


def residue_inventory(topology: app.Topology) -> list[dict[str, Any]]:
    rows = []
    for index, residue in enumerate(topology.residues()):
        atoms = [atom.name for atom in residue.atoms()]
        heavy = [
            atom.name for atom in residue.atoms()
            if atom.element is not None and atom.element.symbol.upper() != "H"
        ]
        rows.append(
            {
                "model_index_0based": index,
                "pdb_residue_id": residue.id,
                "resname": residue.name,
                "atom_names": atoms,
                "heavy_atom_names": heavy,
            }
        )
    return rows


def minimize_like_base_dynamics(
    source: Path,
    destination: Path,
    forcefield_xml: Path,
    temperature_k: float,
) -> dict[str, Any]:
    """Mirror TPS-DPS BaseDynamics setup and its one minimizeEnergy call."""

    cwd = Path.cwd()
    try:
        # The official dynamics refers to data/protein.ff14SBonlysc.xml by a
        # relative path.  Passing its absolute path changes no force-field XML.
        forcefield = app.ForceField(str(forcefield_xml), "implicit/gbn2.xml")
        pdb = app.PDBFile(str(source))
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        for atom_index in range(len(pdb.positions)):
            external_force.addParticle(atom_index, [0.0, 0.0, 0.0])
        system.addForce(external_force)
        integrator = VVVRIntegrator(
            temperature_k * unit.kelvin,
            0.001 / unit.femtoseconds,
            1.0 * unit.femtoseconds,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        initial = simulation.context.getState(getEnergy=True)
        simulation.minimizeEnergy()
        final = simulation.context.getState(getPositions=True, getEnergy=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            app.PDBFile.writeFile(
                pdb.topology,
                final.getPositions(),
                handle,
                keepIds=True,
            )
        return {
            "source": str(source),
            "destination": str(destination),
            "temperature_k": temperature_k,
            "friction_per_fs": 0.001,
            "timestep_fs": 1.0,
            "constraint_tolerance": 0.00001,
            "minimize_energy_call": "Simulation.minimizeEnergy() with OpenMM defaults",
            "zero_external_bias_force_included": True,
            "initial_potential_kj_mol": float(
                initial.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            ),
            "final_potential_kj_mol": float(
                final.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            ),
            "openmm_platform": simulation.context.getPlatform().getName(),
        }
    finally:
        del cwd


def parse_atom37(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    pdb = app.PDBFile(str(path))
    residues = list(pdb.topology.residues())
    positions_a = np.asarray(
        pdb.positions.value_in_unit(unit.angstrom), dtype=np.float32
    )
    xyz = np.zeros((len(residues), 37, 3), dtype=np.float32)
    mask = np.zeros((len(residues), 37), dtype=bool)
    names: list[str] = []
    numbers: list[int] = []
    for residue_index, residue in enumerate(residues):
        if residue.name not in AA3_TO_1:
            raise ValueError(f"unsupported residue {residue.name} in {path}")
        names.append(residue.name)
        numbers.append(int(residue.id))
        for atom in residue.atoms():
            if atom.element is not None and atom.element.symbol.upper() == "H":
                continue
            if atom.name not in ATOM37_INDEX:
                raise ValueError(f"unsupported heavy atom {residue.name}:{atom.name}")
            target = ATOM37_INDEX[atom.name]
            xyz[residue_index, target] = positions_a[atom.index]
            mask[residue_index, target] = True
    return xyz, mask, names, numbers


def kabsch_rmsd_a(moving: np.ndarray, target: np.ndarray) -> float:
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (moving - moving_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.linalg.det(left @ right_t)])
    rotation = left @ correction @ right_t
    aligned = (moving - moving_center) @ rotation + target_center
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=-1))))


def write_heavy_pdb(
    path: Path,
    xyz: np.ndarray,
    mask: np.ndarray,
    residue_names: list[str],
) -> None:
    lines: list[str] = []
    serial = 1
    for residue_index, residue_name in enumerate(residue_names):
        for atom_index, atom_name in enumerate(ATOM37_NAMES):
            if not bool(mask[residue_index, atom_index]):
                continue
            x, y, z = xyz[residue_index, atom_index]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} A"
                f"{residue_index + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {atom_name[0]:>2s}"
            )
            serial += 1
    lines.extend([f"TER   {serial:5d}", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_same_topology(start: app.PDBFile, target: app.PDBFile, molecule: str) -> None:
    """Require identical chemistry while tolerating terminal H/H1 naming aliases.

    TPS-DPS uses both PDBs independently with the same Amber force field.  Its
    Chignolin files differ only in whether the first N-terminal hydrogen is
    named ``H`` or ``H1``.  That alias must not block the heavy-atom mapping,
    but every residue, heavy atom, element, and per-residue hydrogen count is
    still checked explicitly.
    """

    start_inventory = residue_inventory(start.topology)
    target_inventory = residue_inventory(target.topology)
    if len(start_inventory) != len(target_inventory):
        raise ValueError(
            f"{molecule}: residue counts differ: {len(start_inventory)} != {len(target_inventory)}"
        )
    for index, (left, right) in enumerate(zip(start_inventory, target_inventory)):
        if left["resname"] != right["resname"]:
            raise ValueError(
                f"{molecule}: residue {index} differs: {left['resname']} != {right['resname']}"
            )
        if left["heavy_atom_names"] != right["heavy_atom_names"]:
            raise ValueError(
                f"{molecule}: residue {index} heavy atoms differ: "
                f"{left['heavy_atom_names']} != {right['heavy_atom_names']}"
            )
        left_h = [name for name in left["atom_names"] if name.startswith("H")]
        right_h = [name for name in right["atom_names"] if name.startswith("H")]
        if len(left_h) != len(right_h):
            raise ValueError(
                f"{molecule}: residue {index} hydrogen counts differ: {left_h} != {right_h}"
            )

    # Full atom counts/elements must remain equal even when a terminal alias is
    # accepted.  This prevents a missing atom from being hidden by the checks
    # above.
    start_rows = topology_signature(start.topology)
    target_rows = topology_signature(target.topology)
    if len(start_rows) != len(target_rows):
        raise ValueError(
            f"{molecule}: atom counts differ: {len(start_rows)} != {len(target_rows)}"
        )
    if [row[3] for row in start_rows] != [row[3] for row in target_rows]:
        raise ValueError(f"{molecule}: atom element ordering differs")


def tica_projection(
    molecule: str,
    official_root: Path,
    minimized_start: Path,
    minimized_target: Path,
) -> dict[str, Any]:
    original_folded = official_root / "data" / molecule / "folded.pdb"
    topology_traj = md.load(str(original_folded))
    start_traj = md.load(str(minimized_start))
    target_traj = md.load(str(minimized_target))
    if topology_traj.n_atoms != start_traj.n_atoms or topology_traj.n_atoms != target_traj.n_atoms:
        raise ValueError(f"{molecule}: minimized topology atom count changed")
    feature = coor.featurizer(str(original_folded))
    feature.add_backbone_torsions(cossin=True)
    model = joblib.load(official_root / "data" / molecule / "tica_model.pkl")
    start_values = np.asarray(
        model.transform(feature.transform(md.Trajectory(start_traj.xyz, topology_traj.topology)))
    )[0]
    target_values = np.asarray(
        model.transform(feature.transform(md.Trajectory(target_traj.xyz, topology_traj.topology)))
    )[0]
    distance = float(np.linalg.norm(start_values[:2] - target_values[:2]))
    self_distance = float(np.linalg.norm(target_values[:2] - target_values[:2]))
    return {
        "feature": "folded.pdb topology backbone torsions with cossin=True",
        "feature_dimension": int(feature.dimension()),
        "tica_output_dimension": int(len(target_values)),
        "start_first_two": start_values[:2].tolist(),
        "target_first_two": target_values[:2].tolist(),
        "target_self_distance": self_distance,
        "start_to_target_distance": distance,
        "target_hit_cutoff": 0.75,
        "start_outside_target_basin": bool(distance >= 0.75),
    }


def prepare_case(official_root: Path, output_root: Path, molecule: str) -> dict[str, Any]:
    source_root = official_root / "data" / molecule
    case_root = output_root / "prepared" / molecule
    raw_root = output_root / "raw" / molecule
    case_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_DATA_FILES if not (source_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{molecule}: missing official files {missing}")
    for name in REQUIRED_DATA_FILES:
        destination = raw_root / name
        if not destination.exists():
            shutil.copy2(source_root / name, destination)

    forcefield = official_root / "data" / "protein.ff14SBonlysc.xml"
    start_source = source_root / "unfolded.pdb"
    target_source = source_root / "folded.pdb"
    source_start_pdb = app.PDBFile(str(start_source))
    source_target_pdb = app.PDBFile(str(target_source))
    assert_same_topology(source_start_pdb, source_target_pdb, molecule)

    minimized_start = case_root / "unfolded_minimized_allatom.pdb"
    minimized_target = case_root / "folded_minimized_allatom.pdb"
    start_min = minimize_like_base_dynamics(
        start_source, minimized_start, forcefield, TEMPERATURE_K[molecule]
    )
    target_min = minimize_like_base_dynamics(
        target_source, minimized_target, forcefield, TEMPERATURE_K[molecule]
    )
    min_start_pdb = app.PDBFile(str(minimized_start))
    min_target_pdb = app.PDBFile(str(minimized_target))
    assert_same_topology(source_start_pdb, min_start_pdb, molecule)
    assert_same_topology(source_target_pdb, min_target_pdb, molecule)

    start_xyz, start_mask, start_names, start_numbers = parse_atom37(minimized_start)
    target_xyz, target_mask, target_names, target_numbers = parse_atom37(minimized_target)
    if start_names != target_names:
        raise ValueError(f"{molecule}: minimized start/target residue sequences differ")
    if start_numbers != target_numbers:
        raise ValueError(f"{molecule}: minimized start/target residue numbering differs")
    backbone_atoms = np.asarray([ATOM37_INDEX[name] for name in BACKBONE])
    if not np.all(start_mask[:, backbone_atoms]) or not np.all(target_mask[:, backbone_atoms]):
        raise ValueError(f"{molecule}: missing common N/CA/C after minimization")
    common_heavy = start_mask & target_mask
    if not np.array_equal(start_mask, target_mask):
        raise ValueError(f"{molecule}: start/target heavy-atom inventories differ")
    sequence = "".join(AA3_TO_1[name] for name in start_names)
    residue_indices = np.arange(len(start_names), dtype=np.int64)
    rows = np.repeat(residue_indices, 3)
    atoms = np.tile(backbone_atoms, len(residue_indices))
    d0_a = kabsch_rmsd_a(start_xyz[rows, atoms], target_xyz[rows, atoms])
    if not d0_a > 1.0:
        raise ValueError(f"{molecule}: endpoint not separated, d0={d0_a}")

    prepared_start = case_root / "unfolded_minimized_confrover_heavy.pdb"
    prepared_target = case_root / "folded_minimized_target_heavy.pdb"
    write_heavy_pdb(prepared_start, start_xyz, start_mask, start_names)
    write_heavy_pdb(prepared_target, target_xyz, target_mask, target_names)
    reference_npz = case_root / "whole_backbone_reference_atom37.npz"
    np.savez_compressed(
        reference_npz,
        start_atom37_a=start_xyz,
        start_atom37_mask=start_mask,
        reference_names=np.asarray(["folded"]),
        reference_atom37_a=target_xyz[None],
        reference_atom37_mask=target_mask[None],
        core_residue_indices=residue_indices,
        loop_residue_indices=residue_indices,
    )

    mapping_path = case_root / "atom_mapping.csv"
    start_inventory = residue_inventory(min_start_pdb.topology)
    target_inventory = residue_inventory(min_target_pdb.topology)
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_index_0based", "pdb_residue_id", "resname",
                "start_atom_names", "target_atom_names", "heavy_atom_names",
            ],
        )
        writer.writeheader()
        for start_row, target_row in zip(start_inventory, target_inventory):
            writer.writerow(
                {
                    "model_index_0based": start_row["model_index_0based"],
                    "pdb_residue_id": start_row["pdb_residue_id"],
                    "resname": start_row["resname"],
                    "start_atom_names": " ".join(start_row["atom_names"]),
                    "target_atom_names": " ".join(target_row["atom_names"]),
                    "heavy_atom_names": " ".join(start_row["heavy_atom_names"]),
                }
            )

    tica = tica_projection(molecule, official_root, minimized_start, minimized_target)
    pmf = np.load(source_root / "pmf.npy")
    xs = np.load(source_root / "xs.npy")
    ys = np.load(source_root / "ys.npy")
    if pmf.shape != (len(xs), len(ys)) and pmf.T.shape != (len(xs), len(ys)):
        raise ValueError(
            f"{molecule}: PMF/grid shape mismatch {pmf.shape}, {xs.shape}, {ys.shape}"
        )

    optional_reference_files = [
        path for path in (source_root / "path.gro", source_root / f"{molecule}.h5")
        if path.exists()
    ]
    manifest = {
        "schema_version": 1,
        "pilot": "small_protein_transition_pilot",
        "protein": molecule,
        "official_repository": "https://github.com/kiyoung98/tps-dps.git",
        "official_repository_commit": git_commit(official_root),
        "model_length": len(sequence),
        "seqres": sequence,
        "construct_uniprot_residues_inclusive": [1, len(sequence)],
        "mapping_rule": "model_index_0based = prepared PDB residue number - 1",
        "prepared_start": str(prepared_start),
        "prepared_target": str(prepared_target),
        "minimized_start_allatom": str(minimized_start),
        "minimized_target_allatom": str(minimized_target),
        "reference_npz": str(reference_npz),
        "reward_scope": "all residues with common N/CA/C",
        "reward_atoms": list(BACKBONE),
        "core_model_indices_0based": residue_indices.tolist(),
        "loop_model_indices_0based": residue_indices.tolist(),
        "d0_a": d0_a,
        "status": "ready" if tica["start_outside_target_basin"] else "blocked_start_in_target_basin",
        "potential": "-16*(whole_backbone_rmsd/d0)^2; no floor",
        "official_base_dynamics_minimization": {
            "force_fields": ["data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"],
            "nonbonded_method": "NoCutoff",
            "constraints": "HBonds",
            "start": start_min,
            "target": target_min,
        },
        "tica": tica,
        "tica_model": str(source_root / "tica_model.pkl"),
        "pmf": str(source_root / "pmf.npy"),
        "pmf_xs": str(source_root / "xs.npy"),
        "pmf_ys": str(source_root / "ys.npy"),
        "target_hit_definition": "Euclidean distance in first two official TICA coordinates < 0.75",
        "reference_path_coverage": None,
        "reference_path_coverage_na_reason": (
            "path.gro/h5 provenance was not established as an unbiased reference trajectory"
        ),
        "official_source_files": [
            {
                "path": str(source_root / name),
                "size_bytes": (source_root / name).stat().st_size,
                "sha256": sha256(source_root / name),
            }
            for name in REQUIRED_DATA_FILES
        ],
        "official_reference_code_and_forcefield": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (
                official_root / "src" / "utils" / "metrics.py",
                official_root / "src" / "utils" / "utils.py",
                official_root / "src" / "dynamics" / "dynamics.py",
                official_root / "src" / "dynamics" / "base.py",
                official_root / "data" / "protein.ff14SBonlysc.xml",
            )
        ],
        "optional_path_files_not_used_as_reference": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in optional_reference_files
        ],
        "prepared_file_sha256": {
            path.name: sha256(path)
            for path in (
                minimized_start, minimized_target, prepared_start,
                prepared_target, reference_npz, mapping_path,
            )
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "openmm": mm.__version__,
            "mdtraj": md.__version__,
            "pyemma": pyemma.__version__,
            "deeptime": deeptime.__version__,
        },
        "feature_firewall": {
            "sampling_reward": ["whole-protein N/CA/C RMSD to minimized folded target"],
            "evaluation_only": ["official TICA/THP", "heavy-atom RMSD", "PMF overlay", "DTW", "sampled-frame ETS"],
        },
    }
    manifest_path = case_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_configs(config_root: Path, manifest: dict[str, Any]) -> list[Path]:
    molecule = str(manifest["protein"])
    config_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for stride in (16, 128):
        config = {
            "project_root": "${DUET_PROJECT_ROOT}",
            "model": {
                "backend": "confrover",
                "repository_path": "${DUET_ASSET_ROOT}/external/ConfRover",
                "checkpoint": "${DUET_ASSET_ROOT}/data/confrover_cache/confrover_ckpts/confrover_base_20m_v1_0.pt",
                "cache_dir": "${DUET_ASSET_ROOT}/data/confrover_cache",
                "sampler_mode": "sde",
                "physical_lag_in_10ps": stride,
                "reverse_steps": 200,
                "kv_cache_type": "offloaded",
                "decoder_microbatch_size": 1,
                "pairformer_chunk_size": 32,
                "device": "cuda:0",
                "dtype": "float32",
            },
            "trajectory": {
                "case_id": f"small_protein_{molecule}_stride{stride}",
                "seqres": manifest["seqres"],
                "horizon": 32,
                "initial_structure": (
                    "${SMALL_PROTEIN_DATA_ROOT}/prepared/"
                    f"{molecule}/unfolded_minimized_confrover_heavy.pdb"
                ),
            },
            "particles": {
                "outer_k": 4,
                "inner_m": 4,
                "inner_checkpoint_progresses": [0.75, 0.90],
                "outer_resampling": "systematic",
                "outer_resampling_ess_fraction": 0.5,
                "inner_resampling": "systematic",
            },
            "program": {
                "type": "terminal",
                "reward_coefficient": 16.0,
                "reward_log_floor": None,
                "potential_floor": 1.0e-300,
            },
            "pocket": {
                "manifest": f"${{SMALL_PROTEIN_DATA_ROOT}}/prepared/{molecule}/manifest.json",
                "reference_npz": (
                    f"${{SMALL_PROTEIN_DATA_ROOT}}/prepared/{molecule}/"
                    "whole_backbone_reference_atom37.npz"
                ),
            },
            "experiment": {
                "stage": f"stride{stride}_t32",
                "methods": ["frozen", "duet"],
                "method_settings": {
                    "frozen": {"outer_k": 16, "inner_m": 1},
                    "duet": {"outer_k": 4, "inner_m": 4},
                },
                "seeds": [211, 223],
                "analysis_transitions": [8, 16, 32],
                "task_count": 4,
                "decoder_population_budget": 16,
                "output_directory": "${SMALL_PROTEIN_OUTPUT_ROOT}",
            },
        }
        path = config_root / f"{molecule}_stride{stride}_t32.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    preflight = config.copy()
    preflight["model"] = dict(config["model"])
    preflight["model"]["physical_lag_in_10ps"] = 16
    preflight["trajectory"] = dict(config["trajectory"])
    preflight["trajectory"]["case_id"] = f"small_protein_{molecule}_preflight_stride16"
    preflight["trajectory"]["horizon"] = 1
    preflight["experiment"] = dict(config["experiment"])
    preflight["experiment"]["stage"] = "_preflight_stride16_t1"
    preflight["experiment"]["seeds"] = [199]
    preflight["experiment"]["analysis_transitions"] = [1]
    preflight["experiment"]["task_count"] = 2
    preflight_path = config_root / f"{molecule}_preflight_stride16_t1.yaml"
    preflight_path.write_text(
        yaml.safe_dump(preflight, sort_keys=False), encoding="utf-8"
    )
    paths.append(preflight_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    args = parser.parse_args()
    official_root = args.official_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for molecule in ("chignolin", "trpcage", "bba"):
        try:
            manifest = prepare_case(official_root, output_root, molecule)
            config_paths = write_configs(args.config_root.resolve(), manifest)
            results.append(
                {
                    "protein": molecule,
                    "status": manifest["status"],
                    "d0_a": manifest["d0_a"],
                    "tica_start_to_target": manifest["tica"]["start_to_target_distance"],
                    "configs": [str(path) for path in config_paths],
                }
            )
        except Exception as error:
            results.append(
                {"protein": molecule, "status": "blocked", "error": repr(error)}
            )
    report = {
        "official_repository_commit": git_commit(official_root),
        "cases": results,
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not all(row["status"] == "ready" for row in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
