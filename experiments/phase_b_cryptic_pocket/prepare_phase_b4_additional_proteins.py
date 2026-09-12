#!/usr/bin/env python3
"""Prepare/audit the two additional Phase-B4 cryptic-pocket proteins.

This script deliberately refuses to create a runnable reference NPZ when the
requested holo reward segment lacks backbone atoms.  It never masks or shrinks
the reward segment implicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from confmh.duet.observables import ATOM37_INDEX, ATOM37_NAMES
from confmh.duet.phase_b_pockets import (
    BACKBONE_ATOMS,
    PocketEndpointMetric,
    PocketReference,
    _apply_transform,
    _atom_selection,
    _fit_transform,
)


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

CASES = {
    "smarca2": {
        "uniprot": "P51531", "construct": (705, 955), "loop": (852, 860),
        "start_raw": "AF-P51531-F1-model_v6.pdb", "target_raw": "6EG3.pdb",
        "target_name": "6eg3", "ligand": "J7G",
        "description": "AlphaFold DB v6 P51531 residues 705-955 to 6EG3 chain A",
    },
    "pi3ka": {
        "uniprot": "P42336", "construct": (765, 1051), "loop": (931, 957),
        "start_raw": "AF-P42336-F1-model_v6.pdb", "target_raw": "8TSB.pdb",
        "target_name": "8tsb", "ligand": "UIW",
        "description": "AlphaFold DB v6 P42336 residues 765-1051 to 8TSB chain A",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_chain_a(path: Path) -> dict[int, dict[str, Any]]:
    residues: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM  ") or line[21].strip() not in {"", "A"}:
            continue
        if line[16].strip() not in {"", "A"}:
            continue
        number = int(line[22:26])
        if line[26].strip():
            raise ValueError(f"Insertion code at {number}{line[26]} in {path}")
        atom = line[12:16].strip()
        if atom not in ATOM37_INDEX:
            continue
        residue = residues.setdefault(number, {"resname": line[17:20].strip(), "atoms": {}})
        residue["atoms"].setdefault(
            atom,
            np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
        )
    if not residues:
        raise ValueError(f"No chain-A protein atoms in {path}")
    return residues


def atom37(residues: dict[int, dict[str, Any]], first: int, last: int):
    length = last - first + 1
    xyz = np.zeros((length, 37, 3), dtype=np.float32)
    mask = np.zeros((length, 37), dtype=bool)
    names: list[str | None] = [None] * length
    for number, residue in residues.items():
        if first <= number <= last:
            i = number - first
            names[i] = residue["resname"]
            for atom, point in residue["atoms"].items():
                j = ATOM37_INDEX[atom]
                xyz[i, j] = point
                mask[i, j] = True
    return xyz, mask, names


def write_atom37_pdb(path: Path, xyz: np.ndarray, mask: np.ndarray,
                     names: list[str | None], first_number: int) -> None:
    lines, serial = [], 1
    for i, resname in enumerate(names):
        if resname is None:
            continue
        for j, atom in enumerate(ATOM37_NAMES):
            if not mask[i, j]:
                continue
            x, y, z = xyz[i, j]
            lines.append(
                f"ATOM  {serial:5d} {atom:^4s} {resname:>3s} A{first_number + i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom[0]:>2s}"
            )
            serial += 1
    lines += [f"TER   {serial:5d}", "END"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_ligand(raw: Path, out: Path, resname: str) -> tuple[np.ndarray, list[str]]:
    rows, xyz, elements = [], [], []
    for line in raw.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == resname:
            rows.append(line)
            element = line[76:78].strip() or line[12:16].strip()[0]
            if element.upper() != "H":
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                elements.append(element)
    if not rows:
        raise ValueError(f"Ligand {resname} absent from {raw}")
    out.write_text("\n".join(rows + ["END"]) + "\n", encoding="utf-8")
    return np.asarray(xyz, dtype=np.float32), elements


def prepare(data_root: Path, protein: str) -> dict[str, Any]:
    spec = CASES[protein]
    first, last = spec["construct"]
    loop_first, loop_last = spec["loop"]
    raw_root = data_root / "raw_b4"
    case_root = data_root / "prepared" / protein
    source_root = case_root / "source_numbering"
    case_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    start_raw = raw_root / spec["start_raw"]
    target_raw = raw_root / spec["target_raw"]
    start_res = parse_chain_a(start_raw)
    target_res = parse_chain_a(target_raw)
    start_xyz, start_mask, start_names = atom37(start_res, first, last)
    target_xyz, target_mask, target_names = atom37(target_res, first, last)
    bb = [ATOM37_INDEX[x] for x in BACKBONE_ATOMS]
    missing_start_residues = [first + i for i, x in enumerate(start_names) if x is None]
    missing_start_bb = [first + i for i in range(len(start_names)) if not np.all(start_mask[i, bb])]
    missing_target_residues = [first + i for i, x in enumerate(target_names) if x is None]
    missing_target_bb = [first + i for i in range(len(target_names)) if not np.all(target_mask[i, bb])]
    reward_indices = list(range(loop_first - first, loop_last - first + 1))
    missing_reward_bb = [first + i for i in reward_indices if not np.all(target_mask[i, bb])]
    conflicts = [
        {"uniprot_residue": first + i, "start": a, "target": b}
        for i, (a, b) in enumerate(zip(start_names, target_names))
        if a is not None and b is not None and a != b
    ]
    if missing_start_residues or missing_start_bb:
        raise ValueError(f"{protein}: incomplete AlphaFold start: residues={missing_start_residues}, bb={missing_start_bb}")
    if conflicts:
        raise ValueError(f"{protein}: start/target sequence conflicts: {conflicts[:10]}")

    source_start = source_root / f"{Path(spec['start_raw']).stem}_{first}_{last}_start.pdb"
    source_target = source_root / f"{Path(spec['target_raw']).stem}_A_{first}_{last}_holo.pdb"
    write_atom37_pdb(source_start, start_xyz, start_mask, start_names, first)
    write_atom37_pdb(source_target, target_xyz, target_mask, target_names, first)
    clean_start = case_root / f"{protein}_start_model_numbering.pdb"
    write_atom37_pdb(clean_start, start_xyz, start_mask, start_names, 1)

    ligand_pdb = case_root / f"{spec['target_name']}_{spec['ligand']}_ligand.pdb"
    ligand_xyz, ligand_elements = extract_ligand(target_raw, ligand_pdb, spec["ligand"])
    np.savez_compressed(
        case_root / "holo_ligand_heavy_atoms.npz",
        coordinates_a=ligand_xyz,
        elements=np.asarray(ligand_elements),
        ligand_resname=np.asarray(spec["ligand"]),
        target_name=np.asarray(spec["target_name"]),
    )

    sequence = "".join(AA3_TO_1[x] for x in start_names if x is not None)
    mapping = case_root / "residue_mapping.csv"
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "uniprot_residue", "model_index_0based", "prepared_pdb_residue",
            "start_resname", f"{spec['target_name']}_resname",
        ])
        writer.writeheader()
        for i, (a, b) in enumerate(zip(start_names, target_names)):
            writer.writerow({
                "uniprot_residue": first + i, "model_index_0based": i,
                "prepared_pdb_residue": i + 1, "start_resname": a,
                f"{spec['target_name']}_resname": b,
            })

    manifest: dict[str, Any] = {
        "schema_version": 2, "protein": protein, "uniprot_accession": spec["uniprot"],
        "construct_uniprot_residues_inclusive": [first, last], "model_length": len(sequence),
        "seqres": sequence, "mapping_rule": f"model_index_0based = UniProt residue number - {first}",
        "prepared_start": str(clean_start), "prepared_start_description": spec["description"],
        "holo_references": [spec["target_name"]],
        "reward_loop_uniprot_residues_inclusive": [loop_first, loop_last],
        "reward_atoms": list(BACKBONE_ATOMS), "alignment_atoms": list(BACKBONE_ATOMS),
        "missing_start_backbone_uniprot": missing_start_bb,
        "missing_target_residues_uniprot": {spec["target_name"]: missing_target_residues},
        "missing_target_backbone_uniprot": {spec["target_name"]: missing_target_bb},
        "missing_reward_backbone_uniprot": {spec["target_name"]: missing_reward_bb},
        "sequence_conflicts": {spec["target_name"]: conflicts},
        "residue_mapping_csv": str(mapping),
        "ligand": {"resname": spec["ligand"], "pdb": str(ligand_pdb),
                   "heavy_atom_npz": str(case_root / "holo_ligand_heavy_atoms.npz"),
                   "heavy_atom_count": len(ligand_xyz)},
        "sources": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (start_raw, target_raw)
        ],
        "feature_firewall": {
            "sampling_reward": ["core-aligned requested loop N/CA/C RMSD"],
            "evaluation_only": ["holo ligand spatial clearance", "protein-ligand clashes", "contact changes"],
        },
    }

    if missing_reward_bb:
        manifest.update({
            "status": "hold_missing_target_reward_backbone",
            "hold_reason": "The specified holo target cannot define the full requested reward RMSD; no implicit masking was applied.",
        })
    else:
        excluded = set(range(loop_first - 5, loop_last + 6))
        excluded.update(range(first, first + 5))
        excluded.update(range(last - 4, last + 1))
        core = [
            number - first for number in range(first, last + 1)
            if number not in excluded
            and np.all(start_mask[number - first, bb])
            and np.all(target_mask[number - first, bb])
        ]
        start_ref = PocketReference("start", start_xyz, start_mask)
        target_ref = PocketReference(spec["target_name"], target_xyz, target_mask)
        metric = PocketEndpointMetric(
            references=[target_ref],
            core_residue_indices=core,
            loop_residue_indices=reward_indices,
        )
        d0_by_ref = metric.distances_a(start_ref)
        d0_a = min(d0_by_ref.values())
        np.savez_compressed(
            case_root / "pocket_references_atom37.npz",
            start_atom37_a=start_xyz, start_atom37_mask=start_mask,
            reference_names=np.asarray([spec["target_name"]]),
            reference_atom37_a=np.stack([target_xyz]),
            reference_atom37_mask=np.stack([target_mask]),
            core_residue_indices=np.asarray(core, dtype=np.int64),
            loop_residue_indices=np.asarray(reward_indices, dtype=np.int64),
        )
        residues, atoms = _atom_selection(core, BACKBONE_ATOMS)
        transform = _fit_transform(start_xyz[residues, atoms], target_xyz[residues, atoms])
        aligned_start = _apply_transform(start_xyz, *transform)
        overlay = case_root / f"start_{spec['target_name']}_core_aligned_overlay.pdb"
        tmp_start = case_root / ".aligned_start.tmp.pdb"
        tmp_target = case_root / ".target.tmp.pdb"
        write_atom37_pdb(tmp_start, aligned_start, start_mask, start_names, first)
        write_atom37_pdb(tmp_target, target_xyz, target_mask, target_names, first)
        start_lines = [x for x in tmp_start.read_text().splitlines() if x.startswith("ATOM")]
        target_lines = [x for x in tmp_target.read_text().splitlines() if x.startswith("ATOM")]
        overlay.write_text("MODEL        1\n" + "\n".join(start_lines) +
                           "\nENDMDL\nMODEL        2\n" + "\n".join(target_lines) +
                           "\nENDMDL\nEND\n", encoding="utf-8")
        tmp_start.unlink(); tmp_target.unlink()
        manifest.update({
            "status": "ready" if d0_a > 1.0 else "endpoint_not_separated",
            "reference_npz": str(case_root / "pocket_references_atom37.npz"),
            "core_model_indices_0based": core,
            "core_uniprot_residues": [first + i for i in core],
            "loop_model_indices_0based": reward_indices,
            "d0_a": d0_a, "d0_by_reference_a": d0_by_ref,
            "potential": "-a*(d/d0)^2; no reward floor; a in {16,32}",
            "overlays": [str(overlay)],
        })

    (case_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_pi3ka_observed_mask(data_root: Path) -> dict[str, Any]:
    """Create the approved observed-residue RMSD variant without altering the HOLD task."""
    spec = CASES["pi3ka"]
    first, last = spec["construct"]
    moving_first, moving_last = spec["loop"]
    raw_root = data_root / "raw_b4"
    case_root = data_root / "prepared" / "pi3ka_observed_mask"
    source_root = case_root / "source_numbering"
    case_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    start_raw = raw_root / spec["start_raw"]
    target_raw = raw_root / spec["target_raw"]
    start_res = parse_chain_a(start_raw)
    target_res = parse_chain_a(target_raw)
    start_xyz, start_mask, start_names = atom37(start_res, first, last)
    target_xyz, target_mask, target_names = atom37(target_res, first, last)
    bb = [ATOM37_INDEX[name] for name in BACKBONE_ATOMS]

    if any(name is None for name in start_names):
        raise ValueError("pi3ka_observed_mask: AlphaFold start has missing residues")
    missing_start_bb = [
        first + i for i in range(len(start_names)) if not np.all(start_mask[i, bb])
    ]
    if missing_start_bb:
        raise ValueError(
            f"pi3ka_observed_mask: AlphaFold start missing backbone {missing_start_bb}"
        )
    conflicts = [
        {"uniprot_residue": first + i, "start": a, "target": b}
        for i, (a, b) in enumerate(zip(start_names, target_names))
        if a is not None and b is not None and a != b
    ]
    if conflicts:
        raise ValueError(f"pi3ka_observed_mask: sequence conflicts {conflicts[:10]}")

    original_numbers = list(range(moving_first, moving_last + 1))
    observed_numbers = [
        number
        for number in original_numbers
        if np.all(target_mask[number - first, bb])
    ]
    excluded_numbers = [
        number for number in original_numbers if number not in observed_numbers
    ]
    expected_observed = list(range(931, 943)) + list(range(951, 958))
    expected_excluded = list(range(943, 951))
    if observed_numbers != expected_observed or excluded_numbers != expected_excluded:
        raise RuntimeError(
            "PI3Kalpha atom inventory changed: "
            f"observed={observed_numbers}, excluded={excluded_numbers}"
        )

    # Preserve the established fixed-core rule: the full original moving
    # segment (and the existing five-residue guard band) stays out of alignment.
    alignment_excluded = set(range(moving_first - 5, moving_last + 6))
    alignment_excluded.update(range(first, first + 5))
    alignment_excluded.update(range(last - 4, last + 1))
    core = [
        number - first
        for number in range(first, last + 1)
        if number not in alignment_excluded
        and np.all(start_mask[number - first, bb])
        and np.all(target_mask[number - first, bb])
    ]
    reward_indices = [number - first for number in observed_numbers]
    start_ref = PocketReference("start", start_xyz, start_mask)
    target_ref = PocketReference(spec["target_name"], target_xyz, target_mask)
    metric = PocketEndpointMetric(
        references=[target_ref],
        core_residue_indices=core,
        loop_residue_indices=reward_indices,
    )
    d0_by_ref = metric.distances_a(start_ref)
    d0_a = min(d0_by_ref.values())

    source_start = source_root / f"{Path(spec['start_raw']).stem}_{first}_{last}_start.pdb"
    source_target = source_root / f"{Path(spec['target_raw']).stem}_A_{first}_{last}_holo.pdb"
    write_atom37_pdb(source_start, start_xyz, start_mask, start_names, first)
    write_atom37_pdb(source_target, target_xyz, target_mask, target_names, first)
    clean_start = case_root / "pi3ka_observed_mask_start_model_numbering.pdb"
    write_atom37_pdb(clean_start, start_xyz, start_mask, start_names, 1)
    reference_npz = case_root / "pocket_references_atom37.npz"
    np.savez_compressed(
        reference_npz,
        start_atom37_a=start_xyz,
        start_atom37_mask=start_mask,
        reference_names=np.asarray([spec["target_name"]]),
        reference_atom37_a=np.stack([target_xyz]),
        reference_atom37_mask=np.stack([target_mask]),
        core_residue_indices=np.asarray(core, dtype=np.int64),
        loop_residue_indices=np.asarray(reward_indices, dtype=np.int64),
    )

    ligand_pdb = case_root / f"{spec['target_name']}_{spec['ligand']}_ligand.pdb"
    ligand_xyz, ligand_elements = extract_ligand(target_raw, ligand_pdb, spec["ligand"])
    ligand_npz = case_root / "holo_ligand_heavy_atoms.npz"
    np.savez_compressed(
        ligand_npz,
        coordinates_a=ligand_xyz,
        elements=np.asarray(ligand_elements),
        ligand_resname=np.asarray(spec["ligand"]),
        target_name=np.asarray(spec["target_name"]),
    )

    sequence = "".join(AA3_TO_1[name] for name in start_names if name is not None)
    mapping = case_root / "residue_mapping.csv"
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "uniprot_residue", "model_index_0based", "prepared_pdb_residue",
            "start_resname", f"{spec['target_name']}_resname",
            "in_original_moving_region", "in_observed_reward_mask",
            "excluded_from_reward_missing_holo_backbone",
        ])
        writer.writeheader()
        for i, (start_name, target_name) in enumerate(zip(start_names, target_names)):
            number = first + i
            writer.writerow({
                "uniprot_residue": number,
                "model_index_0based": i,
                "prepared_pdb_residue": i + 1,
                "start_resname": start_name,
                f"{spec['target_name']}_resname": target_name,
                "in_original_moving_region": int(number in original_numbers),
                "in_observed_reward_mask": int(number in observed_numbers),
                "excluded_from_reward_missing_holo_backbone": int(number in excluded_numbers),
            })

    residues, atoms = _atom_selection(core, BACKBONE_ATOMS)
    transform = _fit_transform(start_xyz[residues, atoms], target_xyz[residues, atoms])
    aligned_start = _apply_transform(start_xyz, *transform)
    overlay = case_root / "start_8tsb_core_aligned_overlay.pdb"
    tmp_start = case_root / ".aligned_start.tmp.pdb"
    tmp_target = case_root / ".target.tmp.pdb"
    write_atom37_pdb(tmp_start, aligned_start, start_mask, start_names, first)
    write_atom37_pdb(tmp_target, target_xyz, target_mask, target_names, first)
    start_lines = [line for line in tmp_start.read_text().splitlines() if line.startswith("ATOM")]
    target_lines = [line for line in tmp_target.read_text().splitlines() if line.startswith("ATOM")]
    overlay.write_text(
        "MODEL        1\n" + "\n".join(start_lines)
        + "\nENDMDL\nMODEL        2\n" + "\n".join(target_lines)
        + "\nENDMDL\nEND\n",
        encoding="utf-8",
    )
    tmp_start.unlink()
    tmp_target.unlink()

    manifest = {
        "schema_version": 3,
        "protocol_amendment": "PI3Kalpha observed-residue RMSD variant",
        "protein": "pi3ka_observed_mask",
        "parent_blocked_task": "pi3ka",
        "uniprot_accession": spec["uniprot"],
        "construct_uniprot_residues_inclusive": [first, last],
        "model_length": len(sequence),
        "seqres": sequence,
        "mapping_rule": f"model_index_0based = UniProt residue number - {first}",
        "prepared_start": str(clean_start),
        "prepared_start_description": (
            "AlphaFold DB v6 P42336 residues 765-1051; identical prepared input "
            "for a=16 and a=32"
        ),
        "holo_references": [spec["target_name"]],
        "reference_npz": str(reference_npz),
        "metric_name": "observed_loop_rmsd",
        "metric_interpretation": (
            "Holo-like RMSD of the experimentally observed subset only; not full-loop "
            "restoration or pocket-opening success"
        ),
        "original_moving_region_uniprot_inclusive": [moving_first, moving_last],
        "original_moving_region_uniprot_residues": original_numbers,
        "observed_reward_mask_uniprot_residues": observed_numbers,
        "excluded_reward_residues_missing_holo_backbone": excluded_numbers,
        "observed_reward_coverage": {
            "observed_residue_count": len(observed_numbers),
            "original_residue_count": len(original_numbers),
            "fraction": len(observed_numbers) / len(original_numbers),
        },
        "reward_loop_uniprot_residues_inclusive": [moving_first, moving_last],
        "reward_atoms": list(BACKBONE_ATOMS),
        "alignment_atoms": list(BACKBONE_ATOMS),
        "alignment_excludes_full_original_moving_region": True,
        "alignment_excluded_uniprot_residues": sorted(alignment_excluded),
        "core_model_indices_0based": core,
        "core_uniprot_residues": [first + i for i in core],
        "loop_model_indices_0based": reward_indices,
        "d0_a": d0_a,
        "d0_by_reference_a": d0_by_ref,
        "holo_like_threshold_a": 1.5,
        "holo_like_persistence_final_frames": 2,
        "status": "ready" if d0_a > 1.5 else "endpoint_not_separated",
        "potential": "-a*(observed_loop_rmsd/d0)^2; no reward floor; a in {16,32}",
        "sequence_conflicts": {spec["target_name"]: conflicts},
        "residue_mapping_csv": str(mapping),
        "overlays": [str(overlay)],
        "ligand": {
            "resname": spec["ligand"],
            "pdb": str(ligand_pdb),
            "heavy_atom_npz": str(ligand_npz),
            "heavy_atom_count": len(ligand_xyz),
            "generated_full_construct_used_for_space_evaluation": True,
        },
        "sources": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (start_raw, target_raw)
        ],
        "feature_firewall": {
            "sampling_reward": ["core-aligned observed-loop N/CA/C RMSD"],
            "evaluation_only": [
                "holo ligand spatial clearance using the full generated construct",
                "protein-ligand distance cutoff counts",
            ],
        },
    }
    (case_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protein", choices=tuple(CASES), action="append")
    parser.add_argument("--pi3ka-observed-mask", action="store_true")
    args = parser.parse_args()
    proteins = args.protein if args.protein is not None else (
        [] if args.pi3ka_observed_mask else list(CASES)
    )
    for protein in proteins:
        report = prepare(args.data_root.resolve(), protein)
        print(json.dumps({
            "protein": protein, "status": report["status"],
            "model_length": report["model_length"],
            "d0_a": report.get("d0_a"),
            "missing_reward_backbone": report["missing_reward_backbone_uniprot"],
            "ligand_heavy_atoms": report["ligand"]["heavy_atom_count"],
        }, sort_keys=True))
    if args.pi3ka_observed_mask:
        report = prepare_pi3ka_observed_mask(args.data_root.resolve())
        print(json.dumps({
            "protein": report["protein"],
            "status": report["status"],
            "d0_a": report["d0_a"],
            "observed_reward_mask": report["observed_reward_mask_uniprot_residues"],
            "excluded_reward_residues": report[
                "excluded_reward_residues_missing_holo_backbone"
            ],
            "coverage": report["observed_reward_coverage"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
