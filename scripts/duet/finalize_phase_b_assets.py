#!/usr/bin/env python3
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
    hidden_observables,
)


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


CASES = {
    "prmt5": {
        "uniprot": "O14744",
        "construct": (294, 637),
        "loop": (435, 445),
        "start": "7KIC_A_294_637_start.pdb",
        "targets": ["6UXY_A_294_637_holo.pdb", "6UXX_A_294_637_holo.pdb"],
        "source_files": ["7KIC.cif", "6UXY.cif", "6UXX.cif"],
        "start_description": "7KIC author chain A residues 294-637; pseudo-apo PRMT5 domain",
    },
    "prmt6": {
        "uniprot": "Q96LA8",
        "construct": (53, 375),
        "loop": (155, 165),
        "start": "AF-Q96LA8-F1-model_v4_53_375_start.pdb",
        "targets": ["6W6D_A_53_375_holo.pdb"],
        "source_files": ["AF-Q96LA8-F1-model_v4.pdb", "6W6D.cif"],
        "start_description": "AlphaFold AF-Q96LA8-F1-model_v4 residues 53-375",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pdb(path: Path) -> dict[int, dict[str, Any]]:
    residues: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[21].strip() not in {"", "A"}:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            number = int(line[22:26])
            insertion = line[26].strip()
            if insertion:
                raise ValueError(f"Unsupported insertion code {number}{insertion} in {path}")
            atom_name = line[12:16].strip()
            if atom_name not in ATOM37_INDEX:
                continue
            residue = residues.setdefault(
                number, {"resname": line[17:20].strip(), "atoms": {}}
            )
            if residue["resname"] != line[17:20].strip():
                raise ValueError(f"Conflicting residue identity at {number} in {path}")
            residue["atoms"].setdefault(
                atom_name,
                np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            )
    if not residues:
        raise ValueError(f"No chain-A protein atoms found in {path}")
    return residues


def atom37(
    residues: dict[int, dict[str, Any]], first: int, last: int
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    length = last - first + 1
    coords = np.zeros((length, 37, 3), dtype=np.float32)
    mask = np.zeros((length, 37), dtype=bool)
    names: list[str | None] = [None] * length
    for number, residue in residues.items():
        if number < first or number > last:
            continue
        index = number - first
        names[index] = str(residue["resname"])
        for atom_name, xyz in residue["atoms"].items():
            atom_index = ATOM37_INDEX[atom_name]
            coords[index, atom_index] = xyz
            mask[index, atom_index] = True
    return coords, mask, names


def write_start_pdb(
    path: Path, coords: np.ndarray, mask: np.ndarray, residue_names: list[str | None]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    serial = 1
    for index, resname in enumerate(residue_names):
        if resname is None:
            raise ValueError(f"Start structure is missing residue index {index}")
        for atom_index, atom_name in enumerate(ATOM37_NAMES):
            if not mask[index, atom_index]:
                continue
            x, y, z = coords[index, atom_index]
            element = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} A{index + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
            )
            serial += 1
    lines.extend([f"TER   {serial:5d}", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _model_lines(
    coords: np.ndarray,
    mask: np.ndarray,
    residue_names: list[str | None],
    first: int,
    model: int,
) -> list[str]:
    lines = [f"MODEL     {model:4d}"]
    serial = 1
    for index, resname in enumerate(residue_names):
        if resname is None:
            continue
        for atom_index, atom_name in enumerate(ATOM37_NAMES):
            if not mask[index, atom_index]:
                continue
            x, y, z = coords[index, atom_index]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} A{first + index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom_name[0]:>2s}"
            )
            serial += 1
    lines.extend([f"TER   {serial:5d}", "ENDMDL"])
    return lines


def write_overlay(
    path: Path,
    start: PocketReference,
    target: PocketReference,
    start_names: list[str | None],
    target_names: list[str | None],
    first: int,
    core_indices: list[int],
) -> None:
    residues, atoms = _atom_selection(core_indices, BACKBONE_ATOMS)
    transform = _fit_transform(
        start.atom37_a[residues, atoms], target.atom37_a[residues, atoms]
    )
    aligned_start = _apply_transform(start.atom37_a, *transform)
    lines = _model_lines(aligned_start, start.atom37_mask, start_names, first, 1)
    lines += _model_lines(target.atom37_a, target.atom37_mask, target_names, first, 2)
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_case(data_root: Path, protein: str) -> dict[str, Any]:
    spec = CASES[protein]
    first, last = spec["construct"]
    loop_first, loop_last = spec["loop"]
    case_root = data_root / "prepared" / protein
    source_root = case_root / "source_numbering"
    raw_root = data_root / "raw"
    start_path = source_root / spec["start"]
    target_paths = [source_root / name for name in spec["targets"]]
    for path in [start_path, *target_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    start_residues = parse_pdb(start_path)
    target_residues = [parse_pdb(path) for path in target_paths]
    start_coords, start_mask, start_names = atom37(start_residues, first, last)
    target_payloads = [atom37(residues, first, last) for residues in target_residues]
    target_coords = [item[0] for item in target_payloads]
    target_masks = [item[1] for item in target_payloads]
    target_names = [item[2] for item in target_payloads]

    missing_start_residues = [first + i for i, name in enumerate(start_names) if name is None]
    if missing_start_residues:
        raise ValueError(
            f"{protein}: start has internal/terminal missing residues in requested construct: "
            f"{missing_start_residues}"
        )
    backbone_indices = [ATOM37_INDEX[name] for name in BACKBONE_ATOMS]
    missing_start_backbone = [
        first + i for i in range(len(start_names)) if not np.all(start_mask[i, backbone_indices])
    ]
    if missing_start_backbone:
        raise ValueError(f"{protein}: start missing backbone at {missing_start_backbone}")

    loop_numbers = list(range(loop_first, loop_last + 1))
    loop_indices = [number - first for number in loop_numbers]
    excluded = set(range(loop_first - 5, loop_last + 6))
    excluded.update(range(first, first + 5))
    excluded.update(range(last - 4, last + 1))
    core_indices = []
    for number in range(first, last + 1):
        index = number - first
        if number in excluded:
            continue
        all_present = np.all(start_mask[index, backbone_indices]) and all(
            np.all(mask[index, backbone_indices]) for mask in target_masks
        )
        if all_present:
            core_indices.append(index)
    if len(core_indices) < 3:
        raise ValueError(f"{protein}: insufficient fixed alignment core")
    for label, mask in [("start", start_mask), *zip(spec["targets"], target_masks)]:
        residues, atoms = _atom_selection(loop_indices, BACKBONE_ATOMS)
        if not np.all(mask[residues, atoms]):
            raise ValueError(f"{protein}: {label} is missing reward-loop N/CA/C")

    start_reference = PocketReference("start", start_coords, start_mask)
    references = [
        PocketReference(Path(name).stem.split("_")[0].lower(), coords, mask)
        for name, coords, mask in zip(spec["targets"], target_coords, target_masks)
    ]
    metric = PocketEndpointMetric(
        references=references,
        core_residue_indices=core_indices,
        loop_residue_indices=loop_indices,
    )
    d0_by_reference = metric.distances_a(start_reference)
    d0_a = min(d0_by_reference.values())

    sequence = "".join(AA3_TO_1[name] for name in start_names if name is not None)
    clean_start = case_root / f"{protein}_start_model_numbering.pdb"
    write_start_pdb(clean_start, start_coords, start_mask, start_names)
    np.savez_compressed(
        case_root / "pocket_references_atom37.npz",
        start_atom37_a=start_coords,
        start_atom37_mask=start_mask,
        reference_names=np.asarray([reference.name for reference in references]),
        reference_atom37_a=np.stack(target_coords),
        reference_atom37_mask=np.stack(target_masks),
        core_residue_indices=np.asarray(core_indices, dtype=np.int64),
        loop_residue_indices=np.asarray(loop_indices, dtype=np.int64),
    )

    mapping_path = case_root / "residue_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "uniprot_residue", "model_index_0based", "prepared_pdb_residue",
            "start_resname", *[f"{reference.name}_resname" for reference in references],
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, start_name in enumerate(start_names):
            row = {
                "uniprot_residue": first + index,
                "model_index_0based": index,
                "prepared_pdb_residue": index + 1,
                "start_resname": start_name,
            }
            for reference, names in zip(references, target_names):
                row[f"{reference.name}_resname"] = names[index]
            writer.writerow(row)

    overlays = []
    for reference, names in zip(references, target_names):
        overlay_path = case_root / f"start_{reference.name}_core_aligned_overlay.pdb"
        write_overlay(
            overlay_path,
            start_reference,
            reference,
            start_names,
            names,
            first,
            core_indices,
        )
        overlays.append(str(overlay_path))

    uniprot_to_model = {number: number - first for number in range(first, last + 1)}
    sequence_conflicts = {}
    missing_target_backbone = {}
    for reference, names, mask in zip(references, target_names, target_masks):
        sequence_conflicts[reference.name] = [
            {
                "uniprot_residue": first + index,
                "start": start_names[index],
                "target": target_name,
            }
            for index, target_name in enumerate(names)
            if target_name is not None and target_name != start_names[index]
        ]
        missing_target_backbone[reference.name] = [
            first + index
            for index in range(len(names))
            if not np.all(mask[index, backbone_indices])
        ]

    hidden_start = hidden_observables(
        start_reference,
        protein=protein,
        uniprot_to_model_index=uniprot_to_model,
        metric=metric,
    )
    hidden_targets = {
        reference.name: hidden_observables(
            reference,
            protein=protein,
            uniprot_to_model_index=uniprot_to_model,
            metric=metric,
        )
        for reference in references
    }
    source_rows = []
    for name in spec["source_files"]:
        path = raw_root / name
        source_rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() else None,
            }
        )
    manifest = {
        "schema_version": 1,
        "protein": protein,
        "uniprot_accession": spec["uniprot"],
        "construct_uniprot_residues_inclusive": [first, last],
        "model_length": len(sequence),
        "seqres": sequence,
        "mapping_rule": f"model_index_0based = UniProt residue number - {first}",
        "prepared_start": str(clean_start),
        "prepared_start_description": spec["start_description"],
        "holo_references": [reference.name for reference in references],
        "reference_npz": str(case_root / "pocket_references_atom37.npz"),
        "reward_loop_uniprot_residues_inclusive": [loop_first, loop_last],
        "reward_atoms": list(BACKBONE_ATOMS),
        "alignment_atoms": list(BACKBONE_ATOMS),
        "core_model_indices_0based": core_indices,
        "core_uniprot_residues": [first + index for index in core_indices],
        "loop_model_indices_0based": loop_indices,
        "d0_a": d0_a,
        "d0_by_reference_a": d0_by_reference,
        "status": "ready" if d0_a > 1.0 else "endpoint_not_separated",
        "potential": "max(-30, -4*(d/d0)^2)",
        "sequence_conflicts": sequence_conflicts,
        "missing_start_backbone_uniprot": missing_start_backbone,
        "missing_target_backbone_uniprot": missing_target_backbone,
        "residue_mapping_csv": str(mapping_path),
        "overlays": overlays,
        "hidden_observables_start": hidden_start,
        "hidden_observables_holo": hidden_targets,
        "sources": source_rows,
        "feature_firewall": {
            "sampling_reward": ["core-aligned loop N/CA/C RMSD"],
            "evaluation_only": list(hidden_start),
        },
    }
    (case_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/phase_b_pockets"))
    parser.add_argument("--protein", choices=tuple(CASES), action="append")
    args = parser.parse_args()
    proteins = args.protein or list(CASES)
    reports = [prepare_case(args.data_root.resolve(), protein) for protein in proteins]
    for report in reports:
        print(
            f"protein={report['protein']} status={report['status']} "
            f"length={report['model_length']} core={len(report['core_model_indices_0based'])} "
            f"d0_a={report['d0_a']:.6f}"
        )


if __name__ == "__main__":
    main()
