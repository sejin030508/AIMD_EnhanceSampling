#!/usr/bin/env python3
"""Export fixed, favorable corrected-audit examples as raw/relaxed PDB pairs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from openmm import app, unit

from run_restrained_relaxation_audit import RestrainedRelaxer


# These choices were ranked from the corrected audit before re-export:
# small proteins: retained raw+relaxed TICA hit, no relaxed gross C-N/clash,
# then lowest backbone RMSD; PRMT6: no relaxed C-N/clash, then lowest max
# displacement, bond and angle residuals.  They are examples, not a rate.
SMALL = {
    "trpcage": [
        ("duet", "stride128_t32", 211, 3, 18, "duet_lowest_backbone_change"),
        ("duet", "stride128_t32", 211, 3, 32, "duet_final_endpoint_retained"),
        ("frozen", "stride16_t32", 223, 6, 25, "frozen_best_retained_hit"),
    ],
    "bba": [
        ("duet", "stride16_t32", 223, 1, 6, "duet_lowest_backbone_change"),
        ("frozen", "stride16_t32", 211, 0, 30, "frozen_lowest_backbone_change"),
        ("frozen", "stride16_t32", 211, 0, 10, "frozen_best_relaxed_tica_distance"),
    ],
}
PRMT6 = [
    ("official_forward", 3, 1, "official_lowest_displacement"),
    ("official_forward", 2, 1, "official_second_lowest_displacement"),
    ("current_custom_sde", 3, 2, "custom_lowest_displacement"),
]


def write_pdb(path: Path, topology, xyz_a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        app.PDBFile.writeFile(topology, xyz_a * unit.angstrom, handle, keepIds=True)


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def relaxer_small(root: Path, protein: str):
    prep = root / "data/small_protein_transition_pilot/prepared" / protein
    manifest = json.loads((prep / "manifest.json").read_text())
    sources = {Path(x["path"]).name: Path(x["path"]) for x in manifest["official_source_files"]}
    ffroot = Path(manifest["tica_model"]).parents[2]
    return RestrainedRelaxer(sources["folded.pdb"], manifest["minimized_target_allatom"], manifest["model_length"], [ffroot / "data/protein.ff14SBonlysc.xml", Path("implicit/gbn2.xml")]), prep


def do_small(root: Path, output: Path, protein: str) -> list[dict]:
    relaxer, prep = relaxer_small(root, protein)
    rows = []
    for method, run_name, seed, path_i, frame_i, reason in SMALL[protein]:
        run = root / "outputs/small_protein_transition_pilot" / protein / run_name / method / f"seed_{seed}"
        pop = np.load(run / "pre_final_population_atom37.npz")
        frame = np.asarray(pop["trajectories_atom37_a"][path_i, frame_i], float)
        mask = np.asarray(pop["trajectories_atom37_mask"][path_i, frame_i], bool)
        raw, relaxed, metrics = relaxer.relax(frame, mask)
        stem = f"{protein}_{method}_seed{seed}_p{path_i}_f{frame_i}"
        write_pdb(output / protein / f"{stem}_raw.pdb", relaxer.topology, raw)
        write_pdb(output / protein / f"{stem}_relaxed.pdb", relaxer.topology, relaxed)
        rows.append({"label": stem, "selection_reason": reason, "source_run": str(run), "path_index": path_i, "frame_index": frame_i, **metrics})
    # Include the physically valid folded target solely as a visual reference.
    ref = np.load(prep / "whole_backbone_reference_atom37.npz")
    ref_a, _ = relaxer._complete(ref["reference_atom37_a"][0], ref["reference_atom37_mask"][0])
    write_pdb(output / protein / f"{protein}_folded_target_reference.pdb", relaxer.topology, ref_a)
    return rows


def do_prmt6(root: Path, output: Path) -> list[dict]:
    prep = Path("/workspace/sejin/phase_b_pockets_recovery/prepared/prmt6")
    manifest = json.loads((prep / "manifest.json").read_text())
    relaxer = RestrainedRelaxer(prep / "prmt6_start_model_numbering.pdb", prep / "prmt6_start_model_numbering.pdb", manifest["model_length"], [root / "external/tps-dps/data/protein.ff14SBonlysc.xml", Path("implicit/gbn2.xml")])
    rows = []
    for arm, path_i, frame_i, reason in PRMT6:
        src = root / "outputs/bundle_a_cross_clock_audit/sampler_contrast/prmt6" / arm / "trajectories_atom37.npz"
        arr = np.load(src); frame = np.asarray(arr["trajectories_atom37_a"][path_i, frame_i], float); mask = np.asarray(arr["trajectories_atom37_mask"][path_i, frame_i], bool)
        raw, relaxed, metrics = relaxer.relax(frame, mask)
        stem = f"prmt6_{arm}_p{path_i}_f{frame_i}"
        write_pdb(output / "prmt6" / f"{stem}_raw.pdb", relaxer.topology, raw)
        write_pdb(output / "prmt6" / f"{stem}_relaxed.pdb", relaxer.topology, relaxed)
        rows.append({"label": stem, "selection_reason": reason, "source_npz": str(src), "path_index": path_i, "frame_index": frame_i, **metrics})
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    result = {"selection_note": "Examples selected from corrected audit only; not representative-rate estimates.", "trpcage": do_small(a.root, a.output, "trpcage"), "bba": do_small(a.root, a.output, "bba"), "prmt6": do_prmt6(a.root, a.output)}
    dump(a.output / "selection_and_recomputed_metrics.json", result)


if __name__ == "__main__":
    main()
