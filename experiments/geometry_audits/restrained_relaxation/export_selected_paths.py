#!/usr/bin/env python3
"""Export complete raw/relaxed physical paths as multi-model PDB files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openmm import app, unit

from export_selected_examples import relaxer_small
from run_restrained_relaxation_audit import RestrainedRelaxer


# All small-protein choices are raw-valid final TICA-hit paths.  The first two
# are the fixed audit paths; the third is an independent-seed hit for viewing.
SMALL_PATHS = {
    "trpcage": [
        ("duet", "stride128_t32", 211, 3, "duet_fixed_audit_hit"),
        ("frozen", "stride16_t32", 223, 6, "frozen_fixed_audit_hit"),
        ("duet", "stride128_t32", 223, 3, "duet_independent_seed_hit"),
    ],
    "bba": [
        ("duet", "stride16_t32", 223, 1, "duet_fixed_audit_hit"),
        ("frozen", "stride16_t32", 211, 0, "frozen_fixed_audit_hit"),
        ("duet", "stride128_t32", 211, 1, "duet_independent_seed_hit"),
    ],
}
PRMT6_PATHS = [
    ("official_forward", 3, "official_forward_selected_low_displacement"),
    ("official_forward", 2, "official_forward_selected_low_displacement_2"),
    ("current_custom_sde", 3, "custom_sde_selected_low_displacement"),
]


def write_models(path: Path, topology, models_a: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        app.PDBFile.writeHeader(topology, handle)
        for index, xyz in enumerate(models_a, 1):
            app.PDBFile.writeModel(topology, xyz * unit.angstrom, handle, modelIndex=index, keepIds=True)
        app.PDBFile.writeFooter(topology, handle)


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_small(root: Path, output: Path, protein: str):
    relaxer, _ = relaxer_small(root, protein)
    results = []
    for method, run_name, seed, path_i, label in SMALL_PATHS[protein]:
        run = root / "outputs/small_protein_transition_pilot" / protein / run_name / method / f"seed_{seed}"
        pop = np.load(run / "pre_final_population_atom37.npz")
        frames = np.asarray(pop["trajectories_atom37_a"][path_i], float)
        masks = np.asarray(pop["trajectories_atom37_mask"][path_i], bool)
        raw_models=[]; relaxed_models=[]; metrics=[]
        for fi,(frame,mask) in enumerate(zip(frames,masks)):
            raw, relaxed, record = relaxer.relax(frame, mask)
            raw_models.append(raw); relaxed_models.append(relaxed); metrics.append({"frame_index":fi, **record})
        stem=f"{protein}_{method}_seed{seed}_p{path_i}"
        write_models(output/protein/f"{stem}_raw_path.pdb",relaxer.topology,raw_models)
        write_models(output/protein/f"{stem}_relaxed_path.pdb",relaxer.topology,relaxed_models)
        results.append({"label":stem,"selection_reason":label,"source_run":str(run),"path_index":path_i,"frame_count":len(frames),"frames":metrics})
    dump(output/protein/"path_metrics.json",results)


def run_prmt6(root: Path, output: Path):
    prep=Path("/workspace/sejin/phase_b_pockets_recovery/prepared/prmt6"); manifest=json.loads((prep/"manifest.json").read_text())
    relaxer=RestrainedRelaxer(prep/"prmt6_start_model_numbering.pdb",prep/"prmt6_start_model_numbering.pdb",manifest["model_length"],[root/"external/tps-dps/data/protein.ff14SBonlysc.xml",Path("implicit/gbn2.xml")])
    results=[]
    for arm,path_i,label in PRMT6_PATHS:
        src=root/"outputs/bundle_a_cross_clock_audit/sampler_contrast/prmt6"/arm/"trajectories_atom37.npz"; arr=np.load(src)
        frames=np.asarray(arr["trajectories_atom37_a"][path_i],float); masks=np.asarray(arr["trajectories_atom37_mask"][path_i],bool)
        raw_models=[]; relaxed_models=[]; metrics=[]
        for fi,(frame,mask) in enumerate(zip(frames,masks)):
            raw,relaxed,record=relaxer.relax(frame,mask);raw_models.append(raw);relaxed_models.append(relaxed);metrics.append({"frame_index":fi,**record})
        stem=f"prmt6_{arm}_p{path_i}";write_models(output/"prmt6"/f"{stem}_raw_path.pdb",relaxer.topology,raw_models);write_models(output/"prmt6"/f"{stem}_relaxed_path.pdb",relaxer.topology,relaxed_models)
        results.append({"label":stem,"selection_reason":label,"source_npz":str(src),"path_index":path_i,"frame_count":len(frames),"frames":metrics})
    dump(output/"prmt6"/"path_metrics.json",results)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--scope",choices=["trpcage","bba","prmt6"],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.scope in {"trpcage","bba"}:run_small(a.root,a.output,a.scope)
    else:run_prmt6(a.root,a.output)

if __name__=="__main__":main()
