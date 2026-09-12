#!/usr/bin/env python3
"""Independent name-based C-alpha chirality recheck; no minimization."""
from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path

import mdtraj as md
import numpy as np
from openmm import app
from rdkit import Chem


TOLERANCE_A3 = 0.5
PATHS = {
    "trpcage": [
        "trpcage/trpcage_duet_seed211_p3",
        "trpcage/trpcage_frozen_seed223_p6",
        "trpcage/trpcage_duet_seed223_p3",
    ],
    "bba": [
        "bba/bba_duet_seed223_p1",
        "bba/bba_frozen_seed211_p0",
        "bba/bba_duet_seed211_p1",
    ],
    "prmt6": [
        "prmt6/prmt6_official_forward_p3",
        "prmt6/prmt6_official_forward_p2",
        "prmt6/prmt6_current_custom_sde_p3",
    ],
}


def residue_maps(topology, xyz):
    groups = {}
    for atom in topology.atoms:
        key = (atom.residue.chain.index, int(atom.residue.resSeq), atom.residue.name)
        groups.setdefault(key, {})[atom.name.strip().upper()] = xyz[atom.index]
    return groups


def volume(group):
    try:
        n, ca, c, cb = (group[x] for x in ("N", "CA", "C", "CB"))
    except KeyError:
        return None
    return float(np.dot(n-ca, np.cross(c-ca, cb-ca)))


def sign(v):
    if v is None or not np.isfinite(v) or abs(v) < TOLERANCE_A3:
        return None
    return 1 if v > 0 else -1


def describe(key):
    chain, resseq, name = key
    return {"chain_index": int(chain), "residue_number": int(resseq), "residue_name": name}


def atom_coordinates(traj, frame_index, key):
    """Return N/CA/C/CB coordinates in Å using explicit residue identity."""
    chain, resseq, resname = key
    out = {}
    for atom in traj.topology.atoms:
        r = atom.residue
        if r.chain.index == chain and int(r.resSeq) == int(resseq) and r.name == resname:
            name = atom.name.strip().upper()
            if name in {"N", "CA", "C", "CB"}:
                out[name] = (traj.xyz[frame_index, atom.index] * 10.0).tolist()
    return out


def analyze_pair(raw_path, relaxed_path, ref_traj, system, label):
    raw = md.load(str(raw_path)); relaxed = md.load(str(relaxed_path))
    ref_groups = residue_maps(ref_traj.topology, ref_traj.xyz[0] * 10.0)
    rows=[]; frame_summary=[]
    for side, traj in (("raw",raw),("relaxed",relaxed)):
        for fi in range(traj.n_frames):
            groups = residue_maps(traj.topology, traj.xyz[fi] * 10.0)
            for key, group in groups.items():
                v = volume(group); rv = volume(ref_groups.get(key, {})); s=sign(v); rs=sign(rv)
                status = "missing_or_gly_or_planar" if s is None or rs is None else ("inverted_vs_reference" if s != rs else "normal_vs_reference")
                rows.append({"system":system,"path":label,"frame_index":fi,"side":side,**describe(key),"signed_volume_a3":v,"reference_signed_volume_a3":rv,"sign":s,"reference_sign":rs,"status":status,"tolerance_a3":TOLERANCE_A3})
            current=[r for r in rows if r["system"]==system and r["path"]==label and r["side"]==side and r["frame_index"]==fi]
            evaluable=[r for r in current if r["sign"] is not None and r["reference_sign"] is not None]
            frame_summary.append({"system":system,"path":label,"frame_index":fi,"side":side,"evaluable_ca":len(evaluable),"inverted_ca":sum(r["status"]=="inverted_vs_reference" for r in evaluable),"frame_evaluable":bool(evaluable),"frame_has_inversion":any(r["status"]=="inverted_vs_reference" for r in evaluable)})
    # raw -> relaxed change by the same explicit chain/residue/name key.
    index={(r["system"],r["path"],r["frame_index"],r["chain_index"],r["residue_number"],r["residue_name"]):r for r in rows}
    for r in rows:
        if r["side"]!="raw": continue
        k=(system,label,r["frame_index"],r["chain_index"],r["residue_number"],r["residue_name"]); q=index.get(k[:-1]+(r["residue_name"],))
        # Relaxed key has same tuple except side; build explicit lookup.
        q=next((x for x in rows if x["side"]=="relaxed" and x["frame_index"]==r["frame_index"] and x["chain_index"]==r["chain_index"] and x["residue_number"]==r["residue_number"] and x["residue_name"]==r["residue_name"]),None)
        if q is not None:
            r["raw_to_relaxed_sign_change"]=(r["sign"] is not None and q["sign"] is not None and r["sign"]!=q["sign"])
            r["relaxed_status_for_pair"]=q["status"]
    return rows,frame_summary,raw,relaxed


def rdkit_chiral_tags(path, frame_index):
    """Build RDKit graph from the existing OpenMM topology bonds.

    No bond is inferred from the coordinates.  PDB parsing is used only for
    the coordinates; the topology graph comes from OpenMM's standard PDB
    topology (the same bond graph used by the relaxation system).
    """
    traj=md.load(str(path)); pdb=app.PDBFile(str(path)); topology=pdb.topology
    try:
        rw=Chem.RWMol();
        for atom in topology.atoms():
            rd=Chem.Atom(atom.element.atomic_number if atom.element is not None else 0); rd.SetNoImplicit(True); rw.AddAtom(rd)
        for left,right in topology.bonds(): rw.AddBond(left.index,right.index,Chem.BondType.SINGLE)
        mol=rw.GetMol(); conf=Chem.Conformer(topology.getNumAtoms()); xyz=traj.xyz[frame_index]*10.0
        for i,coord in enumerate(xyz): conf.SetAtomPosition(i,tuple(float(x) for x in coord))
        mol.AddConformer(conf,assignId=True); Chem.RemoveStereochemistry(mol); Chem.AssignAtomChiralTagsFromStructure(mol,confId=0,replaceExistingTags=True)
        out={}
        for residue in topology.residues():
            ca=next((atom for atom in residue.atoms() if atom.name.strip().upper()=="CA"),None)
            if ca is None: continue
            key=(residue.chain.index,int(residue.id) if str(residue.id).lstrip("-").isdigit() else residue.index,residue.name)
            out[str(key)]=str(mol.GetAtomWithIdx(ca.index).GetChiralTag())
        return {"available":True,"ca_tags":out,"bond_count":mol.GetNumBonds()}
    except Exception as exc: return {"available":False,"reason":repr(exc),"bond_count":sum(1 for _ in topology.bonds())}


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--path-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    all_rows=[]; frame_rows=[]; controls=[]; rdkit_rows=[]; representative_cases=[]
    refs={"trpcage":a.root/"data/small_protein_transition_pilot/prepared/trpcage/folded_minimized_allatom.pdb","bba":a.root/"data/small_protein_transition_pilot/prepared/bba/folded_minimized_allatom.pdb","prmt6":Path("/workspace/sejin/phase_b_pockets_recovery/prepared/prmt6/prmt6_start_model_numbering.pdb")}
    for system, stems in PATHS.items():
        ref=md.load(str(refs[system]))
        for stem in stems:
            raw_path=a.path_root/f"{stem}_raw_path.pdb"; rel_path=a.path_root/f"{stem}_relaxed_path.pdb"; label=stem.split("/",1)[1]
            rows,fs,raw,rel=analyze_pair(raw_path,rel_path,ref,system,label); all_rows.extend(rows); frame_rows.extend(fs)
            # Up to three representative independent-method inversions per system.
            candidates=[r for r in rows if r["side"]=="relaxed" and r["status"]=="inverted_vs_reference"]
            for r in candidates[:3]:
                ri=rdkit_chiral_tags(raw_path,r["frame_index"]); qi=rdkit_chiral_tags(rel_path,r["frame_index"]); rdkit_rows.append({"system":system,"path":label,"frame_index":r["frame_index"],**describe((r["chain_index"],r["residue_number"],r["residue_name"])),"raw_rdkit":ri,"relaxed_rdkit":qi})
                key=(r["chain_index"],r["residue_number"],r["residue_name"])
                lookup=lambda tags: tags.get(str(key))
                representative_cases.append({
                    "system": system, "path": label, "frame_index": r["frame_index"], **describe(key),
                    "raw_signed_volume_a3": next(x["signed_volume_a3"] for x in rows if x["side"]=="raw" and x["frame_index"]==r["frame_index"] and x["chain_index"]==r["chain_index"] and x["residue_number"]==r["residue_number"] and x["residue_name"]==r["residue_name"]),
                    "relaxed_signed_volume_a3": r["signed_volume_a3"], "reference_signed_volume_a3": r["reference_signed_volume_a3"],
                    "raw_group_coordinates_A": atom_coordinates(raw, r["frame_index"], key),
                    "relaxed_group_coordinates_A": atom_coordinates(rel, r["frame_index"], key),
                    "raw_rdkit_ca_tag": lookup(ri["ca_tags"]), "relaxed_rdkit_ca_tag": lookup(qi["ca_tags"]),
                    "raw_bond_count": ri.get("bond_count"), "relaxed_bond_count": qi.get("bond_count"),
                })
    for system, refpath in refs.items():
        ref=md.load(str(refpath)); groups=residue_maps(ref.topology,ref.xyz[0]*10.0); vals=[sign(volume(g)) for g in groups.values()]; controls.append({"system":system,"control":"reference_identity","evaluable_ca":sum(x is not None for x in vals),"inverted_ca":0})
        # Proper rigid transform must preserve signs; mirror in x must invert all determinate signs.
        xyz=ref.xyz[0]*10.0; rot=xyz@np.array([[0,-1,0],[1,0,0],[0,0,1.]])+np.array([7.,-3.,2.]); mirror=xyz*np.array([-1.,1.,1.]);
        for name,coords,expected in (("rotation_translation",rot,False),("mirror_x",mirror,True)):
            gs=residue_maps(ref.topology,coords); matched=sum(sign(volume(g)) is not None and sign(volume(g))!=vals[i] for i,g in enumerate(gs.values()))
            controls.append({"system":system,"control":name,"evaluable_ca":sum(sign(volume(g)) is not None for g in gs.values()),"inverted_ca":matched,"expected_mirror_inversions":expected})
    # Merge frame summary so each frame's raw and relaxed counts can be compared.
    with (a.output/"chirality_residue_rows.csv").open("w",newline="") as f:
        fields=sorted({k for r in all_rows for k in r}); w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
    with (a.output/"chirality_frame_rows.csv").open("w",newline="") as f:
        fields=sorted({k for r in frame_rows for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(frame_rows)
    summary={"tolerance_a3":TOLERANCE_A3,"coordinate_scope":"213 stored complete-path frames (129 appearances unique small frames are not all present in export; 32 PRMT6 path frames; controls included)","raw_relaxed_rows":len(all_rows),"controls":controls,"rdkit_crosscheck":rdkit_rows,"representative_cases":representative_cases}
    for system in PATHS:
        fr=[x for x in frame_rows if x["system"]==system]; rr=[x for x in all_rows if x["system"]==system]; summary[system]={}
        for side in ("raw","relaxed"):
            f=[x for x in fr if x["side"]==side]; q=[x for x in rr if x["side"]==side]; evaluable=[x for x in q if x["sign"] is not None and x["reference_sign"] is not None]; inv=[x for x in evaluable if x["status"]=="inverted_vs_reference"]; summary[system][side]={"inversion_frames":sum(x["frame_has_inversion"] for x in f),"evaluable_frames":sum(x["frame_evaluable"] for x in f),"inverted_ca":len(inv),"evaluable_ca":len(evaluable),"not_evaluable_ca":len(q)-len(evaluable)}
    (a.output/"chirality_recheck_summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":main()
