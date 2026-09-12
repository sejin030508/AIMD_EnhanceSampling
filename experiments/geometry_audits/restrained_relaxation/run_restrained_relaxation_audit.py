#!/usr/bin/env python3
"""Post-hoc restrained OpenMM relaxation audit for existing Bundle-A outputs.

This program deliberately never edits sampling outputs or their labels.  It
selects paths before any minimization, applies a generated-frame-referenced
heavy-atom restraint, and writes raw/relaxed measurements side by side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import mdtraj as md
import openmm as mm
from openmm import app, unit

# Keep this after the active environment's site-packages so an auxiliary TICA
# package source cannot shadow that environment's OpenMM CUDA runtime.
_tica_site = os.environ.get("RELAX_TICA_SITEPACKAGES")
if _tica_site and _tica_site not in sys.path:
    sys.path.append(_tica_site)


K_RESTRAINT = 10.0 * 418.4  # 10 kcal mol^-1 A^-2 in kJ mol^-1 nm^-2
TOLERANCE = 10.0  # kJ mol^-1 nm^-1
MAX_ITER = 2000
# OpenFold/AlphaFold atom37 ordering.  Every available generated heavy atom,
# including side chains, must replace the aligned template coordinate before
# it can be used as a restraint reference.
ATOM37_NAMES = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)
ATOM37 = {name: i for i, name in enumerate(ATOM37_NAMES)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def kabsch(source, target):
    source = np.asarray(source, float); target = np.asarray(target, float)
    a = source.mean(0); b = target.mean(0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1; rot = u @ vt
    return rot, a, b


def rmsd_aligned(a, b):
    if len(a) < 3:
        return None
    rot, ca, cb = kabsch(a, b)
    return float(np.sqrt(np.mean(np.sum(((a-ca) @ rot + cb - b)**2, axis=-1))))


def atom37_topology_map(topology, n_res):
    rows = []
    for atom in topology.atoms():
        if atom.residue.index < n_res and atom.name in ATOM37:
            rows.append((atom.index, atom.residue.index, ATOM37[atom.name]))
    return rows


def all_atom_frame(template_a, mapping, atom37_a, mask):
    """Fill template-only atoms rigidly, replace emitted atom37 coordinates."""
    emitted, generated, reference = [], [], []
    for top, residue, a37 in mapping:
        if mask[residue, a37]:
            emitted.append(top); generated.append(atom37_a[residue, a37]); reference.append(template_a[top])
    if len(emitted) < 3:
        raise RuntimeError("fewer than three emitted atoms available for template alignment")
    rot, ca, cb = kabsch(np.asarray(reference), np.asarray(generated))
    completed = (template_a-ca) @ rot + cb
    completed[np.asarray(emitted)] = np.asarray(generated)
    return completed, np.asarray(emitted, dtype=int)


def force_groups(system):
    manifest = []
    for force in system.getForces():
        name = type(force).__name__
        if name in {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "CMAPTorsionForce"}:
            group, category = 0, "bonded"
        elif name == "NonbondedForce":
            group, category = 1, "nonbonded"
        elif name in {"CustomGBForce", "GBSAOBCForce"}:
            group, category = 2, "solvation"
        else:
            group, category = 3, "other"
        force.setForceGroup(group)
        manifest.append({"force": name, "group": group, "category": category})
    return manifest


class RestrainedRelaxer:
    def __init__(self, topology_pdb, template_pdb, n_res, ff_paths):
        raw_pdb = app.PDBFile(str(topology_pdb))
        raw_template = app.PDBFile(str(template_pdb))
        if len(raw_template.positions) != len(raw_pdb.positions):
            raise RuntimeError("topology and template atom count differ")
        ff = app.ForceField(*map(str, ff_paths))
        # PRMT6's prepared PDB is heavy-atom only.  Add standard hydrogens to
        # the *current evaluated-frame topology* before system construction;
        # they remain unrestrained and fully movable.  This is atom completion,
        # not a target/holo coordinate restraint.
        modeller = app.Modeller(raw_pdb.topology, raw_template.positions)
        modeller.addHydrogens(ff)
        self.topology = modeller.topology
        self.template_a = np.asarray(modeller.positions.value_in_unit(unit.angstrom))
        self.system = ff.createSystem(self.topology, nonbondedMethod=app.NoCutoff,
                                      nonbondedCutoff=1.0*unit.nanometer,
                                      constraints=None, ewaldErrorTolerance=0.0005)
        self.force_manifest = force_groups(self.system)
        self.mapping = atom37_topology_map(self.topology, n_res)
        self.atoms = list(self.topology.atoms())
        self.heavy = np.asarray([a.index for a in self.atoms if a.element and a.element.symbol.upper() != "H"], dtype=int)
        self.bonds = {tuple(sorted((a.index, b.index))) for a,b in self.topology.bonds()}
        heavy_pos = {int(atom): i for i,atom in enumerate(self.heavy)}
        hbond = [(heavy_pos[a], heavy_pos[b]) for a,b in self.bonds if a in heavy_pos and b in heavy_pos]
        self.heavy_bond_left = np.asarray([x[0] for x in hbond],dtype=int)
        self.heavy_bond_right = np.asarray([x[1] for x in hbond],dtype=int)
        self.bond_force = next((f for f in self.system.getForces() if isinstance(f, mm.HarmonicBondForce)), None)
        self.angle_force = next((f for f in self.system.getForces() if isinstance(f, mm.HarmonicAngleForce)), None)
        self.restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        self.restraint.addGlobalParameter("k", K_RESTRAINT)
        for p in ("x0", "y0", "z0"): self.restraint.addPerParticleParameter(p)
        for i in self.heavy: self.restraint.addParticle(int(i), [0.0, 0.0, 0.0])
        self.restraint.setForceGroup(4)
        self.system.addForce(self.restraint)
        self.integrator = mm.VerletIntegrator(1.0*unit.femtoseconds)
        # CUDA is requested only when the installed OpenMM CUDA module supports
        # the host driver.  This H100 image currently returns PTX error 222, so
        # PRMT6 uses the supported CPU backend rather than silently failing.
        self.platform_name = os.environ.get("RELAX_OPENMM_PLATFORM", "CPU")
        self.context = mm.Context(self.system, self.integrator,
                                  mm.Platform.getPlatformByName(self.platform_name))

    def physical_energy(self):
        state = self.context.getState(getEnergy=True, groups={0,1,2,3})
        return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

    def components(self):
        names = (("bonded",0),("nonbonded",1),("solvation",2),("other",3))
        return {name: float(self.context.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)) for name,group in names}

    def _set_restraint_reference(self, positions_a):
        for entry, atom in enumerate(self.heavy):
            self.restraint.setParticleParameters(entry, int(atom), (positions_a[atom]/10.0).tolist())
        self.restraint.updateParametersInContext(self.context)

    def _geometry(self, xyz_a):
        def dist(i,j): return float(np.linalg.norm(xyz_a[i]-xyz_a[j]))
        cn=[]; nca=[]; cac=[]; adjacent=[]
        by_res = {}
        for top, res, a37 in self.mapping: by_res[(res,a37)] = top
        for r in range(max(x[1] for x in self.mapping)+1):
            if (r,ATOM37["N"]) in by_res and (r,ATOM37["CA"]) in by_res: nca.append(dist(by_res[r,ATOM37["N"]],by_res[r,ATOM37["CA"]]))
            if (r,ATOM37["CA"]) in by_res and (r,ATOM37["C"]) in by_res: cac.append(dist(by_res[r,ATOM37["CA"]],by_res[r,ATOM37["C"]]))
            if r and (r-1,ATOM37["C"]) in by_res and (r,ATOM37["N"]) in by_res: cn.append(dist(by_res[r-1,ATOM37["C"]],by_res[r,ATOM37["N"]]))
            if r and (r-1,ATOM37["CA"]) in by_res and (r,ATOM37["CA"]) in by_res: adjacent.append(dist(by_res[r-1,ATOM37["CA"]],by_res[r,ATOM37["CA"]]))
        # Nonbonded heavy clashes. Direct covalent pairs are excluded by definition.
        heavy_xyz=xyz_a[self.heavy]
        hi,hj=np.triu_indices(len(self.heavy),1)
        heavy_dist=np.linalg.norm(heavy_xyz[hi]-heavy_xyz[hj],axis=1)
        direct=np.zeros(len(heavy_dist),dtype=bool)
        # A direct covalent pair is explicitly excluded from the clash definition.
        lookup={(int(i),int(j)): z for z,(i,j) in enumerate(zip(hi,hj))}
        for left,right in zip(self.heavy_bond_left,self.heavy_bond_right):
            key=tuple(sorted((int(left),int(right))))
            if key in lookup: direct[lookup[key]]=True
        nonbond=heavy_dist[~direct]
        clash=int(np.sum(nonbond<1.0)); min_nb=float(np.min(nonbond))
        bond_out=0; max_bond=0.0
        if self.bond_force:
            for i in range(self.bond_force.getNumBonds()):
                a,b,r0,_=self.bond_force.getBondParameters(i); delta=abs(dist(int(a),int(b))/10.0-r0.value_in_unit(unit.nanometer)); max_bond=max(max_bond,delta); bond_out += int(delta > 0.05)
        angle_out=0; max_angle=0.0
        if self.angle_force:
            for i in range(self.angle_force.getNumAngles()):
                a,b,c,t0,_=self.angle_force.getAngleParameters(i); u=xyz_a[int(a)]-xyz_a[int(b)]; v=xyz_a[int(c)]-xyz_a[int(b)]
                theta=math.acos(float(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))); delta=abs(theta-t0.value_in_unit(unit.radian)); max_angle=max(max_angle,delta); angle_out += int(delta > 0.35)
        # L-chirality reference is template's own sign. Glycine has no CB.
        chirality_bad=chirality_tested=0
        for r in range(max(x[1] for x in self.mapping)+1):
            keys=[(r,ATOM37[x]) for x in ("N","CA","C","CB")]
            if all(k in by_res for k in keys):
                ii=[by_res[k] for k in keys]; ref=np.linalg.det(np.stack([self.template_a[ii[0]]-self.template_a[ii[1]],self.template_a[ii[2]]-self.template_a[ii[1]],self.template_a[ii[3]]-self.template_a[ii[1]]]))
                now=np.linalg.det(np.stack([xyz_a[ii[0]]-xyz_a[ii[1]],xyz_a[ii[2]]-xyz_a[ii[1]],xyz_a[ii[3]]-xyz_a[ii[1]]]))
                if abs(ref)>1e-8 and abs(now)>1e-8: chirality_tested+=1; chirality_bad += int(np.sign(ref)!=np.sign(now))
        return {"peptide_c_n_min_max_a": [min(cn),max(cn)] if cn else None, "peptide_c_n_gross_count": sum(x<1.0 or x>1.7 for x in cn), "n_ca_min_max_a": [min(nca),max(nca)] if nca else None, "ca_c_min_max_a": [min(cac),max(cac)] if cac else None, "adjacent_ca_max_a": max(adjacent) if adjacent else None, "nonbonded_heavy_clash_lt_1a_count": clash, "nonbonded_heavy_min_distance_a": min_nb, "bond_deviation_gt_0_5a_count": bond_out, "max_bond_deviation_nm": max_bond, "angle_deviation_gt_20deg_count": angle_out, "max_angle_deviation_rad": max_angle, "chirality_inverted_count": chirality_bad, "chirality_tested_count": chirality_tested, "finite": bool(np.isfinite(xyz_a).all())}

    def relax(self, atom37_a, mask):
        t0=time.perf_counter(); raw_a, emitted=self._complete(atom37_a, mask); prep_s=time.perf_counter()-t0
        self.context.setPositions(raw_a*unit.angstrom); self._set_restraint_reference(raw_a)
        raw_e=self.physical_energy(); raw_components=self.components(); raw_g=self._geometry(raw_a)
        tm=time.perf_counter(); error=None
        try: mm.LocalEnergyMinimizer.minimize(self.context, TOLERANCE*unit.kilojoule_per_mole/unit.nanometer, MAX_ITER)
        except Exception as exc: error=repr(exc)
        min_s=time.perf_counter()-tm
        state=self.context.getState(getPositions=True,getEnergy=True,getForces=True)
        relaxed_a=np.asarray(state.getPositions().value_in_unit(unit.angstrom)); force=np.asarray(state.getForces().value_in_unit(unit.kilojoule_per_mole/unit.nanometer)); max_force=float(np.max(np.linalg.norm(force,axis=1)))
        relax_e=self.physical_energy(); relax_components=self.components(); relax_g=self._geometry(relaxed_a)
        disp=np.linalg.norm(relaxed_a[emitted]-raw_a[emitted],axis=1)
        return raw_a, relaxed_a, {"status":"failed" if error else "complete", "error":error, "preparation_s":prep_s, "minimization_s":min_s, "raw_physical_energy_kj_mol":raw_e, "relaxed_physical_energy_kj_mol":relax_e, "raw_components_kj_mol":raw_components, "relaxed_components_kj_mol":relax_components, "restraint_energy_kj_mol":float(self.context.getState(getEnergy=True,groups={4}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)), "max_force_kj_mol_nm":max_force, "force_tolerance_met": bool(max_force<=TOLERANCE), "max_iterations":MAX_ITER, "iteration_cap_reached_not_observable": bool(max_force>TOLERANCE), "raw_geometry":raw_g, "relaxed_geometry":relax_g, "generated_atom_displacement_a": {"median":float(np.median(disp)),"p95":float(np.quantile(disp,.95)),"max":float(np.max(disp))}, "restraint_reference":"each completed generated frame's own heavy-atom coordinates"}

    def _complete(self, a,m): return all_atom_frame(self.template_a,self.mapping,a,m)


def load_small_selection(root, protein):
    """Freeze lexical selection based only on pre-existing raw labels."""
    candidates=[]
    for run in sorted((root/"outputs/small_protein_transition_pilot"/protein).glob("stride*_t32/*/seed_*")):
        metric=run/"metrics.json"; diag=run/"small_protein_frame_diagnostics.json"; pop=run/"pre_final_population_atom37.npz"
        if not(metric.exists() and diag.exists() and pop.exists()): continue
        base=json.loads(metric.read_text()); d=json.loads(diag.read_text()); valid=np.asarray(base["per_path_valid"],bool); hits=np.asarray(d["target_hit"],bool)
        arr=np.load(pop); key=(int(base["stride_in_10ps"]), int(base["seed"]), str(run))
        for pi in range(len(valid)):
            candidates.append({"sort":key+(pi,),"run":str(run),"method":base["method"],"stride":int(base["stride_in_10ps"]),"seed":int(base["seed"]),"path_index":pi,"valid":bool(valid[pi]),"final_hit":bool(hits[pi,-1]),"any_generated_hit":bool(hits[pi,1:].any())})
    selected=[]
    for method in ("frozen","duet"):
        rows=sorted((x for x in candidates if x["method"]==method),key=lambda x:x["sort"])
        hit=next(x for x in rows if x["valid"] and x["final_hit"])
        non=next(x for x in rows if x["valid"] and not x["any_generated_hit"])
        hit["selection_role"]="final_TICA_hit"; non["selection_role"]="valid_generated_TICA_nonhit"; selected += [hit,non]
    return selected


def small_goal_metrics(tica, target_xy, target, target_mask, official_map, raw_a, relaxed_a, mapping, frame, mask):
    # Project via exact TICA feature mapping by replacing emitted atoms in an atom37 frame.
    inverse={(r,a):top for top,r,a in mapping}; rraw=frame.copy(); rrel=frame.copy()
    for (r,a),top in inverse.items(): rraw[r,a]=raw_a[top]; rrel[r,a]=relaxed_a[top]
    raw_xy, raw_usable = tica.project(rraw[None], mask[None]); rel_xy, rel_usable = tica.project(rrel[None], mask[None])
    tc=np.asarray(target_xy); rd=float(np.linalg.norm(raw_xy[0]-tc)) if raw_usable[0] else float("nan"); ld=float(np.linalg.norm(rel_xy[0]-tc)) if rel_usable[0] else float("nan")
    bb=[ATOM37[x] for x in ("N","CA","C")]
    common=np.asarray([[mask[r,a] and target_mask[r,a] for a in bb] for r in range(frame.shape[0])],bool)
    rawbb=np.asarray([rraw[r,a] for r in range(frame.shape[0]) for a in bb if common[r,bb.index(a)]])
    relbb=np.asarray([rrel[r,a] for r in range(frame.shape[0]) for a in bb if common[r,bb.index(a)]])
    tarbb=np.asarray([target[r,a] for r in range(frame.shape[0]) for a in bb if common[r,bb.index(a)]])
    return {"raw_tica_distance":rd,"relaxed_tica_distance":ld,"raw_tica_hit":bool(rd<0.75),"relaxed_tica_hit":bool(ld<0.75),"raw_target_backbone_rmsd_a":rmsd_aligned(rawbb,tarbb),"relaxed_target_backbone_rmsd_a":rmsd_aligned(relbb,tarbb),"raw_to_relaxed_backbone_rmsd_a":rmsd_aligned(rawbb,relbb)}


def run_small(root, output, protein):
    sys.path.insert(0,str(root/"small_protein_pilot"))
    from evaluate_small_protein_run import OfficialTICA
    prep=root/"data/small_protein_transition_pilot/prepared"/protein; manifest=json.loads((prep/"manifest.json").read_text())
    sources={Path(x["path"]).name:Path(x["path"]) for x in manifest["official_source_files"]}; topology=sources["folded.pdb"]
    ffroot=Path(manifest["tica_model"]).parents[2]
    relaxer=RestrainedRelaxer(topology,manifest["minimized_target_allatom"],manifest["model_length"],[ffroot/"data/protein.ff14SBonlysc.xml",Path("implicit/gbn2.xml")])
    reference=np.load(prep/"whole_backbone_reference_atom37.npz"); target=np.asarray(reference["reference_atom37_a"][0],float); target_mask=np.asarray(reference["reference_atom37_mask"][0],bool); tica=OfficialTICA(manifest)
    target_xy, target_usable = tica.project(target[None], target_mask[None])
    if not target_usable[0]: raise RuntimeError("official target is not TICA-projectable")
    selected=load_small_selection(root,protein); write_json(output/protein/"frozen_path_selection.json", {"selection_rule":"For each method, lexicographically first (stride, seed, run path, population index) raw-valid final TICA hit and raw-valid path with no generated-frame TICA hit. Selection performed before minimization.","paths":selected})
    rows=[]; seen={}
    for choice in selected:
        p=np.load(Path(choice["run"])/"pre_final_population_atom37.npz"); traj=np.asarray(p["trajectories_atom37_a"],float); masks=np.asarray(p["trajectories_atom37_mask"],bool); path=traj[choice["path_index"]]; pmask=masks[choice["path_index"]]
        for fi in range(path.shape[0]):
            token=hashlib.sha256(np.ascontiguousarray(path[fi]).tobytes()+np.ascontiguousarray(pmask[fi]).tobytes()).hexdigest()
            identity={k:v for k,v in choice.items() if k not in {"sort"}}|{"frame_index":fi,"frame_digest":token}
            if token in seen: rows.append(identity|{"reused_from":seen[token]}); continue
            try:
                raw,relaxed,metrics=relaxer.relax(path[fi],pmask[fi])
                metrics["goal"]=small_goal_metrics(tica,target_xy[0],target,target_mask,tica.mapping,raw,relaxed,relaxer.mapping,path[fi],pmask[fi])
                rows.append(identity|metrics)
            except Exception as error:
                # A non-finite/raw-degenerate candidate is itself an audit
                # outcome.  Record it, preserve the selection, and continue
                # rather than allowing one bad frame to erase the full path.
                rows.append(identity|{"status":"failed","failure_stage":"raw_coordinate_alignment_or_relaxation","error":repr(error),"input_emitted_coordinates_finite":bool(np.isfinite(path[fi][pmask[fi]]).all())})
            seen[token]=identity
    # Normal controls are measured but never count as sampled paths.
    controls=[]
    first=np.load(Path(selected[0]["run"])/"pre_final_population_atom37.npz")
    for name,frame,mask in (("normal_start",first["trajectories_atom37_a"][0,0],first["trajectories_atom37_mask"][0,0]),("normal_target",target,target_mask)):
        raw,rel,metric=relaxer.relax(frame,mask); metric["goal"]=small_goal_metrics(tica,target_xy[0],target,target_mask,tica.mapping,raw,rel,relaxer.mapping,frame,mask); controls.append({"name":name}|metric)
    write_json(output/protein/"frames.json", {"scope":"posthoc_only","protein":protein,"forcefield":"protein.ff14SBonlysc + implicit/gbn2; NoCutoff; existing Bundle-A topology","restraint":{"heavy_atoms":"all topology heavy atoms","reference":"the evaluated frame itself","strength_kcal_mol_a2":10.0,"all_atoms_movable":True},"minimizer":{"OpenMM":"LocalEnergyMinimizer L-BFGS","force_tolerance_kj_mol_nm":TOLERANCE,"max_iterations":MAX_ITER},"force_manifest":relaxer.force_manifest,"frames":rows,"normal_controls":controls})


def run_prmt6(root, output):
    # The PRMT6 start PDB is used solely as an atom-complete topology/template;
    # the source trajectories replace every emitted model atom and none is steered to holo.
    prep=Path("/workspace/sejin/phase_b_pockets_recovery/prepared/prmt6"); manifest=json.loads((prep/"manifest.json").read_text()); topology=prep/"prmt6_start_model_numbering.pdb"
    # PDB lacks hydrogens; Modeller/ff would alter topology. Therefore this audit uses
    # exactly its existing protein heavy-atom topology with implicit solvent (all present atoms movable).
    ffroot=root/"external/tps-dps"
    relaxer=RestrainedRelaxer(topology,topology,manifest["model_length"],[ffroot/"data/protein.ff14SBonlysc.xml",Path("implicit/gbn2.xml")])
    rows=[]; seen={}
    partial_path=output/"prmt6"/"frames.partial.json"
    for arm in ("official_forward","current_custom_sde"):
        arr=np.load(root/"outputs/bundle_a_cross_clock_audit/sampler_contrast/prmt6"/arm/"trajectories_atom37.npz"); x=np.asarray(arr["trajectories_atom37_a"],float); m=np.asarray(arr["trajectories_atom37_mask"],bool)
        for pi in range(x.shape[0]):
            for fi in range(1,x.shape[1]): # 16 generated frames per arm; x0 excluded
                token=hashlib.sha256(np.ascontiguousarray(x[pi,fi]).tobytes()+np.ascontiguousarray(m[pi,fi]).tobytes()).hexdigest(); ident={"arm":arm,"path_index":pi,"frame_index":fi,"frame_digest":token}
                if token in seen: rows.append(ident|{"reused_from":seen[token]}); continue
                try:
                    raw,relaxed,metric=relaxer.relax(x[pi,fi],m[pi,fi]); rows.append(ident|metric)
                except Exception as error:
                    rows.append(ident|{"status":"failed","failure_stage":"raw_coordinate_alignment_or_relaxation","error":repr(error),"input_emitted_coordinates_finite":bool(np.isfinite(x[pi,fi][m[pi,fi]]).all())})
                seen[token]=ident
                write_json(partial_path, {"scope":"posthoc_only","protein":"prmt6","completed_unique_frames":len(seen),"expected_unique_frames":32,"frames":rows})
    write_json(output/"prmt6"/"frames.json", {"scope":"posthoc_only","protein":"prmt6","selection":"all 16 generated frames each from existing Bundle-A official_forward and current_custom_sde; no holo/target coordinates used as restraint reference","forcefield":"protein.ff14SBonlysc + implicit/gbn2; existing PRMT6 prepared topology","restraint":{"heavy_atoms":"all topology heavy atoms","reference":"the evaluated frame itself","strength_kcal_mol_a2":10.0,"all_atoms_movable":True},"minimizer":{"force_tolerance_kj_mol_nm":TOLERANCE,"max_iterations":MAX_ITER},"force_manifest":relaxer.force_manifest,"frames":rows})


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--scope",choices=["trpcage","bba","prmt6"],required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    if a.scope in {"trpcage","bba"}: run_small(a.root,a.output,a.scope)
    else: run_prmt6(a.root,a.output)

if __name__ == "__main__": main()
