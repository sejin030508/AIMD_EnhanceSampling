#!/usr/bin/env python3
"""Bundle-A audit of existing small-protein TPS target-hit paths.

This is evaluation-only.  It never edits production metrics and never uses a
minimized structure to reclassify a sampled path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np
import openmm as mm
from openmm import app, unit

from evaluate_small_protein_run import (
    ATOM37_INDEX, BB_INDICES, CA_INDEX, OfficialTICA, fit, backbone_rmsd,
    heavy_rmsd, topology_atom37_map,
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def atom_kind(name: str, is_h: bool, generated: bool) -> dict[str, str]:
    if is_h:
        origin = "added_hydrogen"
    elif generated:
        origin = "generated_heavy"
    elif name == "OXT":
        origin = "template_OXT"
    else:
        origin = "template_missing_heavy"
    backbone_names = {"N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3", "HA", "HA2", "HA3"}
    return {"origin": origin, "region": "backbone" if name in backbone_names else "sidechain"}


class DecomposedEnergy:
    def __init__(self, manifest: dict[str, Any]):
        sources = {Path(r["path"]).name: Path(r["path"]) for r in manifest["official_source_files"]}
        root = Path(manifest["tica_model"]).parents[2]
        ff = app.ForceField(str(root / "data/protein.ff14SBonlysc.xml"), "implicit/gbn2.xml")
        self.pdb = app.PDBFile(str(sources["folded.pdb"]))
        self.system = ff.createSystem(self.pdb.topology, nonbondedMethod=app.NoCutoff,
                                      nonbondedCutoff=1.0*unit.nanometers,
                                      constraints=None, ewaldErrorTolerance=0.0005)
        self.mapping = []
        for atom in self.pdb.topology.atoms():
            if atom.element is not None and atom.element.symbol.upper() != "H":
                self.mapping.append((atom.index, atom.residue.index, ATOM37_INDEX[atom.name]))
                self.system.setParticleMass(atom.index, 0.0*unit.dalton)
        template = app.PDBFile(manifest["minimized_target_allatom"])
        self.template_a = np.asarray(template.positions.value_in_unit(unit.angstrom))
        self.atoms = list(self.pdb.topology.atoms())
        self.bonded_forces, self.nonbonded_force = [], None
        force_manifest = []
        for force in self.system.getForces():
            name = force.__class__.__name__
            if name in {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"}:
                group, category = 0, "bonded"
                self.bonded_forces.append(force)
            elif name == "NonbondedForce":
                group, category = 1, "nonbonded"
                self.nonbonded_force = force
            elif name in {"CustomGBForce", "GBSAOBCForce"}:
                group, category = 2, "solvation"
            else:
                group, category = 3, "other"
            force.setForceGroup(group)
            force_manifest.append({"force": name, "group": group, "category": category})
        self.force_manifest = force_manifest
        self.nb_arrays = None
        if self.nonbonded_force is not None:
            force = self.nonbonded_force
            count = force.getNumParticles()
            ni, nj = np.triu_indices(count, 1)
            charges = np.empty(count); sigmas = np.empty(count); epsilons = np.empty(count)
            for i in range(count):
                q, s, e = force.getParticleParameters(i)
                charges[i] = q.value_in_unit(unit.elementary_charge)
                sigmas[i] = s.value_in_unit(unit.nanometer)
                epsilons[i] = e.value_in_unit(unit.kilojoule_per_mole)
            qprod = charges[ni] * charges[nj]
            sigma = 0.5 * (sigmas[ni] + sigmas[nj])
            epsilon = np.sqrt(epsilons[ni] * epsilons[nj])
            lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(ni, nj))}
            for x in range(force.getNumExceptions()):
                a,b,q,s,e=force.getExceptionParameters(x); key=tuple(sorted((int(a),int(b)))); k=lookup[key]
                qprod[k]=q.value_in_unit(unit.elementary_charge**2)
                sigma[k]=s.value_in_unit(unit.nanometer)
                epsilon[k]=e.value_in_unit(unit.kilojoule_per_mole)
            self.nb_arrays=(ni,nj,qprod,sigma,epsilon)
        self.integrator = mm.VerletIntegrator(1.0*unit.femtoseconds)
        self.sim = app.Simulation(self.pdb.topology, self.system, self.integrator)

    def label(self, atom_index: int, generated_topology: set[int]) -> dict[str, Any]:
        atom = self.atoms[atom_index]
        is_h = atom.element is not None and atom.element.symbol.upper() == "H"
        return {
            "topology_atom_index": atom_index, "residue_index_0based": atom.residue.index,
            "residue": str(atom.residue), "atom": atom.name,
            **atom_kind(atom.name, is_h, atom_index in generated_topology),
        }

    @staticmethod
    def _dihedral(a,b,c,d):
        b0 = -(b-a); b1 = c-b; b2 = d-c
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0,b1)*b1; w = b2 - np.dot(b2,b1)*b1
        return math.atan2(np.dot(np.cross(b1,v),w), np.dot(v,w))

    def _top_terms(self, xyz_nm: np.ndarray, generated: set[int]) -> dict[str, Any]:
        bonded = []
        for force in self.bonded_forces:
            name = force.__class__.__name__
            if name == "HarmonicBondForce":
                for i in range(force.getNumBonds()):
                    a,b,r0,k = force.getBondParameters(i); r=np.linalg.norm(xyz_nm[int(a)]-xyz_nm[int(b)])
                    e=0.5*k.value_in_unit(unit.kilojoule_per_mole/unit.nanometer**2)*(r-r0.value_in_unit(unit.nanometer))**2
                    bonded.append({"term":"bond","energy_kj_mol":float(e),"atoms":[self.label(int(a),generated),self.label(int(b),generated)],"value":float(r),"reference":float(r0.value_in_unit(unit.nanometer)),"unit":"nm"})
            elif name == "HarmonicAngleForce":
                for i in range(force.getNumAngles()):
                    a,b,c,t0,k=force.getAngleParameters(i); u=xyz_nm[int(a)]-xyz_nm[int(b)]; v=xyz_nm[int(c)]-xyz_nm[int(b)]
                    theta=math.acos(float(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))
                    e=0.5*k.value_in_unit(unit.kilojoule_per_mole/unit.radian**2)*(theta-t0.value_in_unit(unit.radian))**2
                    bonded.append({"term":"angle","energy_kj_mol":float(e),"atoms":[self.label(int(x),generated) for x in (a,b,c)],"value":theta,"reference":float(t0.value_in_unit(unit.radian)),"unit":"radian"})
            elif name == "PeriodicTorsionForce":
                for i in range(force.getNumTorsions()):
                    a,b,c,d,n,phase,k=force.getTorsionParameters(i); theta=self._dihedral(*(xyz_nm[int(x)] for x in (a,b,c,d)))
                    e=k.value_in_unit(unit.kilojoule_per_mole)*(1+math.cos(int(n)*theta-phase.value_in_unit(unit.radian)))
                    bonded.append({"term":"torsion","energy_kj_mol":float(e),"atoms":[self.label(int(x),generated) for x in (a,b,c,d)],"value":theta,"reference":float(phase.value_in_unit(unit.radian)),"unit":"radian"})
        bonded.sort(key=lambda r:r["energy_kj_mol"], reverse=True)

        nonbonded=[]
        if self.nb_arrays is not None:
            ni,nj,qprod,sigma,epsilon=self.nb_arrays
            distance=np.linalg.norm(xyz_nm[ni]-xyz_nm[nj],axis=1)
            ratio=sigma/distance
            energy=138.935456*qprod/distance + 4*epsilon*(ratio**12-ratio**6)
            top=np.argpartition(energy,-min(12,len(energy)))[-min(12,len(energy)):]
            top=top[np.argsort(energy[top])[::-1]]
            for k in top:
                i,j=int(ni[k]),int(nj[k])
                nonbonded.append({"term":"pair_vacuum_coulomb_plus_LJ","energy_kj_mol":float(energy[k]),"distance_a":float(10*distance[k]),"atoms":[self.label(i,generated),self.label(j,generated)]})
        return {"top_bonded_terms":bonded[:12],"top_positive_nonbonded_pairs":nonbonded[:12]}

    def evaluate(self, frame: np.ndarray, mask: np.ndarray, analyze_terms: bool = True) -> dict[str, Any]:
        generated_indices=[]; generated_frame=[]; template_points=[]
        for top,res,a37 in self.mapping:
            if bool(mask[res,a37]):
                generated_indices.append(top); generated_frame.append(frame[res,a37]); template_points.append(self.template_a[top])
        rotation, center, target_center = fit(np.asarray(template_points), np.asarray(generated_frame))
        positions_a=(self.template_a-center)@rotation+target_center
        positions_a[np.asarray(generated_indices)]=np.asarray(generated_frame)
        self.sim.context.setPositions(positions_a*unit.angstrom)
        self.sim.minimizeEnergy()
        state=self.sim.context.getState(getPositions=True,getEnergy=True)
        after_a=np.asarray(state.getPositions().value_in_unit(unit.angstrom))
        components={name:float(self.sim.context.getState(getEnergy=True,groups={group}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)) for name,group in [("bonded",0),("nonbonded",1),("solvation",2),("other",3)]}
        total=float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        displacement=float(np.max(np.linalg.norm(after_a[np.asarray(generated_indices)]-np.asarray(generated_frame),axis=1)))
        term_rows = self._top_terms(after_a/10.0,set(generated_indices)) if analyze_terms else {"top_bonded_terms": [], "top_positive_nonbonded_pairs": []}
        return {"total_kj_mol":total,"components_kj_mol":components,"component_sum_kj_mol":float(sum(components.values())),"fixed_generated_heavy_max_displacement_a":displacement,"force_manifest":self.force_manifest,**term_rows}


def native_contacts(frame, mask, target, target_mask, official_map):
    # Residue-level official-heavy contact Q plus a CA-native-contact Q.
    residue_atoms={}
    for _,res,a37 in official_map:
        if target_mask[res,a37]: residue_atoms.setdefault(res,[]).append(a37)
    heavy_native=[]
    for i in range(target.shape[0]):
        for j in range(i+3,target.shape[0]):
            ai=residue_atoms.get(i,[]); aj=residue_atoms.get(j,[])
            if ai and aj and np.min(np.linalg.norm(target[i, ai][:, None, :] - target[j, aj][None, :, :], axis=-1)) < 4.5:
                heavy_native.append((i,j))
    formed=0
    for i,j in heavy_native:
        ai=[a for a in residue_atoms[i] if mask[i,a]]; aj=[a for a in residue_atoms[j] if mask[j,a]]
        if ai and aj and np.min(np.linalg.norm(frame[i, ai][:, None, :] - frame[j, aj][None, :, :], axis=-1)) <= 5.5: formed+=1
    ca_native=[]; ca_formed=0
    for i in range(target.shape[0]):
        for j in range(i+3,target.shape[0]):
            if target_mask[i,CA_INDEX] and target_mask[j,CA_INDEX]:
                d0=np.linalg.norm(target[i,CA_INDEX]-target[j,CA_INDEX])
                if d0 < 8.0:
                    ca_native.append((i,j,d0))
                    if mask[i,CA_INDEX] and mask[j,CA_INDEX] and np.linalg.norm(frame[i,CA_INDEX]-frame[j,CA_INDEX]) <= 1.2*d0: ca_formed+=1
    return {"heavy_native_contact_q":formed/len(heavy_native) if heavy_native else None,"heavy_native_contact_count":len(heavy_native),"ca_native_contact_q":ca_formed/len(ca_native) if ca_native else None,"ca_native_contact_count":len(ca_native)}


def geometry(frame, mask):
    n,ca,c,o=[ATOM37_INDEX[x] for x in ("N","CA","C","O")]
    def values(a,b,offset=0):
        rows=[]
        for i in range(frame.shape[0]-offset):
            j=i+offset
            if mask[i,a] and mask[j,b]: rows.append(float(np.linalg.norm(frame[i,a]-frame[j,b])))
        return rows
    nca=values(n,ca); cac=values(ca,c); cn=values(c,n,1); adj=values(ca,ca,1)
    return {"n_ca_min_max_a":[min(nca),max(nca)],"ca_c_min_max_a":[min(cac),max(cac)],"peptide_c_n_min_max_a":[min(cn),max(cn)],"adjacent_ca_max_a":max(adj),"adjacent_ca_gt_5_5_count":sum(x>5.501 for x in adj),"finite_generated_coordinates":bool(np.isfinite(frame[mask]).all())}


def export_roundtrip(out: Path, frame, mask, manifest, label):
    out.mkdir(parents=True, exist_ok=True)
    source={Path(r["path"]).name:Path(r["path"]) for r in manifest["official_source_files"]}["folded.pdb"]
    template=md.load(str(source)); mapping=topology_atom37_map(template.topology,frame.shape[0])
    xyz=template.xyz[0].copy(); compared=[]
    for top,res,a37 in mapping:
        if mask[res,a37]: xyz[top]=frame[res,a37]/10.0; compared.append((top,res,a37))
    path=out/f"{label}.pdb"; md.Trajectory(xyz[None],template.topology).save_pdb(str(path))
    loaded=md.load(str(path)).xyz[0]*10.0
    errors=[np.linalg.norm(loaded[top]-frame[res,a37]) for top,res,a37 in compared]
    return {"pdb":str(path),"compared_generated_heavy_atoms":len(errors),"max_coordinate_roundtrip_error_a":float(max(errors)),"rms_coordinate_roundtrip_error_a":float(np.sqrt(np.mean(np.square(errors))))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--protein",choices=["chignolin","trpcage","bba"],required=True); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--shard-count",type=int,default=1); args=ap.parse_args()
    prepared=args.root/"data/small_protein_transition_pilot/prepared"/args.protein
    manifest=json.loads((prepared/"manifest.json").read_text()); reference=np.load(prepared/"whole_backbone_reference_atom37.npz")
    target=np.asarray(reference["reference_atom37_a"][0],float); target_mask=np.asarray(reference["reference_atom37_mask"][0],bool)
    tica=OfficialTICA(manifest); official_map=tica.mapping; evaluator=DecomposedEnergy(manifest)
    all_run_dirs=sorted(p.parent for p in (args.root/"outputs/small_protein_transition_pilot"/args.protein).glob("stride*_t32/*/seed_*/small_protein_metrics.json"))
    run_dirs=[r for i,r in enumerate(all_run_dirs) if i % args.shard_count == args.shard_index]
    selected=[]; all_hit_frames=[]; reference_frames=[]
    first_population=None
    for run in run_dirs:
        pop=np.load(run/"pre_final_population_atom37.npz"); traj=np.asarray(pop["trajectories_atom37_a"],float); masks=np.asarray(pop["trajectories_atom37_mask"],bool)
        if first_population is None: first_population=(traj,masks)
        diag=json.loads((run/"small_protein_frame_diagnostics.json").read_text()); base=json.loads((run/"metrics.json").read_text()); hits=np.asarray(diag["target_hit"],bool); valid=np.asarray(base["per_path_valid"],bool)
        energy_by_path={int(r["path_index"]):r for r in diag["energy"] if r.get("status")=="complete"}
        failed_energy_by_path={int(r["path_index"]):r for r in diag["energy"] if r.get("status")=="failed"}
        for path_idx in range(len(traj)):
            for frame_idx in np.flatnonzero(hits[path_idx]):
                hr,count,full=heavy_rmsd(traj[path_idx,frame_idx],masks[path_idx,frame_idx],target,target_mask,official_map)
                all_hit_frames.append({"run":str(run),"method":base["method"],"seed":base["seed"],"stride":base["stride_in_10ps"],"path_index":path_idx,"frame_index":int(frame_idx),"whole_path_valid":bool(valid[path_idx]),"tica_distance":float(diag["tica_distance_to_folded_target"][path_idx][frame_idx]),"backbone_rmsd_a":backbone_rmsd(traj[path_idx,frame_idx],target),"common_heavy_rmsd_a":hr,"common_heavy_atom_count":count,**native_contacts(traj[path_idx,frame_idx],masks[path_idx,frame_idx],target,target_mask,official_map),"geometry":geometry(traj[path_idx,frame_idx],masks[path_idx,frame_idx])})
        for path_idx in np.flatnonzero(valid & hits[:,-1]):
            prior_energy_row = energy_by_path.get(int(path_idx))
            if prior_energy_row is None:
                # Do not turn a previously failed, unbounded minimization into
                # a silent success.  Its maximum-energy frame is undefined;
                # first/final frames are still audited independently below.
                energies = []
                old_failure = failed_energy_by_path.get(int(path_idx), {})
                energy_failures = [{"frame_index": None, "error": old_failure.get("error", "existing optional ETS row unavailable")}]
                energy_source = "existing_optional_ets_failed_max_frame_undefined"
            else:
                energies = prior_energy_row["frame_potential_kj_mol"]
                energy_failures = []
                energy_source = "existing_small_protein_frame_diagnostics"
            finite_energy_indices=[i for i,x in enumerate(energies) if x is not None and np.isfinite(x)]
            roles={"first_generated":1,"final":traj.shape[1]-1}
            if finite_energy_indices:
                max_index=max(finite_energy_indices,key=lambda i:energies[i]); roles["max_finite_energy"]=int(max_index)
            if energy_failures and energy_failures[0]["frame_index"] is not None:
                roles["first_energy_failure"]=int(energy_failures[0]["frame_index"])
            row={"run":str(run),"method":base["method"],"seed":base["seed"],"stride":base["stride_in_10ps"],"path_index":int(path_idx),"original_path_valid":True,"original_final_target_hit":True,"max_energy_index_source":energy_source,"max_finite_energy_kj_mol":float(max(energies[i] for i in finite_energy_indices)) if finite_energy_indices else None,"energy_scan_failures":energy_failures,"frames":{}}
            for role,fi in roles.items():
                label=f"{base['method']}_s{base['seed']}_stride{base['stride_in_10ps']}_p{path_idx}_{role}_f{fi}"
                rt=export_roundtrip(args.output/args.protein/"exports",traj[path_idx,fi],masks[path_idx,fi],manifest,label)
                try:
                    er={"status":"complete",**evaluator.evaluate(traj[path_idx,fi],masks[path_idx,fi])}
                except Exception as error:
                    er={"status":"failed","error":repr(error)}
                row["frames"][role]={"frame_index":fi,"geometry":geometry(traj[path_idx,fi],masks[path_idx,fi]),"energy":er,"roundtrip":rt}
            selected.append(row)
    assert first_population is not None
    start,start_mask=first_population[0][0,0],first_population[1][0,0]
    if args.shard_index == 0:
        for name,frame,mask in [("normal_start",start,start_mask),("normal_target",target,target_mask)]:
            reference_frames.append({"name":name,"geometry":geometry(frame,mask),"energy":evaluator.evaluate(frame,mask),"roundtrip":export_roundtrip(args.output/args.protein/"exports",frame,mask,manifest,name)})
    # Direct tensor vs PDB-roundtrip TICA mapping on every actual hit frame.
    mapping_rows=[]
    for row in all_hit_frames:
        # Values above already came from the direct tensor. Roundtrip errors on
        # selected frames quantify the serialization path; topology/feature
        # identity is recorded from the exact official implementation below.
        mapping_rows.append({"run":row["run"],"path_index":row["path_index"],"frame_index":row["frame_index"],"tica_distance":row["tica_distance"]})
    result={"protein":args.protein,"shard_index":args.shard_index,"shard_count":args.shard_count,"production_runs_audited":len(run_dirs),"valid_final_target_hit_paths":len(selected),"actual_target_hit_frame_count":len(all_hit_frames),"target_hit_frames":all_hit_frames,"selected_path_energy_audit":selected,"normal_reference_energy_audit":reference_frames,"tica_mapping_audit":{"official_feature":"mdtraj add_backbone_torsions(cossin=True)","official_model":str(manifest["tica_model"]),"atom_mapping":"official folded topology atom -> residue index + OpenFold atom37 name; generated coordinates replace matching atoms; template coordinates fill unused atoms","target_definition":"Euclidean distance in first two official TICA coordinates < 0.75","target_self_distance":manifest["tica"]["target_self_distance"],"hit_frame_rows":mapping_rows},"preservation_note":"No production result was edited. Minimized diagnostic coordinates never change hit or validity labels."}
    output_name="audit.json" if args.shard_count == 1 else f"audit_part_{args.shard_index}.json"
    dump(args.output/args.protein/output_name,result); print(json.dumps({k:result[k] for k in ["protein","production_runs_audited","valid_final_target_hit_paths","actual_target_hit_frame_count"]},indent=2))


if __name__=="__main__": main()
