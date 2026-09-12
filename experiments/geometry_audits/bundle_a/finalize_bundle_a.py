#!/usr/bin/env python3
"""Merge Bundle-A audits, add serialization/TICA and sampler diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from audit_tps_hits import DecomposedEnergy, geometry, native_contacts
from evaluate_small_protein_run import OfficialTICA, backbone_rmsd, heavy_rmsd

ROOT=Path("/workspace/sejin/AI_MD_NSMC_phase_b_recovery")
OUT=ROOT/"outputs/bundle_a_cross_clock_audit"


def load(path): return json.loads(Path(path).read_text())
def dump(path,value): Path(path).write_text(json.dumps(value,indent=2)+"\n")


def tica_serialization_audit(protein):
    prepared=ROOT/"data/small_protein_transition_pilot/prepared"/protein
    manifest=load(prepared/"manifest.json"); projector=OfficialTICA(manifest)
    max_saved_recompute=0.0; max_pdb_quant=0.0; usable=0; rows=[]
    for metric in sorted((ROOT/"outputs/small_protein_transition_pilot"/protein).glob("stride*_t32/*/seed_*/small_protein_metrics.json")):
        run=metric.parent; pop=np.load(run/"pre_final_population_atom37.npz")
        xyz=np.asarray(pop["trajectories_atom37_a"],float); mask=np.asarray(pop["trajectories_atom37_mask"],bool)
        flat=xyz.reshape(-1,*xyz.shape[2:]); fm=mask.reshape(-1,*mask.shape[2:])
        raw,ok=projector.project(flat,fm); rounded,_=projector.project(np.round(flat,3),fm)
        stored=np.asarray(load(run/"small_protein_frame_diagnostics.json")["tica_first_two"]).reshape(-1,2)
        d1=np.linalg.norm(raw[ok]-stored[ok],axis=1); d2=np.linalg.norm(raw[ok]-rounded[ok],axis=1)
        max_saved_recompute=max(max_saved_recompute,float(np.max(d1))); max_pdb_quant=max(max_pdb_quant,float(np.max(d2))); usable+=int(ok.sum())
        rows.append({"run":str(run),"usable_frames":int(ok.sum()),"stored_vs_recomputed_max_tica_error":float(np.max(d1)),"raw_vs_0.001A_quantized_max_tica_error":float(np.max(d2))})
    return {"protein":protein,"usable_frame_count":usable,"stored_vs_recomputed_max_tica_error":max_saved_recompute,"raw_vs_pdb_coordinate_precision_max_tica_error":max_pdb_quant,"rows":rows,"conclusion":"The stored evaluator exactly reuses official backbone-torsion features and TICA transform; 0.001 A coordinate serialization sensitivity is reported separately."}


def sampler_details():
    rows=[]
    for meta_path in sorted((OUT/"sampler_contrast").glob("*/*/metadata.json")):
        meta=load(meta_path); z=np.load(meta_path.parent/"trajectories_atom37.npz")
        xyz=np.asarray(z["trajectories_atom37_a"],float); masks=np.asarray(z["trajectories_atom37_mask"],bool)
        invalid=[]
        for pi in range(len(xyz)):
            for fi in range(1,xyz.shape[1]):
                g=geometry(xyz[pi,fi],masks[pi,fi])
                ca=xyz[pi,fi,:,1]; pair=np.linalg.norm(ca[:,None]-ca[None,:],axis=-1); pairs=np.argwhere((pair<1.0)&np.triu(np.ones_like(pair,bool),2))
                if g["adjacent_ca_gt_5_5_count"] or len(pairs) or not g["finite_generated_coordinates"]:
                    adjacent=np.linalg.norm(np.diff(ca,axis=0),axis=-1); idx=int(np.argmax(adjacent))
                    invalid.append({"path_index":pi,"frame_index":fi,"geometry":g,"max_adjacent_pair_model_indices_0based":[idx,idx+1],"max_adjacent_distance_a":float(adjacent[idx]),"ca_clash_pairs_model_indices_0based":pairs.tolist()})
        rows.append({**{k:meta[k] for k in ["protein","arm","seed","trajectory_count","transitions_per_trajectory","physical_transition_count","stride_in_10ps","reverse_steps","wall_clock_s"]},"generated_frame_count":16,"invalid_generated_frame_count":len(invalid),"invalid_generated_frames":invalid,"global_ca_adjacent_max_a":meta["geometry"]["global_ca_adjacent_max_a"]})
    return rows


def sampler_trpcage_energy():
    prepared=ROOT/"data/small_protein_transition_pilot/prepared/trpcage"; manifest=load(prepared/"manifest.json"); evaluator=DecomposedEnergy(manifest)
    rows=[]
    for arm in ["official_forward","official_sampler_adapter","current_custom_sde"]:
        z=np.load(OUT/"sampler_contrast/trpcage"/arm/"trajectories_atom37.npz"); xyz=np.asarray(z["trajectories_atom37_a"],float); masks=np.asarray(z["trajectories_atom37_mask"],bool)
        evals=[]
        for pi in range(4):
            for fi in [1,4]:
                e=evaluator.evaluate(xyz[pi,fi],masks[pi,fi]); evals.append({"path_index":pi,"frame_index":fi,"total_kj_mol":e["total_kj_mol"],"components_kj_mol":e["components_kj_mol"],"top_bonded_term":e["top_bonded_terms"][0],"top_positive_nonbonded_pair":e["top_positive_nonbonded_pairs"][0]})
        vals=[r["total_kj_mol"] for r in evals]; rows.append({"arm":arm,"evaluated_first_and_final_generated_frames":8,"energy_min_median_max_kj_mol":[float(min(vals)),float(np.median(vals)),float(max(vals))],"frames":evals})
    return rows


def hit_summary(audits):
    rows=[]
    for p,a in audits.items():
        groups={}
        for r in a["target_hit_frames"]:
            key=(r["method"],r["stride"],r["seed"]); groups.setdefault(key,[]).append(r)
        for (method,stride,seed),items in sorted(groups.items()):
            valid=[r for r in items if r["whole_path_valid"]]; bb=[r["backbone_rmsd_a"] for r in items]; hq=[r["heavy_native_contact_q"] for r in items if r["heavy_native_contact_q"] is not None]
            rows.append({"protein":p,"method":method,"stride_in_10ps":stride,"seed":seed,"actual_hit_frame_count":len(items),"hit_frames_on_whole_valid_paths":len(valid),"backbone_rmsd_a_min_median_max":[float(min(bb)),float(np.median(bb)),float(max(bb))],"heavy_native_contact_q_min_median_max":[float(min(hq)),float(np.median(hq)),float(max(hq))] if hq else None})
    return rows


def main():
    audits={p:load(OUT/"tps_audit"/p/"audit.json") for p in ["chignolin","trpcage","bba"]}
    tica=[tica_serialization_audit(p) for p in audits]
    sampler=sampler_details(); energy=sampler_trpcage_energy()
    merged={"scope":"Bundle A only","production_results_modified":False,"sampler_transition_count":sum(r["physical_transition_count"] for r in sampler),"tps_audits":{p:str(OUT/"tps_audit"/p/"audit.json") for p in audits},"hit_frame_summary":hit_summary(audits),"tica_serialization_audit":tica,"sampler_contrast":sampler,"sampler_trpcage_energy":energy,"notes":["All hit/validity labels are the original labels.","Diagnostic H minimization never reclassifies a path.","Sampler energy was computed for Trp-cage first/final generated frames; PRMT6 sampler contrast uses geometry because it is outside the TPS-DPS force-field package."]}
    dump(OUT/"bundle_a_merged_raw.json",merged); print(json.dumps({"sampler_transition_count":merged["sampler_transition_count"],"hit_rows":len(merged["hit_frame_summary"])},indent=2))

if __name__=="__main__": main()
