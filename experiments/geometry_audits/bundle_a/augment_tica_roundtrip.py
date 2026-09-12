#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from evaluate_small_protein_run import OfficialTICA
from audit_tps_hits import geometry

root=Path("/workspace/sejin/AI_MD_NSMC_phase_b_recovery")
merged_path=root/"outputs/bundle_a_cross_clock_audit/bundle_a_merged_raw.json"
merged=json.loads(merged_path.read_text())
by_protein={row["protein"]:row for row in merged["tica_serialization_audit"]}
for protein,row in by_protein.items():
    prepared=root/"data/small_protein_transition_pilot/prepared"/protein
    manifest=json.loads((prepared/"manifest.json").read_text()); projector=OfficialTICA(manifest)
    target=np.asarray(manifest["tica"]["target_first_two"],float)
    changed=0; total=0; min_margin=float("inf")
    for metric in sorted((root/"outputs/small_protein_transition_pilot"/protein).glob("stride*_t32/*/seed_*/small_protein_metrics.json")):
        run=metric.parent; pop=np.load(run/"pre_final_population_atom37.npz")
        xyz=np.asarray(pop["trajectories_atom37_a"],float); mask=np.asarray(pop["trajectories_atom37_mask"],bool)
        flat=xyz.reshape(-1,*xyz.shape[2:]); fm=mask.reshape(-1,*mask.shape[2:])
        raw,ok=projector.project(flat,fm); rounded,_=projector.project(np.round(flat,3),fm)
        dr=np.linalg.norm(raw[ok]-target,axis=1); dq=np.linalg.norm(rounded[ok]-target,axis=1)
        changed += int(np.sum((dr<0.75)!=(dq<0.75))); total += len(dr)
        min_margin=min(min_margin,float(np.min(np.abs(dr-0.75))))
    row["target_hit_label_changes_after_0.001A_quantization"]=changed
    row["classification_frame_count"]=total
    row["minimum_raw_distance_margin_to_0.75_cutoff"]=min_margin

for row in merged["sampler_contrast"]:
    path=root/"outputs/bundle_a_cross_clock_audit/sampler_contrast"/row["protein"]/row["arm"]/"trajectories_atom37.npz"
    z=np.load(path); xyz=np.asarray(z["trajectories_atom37_a"],float); masks=np.asarray(z["trajectories_atom37_mask"],bool)
    geoms=[geometry(xyz[i,j],masks[i,j]) for i in range(4) for j in range(1,5)]
    minimum=[g["peptide_c_n_min_max_a"][0] for g in geoms]; maximum=[g["peptide_c_n_min_max_a"][1] for g in geoms]
    row["gross_peptide_cn_frame_count_lt1A_or_gt1_7A"]=sum(a<1.0 or b>1.7 for a,b in zip(minimum,maximum))
    row["peptide_cn_global_min_max_a"]=[float(min(minimum)),float(max(maximum))]
    row["peptide_cn_median_frame_min_max_a"]=[float(np.median(minimum)),float(np.median(maximum))]
merged_path.write_text(json.dumps(merged,indent=2)+"\n")
print(json.dumps([{k:r[k] for k in ["protein","target_hit_label_changes_after_0.001A_quantization","classification_frame_count","minimum_raw_distance_margin_to_0.75_cutoff"]} for r in merged["tica_serialization_audit"]],indent=2))
