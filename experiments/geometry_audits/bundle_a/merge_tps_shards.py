#!/usr/bin/env python3
import json
from pathlib import Path

root=Path("/workspace/sejin/AI_MD_NSMC_phase_b_recovery/outputs/bundle_a_cross_clock_audit/tps_audit/bba")
parts=[json.loads((root/f"audit_part_{i}.json").read_text()) for i in range(3)]
out=dict(parts[0])
out.pop("shard_index",None); out["shard_count"]=3
for key in ["production_runs_audited","valid_final_target_hit_paths","actual_target_hit_frame_count"]:
    out[key]=sum(p[key] for p in parts)
for key in ["target_hit_frames","selected_path_energy_audit"]:
    out[key]=sum((p[key] for p in parts),[])
out["normal_reference_energy_audit"]=sum((p["normal_reference_energy_audit"] for p in parts),[])
out["tica_mapping_audit"]["hit_frame_rows"]=sum((p["tica_mapping_audit"]["hit_frame_rows"] for p in parts),[])
(root/"audit.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:out[k] for k in ["protein","production_runs_audited","valid_final_target_hit_paths","actual_target_hit_frame_count"]},indent=2))
