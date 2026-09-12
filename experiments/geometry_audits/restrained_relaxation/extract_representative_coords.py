#!/usr/bin/env python3
"""Extract raw/relaxed N/CA/C/CB coordinates for listed cases; no minimization."""
import json
from pathlib import Path

ROOT = Path("visual_examples/restrained_relaxation_paths_2026-09-12")
CASES = [
    ("trpcage", "trpcage_duet_seed211_p3", 0, 5, "GLN"),
    ("trpcage", "trpcage_duet_seed211_p3", 0, 6, "TRP"),
    ("trpcage", "trpcage_duet_seed211_p3", 0, 9, "ASP"),
    ("bba", "bba_duet_seed223_p1", 0, 1, "GLU"),
    ("bba", "bba_duet_seed223_p1", 0, 3, "TYR"),
    ("bba", "bba_duet_seed223_p1", 0, 6, "LYS"),
    ("prmt6", "prmt6_official_forward_p3", 1, 1, "ASP"),
    ("prmt6", "prmt6_official_forward_p3", 1, 3, "SER"),
    ("prmt6", "prmt6_official_forward_p3", 1, 6, "GLU"),
]


def read_frame(path, frame, resseq, resname):
    models, current = [], {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("MODEL"):
                current = {}
            elif line.startswith("ENDMDL"):
                models.append(current)
            elif line.startswith(("ATOM  ", "HETATM")):
                name = line[12:16].strip().upper()
                rn = line[17:20].strip()
                try: rs = int(line[22:26])
                except ValueError: continue
                if rs == resseq and rn == resname and name in {"N", "CA", "C", "CB"}:
                    current[name] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    return models[frame]


out=[]
for system, stem, frame, resseq, resname in CASES:
    row={"system":system,"path":stem,"frame_index":frame,"residue_number":resseq,"residue_name":resname}
    for side in ("raw", "relaxed"):
        p=ROOT/system/f"{stem}_{side}_path.pdb"
        row[f"{side}_coordinates_A"] = read_frame(p, frame, resseq, resname)
    out.append(row)
Path("reports/chirality_recheck_2026-09-12_v2/representative_coordinates.json").write_text(json.dumps(out,indent=2)+"\n")
