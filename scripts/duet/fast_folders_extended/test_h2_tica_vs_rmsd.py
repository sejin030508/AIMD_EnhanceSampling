#!/usr/bin/env python
"""H2: does RMSD steering move the population away from the TICA target?

Hit counts are a poor test here because almost every cell scores zero, so the
comparison is made on the continuous quantity behind the hit -- the distance in
the first two TICA coordinates from the folded target -- alongside the RMSD the
reward actually optimises.

If steering helps, both should fall together.  If the reward and the target
disagree, RMSD falls while TICA distance does not, and that is the claim under
test.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

import numpy as np


def load_cell(path: Path) -> dict | None:
    diagnostics = path / "small_protein_frame_diagnostics.json"
    metrics = path / "small_protein_metrics.json"
    if not diagnostics.exists() or not metrics.exists():
        return None
    d = json.loads(diagnostics.read_text(encoding="utf-8"))
    m = json.loads(metrics.read_text(encoding="utf-8"))
    # Diagnostics are (paths, frames) arrays keyed by quantity, not per-path dicts.
    tica = np.asarray(d["tica_distance_to_folded_target"], dtype=float)
    rmsd = np.asarray(d["whole_backbone_rmsd_a"], dtype=float)
    finals = [float(v) for v in tica[:, -1] if np.isfinite(v)]
    rmsds = [float(v) for v in rmsd[:, -1] if np.isfinite(v)]
    hits = int(sum(1 for row in tica if np.nanmin(row) < 0.75))
    if not finals:
        return None
    return {
        "protein": m["protein"],
        "method": m["method"],
        "seed": m["seed"],
        "paths": len(finals),
        "tica_final_mean": st.mean(finals),
        "tica_final_min": min(finals),
        "rmsd_final_mean": st.mean(rmsds) if rmsds else None,
        "anytime_hits": hits,
        "hit_rate_per_path": hits / len(finals),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timing", default="cp7590")
    args = parser.parse_args()

    cells = []
    for run in sorted(glob.glob(str(args.root / args.timing / "*/stride128_t32/*/seed_*"))):
        record = load_cell(Path(run))
        if record:
            cells.append(record)

    header = (
        f"{'protein':9s} {'method':16s} {'paths':>6s} {'TICAd_mean':>11s} "
        f"{'TICAd_min':>10s} {'RMSD':>7s} {'hits':>5s} {'hit/path':>9s}"
    )
    print(f"timing {args.timing}, means over seeds\n")
    print(header)
    print("-" * len(header))
    for protein in ("trpcage", "bba", "bbl"):
        for method in ("frozen", "complete_nested", "duet"):
            group = [c for c in cells if c["protein"] == protein and c["method"] == method]
            if not group:
                continue
            mean = lambda k: st.mean([c[k] for c in group if c[k] is not None])
            print(
                f"{protein:9s} {method:16s} {group[0]['paths']:6d} "
                f"{mean('tica_final_mean'):11.3f} {mean('tica_final_min'):10.3f} "
                f"{mean('rmsd_final_mean'):7.2f} "
                f"{sum(c['anytime_hits'] for c in group):5d} "
                f"{mean('hit_rate_per_path'):9.4f}"
            )
        print()
    (args.root / f"h2_tica_vs_rmsd_{args.timing}.json").write_text(
        json.dumps(cells, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
