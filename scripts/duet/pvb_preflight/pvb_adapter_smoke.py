#!/usr/bin/env python
"""Pre-verify a PVB SDE discretisation and exercise the DuET adapter on it.

Twenty bridge steps put 0.25/0.50/0.75/0.90 on exact boundaries, but changing
the discretisation away from the published ten has to be checked rather than
assumed: a coarser or finer solver can degrade the geometry the whole benchmark
depends on.  This runs the same checks at both settings so the choice is made on
evidence, and exercises every adapter method the outer sampler calls.

Exit status is 0 when the requested ``--sde-step`` passes, 3 when it does not,
so a caller can fall back to ten without reading the report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")


def geometry(frame) -> dict:
    from confmh.duet.observables import ATOM37_INDEX

    atom37 = np.asarray(frame.atom37_a, dtype=float)
    mask = np.asarray(frame.atom37_mask, dtype=bool)
    ca = atom37[:, ATOM37_INDEX["CA"], :]
    adjacent = np.linalg.norm(np.diff(ca, axis=0), axis=-1)
    carbon, nitrogen = ATOM37_INDEX["C"], ATOM37_INDEX["N"]
    lengths = [
        float(np.linalg.norm(atom37[i, carbon] - atom37[i + 1, nitrogen]))
        for i in range(len(atom37) - 1)
        if mask[i, carbon] and mask[i + 1, nitrogen]
    ]
    peptide = np.asarray(lengths) if lengths else np.zeros(0)
    return {
        "ca_adjacent_min_a": float(adjacent.min()),
        "ca_adjacent_max_a": float(adjacent.max()),
        "peptide_cn_min_a": float(peptide.min()) if len(peptide) else None,
        "peptide_cn_max_a": float(peptide.max()) if len(peptide) else None,
        "peptide_cn_violations": int(
            ((peptide < 1.0) | (peptide > 1.7)).sum()
        ) if len(peptide) else None,
    }


def exercise(adapter, particles: int, checkpoints, seed: int) -> dict:
    """Drive the adapter exactly the way the inner Feynman-Kac loop does."""
    from confmh.duet.resampling import systematic_resample

    rng = np.random.default_rng(seed)
    history = adapter.initial_history()
    state = adapter.prepare_history(history)
    seeds = [seed + i for i in range(particles)]
    particle_state = adapter.initialize_inner_particles(state, particles, seeds)

    record: dict = {"checkpoints": []}
    for index, progress in enumerate(checkpoints):
        particle_state = adapter.denoise_to_checkpoint(particle_state, progress)
        predicted = adapter.predict_clean(particle_state)
        if len(predicted) != particles:
            raise SystemExit("predict_clean returned the wrong particle count")
        record["checkpoints"].append(
            {
                "requested": float(progress),
                "step": particle_state.step,
                "realised": particle_state.step / adapter.sde_step,
                "predicted_clean_geometry": geometry(predicted[0]),
            }
        )
        # Collapse onto one ancestor, which is the hardest case for reseeding.
        ancestors = np.zeros(particles, dtype=int)
        particle_state = adapter.resample_particle_state(particle_state, ancestors)
        if index < len(checkpoints) - 1:
            particle_state = adapter.reseed_particle_state(
                particle_state, [seed + 1000 * (index + 1) + i for i in range(particles)]
            )

    particle_state = adapter.denoise_to_end(
        particle_state, [seed + 9000 + i for i in range(particles)]
    )
    frames = adapter.finalize_frames(particle_state)
    validity = adapter.validate_frames(frames)

    endpoints = np.stack([np.asarray(f.atom37_a) for f in frames])
    spread = [
        float(np.sqrt(((endpoints[i] - endpoints[0]) ** 2).sum(-1).mean()))
        for i in range(1, particles)
    ]
    record["final_geometry"] = geometry(frames[0])
    record["validity"] = validity[0]
    record["all_valid"] = bool(all(item["valid"] for item in validity))
    record["branch_rmsd_to_particle_0_a"] = spread
    record["branches_diverge"] = bool(spread and max(spread) > 1e-3)
    record["nfe"] = adapter.accounting.to_dict()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--sde-step", type=int, default=20)
    parser.add_argument("--baseline-sde-step", type=int, default=10)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from confmh.adapters.pvb_duet import PVBDuETAdapter

    checkpoints = [0.25, 0.50, 0.75, 0.90]
    report: dict = {"checkpoint_timings": checkpoints, "arms": {}}

    for label, steps in (
        ("candidate", args.sde_step),
        ("published_baseline", args.baseline_sde_step),
    ):
        adapter = PVBDuETAdapter(
            repository_path=args.repo,
            checkpoint=args.ckpt,
            initial_structure=args.pdb,
            device=args.device,
            sde_step=steps,
        )
        adapter.load_model()
        schedule = adapter.checkpoint_schedule(checkpoints)
        arm = {
            "sde_step": steps,
            "schedule": schedule,
            "all_timings_exact": all(row["exact"] for row in schedule),
            "residues": adapter._residue_count,
            "unmapped_atom_names": adapter.unmapped_atom_names,
        }
        arm.update(exercise(adapter, args.particles, checkpoints, args.seed))
        report["arms"][label] = arm

    candidate = report["arms"]["candidate"]
    baseline = report["arms"]["published_baseline"]
    report["verdict"] = {
        "timings_exact": candidate["all_timings_exact"],
        "geometry_valid": candidate["all_valid"],
        "branches_diverge": candidate["branches_diverge"],
        "baseline_geometry_valid": baseline["all_valid"],
        "peptide_cn_candidate": candidate["final_geometry"]["peptide_cn_violations"],
        "peptide_cn_baseline": baseline["final_geometry"]["peptide_cn_violations"],
    }
    passed = (
        candidate["all_timings_exact"]
        and candidate["all_valid"]
        and candidate["branches_diverge"]
    )
    report["verdict"]["candidate_accepted"] = bool(passed)

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
