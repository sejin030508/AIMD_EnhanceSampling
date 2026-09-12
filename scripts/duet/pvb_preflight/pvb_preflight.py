#!/usr/bin/env python
"""PVB preflight: does the published checkpoint connect to a DuET-style loop?

Three questions, in order of what would stop the project:

1. Does the official rollout run on our own input structure at all, and is the
   geometry it produces usable?
2. Can the inner SDE loop be split into initialize / advance-to-checkpoint /
   resample+reseed / advance-to-end without changing the result?  This is
   checked by running the split loop with resampling disabled and requiring it
   to reproduce the official loop bit-for-bit under the same seed.
3. Is the clean endpoint recoverable at a checkpoint?  The drift head is
   trained against (x1 - xt) / (1 - t), so x1_hat = xt + (1 - t) * drift should
   converge to the realised endpoint as t grows.

Nothing here is a DuET run: no reward, no selection, no outer filter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")


def geometry_report(coords_a: np.ndarray, topology) -> dict:
    """Backbone geometry of one frame, in Angstrom."""
    import mdtraj as md

    top = topology
    ca = [a.index for a in top.atoms if a.name == "CA"]
    report: dict = {"n_atoms": int(coords_a.shape[0]), "n_ca": len(ca)}
    if len(ca) > 1:
        adjacent = np.linalg.norm(np.diff(coords_a[ca], axis=0), axis=-1)
        report["ca_adjacent_min_a"] = float(adjacent.min())
        report["ca_adjacent_max_a"] = float(adjacent.max())
        report["ca_adjacent_mean_a"] = float(adjacent.mean())
        report["ca_adjacent_gt_4p5a"] = int((adjacent > 4.5).sum())

    # peptide C-N: the bond Bundle A found broken in every ConfRover frame.
    residues = list(top.residues)
    lengths = []
    for left, right in zip(residues[:-1], residues[1:]):
        c = next((a.index for a in left.atoms if a.name == "C"), None)
        n = next((a.index for a in right.atoms if a.name == "N"), None)
        if c is not None and n is not None:
            lengths.append(float(np.linalg.norm(coords_a[c] - coords_a[n])))
    if lengths:
        lengths_array = np.asarray(lengths)
        report["peptide_cn_min_a"] = float(lengths_array.min())
        report["peptide_cn_max_a"] = float(lengths_array.max())
        report["peptide_cn_violations"] = int(
            ((lengths_array < 1.0) | (lengths_array > 1.7)).sum()
        )
        report["peptide_cn_count"] = len(lengths)

    if len(ca) > 2:
        pairwise = np.linalg.norm(
            coords_a[ca][:, None, :] - coords_a[ca][None, :, :], axis=-1
        )
        nonneighbor = np.triu(np.ones(pairwise.shape, dtype=bool), k=2)
        report["ca_clash_lt_3a"] = int(((pairwise < 3.0) & nonneighbor).sum())
    report["nonfinite"] = int((~np.isfinite(coords_a)).sum())
    return report


def split_inference(model, batch, sde_step: int, checkpoint_step: int | None,
                    ancestors=None, collect_x1_hat: bool = True):
    """The official inference loop, split so it can be paused and branched.

    Mirrors ``dyVAE.inference`` operation for operation.  With ``ancestors``
    left as None and no reseeding, it must reproduce that method exactly.
    """
    import torch
    from module.graph import construct_edges
    from module.interpolant_matcher import INTERP_MATCHER  # noqa: F401

    z, b, x = batch["atype"], batch["btype"], batch["x0"]
    abid, edge_mask, bond_index = batch["abid"], batch["edge_mask"], batch["bond_index"]
    n_atoms = x.shape[0]

    e_edge_index, e_edge_weight, e_edge_vec, bond_type = construct_edges(
        Z=z, X=x, bid=abid, mask=edge_mask, bond_index=bond_index,
        cutoff_lower=model.cutoff_lower, cutoff_upper=model.cutoff_upper,
        cutoff_H=model.cutoff_H, k_neighbors=model.k_neighbors,
    )
    x_rep, _ = model.encode(
        z, b, x, x, abid, e_edge_index, e_edge_weight, e_edge_vec, bond_type,
        torch.ones_like(z).bool(),
    )

    xt = x_rep.clone()
    grid = np.linspace(0, 1.0 - 1.0 / sde_step, sde_step)
    dt = grid[1] - grid[0]
    ones = torch.ones(n_atoms, 1, dtype=x.dtype, device=x.device)

    x1_hat_trace = []
    for index, t in enumerate(grid):
        if checkpoint_step is not None and index == checkpoint_step and ancestors is not None:
            # Branch: carry x_rep with xt, exactly as the outer filter would.
            order = torch.as_tensor(ancestors, dtype=torch.long, device=xt.device)
            xt = xt[order].clone()
            x_rep = x_rep[order].clone()

        t_diff = ones * t
        d_edge_index, d_edge_weight_t, d_edge_vec_t, bond_type = construct_edges(
            Z=z, X=xt, bid=abid, mask=edge_mask, bond_index=bond_index,
            cutoff_lower=model.cutoff_lower, cutoff_upper=model.cutoff_upper,
            cutoff_H=model.cutoff_H, k_neighbors=model.k_neighbors,
        )
        d_edge_atm_0 = x_rep[d_edge_index.transpose(0, 1)]
        d_edge_vec_0 = d_edge_atm_0[:, 0] - d_edge_atm_0[:, 1]
        d_edge_weight_0 = torch.norm(d_edge_vec_0, dim=-1)

        vel_pred, drf_pred = model.decode(
            z, b, xt, t_diff, abid, d_edge_index, d_edge_weight_0, d_edge_vec_0,
            d_edge_weight_t, d_edge_vec_t, bond_type,
        )
        if collect_x1_hat:
            # Drift head target is (x1 - xt) / (1 - t); invert it.
            x1_hat_trace.append((float(t), (xt + (1.0 - t) * drf_pred).clone()))

        diffusion = model.sigma
        if t != grid[-1]:
            xt = xt + drf_pred * dt + diffusion * np.sqrt(dt) * torch.randn_like(xt)
        else:
            xt = xt + drf_pred * dt
    return xt, x_rep, x1_hat_trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--sde-step", type=int, default=10)
    parser.add_argument("--rollout", type=int, default=8)
    parser.add_argument("--checkpoint-step", type=int, default=7)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    import torch
    import mdtraj as md
    from data import make_batch

    device = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.to(device).eval()

    state0 = md.load(args.pdb)
    batch, (atom_index,) = make_batch(args.pdb, args.particles)
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    topology = state0.atom_slice(atom_index).topology

    result: dict = {
        "checkpoint": args.ckpt,
        "input_pdb": args.pdb,
        "input_atoms_total": int(state0.n_atoms),
        "atoms_kept_by_pvb": int(len(atom_index)),
        "residues_kept": int(topology.n_residues),
        "particles": args.particles,
        "sde_step": args.sde_step,
        "sigma": float(model.sigma),
        "using_ode": bool(model.using_ode),
    }

    # ---- Q1: official rollout ----------------------------------------------
    x0_backup = batch["x0"].clone()
    torch.manual_seed(args.seed)
    started = time.time()
    frames = []
    with torch.no_grad():
        for _ in range(args.rollout):
            x = model.inference(batch, sde_step=args.sde_step)
            frames.append(x.detach().cpu().numpy())
            batch["x0"] = x
    result["official_rollout_seconds"] = round(time.time() - started, 2)
    result["official_rollout_frames"] = len(frames)
    result["frame_shape"] = list(frames[0].shape)
    per_atom = int(frames[0].shape[0] // args.particles)
    result["geometry_first_frame"] = geometry_report(frames[0][:per_atom], topology)
    result["geometry_last_frame"] = geometry_report(frames[-1][:per_atom], topology)
    displacement = np.linalg.norm(frames[-1][:per_atom] - x0_backup.cpu().numpy()[:per_atom], axis=-1)
    result["last_frame_displacement_a"] = {
        "mean": float(displacement.mean()), "max": float(displacement.max())
    }

    # ---- Q2: is the split loop faithful? -----------------------------------
    batch["x0"] = x0_backup.clone()
    with torch.no_grad():
        torch.manual_seed(args.seed)
        official = model.inference(batch, sde_step=args.sde_step)
        torch.manual_seed(args.seed)
        split, _, trace = split_inference(
            model, batch, args.sde_step, checkpoint_step=None, ancestors=None
        )
    difference = (official - split).abs().max().item()
    result["split_loop_max_abs_difference"] = difference
    result["split_loop_reproduces_official"] = bool(difference < 1e-5)

    # ---- Q3: clean endpoint from the drift head ----------------------------
    endpoint = split.detach().cpu().numpy()
    result["x1_hat_vs_realised_endpoint_rmsd_a"] = [
        {
            "t": round(t, 3),
            "rmsd_a": float(
                np.sqrt(((value.detach().cpu().numpy() - endpoint) ** 2).sum(-1).mean())
            ),
        }
        for t, value in trace
    ]

    # ---- Q4: branch at a checkpoint ----------------------------------------
    ancestors = np.zeros(batch["x0"].shape[0], dtype=int)
    stride = batch["x0"].shape[0] // args.particles
    for particle in range(args.particles):
        ancestors[particle * stride:(particle + 1) * stride] = np.arange(stride)
    with torch.no_grad():
        torch.manual_seed(args.seed)
        branched, _, _ = split_inference(
            model, batch, args.sde_step, checkpoint_step=args.checkpoint_step,
            ancestors=ancestors, collect_x1_hat=False,
        )
    branched_np = branched.detach().cpu().numpy().reshape(args.particles, -1, 3)
    spread = [
        float(np.sqrt(((branched_np[i] - branched_np[0]) ** 2).sum(-1).mean()))
        for i in range(1, args.particles)
    ]
    result["branch_test"] = {
        "checkpoint_step": args.checkpoint_step,
        "all_particles_copied_from_particle_0": True,
        "rmsd_to_particle_0_after_resume_a": spread,
        "branches_diverge": bool(spread and max(spread) > 1e-3),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
