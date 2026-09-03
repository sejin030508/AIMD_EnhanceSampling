#!/usr/bin/env python3
"""Extract reproducible protein-only conditioning frames from an ATLAS trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mdtraj as md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--start-output", required=True)
    parser.add_argument("--end-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    args = parser.parse_args()

    topology = Path(args.topology).resolve()
    trajectory = Path(args.trajectory).resolve()
    start_output = Path(args.start_output).resolve()
    end_output = Path(args.end_output).resolve()
    manifest_output = Path(args.manifest_output).resolve()

    with md.open(str(trajectory)) as handle:
        n_frames = len(handle)
    for frame_index in (args.start_frame, args.end_frame):
        if not 0 <= frame_index < n_frames:
            raise IndexError(f"Frame {frame_index} is outside [0, {n_frames})")

    topology_frame = md.load(str(topology))
    protein_atoms = topology_frame.topology.select("protein and chainid 0")
    if protein_atoms.size == 0:
        raise RuntimeError("No atoms matched 'protein and chainid 0'")

    outputs = []
    for frame_index, output in (
        (args.start_frame, start_output),
        (args.end_frame, end_output),
    ):
        frame = md.load_frame(str(trajectory), frame_index, top=str(topology))
        frame = frame.atom_slice(protein_atoms)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.save_pdb(str(output))
        outputs.append(str(output))

    protein_topology = topology_frame.atom_slice(protein_atoms).topology
    manifest = {
        "topology": str(topology),
        "trajectory": str(trajectory),
        "trajectory_frame_count": n_frames,
        "frame_indexing": "zero_based",
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "protein_selection": "protein and chainid 0",
        "protein_atoms": int(protein_atoms.size),
        "protein_residues": int(protein_topology.n_residues),
        "outputs": outputs,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
