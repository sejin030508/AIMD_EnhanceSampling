#!/usr/bin/env python
"""Export the best-observed final-RMSD paths per protein as multi-model PDB
trajectories for local PyMOL inspection, alongside start/target references.

Selection: pool per_path_valid==True candidates across all main-matrix cells
(4 timings x {frozen,complete_nested,duet}), rank by per_path_final_d_a
ascending, keep at most one path per cell (avoid picking resampled siblings),
take the best 5.
"""
import json, glob, sys
from pathlib import Path
import numpy as np
import mdtraj as md

ROOT = Path('/workspace/sejin/AI_MD_NSMC_phase_b_recovery')
OUT = ROOT / 'outputs' / 'pymol_best_paths'
PROTEINS = ['trpcage', 'bba', 'bbl']

ATOM37_NAMES = [
    'N','CA','C','CB','O','CG','CG1','CG2','OG','OG1','SG','CD','CD1','CD2',
    'ND1','ND2','OD1','OD2','SD','CE','CE1','CE2','CE3','NE','NE1','NE2',
    'OE1','OE2','CH2','NH1','NH2','OH','CZ','CZ2','CZ3','NZ','OXT',
]
ATOM37_INDEX = {name: i for i, name in enumerate(ATOM37_NAMES)}


def topology_atom37_map(topology, residue_count):
    mapping = []
    residues = list(topology.residues)
    if len(residues) != residue_count:
        raise ValueError(f'topology residue count {len(residues)} != {residue_count}')
    for atom in topology.atoms:
        if atom.element is None or atom.element.symbol.upper() == 'H':
            continue
        if atom.name not in ATOM37_INDEX:
            raise ValueError(f'unsupported official heavy atom {atom.residue}:{atom.name}')
        mapping.append((atom.index, atom.residue.index, ATOM37_INDEX[atom.name]))
    return mapping


def select_best(protein):
    cand = []
    for f in glob.glob(str(ROOT / 'outputs/pvb_full/*' / protein / 'stride128_t32/*/seed_*/metrics.json')):
        tag = f.split('pvb_full/')[1].split('/')[0]
        if tag in ('k2m32', 'k32m2', 'k1m64'):
            continue
        m = json.load(open(f))
        for i, (v, d) in enumerate(zip(m['per_path_valid'], m['per_path_final_d_a'])):
            if v:
                cand.append((d, tag, m['method'], m['seed'], i, Path(f).parent))
    cand.sort(key=lambda c: c[0])
    seen = set()
    picked = []
    for c in cand:
        key = (c[1], c[2], c[3])
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
        if len(picked) == 5:
            break
    return picked


def main():
    for protein in PROTEINS:
        prepared = ROOT / 'data/small_protein_transition_pilot/prepared' / protein
        manifest = json.loads((prepared / 'manifest.json').read_text())
        src = {Path(r['path']).name: Path(r['path']) for r in manifest['official_source_files']}
        folded_pdb = src['folded.pdb']
        unfolded_pdb = src.get('unfolded.pdb')
        template = md.load(str(folded_pdb))
        topology = template.topology
        mapping = topology_atom37_map(topology, int(manifest['model_length']))
        template_xyz_nm = template.xyz[0].copy()

        out_dir = OUT / protein
        out_dir.mkdir(parents=True, exist_ok=True)

        # References -----------------------------------------------------
        template.save_pdb(str(out_dir / 'ref_target_folded.pdb'))
        if unfolded_pdb and unfolded_pdb.exists():
            md.load(str(unfolded_pdb)).save_pdb(str(out_dir / 'ref_unfolded_official.pdb'))
        start_struct = prepared / 'unfolded_minimized_allatom.pdb'
        if start_struct.exists():
            md.load(str(start_struct)).save_pdb(str(out_dir / 'ref_start_minimized.pdb'))

        picked = select_best(protein)
        summary = []
        for rank, (dist, tag, method, seed, path_idx, cell_dir) in enumerate(picked, start=1):
            npz = np.load(cell_dir / 'pre_final_population_atom37.npz')
            traj_a = npz['trajectories_atom37_a'][path_idx]      # (33, 20, 37, 3) Angstrom
            mask = npz['trajectories_atom37_mask'][path_idx]     # (33, 20, 37)
            n_frames = traj_a.shape[0]
            xyz = np.repeat(template_xyz_nm[None], n_frames, axis=0)  # (33, natoms, 3) nm
            for topo_atom, residue, atom37 in mapping:
                avail = mask[:, residue, atom37]
                xyz[avail, topo_atom] = traj_a[avail, residue, atom37] / 10.0
            traj = md.Trajectory(xyz, topology)
            name = f'rank{rank}_{tag}_{method}_seed{seed}_path{path_idx}_rmsd{dist:.2f}A.pdb'
            traj.save_pdb(str(out_dir / name))
            summary.append(dict(rank=rank, file=name, timing=tag, method=method, seed=seed,
                                 path_index=path_idx, final_backbone_rmsd_a=round(float(dist), 3),
                                 n_frames=n_frames))
            print(f'{protein} rank{rank}: {name}')

        (out_dir / 'selection_summary.json').write_text(json.dumps(summary, indent=1))

        # Simple PyMOL session script -------------------------------------
        pml = [
            f'load {out_dir}/ref_start_minimized.pdb, start',
            f'load {out_dir}/ref_target_folded.pdb, target',
            'hide everything',
            'show cartoon, start or target',
            'color grey70, start',
            'color marine, target',
        ]
        colors = ['red', 'orange', 'yellow', 'green', 'purple']
        for rank, item in enumerate(summary, start=1):
            obj = f'best{rank}'
            pml.append(f'load {out_dir}/{item["file"]}, {obj}')
            pml.append(f'show cartoon, {obj}')
            pml.append(f'color {colors[rank-1]}, {obj}')
            pml.append(f'set all_states, 0, {obj}')
        pml.append('set cartoon_transparency, 0.0')
        pml.append('bg_color white')
        pml.append('orient target')
        (out_dir / 'load_in_pymol.pml').write_text('\n'.join(pml) + '\n')

    print('DONE')


if __name__ == '__main__':
    main()
