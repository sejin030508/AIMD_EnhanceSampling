from pathlib import Path

import mdtraj as md
import numpy as np

from confmh.pca_cv import PCACV, fit_reference_pca, project_reference_trajectories


def _make_topology(n_residues: int = 5):
    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_residues):
        residue = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", md.element.carbon, residue)
    return top


def test_fit_reference_pca(tmp_path: Path):
    top = _make_topology()
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 3)).astype(np.float32)
    xyz = np.stack([base + 0.05 * rng.normal(size=base.shape) for _ in range(30)]).astype(np.float32)
    traj = md.Trajectory(xyz, top)
    pdb = tmp_path / "topology.pdb"
    xtc = tmp_path / "rep.xtc"
    traj[0].save_pdb(str(pdb))
    traj.save_xtc(str(xtc))
    out = tmp_path / "reference"
    model_path = fit_reference_pca(
        topology_pdb=pdb,
        trajectories=[xtc],
        output_dir=out,
        protein_selection="protein and chainid 0",
        n_seed_frames=3,
    )
    model = PCACV.load(model_path)
    scores = model.project_files(pdb, xtc)
    assert scores.shape == (30, 2)
    assert (out / "reference_cv.npz").exists()
    assert len(list((out / "seed_frames").glob("*.pdb"))) == 3

    matched = project_reference_trajectories(
        pca_model=model_path,
        topology_pdb=pdb,
        trajectories=[xtc],
        output_path=out / "reference_cv_stride4.npz",
        protein_selection="protein and chainid 0",
        stride=4,
        burn_in_frames=1,
    )
    projected = np.load(matched)
    assert projected["cv"].shape == (7, 2)
    assert projected["source_frame"].tolist() == [4, 8, 12, 16, 20, 24, 28]
