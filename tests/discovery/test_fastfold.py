from __future__ import annotations

import json

import mdtraj as md
import numpy as np

from confmh.discovery.fastfold import (
    analyze_fastfold_run,
    prepare_fastfold_reference,
    project_pdb_to_tica,
)

from .conftest import write_backbone


def test_reference_only_tica_grid_and_coverage(tmp_path):
    topology = write_backbone(
        tmp_path / "topology.pdb",
        [(3.8 * i, 0.3 * np.sin(i), 0.2 * np.cos(i)) for i in range(10)],
    )
    base = md.load(str(topology))
    rng = np.random.default_rng(7)
    xyz = np.repeat(base.xyz, 60, axis=0)
    xyz += rng.normal(scale=0.015, size=xyz.shape)
    trajectory_path = tmp_path / "reference.xtc"
    md.Trajectory(xyz, base.topology).save_xtc(str(trajectory_path))
    model_path = prepare_fastfold_reference(
        topology_pdb=topology,
        trajectories=[trajectory_path],
        output_path=tmp_path / "reference_tica.npz",
        lag_frames=1,
        n_components=2,
        n_clusters=3,
        grid_bins=5,
    )
    model = np.load(model_path)
    assert model["low_energy_grid_mask"].any()
    projection = project_pdb_to_tica(topology, topology, model_path)
    assert projection.shape == (2,)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = {
        "model_call": 1,
        "candidate_pdb": str(topology),
        "strict_valid": True,
    }
    (run_dir / "candidates.jsonl").write_text(json.dumps(row) + "\n")
    summary_path = analyze_fastfold_run(
        run_dir=run_dir,
        start_pdb=topology,
        reference_model=model_path,
    )
    summary = json.loads(summary_path.read_text())
    assert summary["generated_occupancy_used_for_free_energy"] is False
    assert summary["time_unit"] == "model_calls (not ns)"
