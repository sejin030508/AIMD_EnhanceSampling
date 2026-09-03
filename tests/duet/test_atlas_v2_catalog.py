from pathlib import Path

import numpy as np
import yaml

from confmh.duet import atlas_programs


def test_continuous_contact_targets_do_not_overwrite_terminal_pc_interval(tmp_path, monkeypatch):
    class FakePCA:
        def project_trajectory(self, trajectory):
            return np.linspace(-1.0, 2.0, 20)[:, None]

        def project_files(self, path):
            return np.asarray([[-1.0]])

    class FakeTrajectory:
        xyz = np.zeros((20, 4, 3))

    monkeypatch.setattr(atlas_programs.PCACV, "load", lambda path: FakePCA())
    monkeypatch.setattr(
        atlas_programs, "_load_initial_ca", lambda path: np.zeros((4, 3), dtype=float)
    )
    monkeypatch.setattr(
        atlas_programs,
        "_load_design_trajectories",
        lambda *args, **kwargs: (None, np.arange(4), [FakeTrajectory(), FakeTrajectory()]),
    )
    monkeypatch.setattr(
        atlas_programs,
        "_select_ordered_contacts",
        lambda *args, **kwargs: (
            [
                {"residue_i": 0, "residue_j": 2, "threshold_nm": 0.8,
                 "target_contact": 1, "change_nm": -1.0, "stability": 1.0,
                 "median_crossing": 5.0, "crossings": [5]},
                {"residue_i": 1, "residue_j": 3, "threshold_nm": 1.2,
                 "target_contact": 0, "change_nm": 1.0, "stability": 1.0,
                 "median_crossing": 11.0, "crossings": [11]},
            ],
            {"candidate_count": 2, "top_candidates": []},
        ),
    )
    output = tmp_path / "out"
    (output / "design_pca_r1_r2").mkdir(parents=True)
    (output / "design_pca_r1_r2" / "pca_cv.npz").touch()
    topology = tmp_path / "top.pdb"
    trajectory = tmp_path / "r.xtc"
    start = tmp_path / "start.pdb"
    for path in (topology, trajectory, start):
        path.touch()
    path = atlas_programs.prepare_atlas_programs({
        "project_root": str(tmp_path),
        "reference": {
            "topology": str(topology),
            "design_trajectories": [str(trajectory), str(trajectory)],
            "initial_structure": str(start),
            "stride": 1,
            "basin_half_width": 0.25,
        },
        "trajectory_horizon": 16,
        "program": {"failure_guidance_weight": 1.0},
        "output": {"directory": str(output)},
    })
    catalog = yaml.safe_load(Path(path).read_text())
    terminal = catalog["tasks"]["endpoint"]["events"][0]["target_interval"]
    assert terminal == catalog["selection"]["target_interval"]
    assert catalog["observables"]["pc1"]["pca_model"] == (
        "out/design_pca_r1_r2/pca_cv.npz"
    )
    assert catalog["observables"]["contact_A"]["kind"] == "residue_distance"
    assert catalog["tasks"]["ordered"]["events"][0]["target_interval"] == [0.0, 0.8]


def test_ordered_pair_is_checked_on_every_design_segment():
    left = {"crossings_by_segment": [2, 16]}
    reversed_on_second = {"crossings_by_segment": [22, 10]}
    ordered = {"crossings_by_segment": [10, 23]}

    bad = atlas_programs._ordered_pair_diagnostics(
        left, reversed_on_second, segment_count=2, min_event_separation=6
    )
    good = atlas_programs._ordered_pair_diagnostics(
        left, ordered, segment_count=2, min_event_separation=6
    )

    assert bad["pair_order_stability"] == 0.5
    assert bad["ordered_gaps"] == [20, -6]
    assert good["pair_order_stability"] == 1.0
    assert good["ordered_gaps"] == [8, 7]
