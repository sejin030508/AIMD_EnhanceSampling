from pathlib import Path

import numpy as np
import yaml

from confmh.duet import atlas_programs


def test_transition_path_uses_last_stable_exit_and_fixed_lag():
    scores = np.asarray(
        [0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.7, 1.2, 1.7, 1.8, 2.0, 2.0, 2.0]
    )
    times_ps = np.arange(len(scores), dtype=float) * 10.0

    result = atlas_programs._transition_path_segment(
        scores,
        times_ps,
        (-0.1, 0.1),
        (1.9, 2.1),
        frames=6,
        model_step_ps=10.0,
        start_dwell_steps=2.0,
        target_dwell_steps=2.0,
        duration_ratio=(0.9, 1.1),
        max_time_error_steps=0.0,
    )

    assert result is not None
    indices, report = result
    assert indices.tolist() == [7, 8, 9, 10, 11, 12]
    assert report["start_run"] == [5, 7]
    assert report["start_exit_index"] == 7
    assert report["target_entry_index"] == 12
    assert report["duration_ratio"] == 1.0


def test_transition_path_rejects_duration_instead_of_stretching_it():
    scores = np.asarray([0.0, 0.0] + [1.0] * 7 + [2.0, 2.0])
    times_ps = np.arange(len(scores), dtype=float) * 10.0

    result = atlas_programs._transition_path_segment(
        scores,
        times_ps,
        (-0.1, 0.1),
        (1.9, 2.1),
        frames=5,
        model_step_ps=10.0,
        start_dwell_steps=1.0,
        target_dwell_steps=1.0,
        duration_ratio=(0.9, 1.1),
        max_time_error_steps=0.0,
    )

    assert result is None


def test_route_geometry_uses_last_exit_without_generated_time_matching():
    scores = np.asarray(
        [0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.7, 1.2, 1.7, 2.0, 2.0, 2.0]
    )
    times_ps = np.arange(len(scores), dtype=float) * 10.0

    result = atlas_programs._transition_path_geometry_segment(
        scores,
        times_ps,
        (-0.1, 0.1),
        (1.9, 2.1),
        analysis_frames=5,
        start_dwell_ps=20.0,
        target_dwell_ps=20.0,
        not_before_time_ps=50.0,
        target_not_after_time_ps=120.0,
    )

    assert result is not None
    indices, report = result
    assert indices.tolist() == [7, 8, 9, 10, 11]
    assert report["start_exit_index"] == 7
    assert report["target_entry_index"] == 11
    assert report["reference_first_passage_ps"] == 60.0
    assert report["reference_transition_path_ps"] == 40.0
    assert report["coordinate_interpolation"] is False


def test_reference_first_passage_selects_in_training_range_lag():
    assert atlas_programs._select_model_lag_in_10ps(
        12_000.0, horizon=24, target_compression=2.0
    ) == 25
    assert atlas_programs._select_model_lag_in_10ps(
        1_000.0, horizon=24, target_compression=2.0
    ) == 20
    assert atlas_programs._select_model_lag_in_10ps(
        200_000.0, horizon=24, target_compression=2.0
    ) == 210


def test_one_discovery_route_can_define_a_b_without_r2_support():
    class FakeTrajectory:
        def __init__(self, xyz):
            self.xyz = np.asarray(xyz, dtype=float)

    discovery = np.zeros((5, 4, 3), dtype=float)
    discovery[:, 1, 0] = 10.0
    discovery[:, 2, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]
    discovery[:, 3, 1] = [1.0, 1.0, 2.0, 4.0, 5.0]
    unsupported = np.zeros((5, 4, 3), dtype=float)
    unsupported[:, 1, 0] = 10.0
    unsupported[:, 2, 0] = 1.0
    unsupported[:, 3, 1] = 1.0

    selected, report = atlas_programs._select_ordered_contacts(
        [FakeTrajectory(discovery), FakeTrajectory(unsupported)],
        np.arange(4),
        [np.arange(5), np.arange(5)],
        min_sequence_separation=2,
        min_change_nm=0.5,
        min_event_separation=1,
        stable_fraction=1.0,
        max_event_frame=4,
        discovery_segment_index=0,
        min_event_support_count=1,
        min_pair_support_count=1,
        min_crossing_persistence=2,
    )

    assert [(row["residue_i"], row["residue_j"]) for row in selected] == [(0, 2), (0, 3)]
    assert [row["crossings_by_segment"] for row in selected] == [[3, None], [4, None]]
    assert report["route_support_count"] == 1
    assert report["route_status"] == "single_reference_route"
    assert report["selected_pair_diagnostics"]["ordered_segment_indices"] == [0]


def test_route_v3_catalog_uses_final_terminal_and_r3_default_label(tmp_path, monkeypatch):
    class FakePCA:
        def project_trajectory(self, trajectory):
            return np.linspace(-1.0, 2.0, 8)[:, None]

        def project_files(self, path):
            return np.asarray([[-1.0]])

    class FakeTrajectory:
        xyz = np.zeros((8, 4, 3), dtype=float)
        time = np.arange(8, dtype=float) * 10.0

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
        "_transition_path_segment",
        lambda *args, **kwargs: (np.arange(5), {"start_exit_index": 0}),
    )
    monkeypatch.setattr(
        atlas_programs,
        "_select_ordered_contacts",
        lambda *args, **kwargs: (
            [
                {
                    "residue_i": 0,
                    "residue_j": 2,
                    "threshold_nm": 0.8,
                    "target_contact": 1,
                    "crossings": [1],
                    "crossings_by_segment": [1, None],
                },
                {
                    "residue_i": 0,
                    "residue_j": 3,
                    "threshold_nm": 1.2,
                    "target_contact": 0,
                    "crossings": [2],
                    "crossings_by_segment": [2, None],
                },
            ],
            {
                "route_support_count": 1,
                "route_status": "single_reference_route",
                "selected_pair_diagnostics": {"ordered_segment_indices": [0]},
            },
        ),
    )

    output = tmp_path / "out"
    model_dir = output / "design_pca_r1_r2"
    model_dir.mkdir(parents=True)
    (model_dir / "pca_cv.npz").touch()
    topology = tmp_path / "top.pdb"
    trajectory = tmp_path / "r.xtc"
    held_out = tmp_path / "held_out.xtc"
    start = tmp_path / "start.pdb"
    for path in (topology, trajectory, held_out, start):
        path.touch()

    catalog_path = atlas_programs.prepare_atlas_programs(
        {
            "project_root": str(tmp_path),
            "model": {"physical_lag_in_10ps": 1},
            "trajectory": {"horizon": 4},
            "trajectory_horizon": 4,
            "reference": {
                "topology": str(topology),
                "design_trajectories": [str(trajectory), str(trajectory)],
                "held_out_trajectories": [str(held_out)],
                "initial_structure": str(start),
                "stride": 1,
                "basin_half_width": 0.25,
            },
            "benchmark_protocol": {
                "version": "route_v3",
                "transition_mode": "last_exit_fixed_lag",
                "terminal_mode": "final_frame",
            },
            "output": {"directory": str(output)},
        }
    )
    catalog = yaml.safe_load(Path(catalog_path).read_text())

    assert catalog["protocol_version"] == "route_v3"
    assert catalog["design_split"]["held_out_evaluation"] == ["R3"]
    assert "distance_A" in catalog["observables"]
    assert "contact_A" not in catalog["observables"]
    assert catalog["tasks"]["ordered"]["terminal_event"]["physical_window"] == [4, 4]
    assert catalog["selection"]["route_support_denominator"] == 2


def test_route_v3_1_catalog_uses_persistence_deadline_and_role_split(tmp_path, monkeypatch):
    class FakePCA:
        def project_trajectory(self, trajectory):
            return np.linspace(-1.0, 2.0, 30)[:, None]

        def project_files(self, path):
            return np.asarray([[2.0 if "end" in str(path) else -1.0]])

    class FakeTrajectory:
        xyz = np.zeros((30, 4, 3), dtype=float)
        time = np.arange(30, dtype=float) * 100.0

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
        "_transition_path_geometry_segment",
        lambda *args, **kwargs: (
            np.arange(25),
            {
                "start_exit_index": 0,
                "reference_first_passage_ps": 12_000.0,
                "reference_transition_path_ps": 8_000.0,
            },
        ),
    )
    monkeypatch.setattr(
        atlas_programs,
        "_select_ordered_contacts",
        lambda *args, **kwargs: (
            [
                {
                    "residue_i": 0,
                    "residue_j": 2,
                    "threshold_nm": 0.8,
                    "target_contact": 1,
                    "crossings": [5],
                    "crossings_by_segment": [5, None],
                },
                {
                    "residue_i": 0,
                    "residue_j": 3,
                    "threshold_nm": 1.2,
                    "target_contact": 0,
                    "crossings": [15],
                    "crossings_by_segment": [15, None],
                },
            ],
            {
                "route_support_count": 1,
                "route_status": "single_reference_route",
                "selected_pair_diagnostics": {"ordered_segment_indices": [0]},
            },
        ),
    )

    output = tmp_path / "out"
    model_dir = output / "design_pca_d1_d2"
    model_dir.mkdir(parents=True)
    (model_dir / "pca_cv.npz").touch()
    topology = tmp_path / "top.pdb"
    trajectory = tmp_path / "r.xtc"
    start = tmp_path / "start.pdb"
    endpoint = tmp_path / "end.pdb"
    for path in (topology, trajectory, start, endpoint):
        path.touch()

    catalog_path = atlas_programs.prepare_atlas_programs(
        {
            "project_root": str(tmp_path),
            "model": {"physical_lag_in_10ps": 20},
            "trajectory": {"horizon": 24},
            "trajectory_horizon": 24,
            "reference": {
                "topology": str(topology),
                "design_trajectories": [str(trajectory), str(trajectory)],
                "design_labels": ["D1", "D2"],
                "held_out_labels": ["H"],
                "replicate_roles": {"D1": "R3", "D2": "R1", "H": "R2"},
                "initial_structure": str(start),
                "route_endpoint_structure": str(endpoint),
                "stride": 1,
                "basin_half_width": 0.25,
                "target_basin_half_width": 0.5,
            },
            "benchmark_protocol": {
                "version": "route_v3_1",
                "transition_mode": "last_exit_route_geometry",
                "route_discovery_start_time_ps": 1_000.0,
                "route_discovery_end_time_ps": 20_000.0,
                "terminal_mode": "first_stable_hit_by_deadline",
                "nominal_compression_factor": [1.0, 10.0],
            },
            "output": {"directory": str(output)},
        }
    )
    catalog = yaml.safe_load(Path(catalog_path).read_text())

    assert catalog["protocol_version"] == "route_v3_1"
    assert catalog["design_split"] == {
        "task_design": ["D1", "D2"],
        "held_out_evaluation": ["H"],
    }
    assert "windowed" not in catalog["tasks"]
    ordered = catalog["tasks"]["ordered"]
    assert [event["physical_window"] for event in ordered["events"]] == [
        [1, 24],
        [1, 24],
    ]
    assert [event["persistence_frames"] for event in ordered["events"]] == [2, 2]
    assert ordered["terminal_event"]["physical_window"] == [1, 24]
    assert ordered["terminal_event"]["persistence_frames"] == 2
    assert catalog["selection"]["nominal_challenge"][
        "selected_physical_lag_in_10ps"
    ] == 25
