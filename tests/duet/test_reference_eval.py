import numpy as np

from confmh.duet.reference_eval import (
    _route_v3_1_transition_windows,
    _route_v3_transition_windows,
    _transition_windows,
    dtw_distance,
    kabsch_rmsd_nm,
)


def test_dtw_is_zero_for_identical_path():
    path = np.asarray([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    assert dtw_distance(path, path) == 0.0


def test_kabsch_rmsd_removes_rigid_motion():
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = reference @ rotation + np.asarray([4.0, -2.0, 1.0])
    assert kabsch_rmsd_nm(moved, reference) < 1e-12


def test_reference_windows_have_exact_physical_horizon():
    pc1 = np.linspace(-1.0, 2.0, 20)
    windows = _transition_windows(pc1, (-1.1, -0.5), (1.5, 2.1), horizon=8)
    assert windows
    assert all(len(window) == 9 for window in windows)
    assert all(np.all(np.diff(window) > 0) for window in windows)


def test_route_v3_held_out_window_uses_fixed_lag_and_last_exit():
    pc1 = np.asarray(
        [0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.7, 1.2, 1.7, 1.8, 2.0, 2.0, 2.0]
    )
    times_ps = np.arange(len(pc1), dtype=float) * 10.0
    reference_ca = np.zeros((len(pc1), 3, 3), dtype=float)
    initial_ca = np.zeros((3, 3), dtype=float)
    selection = {
        "transition_mode": "last_exit_fixed_lag",
        "start_interval": [-0.1, 0.1],
        "transition_target_interval": [1.9, 2.1],
        "trajectory_horizon": 5,
        "model_step_ps": 10.0,
        "protocol_parameters": {
            "start_dwell_model_steps": 2.0,
            "target_dwell_model_steps": 2.0,
            "duration_ratio": [0.9, 1.1],
            "max_sampling_time_error_steps": 0.0,
            "max_start_ca_rmsd_nm": 0.3,
        },
    }

    windows = _route_v3_transition_windows(
        pc1, times_ps, reference_ca, initial_ca, selection
    )

    assert len(windows) == 1
    assert windows[0].tolist() == [7, 8, 9, 10, 11, 12]


def test_route_v3_1_held_out_window_uses_geometry_not_model_lag():
    pc1 = np.asarray(
        [0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.7, 1.2, 1.7, 2.0, 2.0, 2.0]
    )
    times_ps = np.arange(len(pc1), dtype=float) * 10.0
    reference_ca = np.zeros((len(pc1), 3, 3), dtype=float)
    initial_ca = np.zeros((3, 3), dtype=float)
    selection = {
        "transition_mode": "last_exit_route_geometry",
        "start_interval": [-0.1, 0.1],
        "transition_target_interval": [1.9, 2.1],
        "trajectory_horizon": 4,
        "model_step_ps": 999.0,
        "protocol_parameters": {
            "reference_start_dwell_ps": 20.0,
            "reference_target_dwell_ps": 20.0,
            "max_start_ca_rmsd_nm": 0.3,
        },
    }

    windows = _route_v3_1_transition_windows(
        pc1, times_ps, reference_ca, initial_ca, selection
    )

    assert len(windows) == 1
    assert windows[0].tolist() == [7, 8, 9, 10, 11]
