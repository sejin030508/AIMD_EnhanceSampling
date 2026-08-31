import numpy as np

from confmh.duet.reference_eval import _transition_windows, dtw_distance, kabsch_rmsd_nm


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
