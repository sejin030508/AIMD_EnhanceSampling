from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_small_protein_run.py")
SPEC = importlib.util.spec_from_file_location("small_protein_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_aligned_rmsd_is_rigid_transform_invariant():
    target = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moving = target @ rotation + np.asarray([4.0, -2.0, 7.0])
    assert MODULE.aligned_rmsd(moving, target) < 1.0e-12


def test_normalized_dtw_identical_curve_is_zero():
    curve = np.asarray([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])
    assert MODULE.dtw_normalized(curve, curve) == 0.0


def test_pre_final_lineage_ignores_terminal_resampling_map():
    ancestry = [
        [0, 0, 2, 3],
        [1, 1, 2, 3],
        [3, 3, 3, 3],  # final map must not change pre-final lineages
    ]
    lineages = MODULE.trace_pre_final_lineages(ancestry, 4)
    assert all(len(row) == 3 for row in lineages)
    assert lineages[0][-1] == 0
    assert lineages[3][-1] == 3
