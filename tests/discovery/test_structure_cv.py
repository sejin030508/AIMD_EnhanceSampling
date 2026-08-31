from __future__ import annotations

import numpy as np

from confmh.discovery.cv import ControlCV
from confmh.discovery.structure import kabsch_rmsd, read_pdb, sequence_alignment_pairs

from .conftest import write_backbone


def test_kabsch_and_alignment_are_rigid_transform_invariant(tmp_path):
    points = np.asarray([[0, 0, 0], [3.8, 0.2, 0], [7.5, -0.1, 0.5], [11.0, 0.5, -0.2]])
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    moved = points @ rotation + np.asarray([4.0, -2.0, 8.0])
    assert kabsch_rmsd(moved, points) < 1e-10
    assert sequence_alignment_pairs("ACDE", "ACXDE") == [(0, 0), (1, 1), (2, 3), (3, 4)]

    start = write_backbone(tmp_path / "start.pdb", [tuple(row) for row in points])
    candidate = write_backbone(tmp_path / "candidate.pdb", [tuple(row) for row in moved])
    cv = ControlCV(start_pdb=start, mode="start_rmsd")
    result = cv.evaluate(candidate)
    assert result.rmsd_start < 1e-6
    assert result.rmsd_target is None
    assert cv.describe()["target_loaded"] is False


def test_target_aware_delta_cv(tmp_path):
    start_points = [(0, 0, 0), (3.8, 0, 0), (7.6, 0, 0), (11.4, 0, 0)]
    target_points = [(0, 0, 0), (3.8, 1, 0), (7.6, 2, 0), (11.4, 4, 0)]
    start = write_backbone(tmp_path / "start.pdb", start_points)
    target = write_backbone(tmp_path / "target.pdb", target_points)
    cv = ControlCV(start_pdb=start, target_pdb=target, mode="delta_rmsd")
    result = cv.evaluate(target)
    assert result.rmsd_target < 1e-6
    assert result.control > 0
    assert len(read_pdb(target)) == 4
