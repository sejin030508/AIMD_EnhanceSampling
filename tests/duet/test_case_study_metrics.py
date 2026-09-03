from __future__ import annotations

import numpy as np
import pytest
import yaml

from confmh.duet.case_study_eval import MechanismBlindEvaluator
from confmh.duet.observables import (
    ATOM37_INDEX,
    atom_pair_distance_a,
    composite_dihedral_distance,
    local_atom_rmsd_a,
    pseudo_dihedral_deg,
)
from confmh.duet.runner import _maximum_aligned_step_nm


def frame_with_points(points):
    coords = np.zeros((8, 37, 3), dtype=float)
    mask = np.zeros((8, 37), dtype=bool)
    for (residue, atom), value in points.items():
        index = ATOM37_INDEX[atom]
        coords[residue, index] = value
        mask[residue, index] = True
    return {"atom37_a": coords, "atom37_mask": mask}


def test_atom_distance_and_composite_dihedral_are_atom37_native():
    atoms = [[0, "CB"], [1, "CA"], [2, "CA"], [3, "CG"]]
    frame = frame_with_points(
        {
            (0, "CB"): [0.0, 1.0, 0.0],
            (1, "CA"): [0.0, 0.0, 0.0],
            (2, "CA"): [1.0, 0.0, 0.0],
            (3, "CG"): [1.0, 0.0, 1.0],
            (4, "NZ"): [0.0, 0.0, 0.0],
            (5, "CD"): [3.0, 4.0, 0.0],
        }
    )
    angle = pseudo_dihedral_deg(frame, atoms)
    assert atom_pair_distance_a(frame, 4, "NZ", 5, "CD") == pytest.approx(5.0)
    assert composite_dihedral_distance(
        frame,
        [{"atoms": atoms, "center_degrees": angle, "scale_degrees": 30.0}],
    ) == pytest.approx(0.0)


def test_local_atom_rmsd_is_rigid_transform_invariant():
    reference = np.zeros((8, 37, 3), dtype=float)
    mask = np.zeros((8, 37), dtype=bool)
    for residue, point in enumerate(([0, 0, 0], [1, 0, 0], [0, 1, 0])):
        reference[residue, ATOM37_INDEX["CA"]] = point
        mask[residue, ATOM37_INDEX["CA"]] = True
    reference[3, ATOM37_INDEX["CG"]] = [0.2, 0.3, 1.0]
    mask[3, ATOM37_INDEX["CG"]] = True
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    moving = reference @ rotation + np.asarray([8.0, -4.0, 2.0])
    frame = {"atom37_a": moving, "atom37_mask": mask}
    assert local_atom_rmsd_a(frame, reference, mask, [0, 1, 2], [[3, "CG"]]) < 1e-12


def test_frame_step_metric_removes_only_global_rigid_motion():
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rigid = reference @ rotation + np.asarray([8.0, -4.0, 2.0])
    assert _maximum_aligned_step_nm(np.stack([reference, rigid])) < 1e-12

    deformed = rigid.copy()
    deformed[2, 2] += 0.5
    assert _maximum_aligned_step_nm(np.stack([reference, deformed])) > 0.05


def test_reward_hidden_feature_firewall_fails_closed(tmp_path):
    np.savez(tmp_path / "reference.npz", placeholder=np.asarray([0]))
    spec = {
        "case_id": "abl1_dfg_flip",
        "reference_features_npz": "reference.npz",
        "feature_firewall": {
            "reward_features": ["endpoint", "hidden_contact"],
            "hidden_evaluation_features": ["hidden_contact"],
        },
    }
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    cfg = {
        "project_root": str(tmp_path),
        "case_study": {"benchmark_spec": "spec.yaml"},
    }
    with pytest.raises(ValueError, match="Reward leakage"):
        MechanismBlindEvaluator.from_config(cfg, ("endpoint",))


def test_program_observable_must_be_declared_as_reward(tmp_path):
    np.savez(tmp_path / "reference.npz", placeholder=np.asarray([0]))
    spec = {
        "case_id": "abl1_dfg_flip",
        "reference_features_npz": "reference.npz",
        "feature_firewall": {
            "reward_features": ["endpoint"],
            "hidden_evaluation_features": ["hidden_contact"],
        },
    }
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    cfg = {
        "project_root": str(tmp_path),
        "case_study": {"benchmark_spec": "spec.yaml"},
    }
    with pytest.raises(ValueError, match="not declared reward features"):
        MechanismBlindEvaluator.from_config(cfg, ("endpoint", "hidden_contact"))
