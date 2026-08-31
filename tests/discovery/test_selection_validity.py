from __future__ import annotations

import inspect

import numpy as np

from confmh.discovery.runner import run_controlled_rollout
from confmh.discovery.selection import select_candidate
from confmh.discovery.validity import evaluate_validity

from .conftest import write_backbone


def test_raw_and_biased_selection_have_no_accept_reject():
    values = np.asarray([0.0, 2.0, 4.0])
    valid = np.asarray([True, True, False])
    raw = select_candidate(values, valid, controller="raw", rng=np.random.default_rng(1))
    assert np.allclose(raw.probabilities, [0.5, 0.5, 0.0])
    guided = select_candidate(
        values,
        valid,
        controller="biased",
        center=2.0,
        kappa=2.0,
        temperature=1.0,
        rng=np.random.default_rng(1),
    )
    assert guided.probabilities[1] > guided.probabilities[0]
    source = inspect.getsource(run_controlled_rollout)
    assert "confmh.mh" not in source
    assert "accept(" not in source


def test_validity_reports_paper_compatible_safety_metrics(tmp_path):
    pdb = write_backbone(
        tmp_path / "candidate.pdb",
        [(0, 0, 0), (3.8, 0.2, 0), (7.6, -0.2, 0), (11.4, 0.3, 0)],
    )
    result = evaluate_validity(pdb)
    assert result.ca_adjacent_max_a < 4.5
    assert result.peptide_cn_max_a < 2.0
    assert result.backbone_clash_min_a > 1.0
    assert result.ca_valid_fraction == 1.0
    assert result.peptide_cn_valid_fraction == 1.0
    assert result.clash_free_fraction == 1.0
    assert result.paper_compliant_0_90
    assert set(result.to_dict()) >= {
        "bond_outlier_fraction",
        "angle_outlier_fraction",
        "rama_outlier_fraction_coarse",
    }
