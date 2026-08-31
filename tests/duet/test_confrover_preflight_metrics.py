from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from confmh.adapters.confrover_duet import seeded_numpy
from confmh.duet.preflight import _trace_metrics, evaluate_preflight_gate


@dataclass
class _Frame:
    ca_nm: np.ndarray


class _Adapter:
    def validate_frames(self, frames):
        return [
            {
                "valid": True,
                "nonfinite_coordinate_count": 0,
                "ca_clash_count_lt_1a": 0,
            }
            for _ in frames
        ]


def test_seeded_numpy_controls_vendor_rng_and_restores_global_state():
    np.random.seed(91)
    before = np.random.get_state()
    with seeded_numpy(7):
        left = np.random.normal(size=8)
    after = np.random.get_state()
    with seeded_numpy(7):
        right = np.random.normal(size=8)
    with seeded_numpy(8):
        different = np.random.normal(size=8)

    assert np.array_equal(left, right)
    assert not np.array_equal(left, different)
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_trace_metrics_aligns_to_first_frame_but_keeps_raw_displacement():
    reference = np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.3, 0.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = reference @ rotation + np.asarray([4.0, -3.0, 2.0])

    metrics = _trace_metrics([[_Frame(reference), _Frame(transformed)]], _Adapter())

    assert metrics["ca_endpoint_displacement_nm"][0] < 1e-12
    assert metrics["temporal_displacement_nm"][0] < 1e-12
    assert metrics["raw_ca_endpoint_displacement_nm"][0] > 1.0
    assert metrics["raw_temporal_displacement_nm"][0] > 1.0


def _variant(mean_max, pc1):
    return {
        "nonfinite_rate": 0.0,
        "clash_rate": 0.0,
        "geometry_valid_rate": 1.0,
        "ca_adjacent_max_a": [mean_max, mean_max],
        "endpoint_pc1": pc1,
    }


def test_explicit_preflight_gate_uses_hard_and_baseline_relative_checks():
    results = {
        "official_ode": _variant(4.7, [-0.5, 0.5]),
        "unconditioned_sde": _variant(4.9, [-0.4, 0.4]),
        "sde_checkpoint_no_resampling": _variant(4.95, [-0.3, 0.3]),
        "sde_constant_resampling": _variant(5.0, [-0.2, 0.2]),
    }
    gate = evaluate_preflight_gate(
        results,
        {
            "max_sde_to_ode_mean_ca_adjacent_ratio": 1.10,
            "max_pc1_wasserstein": 0.5,
        },
    )
    assert gate["passed"]

    results["sde_constant_resampling"] = _variant(5.3, [-0.2, 0.2])
    failed = evaluate_preflight_gate(results, {"max_pc1_wasserstein": 0.5})
    assert not failed["passed"]
    assert not failed["checks"]["sde_mean_max_ca_within_ode_ratio"]
