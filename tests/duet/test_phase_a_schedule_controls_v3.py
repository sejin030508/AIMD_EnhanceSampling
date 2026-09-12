from __future__ import annotations

import numpy as np

from confmh.duet.phase_a_cross_clock import enumerate_target
from confmh.duet.phase_a_diagnosis_v2 import _canonical_informative
from confmh.duet.phase_a_schedule_controls_v3 import (
    _exact_record,
    _method_spec,
    _sampler,
    schedule_control_seed_sequence,
    validate_config,
)


def _config() -> dict:
    return {
        "_config_dir": "/tmp/phase-a-v3-test/configs/duet",
        "project_root": "/tmp/phase-a-v3-test",
        "model": {"dtype": "float64"},
        "trajectory": {"horizon": 5},
        "program": {"potential_floor": 0.05},
        "phase_a_schedule_controls_v3": {
            "frozen_phase_a_root": "outputs/duet_md/phase_a_cross_clock",
            "diagnosis_v2_root": "outputs/duet_md/phase_a_diagnosis_v2",
            "epsilon": 0.05,
            "master_seed": 20260908,
            "rng_namespace": 3001,
            "repetitions": 5000,
            "bootstrap_repetitions": 2000,
            "uniform_schedule": [2, 2, 2, 2, 2],
            "concentrated_schedule": [1, 1, 6, 1, 1],
            "outer_only_schedule": [1, 1, 1, 1, 1],
            "fixed_outer_k": 16,
            "outer_only_k": 32,
            "checkpoint": "Y2",
            "complete_uniform_method_id": 3101,
            "complete_concentrated_method_id": 3102,
            "outer_only_method_id": 3103,
        },
        "experiment": {
            "output_directory": "outputs/duet_md/phase_a_schedule_controls_v3"
        },
    }


def test_config_and_all_three_schedules_have_cost_480() -> None:
    cfg = _config()
    validate_config(cfg)
    for label in ("complete_uniform", "complete_concentrated", "outer_only"):
        spec = _method_spec(cfg, label)
        assert 3 * spec["outer_k"] * sum(spec["schedule"]) == 480


def test_rng_entropy_layout_is_explicit_and_stable() -> None:
    seed = schedule_control_seed_sequence(20260908, 3001, 2, 3103, 17)
    assert seed.entropy == [20260908, 3001, 2, 3103, 17]
    first = np.random.default_rng(seed).integers(0, 2**31, size=8)
    second = np.random.default_rng(
        schedule_control_seed_sequence(20260908, 3001, 2, 3103, 17)
    ).integers(0, 2**31, size=8)
    assert np.array_equal(first, second)


def test_complete_schedules_preserve_target_weights_and_ancestry() -> None:
    cfg = _config()
    process = _canonical_informative()[0]
    exact_before = _exact_record(enumerate_target(process, 0.05))
    for method_index, label in enumerate(
        ("complete_uniform", "complete_concentrated"), start=1
    ):
        spec = _method_spec(cfg, label)
        output = _sampler(
            process,
            0.05,
            schedule_control_seed_sequence(20260908, 3001, 0, 3100 + method_index, 0),
            spec,
        )
        assert output.reverse_transition_evaluations == 480
        assert np.isclose(output.pre_weights.sum(), 1.0)
        assert len(output.pre_paths) == 16
        assert len(output.post_paths) == 16
        assert all(path in output.pre_paths for path in output.post_paths)
        assert all(len(path) == 6 for path in output.pre_paths)
    assert _exact_record(enumerate_target(process, 0.05)) == exact_before


def test_outer_only_m1_degenerate_case_runs_without_inner_selection_change() -> None:
    cfg = _config()
    process = _canonical_informative()[1]
    spec = _method_spec(cfg, "outer_only")
    output = _sampler(
        process,
        0.05,
        schedule_control_seed_sequence(20260908, 3001, 1, 3103, 0),
        spec,
    )
    assert output.reverse_transition_evaluations == 480
    assert output.schedule == [(32, 1)] * 5
    assert len(output.pre_paths) == 32
    assert len(output.post_paths) == 32
    assert np.isclose(output.pre_weights.sum(), 1.0)
    assert all(path in output.pre_paths for path in output.post_paths)
