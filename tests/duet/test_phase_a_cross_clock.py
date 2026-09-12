from __future__ import annotations

import inspect
import json

import numpy as np

from confmh.duet.phase_a_cross_clock import (
    C_L,
    C_R,
    F,
    L,
    O,
    R,
    S,
    _run_adaptive_sampler,
    _run_fixed_sampler,
    canonical_processes,
    choose_allocation,
    endpoint_potentials,
    enumerate_target,
    generate_held_out_processes,
    inner_marginal_error,
    mean_incremental_weight,
    repetition_seed_sequence,
    reverse_kernel,
    run_phase_a_cross_clock,
    transition_spec,
)


PARENTS_BY_STEP = {
    1: [S],
    2: [L, R, F],
    3: [L, R, F],
    4: [C_L, C_R, F],
    5: [O, F],
}


def test_reverse_process_recovers_every_physical_transition_prior():
    processes = canonical_processes() + generate_held_out_processes(20260908, 20)
    maximum = max(
        inner_marginal_error(process, parent, step)
        for process in processes
        for step, parents in PARENTS_BY_STEP.items()
        for parent in parents
    )
    assert maximum < 1e-12


def test_phi_is_base_reverse_conditional_expectation():
    process = next(
        item for item in canonical_processes() if item.name == "rare_low_informative"
    )
    transition = transition_spec(process, L, 3)
    kernel = reverse_kernel(transition)
    psi = endpoint_potentials(transition, 3, 0.05)
    direct_joint = kernel.joint_forward.sum(axis=(1, 3))
    expected = (direct_joint.T @ psi) / direct_joint.sum(axis=0)
    assert np.allclose(kernel.phi(psi, "Y2"), expected, atol=1e-14, rtol=0.0)


def test_no_information_checkpoint_phi_is_constant_and_uniform():
    process = next(
        item
        for item in canonical_processes()
        if item.name == "rare_low_no_information"
    )
    transition = transition_spec(process, L, 3)
    kernel = reverse_kernel(transition)
    phi = kernel.phi(endpoint_potentials(transition, 3, 0.05), "Y2")
    assert np.allclose(phi, phi[0], atol=1e-14, rtol=0.0)
    assert np.allclose(phi / phi.sum(), [0.5, 0.5], atol=1e-14, rtol=0.0)


def test_information_increases_from_y3_to_y2_to_y1():
    process = next(
        item for item in canonical_processes() if item.name == "rare_low_informative"
    )
    transition = transition_spec(process, L, 3)
    kernel = reverse_kernel(transition)
    psi = endpoint_potentials(transition, 3, 0.05)
    chi = [kernel.chi(psi, stage) for stage in ("Y3", "Y2", "Y1")]
    assert np.isclose(chi[0], 0.0, atol=1e-14)
    assert chi[0] < chi[1] < chi[2]


def test_variable_k_normalizer_uses_mean_not_sum():
    assert mean_incremental_weight([0.2, 0.8]) == 0.5
    assert mean_incremental_weight([0.2, 0.8] * 7) == 0.5


def test_probe_and_production_seed_streams_are_independent():
    root_a = repetition_seed_sequence(20260908, 7, 20, 11)
    probe_a, production_a = root_a.spawn(2)
    probe_rng = np.random.default_rng(probe_a)
    _ = probe_rng.random(1000)
    production_after_probe = np.random.default_rng(production_a).random(8)

    root_b = repetition_seed_sequence(20260908, 7, 20, 11)
    _, production_b = root_b.spawn(2)
    production_without_probe = np.random.default_rng(production_b).random(8)
    assert np.array_equal(production_after_probe, production_without_probe)


def test_allocator_has_no_process_or_time_inputs_and_balanced_tie_break():
    assert list(inspect.signature(choose_allocation).parameters) == [
        "outer_need",
        "inner_need",
        "actions",
    ]
    assert choose_allocation(0.0, 0.0) == (6, 4)
    assert choose_allocation(1.0, 0.0) == (12, 2)
    assert choose_allocation(0.0, 1.0) == (3, 8)


def test_exact_target_has_expected_success_and_route_mass():
    process = next(
        item for item in canonical_processes() if item.name == "rare_high_informative"
    )
    exact = enumerate_target(process)
    base_success = 0.8 * 0.02 * 0.9 * 0.9
    expected_z = base_success + 0.05 * (1.0 - base_success)
    assert np.isclose(exact.base_success_probability, base_success)
    assert np.isclose(exact.normalizer, expected_z)
    assert np.isclose(exact.target_success_probability, base_success / expected_z)
    assert np.isclose(exact.route_l_mass + exact.route_r_mass, 1.0)


def test_every_fixed_method_and_adaptive_sampler_use_480_reverse_transitions():
    process = next(
        item for item in canonical_processes() if item.name == "rare_high_informative"
    )
    settings = {
        "frozen": (32, 1),
        "outer_only": (32, 1),
        "inner_only": (1, 32),
        "complete_nested": (8, 4),
        "fixed_duet": (8, 4),
    }
    for method, (outer_k, inner_m) in settings.items():
        result = _run_fixed_sampler(
            process,
            method,
            outer_k,
            inner_m,
            0.05,
            repetition_seed_sequence(20260908, process.instance_id, 1, 0),
        )
        assert result.reverse_transition_evaluations == 480
    adaptive = _run_adaptive_sampler(
        process,
        0.05,
        repetition_seed_sequence(20260908, process.instance_id, 20, 0),
    )
    assert adaptive.reverse_transition_evaluations == 480
    assert adaptive.schedule[0] == (32, 1)
    assert all(item in {(12, 2), (6, 4), (3, 8)} for item in adaptive.schedule[1:])


def test_small_end_to_end_runner_writes_resumable_raw_outputs(tmp_path, monkeypatch):
    config_path = tmp_path / "phase_a_test.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    cfg = {
        "_config_path": str(config_path),
        "_config_dir": str(tmp_path),
        "project_root": str(tmp_path),
        "model": {"backend": "phase_a_cross_clock", "reverse_steps": 3, "dtype": "float64"},
        "trajectory": {"horizon": 5},
        "particles": {"outer_k": 8, "inner_m": 4},
        "program": {"type": "terminal", "potential_floor": 0.05},
        "experiment": {"output_directory": str(tmp_path / "output")},
        "phase_a": {
            "epsilon": 0.05,
            "repetitions": 2,
            "master_seed": 20260908,
            "held_out_count": 2,
            "fixed_methods": {
                "frozen": [32, 1],
                "outer_only": [32, 1],
                "inner_only": [1, 32],
                "complete_nested": [8, 4],
                "fixed_duet": [8, 4],
            },
            "static_allocations": [[16, 2], [8, 4], [4, 8], [2, 16]],
            "adaptive": {"actions": [[12, 2], [6, 4], [3, 8]]},
        },
    }
    monkeypatch.setattr(
        "confmh.duet.phase_a_cross_clock.validate_phase_a_config",
        lambda unused: {"valid": True, "test_override": True},
    )
    output = run_phase_a_cross_clock(cfg)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["held_out"]["process_count"] == 2
    assert metrics["development_allocation"]["selected_global_static"].startswith("K")
    assert (
        output
        / "cells"
        / "held_out"
        / "held_out_00"
        / "adaptive"
        / "raw_repetitions.npz"
    ).exists()
    resumed = run_phase_a_cross_clock(cfg, resume=True)
    assert resumed == output


def test_held_out_generation_is_deterministic_and_in_bounds():
    first = generate_held_out_processes(20260908, 20)
    second = generate_held_out_processes(20260908, 20)
    assert first == second
    assert len(first) == 20
    for process in first:
        assert 0.01 <= process.commitment_probability <= 0.25
        assert 0.62 <= min(process.s_l, process.s_r) <= 0.8
        assert 0.8 <= max(process.s_l, process.s_r) <= 0.98
        assert 0.1 <= process.commitment_beta2 <= 0.5
