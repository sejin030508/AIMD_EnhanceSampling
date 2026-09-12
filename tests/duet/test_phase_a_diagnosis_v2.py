from __future__ import annotations

import inspect

import numpy as np

from confmh.duet.phase_a_cross_clock import L, canonical_processes, repetition_seed_sequence
from confmh.duet.phase_a_diagnosis_v2 import (
    ACTIONS,
    METHOD_IDS,
    _run_adaptive_variant,
    _run_fixed_schedule,
    exact_population_diagnostics,
)


def _process(name: str):
    return next(item for item in canonical_processes() if item.name == name)


def test_method_ids_form_a_new_nonoverlapping_seed_namespace():
    assert len(set(METHOD_IDS.values())) == len(METHOD_IDS)
    assert min(METHOD_IDS.values()) >= 2000


def test_exact_statistics_are_zero_at_the_no_information_commitment_checkpoint():
    process = _process("rare_low_no_information")
    diagnostics = exact_population_diagnostics(process, [(0, L)] * 8, 3, 0.05)
    assert np.isclose(diagnostics["rho_squared"], 0.0, atol=1e-14)
    assert np.isclose(diagnostics["inner_need"], 0.0, atol=1e-14)
    assert diagnostics["zero_phi_variance"] == 1.0


def test_exact_statistics_allocator_exposes_only_statistics_to_allocation_rule():
    source = inspect.getsource(exact_population_diagnostics)
    assert "choose_allocation(outer_need, inner_need" not in source
    # The wrapper computes a decision only after reducing the oracle state to the two
    # permitted scalar statistics.
    assert "_allocation_decision(outer_need, inner_need)" in source
    assert ACTIONS == ((12, 2), (6, 4), (3, 8))


def test_all_adaptive_diagnostic_variants_cost_exactly_480_reverse_transitions():
    process = _process("rare_high_informative")
    for index, variant in enumerate(("current_probe", "exact_stat", "matched")):
        output = _run_adaptive_variant(
            process,
            0.05,
            repetition_seed_sequence(20260908, process.instance_id, 2000 + index, 0),
            variant,
        )
        assert output.reverse_transition_evaluations == 480
        assert output.schedule[0] == (32, 1)
        assert np.isclose(output.pre_weights.sum(), 1.0)
        assert all(action in ACTIONS for action in output.schedule[1:])


def test_matched_production_uses_the_frozen_decomposition_without_using_probes():
    process = _process("common_low_informative")
    output = _run_adaptive_variant(
        process,
        0.05,
        repetition_seed_sequence(20260908, process.instance_id, 2002, 0),
        "matched",
    )
    assert output.schedule == [(32, 1)] + [(12, 2)] * 4
    assert output.reverse_transition_evaluations == 480


def test_every_temporal_budget_schedule_has_fixed_k_and_equal_cost():
    process = _process("rare_low_informative")
    schedules = [[2, 2, 2, 2, 2]]
    for concentrated_step in range(5):
        schedule = [1, 1, 1, 1, 1]
        schedule[concentrated_step] = 6
        schedules.append(schedule)
    for index, schedule in enumerate(schedules):
        output = _run_fixed_schedule(
            process,
            0.05,
            repetition_seed_sequence(20260908, process.instance_id, 2200 + index, 0),
            schedule,
        )
        assert sum(schedule) == 10
        assert output.reverse_transition_evaluations == 480
        assert all(k == 16 for k, _ in output.schedule)
        assert [m for _, m in output.schedule] == schedule
        assert np.isclose(output.pre_weights.sum(), 1.0)
