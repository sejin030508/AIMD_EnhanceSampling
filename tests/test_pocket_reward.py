import numpy as np

from confmh.adapters.confrover_duet import ConfRoverFrame
from confmh.adapters.mock import MockIterativeFrameAdapter
from confmh.duet.observables import ATOM37_INDEX
from confmh.duet.outer_smc import OuterSMC
from confmh.duet.phase_b_pockets import (
    PocketEndpointMetric,
    PocketEndpointPotential,
    PocketReference,
)
from confmh.duet.programs import ProgressState


def _frame(loop_shift=None):
    residues = 8
    coords = np.zeros((residues, 37, 3), dtype=float)
    mask = np.zeros((residues, 37), dtype=bool)
    for residue in range(residues):
        for offset, name in enumerate(("N", "CA", "C")):
            atom = ATOM37_INDEX[name]
            coords[residue, atom] = [3.8 * residue, 0.7 * offset, 0.2 * residue * offset]
            mask[residue, atom] = True
    if loop_shift is not None:
        for residue in (5, 6):
            for name in ("N", "CA", "C"):
                coords[residue, ATOM37_INDEX[name]] += np.asarray(loop_shift)
    return ConfRoverFrame(coords, mask, np.zeros(residues, dtype=int))


def _metric(reference):
    return PocketEndpointMetric(
        references=[PocketReference("holo", reference.atom37_a, reference.atom37_mask)],
        core_residue_indices=[0, 1, 2, 3],
        loop_residue_indices=[5, 6],
    )


def test_pocket_reward_is_rigid_rotation_translation_invariant():
    reference = _frame()
    moving = _frame()
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    moving.atom37_a = moving.atom37_a @ rotation + np.asarray([12.0, -4.0, 8.0])
    assert _metric(reference).distance_a(moving) < 1.0e-10


def test_pocket_reward_measures_loop_after_core_alignment():
    reference = _frame()
    moving = _frame(loop_shift=[0.0, 0.0, 2.0])
    assert np.isclose(_metric(reference).distance_a(moving), 2.0, atol=1.0e-10)


def test_fixed_phase_b_potential_and_clipping():
    reference = _frame()
    metric = _metric(reference)
    potential = PocketEndpointPotential(metric, d0_a=2.0)
    state = ProgressState()
    value, returned_state, diagnostics = potential.candidate_log_psi(
        [reference], state, _frame(loop_shift=[0.0, 0.0, 2.0]), 1
    )
    assert np.isclose(value, -4.0)
    assert returned_state is state
    assert np.isclose(diagnostics["endpoint_distance_over_d0"], 1.0)
    clipped, _, _ = potential.candidate_log_psi(
        [reference], state, _frame(loop_shift=[0.0, 0.0, 20.0]), 1
    )
    assert clipped == -30.0
    assert potential.clipped_evaluations == 1


class _ConcentratingPotential:
    def log_psi(self, history, state, t):
        del history, state, t
        return 0.0

    def candidate_log_psi(self, history, state, frame, t):
        del history, t
        value = float(np.asarray(frame)[0, 0])
        return 1000.0 * value, state, {"value": value}


def test_outer_history_forks_are_independent_after_duplicate_resampling():
    result = OuterSMC(
        adapter=MockIterativeFrameAdapter(reverse_steps=4, residues=4),
        potential=_ConcentratingPotential(),
        method="outer_only",
        outer_k=2,
        inner_m=1,
        checkpoint_progress=0.75,
        seed=73,
        outer_resampling_ess_fraction=0.5,
    ).run([np.zeros((4, 3))], horizon=2)
    pre = result.pre_final_particles
    assert len(pre) == 2
    # The first forced concentration duplicates one parent/context.
    assert np.array_equal(pre[0].history[1], pre[1].history[1])
    # Independent proposal RNG after the fork makes the next frames differ.
    assert not np.array_equal(pre[0].history[2], pre[1].history[2])
    original = pre[1].history[1].copy()
    pre[0].history[1][0, 0] += 123.0
    assert np.array_equal(pre[1].history[1], original)
    assert len(result.pre_final_normalized_weights) == 2
