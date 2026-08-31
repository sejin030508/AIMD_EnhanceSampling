import numpy as np

from confmh.adapters.mock import MockIterativeFrameAdapter
from confmh.duet.observables import ObservableRegistry
from confmh.duet.outer_smc import OuterSMC
from confmh.duet.potentials import PotentialCoefficients, PrefixPotential
from confmh.duet.programs import Event, TemporalProgram


def test_mock_pipeline_runs_all_nested_methods():
    initial = [np.zeros((4, 3))]
    registry = ObservableRegistry({"motion": lambda frame: float(np.mean(np.asarray(frame)[:, 0]))})
    program = TemporalProgram("terminal", (Event("end", "motion", (-0.5, 0.5), (3, 3)),))
    for method in ("outer_only", "inner_only", "naive_dual", "complete_nested", "duet"):
        adapter = MockIterativeFrameAdapter(reverse_steps=4, residues=4)
        potential = PrefixPotential(
            program,
            registry,
            PotentialCoefficients(potential_floor=1e-30),
            horizon=3,
        )
        k = 1 if method == "inner_only" else 3
        m = 1 if method == "outer_only" else 3
        result = OuterSMC(
            adapter=adapter,
            potential=potential,
            method=method,
            outer_k=k,
            inner_m=m,
            checkpoint_progress=0.75,
            seed=19,
        ).run(initial, horizon=3)
        assert len(result.particles) == k
        assert all(len(particle.history) == 4 for particle in result.particles)
        assert len(result.records) == 3 * k


def test_constant_potential_outer_normalizer_is_one():
    initial = [np.zeros((4, 3))]
    registry = ObservableRegistry({"motion": lambda frame: float(np.mean(np.asarray(frame)[:, 0]))})
    program = TemporalProgram("terminal", (Event("end", "motion", (-0.5, 0.5), (3, 3)),))
    for method in ("outer_only", "complete_nested", "duet"):
        adapter = MockIterativeFrameAdapter(reverse_steps=4, residues=4)
        potential = PrefixPotential(
            program,
            registry,
            PotentialCoefficients(lambda_program=0.0, potential_floor=1e-30),
            horizon=3,
        )
        result = OuterSMC(
            adapter=adapter,
            potential=potential,
            method=method,
            outer_k=4,
            inner_m=1 if method == "outer_only" else 3,
            checkpoint_progress=0.75,
            seed=23,
        ).run(initial, horizon=3)
        assert abs(result.log_normalizer_estimate) < 1e-12
        assert np.max(np.abs(result.log_normalizer_increments)) < 1e-12
