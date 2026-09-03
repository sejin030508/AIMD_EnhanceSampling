from confmh.duet.observables import ObservableRegistry
from confmh.duet.potentials import PotentialCoefficients, PrefixPotential
from confmh.duet.programs import Event, ProgressState, TemporalProgram


def test_failed_prefix_keeps_terminal_distance_guidance_when_enabled():
    registry = ObservableRegistry({"x": lambda frame: float(frame)})
    program = TemporalProgram(
        "windowed",
        (Event("middle", "x", (1.0, 1.0), (1, 2)),),
        terminal_event=Event("end", "x", (3.0, 3.0), (4, 4)),
    )
    potential = PrefixPotential(
        program,
        registry,
        PotentialCoefficients(
            failure_penalty=4.0,
            failure_guidance_weight=1.0,
            distance_scale=1.0,
        ),
        horizon=4,
    )
    failed = ProgressState(failed=True, reason="deadline:middle")
    assert potential.prefix_cost([2.5], failed, 3) < potential.prefix_cost([0.0], failed, 3)


def test_failed_prefix_preserves_legacy_flat_penalty_by_default():
    registry = ObservableRegistry({"x": lambda frame: float(frame)})
    program = TemporalProgram("terminal", (Event("end", "x", (3.0, 3.0), (4, 4)),))
    potential = PrefixPotential(
        program,
        registry,
        PotentialCoefficients(failure_penalty=7.0),
        horizon=4,
    )
    failed = ProgressState(failed=True, reason="terminal_deadline")
    assert potential.prefix_cost([0.0], failed, 3) == 7.0
    assert potential.prefix_cost([2.5], failed, 3) == 7.0
