from __future__ import annotations

import numpy as np
import pytest

from confmh.duet.observables import ObservableRegistry
from confmh.duet.potentials import PotentialCoefficients, PrefixPotential
from confmh.duet.programs import Event, TemporalProgram


@pytest.fixture
def scalar_registry():
    return ObservableRegistry({"x": lambda frame: float(frame)})


@pytest.fixture
def ordered_program():
    return TemporalProgram(
        kind="ordered",
        events=(
            Event("A", "x", (1.0, 1.0), (1, 2)),
            Event("B", "x", (2.0, 2.0), (2, 4)),
        ),
        terminal_event=Event("terminal", "x", (3.0, 3.0), (5, 5)),
    )


@pytest.fixture
def scalar_potential(ordered_program, scalar_registry):
    return PrefixPotential(
        ordered_program,
        scalar_registry,
        PotentialCoefficients(
            lambda_program=1.0,
            distance_weight=1.0,
            remaining_event_weight=0.1,
            deadline_weight=1.0,
            failure_penalty=20.0,
            terminal_failure_penalty=20.0,
            distance_scale=1.0,
            potential_floor=1e-30,
        ),
        horizon=5,
    )

