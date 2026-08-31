from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from confmh.duet.observables import ObservableRegistry
from confmh.duet.programs import ProgressState, TemporalProgram


def distance_to_interval(value: float, interval: tuple[float, float]) -> float:
    lower, upper = interval
    if value < lower:
        return float(lower - value)
    if value > upper:
        return float(value - upper)
    return 0.0


@dataclass(frozen=True)
class PotentialCoefficients:
    lambda_program: float = 1.0
    distance_weight: float = 1.0
    remaining_event_weight: float = 0.0
    deadline_weight: float = 1.0
    failure_penalty: float = 25.0
    terminal_failure_penalty: float = 25.0
    distance_scale: float = 1.0
    potential_floor: float = 1e-30

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PotentialCoefficients":
        return cls(**{field: float(cfg.get(field, getattr(cls(), field))) for field in cls.__dataclass_fields__})


class PrefixPotential:
    """Minimal documented prefix potential used by the first implementation."""

    def __init__(
        self,
        program: TemporalProgram,
        observables: ObservableRegistry,
        coefficients: PotentialCoefficients,
        horizon: int,
    ) -> None:
        self.program = program
        self.observables = observables
        self.coefficients = coefficients
        self.horizon = int(horizon)
        self.evaluations = 0

    def values(self, frame: Any) -> dict[str, float]:
        return self.observables.evaluate(frame, self.program.all_observables())

    def advance(self, state: ProgressState, frame: Any, t: int) -> tuple[ProgressState, dict[str, float]]:
        values = self.values(frame)
        return self.program.update(state, values, t), values

    def prefix_cost(
        self,
        history: Sequence[Any],
        state: ProgressState,
        t: int,
        values: Mapping[str, float] | None = None,
    ) -> float:
        self.evaluations += 1
        c = self.coefficients
        if state.failed:
            return c.failure_penalty
        event = self.program.next_event(state)
        if event is None:
            return 0.0
        if values is None:
            values = self.values(history[-1])
        distance = distance_to_interval(values[event.observable], event.target)
        scale = max(c.distance_scale, np.finfo(float).tiny)
        remaining = max(event.window[1] - int(t), 0)
        deadline_factor = 1.0 + c.deadline_weight / float(remaining + 1)
        uncompleted = max(len(self.program.events) - state.stage, 0)
        return float(
            c.distance_weight * (distance / scale) ** 2 * deadline_factor
            + c.remaining_event_weight * uncompleted
        )

    def final_cost(
        self,
        history: Sequence[Any],
        state: ProgressState,
        t: int,
        values: Mapping[str, float] | None = None,
    ) -> float:
        self.evaluations += 1
        c = self.coefficients
        if values is None:
            values = self.values(history[-1])
        if state.failed:
            return c.failure_penalty
        if self.program.successful(state, values, t):
            return 0.0
        event = self.program.next_event(state) or self.program.final_event
        if event is None:
            return c.terminal_failure_penalty
        distance = distance_to_interval(values[event.observable], event.target)
        scale = max(c.distance_scale, np.finfo(float).tiny)
        return float(c.terminal_failure_penalty + c.distance_weight * (distance / scale) ** 2)

    def log_psi(
        self,
        history: Sequence[Any],
        state: ProgressState,
        t: int,
        values: Mapping[str, float] | None = None,
    ) -> float:
        cost = (
            self.final_cost(history, state, t, values)
            if int(t) >= self.horizon
            else self.prefix_cost(history, state, t, values)
        )
        raw = -self.coefficients.lambda_program * cost
        return float(max(raw, np.log(self.coefficients.potential_floor)))

    def candidate_log_psi(
        self,
        parent_history: Sequence[Any],
        parent_state: ProgressState,
        frame: Any,
        t: int,
    ) -> tuple[float, ProgressState, dict[str, float]]:
        state, values = self.advance(parent_state, frame, t)
        history = list(parent_history) + [frame]
        return self.log_psi(history, state, t, values), state, values

