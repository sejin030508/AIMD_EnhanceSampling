from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


Observable = Callable[[Any], float]


@dataclass(frozen=True)
class Event:
    name: str
    observable: str
    target: tuple[float, float]
    window: tuple[int, int]
    persistence: int = 1

    def __post_init__(self) -> None:
        if self.target[0] > self.target[1]:
            raise ValueError(f"Event {self.name}: invalid target interval {self.target}")
        if self.window[0] < 0 or self.window[0] > self.window[1]:
            raise ValueError(f"Event {self.name}: invalid physical window {self.window}")
        if int(self.persistence) < 1:
            raise ValueError(f"Event {self.name}: persistence must be positive")

    def active(self, t: int) -> bool:
        return self.window[0] <= t <= self.window[1]

    def satisfied(self, value: float, t: int) -> bool:
        return self.active(t) and self.target[0] <= float(value) <= self.target[1]


@dataclass(frozen=True)
class ProgressState:
    stage: int = 0
    failed: bool = False
    completed_frames: tuple[int, ...] = field(default_factory=tuple)
    reason: str | None = None
    current_streak: int = 0
    out_of_order_streak: int = 0

    @property
    def completed(self) -> int:
        return self.stage


@dataclass(frozen=True)
class TemporalProgram:
    kind: str
    events: tuple[Event, ...]
    terminal_event: Event | None = None
    allow_same_frame: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"terminal", "windowed", "ordered"}:
            raise ValueError(f"Unsupported program kind: {self.kind}")
        if self.kind == "terminal" and (len(self.events) != 1 or self.terminal_event is not None):
            raise ValueError("terminal program requires exactly one event")
        if self.kind == "windowed" and len(self.events) != 1:
            raise ValueError("windowed program requires exactly one intermediate event")
        if self.kind == "ordered" and len(self.events) != 2:
            raise ValueError("ordered program requires exactly events A and B")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "TemporalProgram":
        def parse_event(raw: Mapping[str, Any]) -> Event:
            return Event(
                name=str(raw["name"]),
                observable=str(raw["observable"]),
                target=(float(raw["target_interval"][0]), float(raw["target_interval"][1])),
                window=(int(raw["physical_window"][0]), int(raw["physical_window"][1])),
                persistence=int(raw.get("persistence_frames", 1)),
            )

        events = tuple(parse_event(item) for item in cfg.get("events", []))
        terminal = cfg.get("terminal_event")
        return cls(
            kind=str(cfg["type"]),
            events=events,
            terminal_event=None if terminal is None else parse_event(terminal),
            allow_same_frame=bool(cfg.get("allow_same_frame", False)),
        )

    @property
    def ordered_events(self) -> tuple[Event, ...]:
        if self.kind == "terminal":
            return ()
        return self.events

    @property
    def final_event(self) -> Event | None:
        return self.events[0] if self.kind == "terminal" else self.terminal_event

    def update(
        self,
        state: ProgressState,
        values: Mapping[str, float],
        t: int,
    ) -> ProgressState:
        if state.failed:
            return state
        if self.kind == "terminal":
            event = self.events[0]
            if state.stage >= 1:
                return state
            if t > event.window[1] and state.stage == 0:
                return ProgressState(
                    state.stage,
                    True,
                    state.completed_frames,
                    "terminal_deadline",
                    state.current_streak,
                    state.out_of_order_streak,
                )
            hit = event.satisfied(values[event.observable], t)
            streak = state.current_streak + 1 if hit else 0
            if streak >= event.persistence:
                return ProgressState(1, False, state.completed_frames + (t,))
            if t >= event.window[1]:
                return ProgressState(
                    state.stage,
                    True,
                    state.completed_frames,
                    "terminal_deadline",
                    streak,
                    0,
                )
            return ProgressState(
                state.stage,
                False,
                state.completed_frames,
                None,
                streak,
                0,
            )

        events = self.events
        stage = state.stage
        completed = state.completed_frames
        streak = state.current_streak
        out_of_order_streak = state.out_of_order_streak
        advanced_event = False
        if stage < len(events):
            current = events[stage]
            if t > current.window[1]:
                return ProgressState(
                    stage,
                    True,
                    completed,
                    f"deadline:{current.name}",
                    streak,
                    out_of_order_streak,
                )
            current_hit = current.satisfied(values[current.observable], t)
            streak = streak + 1 if current_hit else 0

            if self.kind == "ordered" and stage == 0:
                second = events[1]
                second_hit = second.satisfied(values[second.observable], t)
                out_of_order_streak = out_of_order_streak + 1 if second_hit else 0

            if streak >= current.persistence:
                stage += 1
                completed = completed + (t,)
                streak = 0
                out_of_order_streak = 0
                advanced_event = True
                if self.kind == "ordered" and self.allow_same_frame and stage < len(events):
                    following = events[stage]
                    if following.satisfied(values[following.observable], t):
                        streak = 1
                        if streak >= following.persistence:
                            stage += 1
                            completed = completed + (t,)
                            streak = 0
            elif (
                self.kind == "ordered"
                and stage == 0
                and out_of_order_streak >= events[1].persistence
            ):
                return ProgressState(
                    stage,
                    True,
                    completed,
                    "B_before_A",
                    streak,
                    out_of_order_streak,
                )
            elif t >= current.window[1]:
                return ProgressState(
                    stage,
                    True,
                    completed,
                    f"deadline:{current.name}",
                    streak,
                    out_of_order_streak,
                )
        if (
            self.terminal_event is not None
            and stage == len(events)
            and (not advanced_event or self.allow_same_frame)
        ):
            terminal = self.terminal_event
            if t > terminal.window[1]:
                return ProgressState(
                    stage,
                    True,
                    completed,
                    "terminal_deadline",
                    streak,
                    0,
                )
            terminal_hit = terminal.satisfied(values[terminal.observable], t)
            streak = streak + 1 if terminal_hit else 0
            if streak >= terminal.persistence:
                stage += 1
                completed = completed + (t,)
                streak = 0
            elif t >= terminal.window[1]:
                return ProgressState(
                    stage,
                    True,
                    completed,
                    "terminal_deadline",
                    streak,
                    0,
                )
        elif (
            self.terminal_event is not None
            and stage == len(events)
            and advanced_event
            and not self.allow_same_frame
            and t >= self.terminal_event.window[1]
        ):
            return ProgressState(
                stage,
                True,
                completed,
                "terminal_deadline",
                streak,
                0,
            )
        return ProgressState(stage, False, completed, None, streak, out_of_order_streak)

    def successful(self, state: ProgressState, values: Mapping[str, float], t: int) -> bool:
        if state.failed:
            return False
        if self.kind == "terminal":
            return state.stage >= 1
        required = len(self.events) + int(self.terminal_event is not None)
        return state.stage >= required

    def next_event(self, state: ProgressState) -> Event | None:
        if state.failed:
            return None
        if self.kind == "terminal":
            return None if state.stage >= 1 else self.events[0]
        if state.stage < len(self.events):
            return self.events[state.stage]
        return self.terminal_event if state.stage == len(self.events) else None

    def all_observables(self) -> tuple[str, ...]:
        names = [event.observable for event in self.events]
        if self.terminal_event is not None:
            names.append(self.terminal_event.observable)
        return tuple(dict.fromkeys(names))
