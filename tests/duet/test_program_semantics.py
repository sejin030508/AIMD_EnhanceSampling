from confmh.duet.programs import Event, ProgressState, TemporalProgram


def test_terminal_success_and_failure():
    program = TemporalProgram("terminal", (Event("end", "x", (2, 3), (3, 3)),))
    success = program.update(ProgressState(), {"x": 2.5}, 3)
    failure = program.update(ProgressState(), {"x": 0.0}, 4)
    assert program.successful(success, {"x": 2.5}, 3)
    assert failure.failed


def test_windowed_intermediate_success_and_deadline():
    program = TemporalProgram("windowed", (Event("middle", "x", (1, 1), (2, 3)),))
    state = program.update(ProgressState(), {"x": 1.0}, 2)
    assert state.stage == 1
    missed = program.update(ProgressState(), {"x": 0.0}, 4)
    assert missed.failed and missed.reason == "deadline:middle"


def test_ordered_success_and_b_before_a_failure(ordered_program):
    state = ordered_program.update(ProgressState(), {"x": 1.0}, 1)
    state = ordered_program.update(state, {"x": 2.0}, 3)
    assert state.stage == 2 and not state.failed
    failed = ordered_program.update(ProgressState(), {"x": 2.0}, 2)
    assert failed.failed and failed.reason == "B_before_A"


def test_same_frame_cannot_complete_both_unless_enabled():
    events = (
        Event("A", "x", (1, 2), (1, 2)),
        Event("B", "x", (2, 3), (1, 3)),
    )
    strict = TemporalProgram("ordered", events, allow_same_frame=False)
    permissive = TemporalProgram("ordered", events, allow_same_frame=True)
    assert strict.update(ProgressState(), {"x": 2.0}, 1).stage == 1
    assert permissive.update(ProgressState(), {"x": 2.0}, 1).stage == 2


def test_terminal_window_hit_is_absorbing_success_for_path_program():
    program = TemporalProgram(
        "windowed",
        (Event("middle", "x", (1, 1), (2, 3)),),
        terminal_event=Event("end", "x", (3, 4), (4, 6)),
    )
    state = program.update(ProgressState(), {"x": 1.0}, 2)
    state = program.update(state, {"x": 3.5}, 4)
    state = program.update(state, {"x": 0.0}, 6)
    assert state.stage == 2
    assert program.successful(state, {"x": 0.0}, 6)


def test_persistent_ordered_events_and_terminal_record_completion_frame():
    program = TemporalProgram(
        "ordered",
        (
            Event("A", "a", (1, 1), (1, 8), persistence=2),
            Event("B", "b", (1, 1), (1, 8), persistence=2),
        ),
        terminal_event=Event("target", "z", (1, 1), (1, 8), persistence=2),
    )
    state = ProgressState()
    values = (
        {"a": 1, "b": 0, "z": 0},
        {"a": 1, "b": 0, "z": 0},
        {"a": 0, "b": 1, "z": 0},
        {"a": 0, "b": 1, "z": 0},
        {"a": 0, "b": 0, "z": 1},
        {"a": 0, "b": 0, "z": 1},
    )
    for t, row in enumerate(values, start=1):
        state = program.update(state, row, t)
    assert program.successful(state, values[-1], 6)
    assert state.completed_frames == (2, 4, 6)


def test_one_frame_spikes_do_not_complete_persistent_event():
    program = TemporalProgram(
        "terminal",
        (Event("target", "x", (1, 1), (1, 4), persistence=2),),
    )
    state = ProgressState()
    for t, value in enumerate((1, 0, 1, 0), start=1):
        state = program.update(state, {"x": value}, t)
    assert not program.successful(state, {"x": 0}, 4)
    assert state.failed and state.reason == "terminal_deadline"


def test_persistent_b_before_a_is_failure_but_single_b_spike_is_not():
    program = TemporalProgram(
        "ordered",
        (
            Event("A", "a", (1, 1), (1, 6), persistence=2),
            Event("B", "b", (1, 1), (1, 6), persistence=2),
        ),
    )
    state = program.update(ProgressState(), {"a": 0, "b": 1}, 1)
    assert not state.failed
    state = program.update(state, {"a": 0, "b": 0}, 2)
    assert not state.failed
    state = program.update(state, {"a": 0, "b": 1}, 3)
    state = program.update(state, {"a": 0, "b": 1}, 4)
    assert state.failed and state.reason == "B_before_A"


def test_from_config_parses_event_persistence():
    program = TemporalProgram.from_config(
        {
            "type": "terminal",
            "events": [
                {
                    "name": "target",
                    "observable": "x",
                    "target_interval": [0, 1],
                    "physical_window": [1, 3],
                    "persistence_frames": 2,
                }
            ],
        }
    )
    assert program.events[0].persistence == 2
