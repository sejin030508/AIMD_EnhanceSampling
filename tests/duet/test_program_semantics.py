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

