from confmh.duet.programs import ProgressState


def test_progress_state_is_immutable_and_records_frames(ordered_program):
    initial = ProgressState()
    after_a = ordered_program.update(initial, {"x": 1.0}, 1)
    after_b = ordered_program.update(after_a, {"x": 2.0}, 2)
    assert initial.stage == 0
    assert after_a.completed_frames == (1,)
    assert after_b.completed_frames == (1, 2)


def test_failed_state_is_absorbing(ordered_program):
    failed = ordered_program.update(ProgressState(), {"x": 2.0}, 2)
    assert ordered_program.update(failed, {"x": 1.0}, 3) == failed

