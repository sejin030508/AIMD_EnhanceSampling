import numpy as np
import pytest

from confmh.adapters.mock import MockIterativeFrameAdapter
from confmh.duet.inner_fkc import multi_checkpoint_inner_step
from confmh.duet.programs import ProgressState


def _candidate(frame):
    value = float(np.mean(np.asarray(frame, dtype=float)))
    return -abs(value), ProgressState(), {"x": value}


def test_multi_checkpoint_telescope_and_decoder_budget():
    adapter = MockIterativeFrameAdapter(reverse_steps=8, residues=4)
    history_state = adapter.prepare_history([np.zeros((4, 3))])
    result = multi_checkpoint_inner_step(
        adapter=adapter,
        history_state=history_state,
        count=5,
        seeds=range(5),
        continuation_seed_families=[range(100, 105), range(200, 205)],
        checkpoint_progresses=[0.5, 0.75],
        candidate_potential=_candidate,
        rng=np.random.default_rng(9),
    )

    assert result.diagnostics.telescoping_max_abs_log_error < 1e-12
    assert result.diagnostics.checkpoint_progresses == (0.5, 0.75)
    assert len(result.diagnostics.checkpoint_ess) == 2
    assert adapter.accounting.reverse_decoder_evaluations == 5 * 8
    assert adapter.accounting.predicted_clean_evaluations == 5 * 2


@pytest.mark.parametrize(
    "progresses",
    ([], [0.0, 0.9], [0.9, 0.9], [0.95, 0.85]),
)
def test_multi_checkpoint_rejects_invalid_schedules(progresses):
    adapter = MockIterativeFrameAdapter(reverse_steps=8, residues=4)
    history_state = adapter.prepare_history([np.zeros((4, 3))])
    with pytest.raises(ValueError):
        multi_checkpoint_inner_step(
            adapter=adapter,
            history_state=history_state,
            count=2,
            seeds=[1, 2],
            continuation_seed_families=[[3, 4] for _ in progresses],
            checkpoint_progresses=progresses,
            candidate_potential=_candidate,
            rng=np.random.default_rng(9),
        )
