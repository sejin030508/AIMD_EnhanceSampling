import numpy as np

from confmh.duet.inner_fkc import one_checkpoint_inner_step
from confmh.duet.programs import ProgressState
from confmh.duet.toy import FiniteStateAdapter


def test_one_checkpoint_weights_telescope():
    adapter = FiniteStateAdapter()
    history_state = adapter.prepare_history([0])
    psi = np.asarray([0.2, 0.4, 1.0, 2.0, 3.0, 4.0])
    candidate = lambda x: (float(np.log(psi[int(x)])), ProgressState(), {"x": float(x)})
    result = one_checkpoint_inner_step(
        adapter=adapter,
        history_state=history_state,
        count=8,
        seeds=range(8),
        continuation_seeds=range(100, 108),
        checkpoint_progress=0.75,
        candidate_potential=candidate,
        rng=np.random.default_rng(12),
    )
    assert result.diagnostics.telescoping_max_abs_log_error < 1e-12

