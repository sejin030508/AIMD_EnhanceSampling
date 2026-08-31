import numpy as np

from confmh.duet.inner_fkc import one_checkpoint_inner_step
from confmh.duet.programs import ProgressState
from confmh.duet.toy import FiniteStateAdapter, total_variation


def test_constant_potential_preserves_base_endpoint_distribution():
    adapter = FiniteStateAdapter()
    history = [0]
    state = adapter.prepare_history(history)
    counts = np.zeros(6)
    for replicate in range(3000):
        base = 20000 + replicate * 17
        result = one_checkpoint_inner_step(
            adapter=adapter,
            history_state=state,
            count=4,
            seeds=[base + i for i in range(4)],
            continuation_seeds=[base + 50 + i for i in range(4)],
            checkpoint_progress=0.75,
            candidate_potential=lambda x: (0.0, ProgressState(), {"x": float(x)}),
            rng=np.random.default_rng(base + 90),
        )
        counts[int(result.frame)] += 1
        assert abs(result.log_z_hat) < 1e-12
    assert total_variation(counts / counts.sum(), adapter.exact_q(history)) < 0.045

