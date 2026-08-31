import numpy as np

from confmh.duet.inner_fkc import one_checkpoint_inner_step
from confmh.duet.programs import ProgressState
from confmh.duet.toy import FiniteStateAdapter


def test_local_normalizer_is_unbiased_with_monte_carlo_error():
    adapter = FiniteStateAdapter()
    history = [0]
    history_state = adapter.prepare_history(history)
    psi = np.asarray([0.3, 0.7, 1.1, 1.8, 2.4, 3.0])
    exact = float(adapter.exact_q(history) @ psi)
    estimates = []
    for replicate in range(2500):
        base = replicate * 31
        result = one_checkpoint_inner_step(
            adapter=adapter,
            history_state=history_state,
            count=4,
            seeds=[base + i for i in range(4)],
            continuation_seeds=[base + 100 + i for i in range(4)],
            checkpoint_progress=0.75,
            candidate_potential=lambda x: (
                float(np.log(psi[int(x)])), ProgressState(), {"x": float(x)}
            ),
            rng=np.random.default_rng(base + 999),
        )
        estimates.append(np.exp(result.log_z_hat))
    estimates = np.asarray(estimates)
    standard_error = estimates.std(ddof=1) / np.sqrt(len(estimates))
    assert abs(estimates.mean() - exact) < 4.5 * standard_error + 0.01

