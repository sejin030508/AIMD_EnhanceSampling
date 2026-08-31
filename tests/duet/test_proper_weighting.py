import numpy as np

from confmh.duet.inner_fkc import one_checkpoint_inner_step
from confmh.duet.programs import ProgressState
from confmh.duet.toy import FiniteStateAdapter


def test_joint_selected_child_and_normalizer_identity():
    adapter = FiniteStateAdapter()
    history = [0]
    state = adapter.prepare_history(history)
    q = adapter.exact_q(history)
    psi = np.asarray([0.25, 0.5, 1.0, 1.5, 2.5, 4.0])
    functions = [
        lambda x: 1.0,
        lambda x: float(x) / 5.0,
        lambda x: float(int(x) in {1, 4}),
    ]
    samples = [[] for _ in functions]
    repetitions = 3500
    for replicate in range(repetitions):
        base = 10000 + replicate * 43
        result = one_checkpoint_inner_step(
            adapter=adapter,
            history_state=state,
            count=3,
            seeds=[base + i for i in range(3)],
            continuation_seeds=[base + 100 + i for i in range(3)],
            checkpoint_progress=0.75,
            candidate_potential=lambda x: (
                float(np.log(psi[int(x)])), ProgressState(), {"x": float(x)}
            ),
            rng=np.random.default_rng(base + 300),
        )
        zhat = np.exp(result.log_z_hat)
        for values, function in zip(samples, functions):
            values.append(zhat * function(result.frame))
    for values, function in zip(samples, functions):
        exact = float(sum(q[x] * psi[x] * function(x) for x in range(6)))
        values = np.asarray(values)
        standard_error = values.std(ddof=1) / np.sqrt(repetitions)
        assert abs(values.mean() - exact) < 4.5 * standard_error + 0.012

