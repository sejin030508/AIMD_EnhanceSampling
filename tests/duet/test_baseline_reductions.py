import numpy as np

from confmh.duet.baselines import complete_frame_nested_step
from confmh.duet.inner_fkc import one_checkpoint_inner_step
from confmh.duet.programs import ProgressState
from confmh.duet.toy import FiniteStateAdapter


def test_m_one_returns_base_child_and_psi_normalizer():
    adapter = FiniteStateAdapter()
    state = adapter.prepare_history([0])
    psi = np.asarray([0.3, 0.4, 0.8, 1.2, 2.0, 3.0])
    result = one_checkpoint_inner_step(
        adapter=adapter,
        history_state=state,
        count=1,
        seeds=[3],
        continuation_seeds=[4],
        checkpoint_progress=0.75,
        candidate_potential=lambda x: (
            float(np.log(psi[int(x)])), ProgressState(), {"x": float(x)}
        ),
        rng=np.random.default_rng(5),
    )
    assert np.isclose(result.log_z_hat, np.log(psi[int(result.frame)]))


def test_complete_nested_returns_selected_child_and_mean_potential():
    adapter = FiniteStateAdapter()
    state = adapter.prepare_history([0])
    psi = np.asarray([0.2, 0.4, 0.8, 1.6, 2.5, 3.5])
    result = complete_frame_nested_step(
        adapter=adapter,
        history_state=state,
        count=6,
        seeds=range(6),
        checkpoint_progress=0.75,
        candidate_potential=lambda x: (
            float(np.log(psi[int(x)])), ProgressState(), {"x": float(x)}
        ),
        rng=np.random.default_rng(8),
    )
    assert np.isclose(np.exp(result.log_z_hat), np.mean(np.exp(result.candidate_log_potentials)))
    assert np.isclose(result.log_psi, result.candidate_log_potentials[result.selected_index])

