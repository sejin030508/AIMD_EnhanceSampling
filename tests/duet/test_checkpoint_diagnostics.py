import numpy as np

from confmh.duet.checkpoint_diagnostics import summarize_checkpoint_fidelity


def test_checkpoint_fidelity_detects_perfect_ranking_and_selection_gain():
    checkpoint = np.asarray([-4.0, -3.0, -2.0, -1.0])
    children = np.asarray(
        [
            [-4.2, -3.8],
            [-3.2, -2.8],
            [-2.2, -1.8],
            [-1.2, -0.8],
        ]
    )
    summary = summarize_checkpoint_fidelity(
        checkpoint,
        children,
        child_advanced=children > -2.0,
        child_failed=children < -4.0,
    )

    assert summary["checkpoint_to_conditional_log_psi_spearman"] == 1.0
    assert summary["endpoint_best_rank_by_checkpoint"] == 1
    assert summary["top_checkpoint_overlap_fraction"] == 1.0
    assert summary["checkpoint_selection_log_gain_vs_uniform"] > 0.0
    assert summary["checkpoint_weighted_advance_probability"] > summary[
        "uniform_advance_probability"
    ]


def test_checkpoint_fidelity_reports_uninformative_constant_score():
    checkpoint = np.zeros(4)
    children = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    summary = summarize_checkpoint_fidelity(checkpoint, children)

    assert summary["checkpoint_to_conditional_log_psi_spearman"] is None
    assert summary["checkpoint_selection_log_gain_vs_uniform"] == 0.0
    assert summary["endpoint_best_rank_by_checkpoint"] == 4
