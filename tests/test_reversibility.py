import numpy as np

from confmh.diagnostics import (
    _fit_reversible_null,
    _projection_summary,
    _transition_residual,
)


def test_reversible_null_has_requested_stationary_distribution():
    stationary = np.asarray([0.2, 0.3, 0.5])
    transition = np.asarray(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ]
    )
    null = _fit_reversible_null(stationary, transition)
    flux = stationary[:, None] * null
    assert np.allclose(null.sum(axis=1), 1.0)
    assert np.allclose(flux, flux.T)
    assert _transition_residual(stationary, null) < 1.0e-10


def test_projection_summary_reports_balanced_counts_and_null():
    rng = np.random.default_rng(7)
    reference = np.linspace(-2.0, 2.0, 1000)
    source = np.repeat([-1.5, -0.5, 0.5, 1.5], 20)
    destination = source + rng.normal(scale=0.2, size=len(source))
    groups = np.repeat(np.arange(20), 4)
    summary, arrays = _projection_summary(
        reference_pc1=reference,
        source_pc1=source,
        destination_pc1=destination,
        group_ids=groups,
        n_bins=4,
        bootstrap_replicates=100,
        seed=11,
    )
    assert summary["num_transitions"] == 80
    assert summary["num_source_structures"] == 20
    assert sum(summary["row_transition_counts"]) == 80
    assert arrays["counts"].shape == (4, 4)
    assert 0.0 <= summary["reversible_null_tail_probability"] <= 1.0
