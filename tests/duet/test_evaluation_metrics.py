import numpy as np

from confmh.duet.evaluation_metrics import summarize_pre_resampling_population


def _record(t, particle, increment, stage, failed=False):
    return {
        "t": t,
        "particle": particle,
        "log_increment": increment,
        "progress_stage": stage,
        "progress_failed": failed,
    }


def test_pre_resampling_success_mass_does_not_count_output_duplicates():
    records = [
        _record(1, 0, 0.0, 0),
        _record(1, 1, 0.0, 0),
        _record(2, 0, np.log(9.0), 3),
        _record(2, 1, 0.0, 0, failed=True),
    ]
    summary = summarize_pre_resampling_population(
        records,
        outer_k=2,
        horizon=2,
        method="duet",
        resampling_ess_fraction=0.25,
        required_success_stage=3,
    )

    assert summary["pre_resampling_unique_success_count"] == 1
    assert summary["pre_resampling_unique_success_rate"] == 0.5
    assert np.isclose(summary["pre_resampling_success_weight_mass"], 0.9)
    assert np.isclose(summary["expected_success_copies_after_final_resampling"], 1.8)
    assert summary["preterminal_outer_resampling_count"] == 0
