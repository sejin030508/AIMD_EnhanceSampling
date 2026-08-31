import numpy as np

from confmh.duet.toy import total_variation


def _normalized(values):
    values = np.asarray(values, dtype=float)
    return values / values.sum()


def test_naive_ideal_limit_matches_q_psi_squared_not_q_psi():
    q = _normalized([0.5, 0.3, 0.2])
    for psi, minimum_difference in [([1.0, 1.05, 0.95], 0.005), ([0.2, 1.0, 4.0], 0.10)]:
        intended = _normalized(q * np.asarray(psi))
        naive = _normalized(q * np.asarray(psi) ** 2)
        assert total_variation(intended, naive) > minimum_difference

