import numpy as np

from confmh.analysis import wham_weights


def test_wham_weights_are_normalized():
    reduced = np.array([[0.0, 0.2, 1.0, 2.0], [2.0, 1.0, 0.2, 0.0]])
    weights, free_energies = wham_weights(reduced, np.array([2, 2]))
    assert np.isclose(weights.sum(), 1.0)
    assert weights.shape == (4,)
    assert free_energies.shape == (2,)
