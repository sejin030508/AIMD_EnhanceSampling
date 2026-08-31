import math

import numpy as np

from confmh.mh import accept, acceptance_probability, bias_only_log_alpha


def test_bias_only_acceptance():
    assert bias_only_log_alpha(-2.0, 1.0) == 0.0
    assert math.isclose(bias_only_log_alpha(2.0, 1.0), -2.0)
    assert math.isclose(acceptance_probability(-2.0), math.exp(-2.0))


def test_accept_is_deterministic_with_seed():
    rng = np.random.default_rng(1)
    outcomes = [accept(-1.0, rng) for _ in range(5)]
    assert outcomes == [False, False, True, False, True]
