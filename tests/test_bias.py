import numpy as np

from confmh.bias import HarmonicBias, OPES1D


def test_harmonic_bias():
    bias = HarmonicBias(center=1.0, kappa_kj_mol=2.0)
    assert np.allclose(bias.energy(np.array([0.0, 1.0, 2.0])), [1.0, 0.0, 1.0])


def test_opes_raises_sampled_basin_and_freezes():
    bias = OPES1D(
        grid_min=-3,
        grid_max=3,
        n_grid=101,
        bandwidth=0.2,
        kbt_kj_mol=2.5,
        bias_factor=10,
        barrier_kj_mol=10,
        update_interval=20,
        adapt_steps=40,
    )
    rng = np.random.default_rng(0)
    for step, value in enumerate(rng.normal(0, 0.15, size=40), start=1):
        energy = float(bias.energy(value))
        bias.observe(value, energy, step)
        bias.maybe_update(step)
    assert bias.frozen
    assert float(bias.energy(0.0)) > float(bias.energy(2.5))
    assert float(np.max(bias.bias_grid)) <= 10.0 + 1e-8
