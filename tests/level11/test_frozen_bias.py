from pathlib import Path

import numpy as np
import pytest

from confmh.level11.frozen_bias import FrozenGridBias


def test_frozen_bias_build_save_and_load(tmp_path: Path):
    rng = np.random.default_rng(4)
    values = np.concatenate([rng.normal(-1, 0.3, 500), rng.normal(1, 0.2, 300)])
    bias = FrozenGridBias.from_reference(
        values,
        kbt_kj_mol=2.494,
        bias_factor=10.0,
        barrier_kj_mol=15.0,
        n_grid=101,
    )
    assert np.max(bias.bias_grid_kj_mol) <= 1e-12
    assert np.min(bias.bias_grid_kj_mol) >= -15.0 - 1e-12
    loaded = FrozenGridBias.load(bias.save(tmp_path / "bias.npz"))
    probes = np.linspace(-3, 3, 20)
    assert np.allclose(loaded.energy(probes), bias.energy(probes))


def test_torch_and_numpy_interpolation_agree():
    torch = pytest.importorskip("torch")
    bias = FrozenGridBias(np.linspace(-2, 2, 9), np.linspace(0, -4, 9) ** 2, 10, 15)
    probes = np.linspace(-3, 3, 101)
    tensor = torch.tensor(probes, dtype=torch.float64, requires_grad=True)
    values = bias.torch_energy(tensor)
    assert np.allclose(values.detach().numpy(), bias.energy(probes), atol=1e-12)
    values.sum().backward()
    assert torch.isfinite(tensor.grad).all()
