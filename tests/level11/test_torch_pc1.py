import numpy as np
import pytest

torch = pytest.importorskip("torch")

from confmh.level11.torch_pc1 import TorchPC1
from confmh.pca_cv import PCACV


def _models():
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.1, 1.2, 0.2], [0.2, 0.3, 1.3]],
        dtype=np.float64,
    )
    mean = reference.reshape(-1) + np.linspace(-0.02, 0.03, reference.size)
    component = np.linspace(-1.0, 1.0, reference.size)
    component /= np.linalg.norm(component)
    numpy_model = PCACV(reference, mean, component[None], np.array([0.2]), np.array([1.7]))
    torch_model = TorchPC1(reference, mean, component, 0.2, 1.7)
    return numpy_model, torch_model, reference


def test_torch_pc1_matches_numpy_and_has_finite_gradient():
    numpy_model, torch_model, reference = _models()
    coords_np = reference + np.array(
        [[0.01, -0.02, 0.03], [0.02, 0.01, -0.01], [-0.01, 0.03, 0.02], [0.0, -0.01, 0.01]]
    )
    expected = float(numpy_model.project_ca(coords_np)[0])
    coords = torch.tensor(coords_np, dtype=torch.float64, requires_grad=True)
    value = torch_model.project(coords)
    value.backward()
    assert np.isclose(float(value.detach()), expected, atol=1e-8)
    assert torch.isfinite(coords.grad).all()


def test_torch_pc1_is_rigid_transform_invariant():
    _, model, reference = _models()
    angle = 0.63
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    first = model.project(torch.tensor(reference, dtype=torch.float64))
    transformed = reference @ rotation + np.array([4.0, -2.0, 3.0])
    second = model.project(torch.tensor(transformed, dtype=torch.float64))
    assert torch.allclose(first, second, atol=1e-8)
