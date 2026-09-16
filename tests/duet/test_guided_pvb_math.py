import numpy as np
import pytest

from confmh.duet.guided_weights import GuidedInnerWeights


def test_guided_weights_include_proposal_ratio_and_potential():
    tracker = GuidedInnerWeights(3)
    ratio = np.array([0.2, -0.1, 0.4])
    phi = np.array([-1.0, -0.5, -2.0])
    tracker.add_proposal_ratio(ratio)
    observation = tracker.observe(phi)
    expected = np.exp(ratio + phi)
    expected /= expected.sum()
    assert np.allclose(observation.weights, expected)
    assert np.isclose(
        tracker.log_z_hat,
        np.log(np.mean(np.exp(ratio + phi))),
    )
    assert tracker.potential_telescoping_error(phi) < 1.0e-14


def test_guided_weights_follow_resampled_lineage_and_telescope():
    tracker = GuidedInnerWeights(3)
    tracker.add_proposal_ratio(np.array([0.1, 0.2, -0.3]))
    first = np.array([-1.0, -2.0, -3.0])
    tracker.observe(first)
    ancestors = np.array([1, 1, 0])
    tracker.resample(ancestors)
    tracker.add_proposal_ratio(np.array([0.4, -0.2, 0.3]))
    final = np.array([-0.4, -0.7, -0.1])
    tracker.observe(final)
    assert tracker.potential_telescoping_error(final) < 1.0e-14
    assert np.allclose(
        tracker.path_log_proposal_ratio,
        np.array([0.2, 0.2, 0.1]) + np.array([0.4, -0.2, 0.3]),
    )


def test_gaussian_shift_log_ratio_matches_direct_normal_density():
    torch = pytest.importorskip("torch")
    from confmh.adapters.pvb_duet import PVBDuETAdapter

    noise = torch.tensor(
        [[0.2, -0.1, 0.3], [0.0, 0.4, -0.2], [-0.3, 0.1, 0.2], [0.5, 0.2, 0.0]],
        dtype=torch.float64,
    )
    shift = torch.tensor(
        [[0.1, 0.0, -0.2], [0.2, -0.1, 0.0], [0.0, 0.3, 0.1], [-0.2, 0.0, 0.2]],
        dtype=torch.float64,
    )
    std = torch.tensor(0.7, dtype=torch.float64)
    actual = PVBDuETAdapter._gaussian_shift_log_ratio(noise, shift, std, 2)
    guided_draw = shift + std * noise
    direct = (
        -0.5 * (guided_draw / std).square().reshape(2, -1).sum(-1)
        + 0.5 * noise.square().reshape(2, -1).sum(-1)
    )
    assert torch.allclose(actual, direct, atol=1.0e-12, rtol=0.0)


def test_eta_zero_guided_update_is_exactly_the_base_update():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from confmh.adapters.pvb_duet import PVBDuETAdapter, PVBParticleState
    from confmh.duet.accounting import NFEAccounting

    def adapter():
        value = object.__new__(PVBDuETAdapter)
        value.sde_step = 4
        value.model = SimpleNamespace(sigma=0.2)
        value._guidance_strength = 0.0
        value._guidance_update_steps = frozenset(range(3))
        value.accounting = NFEAccounting()

        def decode(state):
            state.cached_drift = torch.full_like(state.xt, 0.25)

        value._decode = decode
        value._particle_noise = lambda state: torch.full_like(state.xt, -0.4)
        return value

    initial = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 10.0
    base_state = PVBParticleState(
        x_rep=initial.clone(), xt=initial.clone(), step=0, count=2,
        noise_seeds=np.array([3, 5]),
    )
    guided_state = PVBParticleState(
        x_rep=initial.clone(), xt=initial.clone(), step=0, count=2,
        noise_seeds=np.array([3, 5]),
    )
    base = adapter()
    guided = adapter()
    base._integrate(base_state)
    correction = guided._guided_integrate(guided_state)
    assert torch.equal(base_state.xt, guided_state.xt)
    assert torch.equal(correction, torch.zeros(2, dtype=torch.float64))


def test_torch_ca_tica_gradient_matches_finite_difference():
    torch = pytest.importorskip("torch")
    from confmh.duet.torch_tica import TorchTicaEndpointPotential

    potential = TorchTicaEndpointPotential(
        atom_count=3,
        kind="ca_distance",
        indices=np.array([[0, 1], [0, 2], [1, 2]]),
        mean=np.array([1.0, 1.5, 1.0]),
        components=np.array([[1.0, 0.1], [0.2, -0.4], [-0.3, 0.7]]),
        target=np.array([0.2, -0.1]),
        coefficient=4.0,
        d0=2.0,
    )
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.4, 0.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    value = potential(coordinates).sum()
    value.backward()
    analytic = float(coordinates.grad[0, 1, 0])
    epsilon = 1.0e-6
    plus = coordinates.detach().clone()
    minus = coordinates.detach().clone()
    plus[0, 1, 0] += epsilon
    minus[0, 1, 0] -= epsilon
    finite = float((potential(plus) - potential(minus)) / (2.0 * epsilon))
    assert np.isclose(analytic, finite, atol=1.0e-6, rtol=1.0e-5)


def test_torch_torsion_features_are_finite_and_differentiable():
    torch = pytest.importorskip("torch")
    from confmh.duet.torch_tica import TorchTicaEndpointPotential

    potential = TorchTicaEndpointPotential(
        atom_count=4,
        kind="torsion",
        indices=np.array([[0, 1, 2, 3]]),
        mean=np.zeros(2),
        components=np.eye(2),
        target=np.array([0.5, -0.2]),
        coefficient=2.0,
        d0=1.3,
    )
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.6, 1.0, 0.2], [2.1, 1.3, 1.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    value = potential(coordinates).sum()
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(coordinates.grad).all()
    assert float(torch.linalg.vector_norm(coordinates.grad)) > 0.0
