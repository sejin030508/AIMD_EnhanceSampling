import numpy as np
import pytest

torch = pytest.importorskip("torch")

from confmh.level11.config import GuidanceConfig
from confmh.level11.frozen_bias import HarmonicPotential
from confmh.level11.sampler import GuidedEulerSampler
from confmh.level11.torch_pc1 import TorchPC1


class _Rotation:
    pass


class _Rigid:
    def __init__(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def get_rots(self):
        return _Rotation()


class _R3Config:
    coordinate_scaling = 0.1


class _R3:
    _r3_conf = _R3Config()


class _Diffuser:
    def __init__(self):
        self._r3_diffuser = _R3()
        self.scores = []

    def calc_rot_score(self, *args, **kwargs):
        return torch.zeros(1, 4, 3)

    def calc_trans_score(self, *args, **kwargs):
        return torch.ones(1, 4, 3)

    def reverse(self, *, rigids_t, trans_score, **kwargs):
        self.scores.append(trans_score.detach().clone())
        return rigids_t


class _Model(torch.nn.Module):
    def __init__(self, clean):
        super().__init__()
        self.clean = clean

    def forward(self, **kwargs):
        return {"pred_rigids_0": _Rigid(self.clean), "pred_atom14": self.clean[:, :, None, :]}


class _BaseSampler:
    diffusion_steps = 4
    early_stop = 0
    mode = "ode"
    tmin = 0.0
    tmax = 1.0


def _sampler(clip_ratio):
    reference = np.array(
        [[0, 0, 0], [0.12, 0.01, 0], [0.01, 0.11, 0.02], [0.02, 0.03, 0.13]],
        dtype=float,
    )
    component = np.linspace(-1, 1, reference.size)
    component /= np.linalg.norm(component)
    pca = TorchPC1(reference, reference.reshape(-1), component, 0.0, 1.0)
    cfg = GuidanceConfig(active_fraction=0.5, clip_ratio=clip_ratio)
    return GuidedEulerSampler(_BaseSampler(), pca, HarmonicPotential(1.0, 10.0), cfg, 0.4)


def test_mock_sampler_zero_guidance_matches_base_and_schedule():
    clean = torch.tensor(
        [[[0.0, 0, 0], [1.2, 0.1, 0], [0.1, 1.1, 0.2], [0.2, 0.3, 1.3]]]
    )
    feats = {"rigids_mask": torch.ones(1, 4), "aatype": torch.zeros(1, 4, dtype=torch.long)}
    diffuser = _Diffuser()
    sampler = _sampler(0.0)
    with torch.inference_mode():
        sampler.reverse_sample(
            _Model(clean).eval(),
            diffuser,
            _Rigid(torch.zeros_like(clean)),
            torch.zeros(1, 4, 2),
            torch.zeros(1, 4, 4, 2),
            **feats,
        )
    assert len(diffuser.scores) == 4
    assert all(torch.equal(score, torch.ones_like(score)) for score in diffuser.scores)
    assert sampler.diagnostics()["guided_step_count"] == 2
