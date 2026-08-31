import numpy as np

from confmh.adapters.mock import MockSE3ParticleState
from confmh.duet.resampling import gather_state


def test_every_particle_state_component_is_resampled_consistently():
    count, residues = 3, 2
    state = MockSE3ParticleState(
        translations=np.arange(count * residues * 3).reshape(count, residues, 3),
        rotations=np.arange(count * residues * 9).reshape(count, residues, 3, 3),
        residue_mask=np.arange(count * residues).reshape(count, residues),
        time_index=np.arange(count),
        latent=np.arange(count * residues * 2).reshape(count, residues, 2),
        conditioning_ids=np.asarray([10, 11, 12]),
        ancestor_metadata=np.asarray([20, 21, 22]),
    )
    ancestors = np.asarray([2, 0, 2, 1])
    selected = gather_state(state, ancestors)
    for field in (
        "translations",
        "rotations",
        "residue_mask",
        "time_index",
        "latent",
        "conditioning_ids",
        "ancestor_metadata",
    ):
        assert np.array_equal(getattr(selected, field), getattr(state, field)[ancestors])

