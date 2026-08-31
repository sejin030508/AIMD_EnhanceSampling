import numpy as np

from confmh.adapters.mock import MockIterativeFrameAdapter


def _sample():
    adapter = MockIterativeFrameAdapter(reverse_steps=6, residues=3)
    state = adapter.prepare_history([np.zeros((3, 3))])
    return adapter.sample_complete_frames(state, 3, [1, 2, 3], checkpoint_progress=0.75)


def test_same_seed_family_reproduces_outputs():
    for left, right in zip(_sample(), _sample()):
        assert np.array_equal(left, right)

