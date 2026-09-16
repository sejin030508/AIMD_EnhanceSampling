from dataclasses import dataclass

import numpy as np
import pytest

from confmh.duet.guided_inner import guided_multi_checkpoint_inner_step
from confmh.duet.programs import ProgressState


@dataclass
class _State:
    x: np.ndarray
    stage: int = 0


class _GuidedAdapter:
    def __init__(self):
        self.reseed_calls = 0
        self.end_seeds = None

    def initialize_inner_particles(self, history_state, count, seeds):
        del history_state
        return _State(np.asarray(seeds, dtype=float) / 10.0)

    def guided_denoise_to_checkpoint(self, state, progress):
        state.x = state.x + float(progress) * 0.1
        state.stage += 1
        return state, np.linspace(-0.2, 0.2, len(state.x))

    def guided_checkpoint_log_potential(self, state):
        return -np.square(state.x)

    def resample_particle_state(self, state, ancestors):
        return _State(state.x[np.asarray(ancestors)].copy(), state.stage)

    def reseed_particle_state(self, state, seeds):
        assert seeds is not None
        self.reseed_calls += 1
        return state

    def guided_denoise_to_end(self, state, seeds):
        self.end_seeds = seeds
        state.x = state.x + 0.05
        return state, np.linspace(0.1, -0.1, len(state.x))

    def finalize_frames(self, state):
        return [np.asarray([value]) for value in state.x]

    def guided_endpoint_log_potential(self, state):
        return -np.square(state.x)


@pytest.mark.parametrize("resampling_enabled", [False, True])
def test_guided_inner_runs_with_and_without_checkpoint_resampling(
    resampling_enabled,
):
    adapter = _GuidedAdapter()

    def potential(frame):
        log_phi = -float(np.asarray(frame)[0] ** 2)
        return log_phi, ProgressState(), {"x": float(frame[0])}

    result = guided_multi_checkpoint_inner_step(
        adapter=adapter,
        history_state=None,
        count=4,
        seeds=[1, 2, 3, 4],
        continuation_seed_families=[[11, 12, 13, 14], [21, 22, 23, 24]],
        checkpoint_progresses=[0.5, 0.75],
        candidate_potential=potential,
        rng=np.random.default_rng(7),
        resampling_enabled=resampling_enabled,
    )
    assert np.isfinite(result.log_z_hat)
    assert result.diagnostics.resampling_enabled is resampling_enabled
    assert len(result.diagnostics.checkpoint_ess) == 2
    assert np.all(np.isfinite(result.diagnostics.path_log_proposal_ratios))
    assert result.diagnostics.potential_telescoping_max_abs_log_error < 1.0e-12
    if resampling_enabled:
        assert adapter.reseed_calls == 1
        assert adapter.end_seeds == [21, 22, 23, 24]
    else:
        assert adapter.reseed_calls == 0
        assert adapter.end_seeds is None
