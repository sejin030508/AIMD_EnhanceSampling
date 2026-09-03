from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from confmh.adapters.base_iterative_frame import IterativeFrameAdapter
from confmh.duet.resampling import gather_state


@dataclass
class MockSE3ParticleState:
    translations: np.ndarray
    rotations: np.ndarray
    residue_mask: np.ndarray
    time_index: np.ndarray
    latent: np.ndarray
    conditioning_ids: np.ndarray
    ancestor_metadata: np.ndarray
    endpoint: np.ndarray | None = None


class MockIterativeFrameAdapter(IterativeFrameAdapter):
    """Dependency-light stochastic coordinate adapter used by pipeline tests."""

    def __init__(self, reverse_steps: int = 8, residues: int = 4) -> None:
        super().__init__()
        self.reverse_steps = int(reverse_steps)
        self.residues = int(residues)

    def load_model(self) -> None:
        return None

    def prepare_history(self, history: Sequence[Any]) -> dict[str, Any]:
        self.accounting.temporal_encoder_evaluations += 1
        return {"history": [np.asarray(frame, dtype=float).copy() for frame in history]}

    def initialize_inner_particles(
        self, history_state: Any, count: int, seeds: Sequence[int]
    ) -> MockSE3ParticleState:
        last = np.asarray(history_state["history"][-1], dtype=float)
        translations = np.stack(
            [
                last
                + np.random.default_rng(np.random.SeedSequence([int(seed), 0])).normal(
                    0.0, 0.2, last.shape
                )
                for seed in seeds
            ]
        )
        rotations = np.tile(np.eye(3), (count, self.residues, 1, 1))
        return MockSE3ParticleState(
            translations=translations,
            rotations=rotations,
            residue_mask=np.ones((count, self.residues), dtype=bool),
            time_index=np.full(count, self.reverse_steps, dtype=int),
            latent=np.stack([np.full((self.residues, 2), seed % 17) for seed in seeds]),
            conditioning_ids=np.arange(count, dtype=int),
            ancestor_metadata=np.arange(count, dtype=int),
        )

    def denoise_to_checkpoint(
        self, particle_state: MockSE3ParticleState, checkpoint_progress: float
    ) -> MockSE3ParticleState:
        target = int(round(self.reverse_steps * float(checkpoint_progress)))
        completed = self.reverse_steps - particle_state.time_index
        steps_per_particle = np.maximum(target - completed, 0)
        if not np.all(steps_per_particle == steps_per_particle[0]):
            raise RuntimeError("Mock particles must remain synchronized in diffusion time")
        steps = int(steps_per_particle[0])
        particle_state.translations *= 1.0 - 0.03 * steps
        particle_state.time_index -= steps
        self.accounting.reverse_decoder_evaluations += len(particle_state.time_index) * steps
        return particle_state

    def predict_clean(self, particle_state: MockSE3ParticleState) -> list[Any]:
        self.accounting.predicted_clean_evaluations += len(particle_state.time_index)
        return [coords.copy() for coords in particle_state.translations]

    def resample_particle_state(
        self, particle_state: MockSE3ParticleState, ancestor_indices: Any
    ) -> MockSE3ParticleState:
        return gather_state(particle_state, np.asarray(ancestor_indices, dtype=int))

    def reseed_particle_state(
        self,
        particle_state: MockSE3ParticleState,
        independent_seeds: Sequence[int],
    ) -> MockSE3ParticleState:
        if len(independent_seeds) != len(particle_state.time_index):
            raise ValueError("Expected one independent continuation seed per particle")
        return particle_state

    def denoise_to_end(
        self, particle_state: MockSE3ParticleState, independent_seeds: Sequence[int]
    ) -> MockSE3ParticleState:
        self.reseed_particle_state(particle_state, independent_seeds)
        remaining = particle_state.time_index.copy()
        endpoint = []
        for coords, steps, seed in zip(
            particle_state.translations, remaining, independent_seeds
        ):
            noise = np.random.default_rng(np.random.SeedSequence([int(seed), 1])).normal(
                0.0, 0.01 * max(int(steps), 1), coords.shape
            )
            endpoint.append(coords + noise)
        particle_state.endpoint = np.stack(endpoint)
        self.accounting.reverse_decoder_evaluations += int(np.sum(remaining))
        particle_state.time_index[:] = 0
        return particle_state

    def finalize_frames(self, particle_state: MockSE3ParticleState) -> list[Any]:
        if particle_state.endpoint is None:
            raise RuntimeError("Mock particles have not been denoised to the endpoint")
        self.accounting.generated_complete_frames += len(particle_state.endpoint)
        return [frame.copy() for frame in particle_state.endpoint]

    def validate_frames(self, frames: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {"valid": bool(np.all(np.isfinite(frame))), "nonfinite": int(np.size(frame) - np.isfinite(frame).sum())}
            for frame in frames
        ]
