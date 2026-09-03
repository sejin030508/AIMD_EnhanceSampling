from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from confmh.adapters.base_iterative_frame import IterativeFrameAdapter
from confmh.duet.resampling import gather_state


@dataclass
class FiniteInnerState:
    checkpoint_latent: np.ndarray
    parent_state: int
    route_flag: int
    endpoint: np.ndarray | None = None


class FiniteStateAdapter(IterativeFrameAdapter):
    """Exactly enumerable inner chain and history-dependent outer transition."""

    latent_probabilities = np.asarray([0.45, 0.35, 0.20], dtype=float)
    endpoint_given_latent = np.asarray(
        [
            [0.70, 0.20, 0.07, 0.03, 0.00, 0.00],
            [0.10, 0.35, 0.30, 0.15, 0.08, 0.02],
            [0.02, 0.08, 0.15, 0.25, 0.30, 0.20],
        ],
        dtype=float,
    )

    def __init__(self) -> None:
        super().__init__()

    def load_model(self) -> None:
        return None

    @staticmethod
    def _route_flag(history: Sequence[int]) -> int:
        return int(1 in history and 3 not in history[: history.index(1) + 1])

    def prepare_history(self, history: Sequence[Any]) -> dict[str, Any]:
        values = tuple(int(item) for item in history)
        self.accounting.temporal_encoder_evaluations += 1
        return {
            "history": values,
            "parent_state": values[-1],
            "route_flag": self._route_flag(values),
        }

    def _latent_probs(self, parent_state: int, route_flag: int) -> np.ndarray:
        adjustment = np.asarray([0.04 * (parent_state % 2), -0.02, -0.02 * (parent_state % 2)])
        if route_flag:
            adjustment += np.asarray([-0.05, 0.00, 0.05])
        probabilities = np.clip(self.latent_probabilities + adjustment, 1e-6, None)
        return probabilities / probabilities.sum()

    def _endpoint_probs(self, latent: int, parent_state: int, route_flag: int) -> np.ndarray:
        probabilities = self.endpoint_given_latent[int(latent)].copy()
        probabilities = np.roll(probabilities, int(parent_state > 2))
        if route_flag:
            probabilities *= np.asarray([0.7, 0.8, 1.0, 1.1, 1.2, 1.3])
        probabilities = np.clip(probabilities, 1e-9, None)
        return probabilities / probabilities.sum()

    def exact_q(self, history: Sequence[int]) -> np.ndarray:
        parent, route = int(history[-1]), self._route_flag(history)
        result = np.zeros(6, dtype=float)
        for latent, probability in enumerate(self._latent_probs(parent, route)):
            result += probability * self._endpoint_probs(latent, parent, route)
        return result / result.sum()

    def initialize_inner_particles(
        self, history_state: Any, count: int, seeds: Sequence[int]
    ) -> FiniteInnerState:
        probabilities = self._latent_probs(history_state["parent_state"], history_state["route_flag"])
        latent = np.asarray(
            [
                np.random.default_rng(np.random.SeedSequence([int(seed), 0])).choice(
                    3, p=probabilities
                )
                for seed in seeds
            ],
            dtype=int,
        )
        return FiniteInnerState(
            checkpoint_latent=latent,
            parent_state=int(history_state["parent_state"]),
            route_flag=int(history_state["route_flag"]),
        )

    def denoise_to_checkpoint(
        self, particle_state: FiniteInnerState, checkpoint_progress: float
    ) -> FiniteInnerState:
        self.accounting.reverse_decoder_evaluations += len(particle_state.checkpoint_latent)
        return particle_state

    def predict_clean(self, particle_state: FiniteInnerState) -> list[Any]:
        self.accounting.predicted_clean_evaluations += len(particle_state.checkpoint_latent)
        return [int(value * 2) for value in particle_state.checkpoint_latent]

    def resample_particle_state(
        self, particle_state: FiniteInnerState, ancestor_indices: Any
    ) -> FiniteInnerState:
        return gather_state(particle_state, np.asarray(ancestor_indices, dtype=int))

    def reseed_particle_state(
        self,
        particle_state: FiniteInnerState,
        independent_seeds: Sequence[int],
    ) -> FiniteInnerState:
        if len(independent_seeds) != len(particle_state.checkpoint_latent):
            raise ValueError("Expected one independent continuation seed per particle")
        return particle_state

    def denoise_to_end(
        self, particle_state: FiniteInnerState, independent_seeds: Sequence[int]
    ) -> FiniteInnerState:
        self.reseed_particle_state(particle_state, independent_seeds)
        endpoint = []
        for latent, seed in zip(particle_state.checkpoint_latent, independent_seeds):
            probabilities = self._endpoint_probs(
                int(latent), particle_state.parent_state, particle_state.route_flag
            )
            endpoint.append(
                np.random.default_rng(np.random.SeedSequence([int(seed), 1])).choice(
                    6, p=probabilities
                )
            )
        particle_state.endpoint = np.asarray(endpoint, dtype=int)
        self.accounting.reverse_decoder_evaluations += len(endpoint)
        return particle_state

    def finalize_frames(self, particle_state: FiniteInnerState) -> list[Any]:
        if particle_state.endpoint is None:
            raise RuntimeError("Finite particles have no endpoint")
        self.accounting.generated_complete_frames += len(particle_state.endpoint)
        return [int(value) for value in particle_state.endpoint]

    def validate_frames(self, frames: Sequence[Any]) -> list[dict[str, Any]]:
        return [{"valid": 0 <= int(frame) < 6} for frame in frames]


@dataclass
class ExactPathTarget:
    paths: list[tuple[int, ...]]
    base_probabilities: np.ndarray
    rewards: np.ndarray
    target_probabilities: np.ndarray
    normalizer: float

    def marginal(self, t: int, states: int = 6) -> np.ndarray:
        result = np.zeros(states, dtype=float)
        for path, probability in zip(self.paths, self.target_probabilities):
            result[path[t]] += probability
        return result


def enumerate_path_target(
    adapter: FiniteStateAdapter,
    initial_state: int,
    horizon: int,
    reward: Callable[[tuple[int, ...]], float],
) -> ExactPathTarget:
    paths: list[tuple[int, ...]] = []
    base: list[float] = []
    rewards: list[float] = []
    for continuation in itertools.product(range(6), repeat=int(horizon)):
        path = (int(initial_state),) + tuple(int(item) for item in continuation)
        probability = 1.0
        for t in range(1, len(path)):
            probability *= adapter.exact_q(path[:t])[path[t]]
        paths.append(path)
        base.append(probability)
        rewards.append(float(reward(path)))
    base_values = np.asarray(base, dtype=float)
    reward_values = np.asarray(rewards, dtype=float)
    unnormalized = base_values * reward_values
    normalizer = float(unnormalized.sum())
    return ExactPathTarget(
        paths=paths,
        base_probabilities=base_values,
        rewards=reward_values,
        target_probabilities=unnormalized / normalizer,
        normalizer=normalizer,
    )


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def finite_kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    if np.any(q[mask] <= 0):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
