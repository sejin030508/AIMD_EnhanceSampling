from __future__ import annotations

"""Sequential importance weights for guided inner bridge proposals."""

from dataclasses import dataclass

import numpy as np

from confmh.duet.resampling import effective_sample_size, normalize_log_weights


@dataclass(frozen=True)
class GuidedWeightObservation:
    weights: np.ndarray
    ess: float
    log_normalizer_increment: float
    log_potential: np.ndarray


class GuidedInnerWeights:
    """One physical parent with ``M`` guided candidates.

    Proposal ratios are ``log(p_base / q_guided)``.  Checkpoint potentials
    telescope along ancestry, while proposal ratios remain in the path weight.
    """

    def __init__(self, count: int) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        self.count = int(count)
        self.log_weights = np.full(count, -np.log(count), dtype=np.float64)
        self.previous_log_potential = np.zeros(count, dtype=np.float64)
        self.pending_log_proposal_ratio = np.zeros(count, dtype=np.float64)
        self.path_log_proposal_ratio = np.zeros(count, dtype=np.float64)
        self.path_log_potential_increment = np.zeros(count, dtype=np.float64)
        self.log_z_hat = 0.0

    @staticmethod
    def _finite(values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError(f"Non-finite {name}")
        return result

    def add_proposal_ratio(self, log_ratio: np.ndarray) -> None:
        values = self._finite(log_ratio, "proposal log ratio")
        if values.shape != (self.count,):
            raise ValueError("Expected one proposal ratio per inner candidate")
        self.pending_log_proposal_ratio += values
        self.path_log_proposal_ratio += values

    def observe(self, log_potential: np.ndarray) -> GuidedWeightObservation:
        potential = self._finite(log_potential, "log potential")
        if potential.shape != (self.count,):
            raise ValueError("Expected one log potential per inner candidate")
        potential_increment = potential - self.previous_log_potential
        unnormalized = (
            self.log_weights
            + self.pending_log_proposal_ratio
            + potential_increment
        )
        weights, log_total = normalize_log_weights(unnormalized)
        self.log_z_hat += float(log_total)
        # Keep the normalized weights in log space.  Converting the exponentiated
        # values back with ``log`` loses finite, very small weights to ``-inf``.
        self.log_weights = unnormalized - float(log_total)
        self.path_log_potential_increment += potential_increment
        self.previous_log_potential = potential.copy()
        self.pending_log_proposal_ratio.fill(0.0)
        return GuidedWeightObservation(
            weights=weights.copy(),
            ess=effective_sample_size(weights),
            log_normalizer_increment=float(log_total),
            log_potential=potential.copy(),
        )

    def resample(self, ancestors: np.ndarray) -> None:
        indices = np.asarray(ancestors, dtype=np.int64)
        if indices.shape != (self.count,):
            raise ValueError("Expected one ancestor per inner candidate")
        if np.any(indices < 0) or np.any(indices >= self.count):
            raise ValueError("Ancestor index outside inner population")
        if np.any(self.pending_log_proposal_ratio != 0.0):
            raise RuntimeError("observe() must be called before resampling")
        self.previous_log_potential = self.previous_log_potential[indices].copy()
        self.path_log_proposal_ratio = self.path_log_proposal_ratio[indices].copy()
        self.path_log_potential_increment = (
            self.path_log_potential_increment[indices].copy()
        )
        self.log_weights.fill(-np.log(self.count))

    def potential_telescoping_error(self, final_log_potential: np.ndarray) -> float:
        final = self._finite(final_log_potential, "final log potential")
        if final.shape != (self.count,):
            raise ValueError("Expected one final log potential per candidate")
        return float(np.max(np.abs(self.path_log_potential_increment - final)))
