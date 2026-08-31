from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from confmh.duet.accounting import NFEAccounting


class IterativeFrameAdapter(ABC):
    """Minimal interface between DuET-MD and an iterative frame emulator."""

    def __init__(self) -> None:
        self.accounting = NFEAccounting()

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_history(self, history: Sequence[Any]) -> Any:
        """Encode the complete physical-frame history for one outer particle."""
        raise NotImplementedError

    def clone_or_reencode_history_state(self, history_state: Any, history: Sequence[Any]) -> Any:
        return self.prepare_history(history)

    @abstractmethod
    def initialize_inner_particles(
        self, history_state: Any, count: int, seeds: Sequence[int]
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def denoise_to_checkpoint(self, particle_state: Any, checkpoint_progress: float) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict_clean(self, particle_state: Any) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def resample_particle_state(self, particle_state: Any, ancestor_indices: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def denoise_to_end(self, particle_state: Any, independent_seeds: Sequence[int]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def finalize_frames(self, particle_state: Any) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def validate_frames(self, frames: Sequence[Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def count_decoder_evaluations(self) -> int:
        return int(self.accounting.reverse_decoder_evaluations)

    def sample_complete_frames(
        self,
        history_state: Any,
        count: int,
        seeds: Sequence[int],
        checkpoint_progress: float,
    ) -> list[Any]:
        state = self.initialize_inner_particles(history_state, count, seeds)
        state = self.denoise_to_checkpoint(state, checkpoint_progress)
        state = self.denoise_to_end(state, seeds)
        return self.finalize_frames(state)

