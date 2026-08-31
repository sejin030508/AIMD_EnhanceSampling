from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class NFEAccounting:
    reverse_decoder_evaluations: int = 0
    predicted_clean_evaluations: int = 0
    temporal_encoder_evaluations: int = 0
    reward_evaluations: int = 0
    generated_complete_frames: int = 0

    def add(self, other: "NFEAccounting") -> None:
        for key, value in asdict(other).items():
            setattr(self, key, int(getattr(self, key)) + int(value))

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}

