from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from confmh.config import resolve_path


@dataclass(frozen=True)
class GuidanceConfig:
    enabled: bool = True
    eta: float = 1.0
    active_fraction: float = 0.25
    clip_ratio: float = 0.25
    clip_eps: float = 1.0e-8
    remove_global_translation: bool = True
    degrees_of_freedom: str = "translation"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "GuidanceConfig":
        values = {} if mapping is None else dict(mapping)
        obj = cls(
            enabled=bool(values.get("enabled", True)),
            eta=float(values.get("eta", 1.0)),
            active_fraction=float(values.get("active_fraction", 0.25)),
            clip_ratio=float(values.get("clip_ratio", 0.25)),
            clip_eps=float(values.get("clip_eps", 1.0e-8)),
            remove_global_translation=bool(values.get("remove_global_translation", True)),
            degrees_of_freedom=str(values.get("degrees_of_freedom", "translation")).lower(),
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.eta < 0:
            raise ValueError("level11.guidance.eta must be >= 0")
        if not 0 < self.active_fraction <= 1:
            raise ValueError("level11.guidance.active_fraction must be in (0, 1]")
        if self.clip_ratio < 0:
            raise ValueError("level11.guidance.clip_ratio must be >= 0")
        if self.clip_eps <= 0:
            raise ValueError("level11.guidance.clip_eps must be positive")
        if self.degrees_of_freedom != "translation":
            raise ValueError("Level 1.1 MVP supports translation guidance only")


def level11_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("level11", {}).get("enabled", False))


def validate_level11_config(cfg: dict[str, Any], *, require_paths: bool = True) -> GuidanceConfig:
    level11 = cfg.get("level11", {})
    if not bool(level11.get("enabled", False)):
        raise ValueError("level11.enabled must be true for a Level 1.1 command")
    backend = str(cfg.get("model", {}).get("backend", "confrover")).lower()
    if backend != "confrover":
        raise ValueError(
            f"Level 1.1 MVP is ConfRover-only; got model.backend={backend!r}. "
            "The existing ProAR path is intentionally unchanged."
        )
    acceptance = str(level11.get("acceptance", {}).get("type", "bias_only")).lower()
    if acceptance != "bias_only":
        raise ValueError("Level 1.1 MVP implements bias-only accept/reject only")
    guidance = GuidanceConfig.from_mapping(level11.get("guidance"))

    if require_paths:
        required = {
            "ConfRover repository": resolve_path(
                cfg, cfg.get("model", {}).get("repo", "external/ConfRover")
            ),
            "PCA model": resolve_path(cfg, cfg["reference"]["pca_model"]),
            "reference CV": resolve_path(cfg, cfg["reference"]["reference_cv"]),
            "starting PDB": resolve_path(cfg, cfg["system"]["start_pdb"]),
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Missing Level 1.1 assets:\n" + "\n".join(missing))
    return guidance
