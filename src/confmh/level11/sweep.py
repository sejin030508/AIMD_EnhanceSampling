from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from confmh.config import load_config, resolve_path, save_config
from confmh.level11.config import validate_level11_config


def make_level11_umbrella_configs(
    template_path: str | Path,
    output_dir: str | Path,
    *,
    hard: bool = False,
) -> list[Path]:
    cfg = load_config(template_path)
    validate_level11_config(cfg)
    if str(cfg["experiment"]["mode"]).lower() != "umbrella":
        raise ValueError("Level 1.1 umbrella template must use experiment.mode=umbrella")
    reference = np.load(resolve_path(cfg, cfg["reference"]["reference_cv"]))["cv"][:, 0]
    quantiles = np.asarray(
        cfg.get("level11", {}).get("umbrella", {}).get(
            "quantiles", [0.1, 0.3, 0.5, 0.7, 0.9]
        ),
        dtype=float,
    )
    centers = np.quantile(reference, quantiles)
    if hard:
        centers = centers[-1:]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    base_kappa = float(cfg["experiment"]["kappa_kj_mol"])
    for index, center in enumerate(centers):
        window_cfg = copy.deepcopy(cfg)
        window_cfg["experiment"]["center"] = float(center)
        window_cfg["experiment"]["kappa_kj_mol"] = base_kappa * (2.0 if hard else 1.0)
        seed_offset = len(quantiles) - 1 if hard else index
        window_cfg.setdefault("run", {})["seed"] = int(
            cfg.get("run", {}).get("seed", 20260827)
        ) + seed_offset
        name = "hard_window" if hard else f"window_{index:02d}"
        suite = "umbrella_hard" if hard else "umbrella_standard"
        window_cfg["run"]["output_dir"] = f"outputs/level11/{suite}/{name}"
        path = output_dir / f"{name}.yaml"
        save_config(window_cfg, path)
        paths.append(path)
    return paths
