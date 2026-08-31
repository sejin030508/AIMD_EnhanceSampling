from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from confmh.config import load_config, project_root, resolve_path, save_config


def make_umbrella_configs(
    template_config: str | Path,
    output_dir: str | Path,
    quantiles: list[float] | None = None,
) -> list[Path]:
    template_config = Path(template_config).expanduser().resolve()
    cfg = load_config(template_config)
    if str(cfg["experiment"]["mode"]).lower() != "umbrella":
        raise ValueError("Template must have experiment.mode=umbrella")
    quantiles = quantiles or [0.10, 0.30, 0.50, 0.70, 0.90]
    reference = np.load(resolve_path(cfg, cfg["reference"]["reference_cv"]))["cv"][:, 0]
    centers = np.quantile(reference, quantiles)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root(cfg)
    template_run_dir = Path(str(cfg["run"].get("output_dir", "outputs/umbrella/template")))
    suite_run_dir = template_run_dir.parent
    paths = []
    for index, (quantile, center) in enumerate(zip(quantiles, centers)):
        child = copy.deepcopy(cfg)
        child["project_root"] = str(root)
        child["experiment"]["center"] = float(center)
        child["experiment"]["center_quantile"] = float(quantile)
        child["run"]["seed"] = int(child["run"].get("seed", 20260827)) + index
        child["run"]["output_dir"] = str(suite_run_dir / f"window_{index:02d}")
        path = output_dir / f"window_{index:02d}.yaml"
        save_config(child, path)
        paths.append(path)
    return paths
