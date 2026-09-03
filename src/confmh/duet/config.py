from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("model", "trajectory", "particles", "program", "experiment")
SUPPORTED_METHODS = {
    "frozen",
    "best_of_budget",
    "outer_only",
    "inner_only",
    "naive_dual",
    "complete_nested",
    "duet",
}


def load_duet_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg = expand_env(cfg)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    validate_duet_config(cfg)
    return cfg


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def project_root(cfg: dict[str, Any]) -> Path:
    root = Path(str(cfg.get("project_root", cfg["_config_dir"]))).expanduser()
    if not root.is_absolute():
        root = Path(cfg["_config_dir"]) / root
    return root.resolve()


def resolve_config_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root(cfg) / path).resolve()


def inner_checkpoint_progresses(cfg: dict[str, Any]) -> tuple[float, ...]:
    particles = cfg["particles"]
    raw_progresses = particles.get("inner_checkpoint_progresses")
    if raw_progresses is None:
        return (float(particles.get("inner_checkpoint_progress", 0.75)),)
    if not isinstance(raw_progresses, list) or not raw_progresses:
        raise ValueError("particles.inner_checkpoint_progresses must be a non-empty list")
    return tuple(float(item) for item in raw_progresses)


def validate_duet_config(cfg: dict[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise KeyError(f"Missing required config sections: {missing}")
    model = cfg["model"]
    particles = cfg["particles"]
    trajectory = cfg["trajectory"]
    program = cfg["program"]
    experiment = cfg["experiment"]
    if str(model.get("sampler_mode", "sde")) not in {"ode", "sde", "exact"}:
        raise ValueError("model.sampler_mode must be ode, sde, or exact")
    if int(model.get("reverse_steps", 1)) < 1:
        raise ValueError("model.reverse_steps must be positive")
    if int(model.get("decoder_microbatch_size", 1)) < 1:
        raise ValueError("model.decoder_microbatch_size must be positive")
    if int(trajectory.get("horizon", 0)) < 1:
        raise ValueError("trajectory.horizon must be positive")
    for key in ("outer_k", "inner_m"):
        if int(particles.get(key, 0)) < 1:
            raise ValueError(f"particles.{key} must be positive")
    progresses = inner_checkpoint_progresses(cfg)
    if any(not 0.0 < progress < 1.0 for progress in progresses):
        raise ValueError("Every inner checkpoint progress must be in (0, 1)")
    if any(right <= left for left, right in zip(progresses, progresses[1:])):
        raise ValueError("Inner checkpoint progresses must be strictly increasing")
    if particles.get("outer_resampling", "systematic") != "systematic":
        raise ValueError("The correctness protocol requires systematic outer resampling")
    ess_fraction = float(particles.get("outer_resampling_ess_fraction", 1.0))
    if not 0.0 < ess_fraction <= 1.0:
        raise ValueError("particles.outer_resampling_ess_fraction must be in (0, 1]")
    if program.get("type") not in {"terminal", "windowed", "ordered"}:
        raise ValueError("program.type must be terminal, windowed, or ordered")
    if float(program.get("potential_floor", 0.0)) <= 0.0:
        raise ValueError("program.potential_floor must be positive")
    methods = experiment.get("methods", [experiment.get("method", "duet")])
    unknown = set(methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")


def resolved_copy(cfg: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    result["project_root"] = str(project_root(cfg))
    return result


def save_resolved_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_copy(cfg), handle, sort_keys=False)


def estimate_decoder_nfe(cfg: dict[str, Any], method: str | None = None) -> int:
    method = method or str(cfg["experiment"].get("method", "duet"))
    horizon = int(cfg["trajectory"]["horizon"])
    reverse_steps = int(cfg["model"].get("reverse_steps", 1))
    settings = cfg["experiment"].get("method_settings", {}).get(method, {})
    k = int(settings.get("outer_k", cfg["particles"]["outer_k"]))
    m = int(settings.get("inner_m", cfg["particles"]["inner_m"]))
    seeds = len(cfg["experiment"].get("seeds", [cfg["experiment"].get("seed", 0)]))
    tasks = int(cfg["experiment"].get("task_count", 1))
    if method in {"frozen", "best_of_budget"}:
        population = int(cfg["experiment"].get("decoder_population_budget", k * m))
    elif method == "outer_only":
        population = k
    elif method == "inner_only":
        population = m
    else:
        population = k * m
    return int(horizon * reverse_steps * population * seeds * tasks)


def missing_assets(cfg: dict[str, Any]) -> list[str]:
    values: list[tuple[str, Any]] = []
    model = cfg["model"]
    trajectory = cfg["trajectory"]
    reference = cfg.get("reference", {})
    program = cfg.get("program", {})
    for label, key in (
        ("model.repository_path", "repository_path"),
        ("model.checkpoint", "checkpoint"),
        ("model.interpolator_checkpoint", "interpolator_checkpoint"),
    ):
        if model.get(key):
            values.append((label, model[key]))
    if trajectory.get("initial_structure"):
        values.append(("trajectory.initial_structure", trajectory["initial_structure"]))
    if trajectory.get("initial_history"):
        values.append(("trajectory.initial_history", trajectory["initial_history"]))
    for key in ("topology", "pca_model", "reference_cv", "case_manifest"):
        if reference.get(key):
            values.append((f"reference.{key}", reference[key]))
    for key in ("held_out_trajectory",):
        if reference.get(key):
            values.append((f"reference.{key}", reference[key]))
    for key in ("design_trajectories", "held_out_trajectories"):
        for index, item in enumerate(reference.get(key, [])):
            values.append((f"reference.{key}[{index}]", item))
    if program.get("catalog"):
        values.append(("program.catalog", program["catalog"]))
    case_study = cfg.get("case_study", {})
    if case_study.get("benchmark_spec"):
        values.append(("case_study.benchmark_spec", case_study["benchmark_spec"]))
    missing = []
    for label, value in values:
        if not resolve_config_path(cfg, value).exists():
            missing.append(f"{label}: {resolve_config_path(cfg, value)}")
    return missing
