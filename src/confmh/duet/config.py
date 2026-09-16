from __future__ import annotations

import copy
import math
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
    "guided_duet",
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


def guidance_update_steps(cfg: dict[str, Any]) -> tuple[int, ...] | None:
    """Resolve optional zero-based guided PVB stochastic update indices."""
    guidance = cfg.get("guidance", {})
    schedule = guidance.get("schedule", "all_stochastic")
    if schedule == "all_stochastic":
        if guidance.get("update_steps") is not None:
            raise ValueError(
                "guidance.update_steps must be omitted for all_stochastic schedule"
            )
        return None
    if schedule != "explicit":
        raise ValueError("guidance.schedule must be all_stochastic or explicit")
    raw = guidance.get("update_steps")
    if not isinstance(raw, list) or not raw:
        raise ValueError("explicit guidance schedule requires update_steps")
    steps = tuple(int(item) for item in raw)
    if len(set(steps)) != len(steps) or tuple(sorted(steps)) != steps:
        raise ValueError("guidance.update_steps must be unique and increasing")
    return steps


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
    potential = str(program.get("potential", "rmsd"))
    if potential not in {"rmsd", "tica"}:
        raise ValueError("program.potential must be rmsd or tica")
    if potential == "tica":
        tica = program.get("tica", {})
        if not isinstance(tica, dict):
            raise ValueError("program.tica must be a mapping")
        if int(tica.get("dimensions", 2)) < 1:
            raise ValueError("program.tica.dimensions must be positive")
    if float(program.get("potential_floor", 0.0)) <= 0.0:
        raise ValueError("program.potential_floor must be positive")
    methods = experiment.get("methods", [experiment.get("method", "duet")])
    unknown = set(methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")
    if "guided_duet" in methods:
        if str(model.get("backend", "confrover")) != "pvb":
            raise ValueError("guided_duet currently supports only model.backend=pvb")
        if potential != "tica":
            raise ValueError("guided_duet requires program.potential=tica")
        guidance = cfg.get("guidance")
        if not isinstance(guidance, dict):
            raise ValueError("guided_duet requires a top-level guidance mapping")
        if guidance.get("enabled", True) is not True:
            raise ValueError("guided_duet requires guidance.enabled=true")
        strength = float(guidance.get("strength", -1.0))
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("guidance.strength must be non-negative")
        if not isinstance(guidance.get("inner_resampling", True), bool):
            raise ValueError("guidance.inner_resampling must be boolean")
        steps = guidance_update_steps(cfg)
        if steps is not None:
            sde_step = int(model.get("sde_step", 20))
            if any(item < 0 or item >= sde_step - 1 for item in steps):
                raise ValueError(
                    "guidance.update_steps may only select stochastic PVB updates"
                )


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
    if isinstance(program.get("tica"), dict) and program["tica"].get(
        "projection_npz"
    ):
        values.append(("program.tica.projection_npz", program["tica"]["projection_npz"]))
    case_study = cfg.get("case_study", {})
    if case_study.get("benchmark_spec"):
        values.append(("case_study.benchmark_spec", case_study["benchmark_spec"]))
    missing = []
    for label, value in values:
        if not resolve_config_path(cfg, value).exists():
            missing.append(f"{label}: {resolve_config_path(cfg, value)}")
    return missing
