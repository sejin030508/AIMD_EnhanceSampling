from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from confmh.duet.config import load_duet_config, resolve_config_path, save_resolved_config
from confmh.duet.phase_a_cross_clock import (
    ExactTarget,
    ProcessParameters,
    _normalizer_summary,
    _target_tv,
    enumerate_target,
)
from confmh.duet.phase_a_diagnosis_v2 import (
    DiagnosticOutput,
    _canonical_informative,
    _estimates,
    _rmse,
    _route_rmse,
    _run_fixed_schedule,
)
from confmh.duet.records import write_json


METHODS = (
    "duet_uniform",
    "duet_concentrated",
    "complete_uniform",
    "complete_concentrated",
    "outer_only",
)
NEW_METHODS = ("complete_uniform", "complete_concentrated", "outer_only")
PRIMARY_POPULATION = "pre_outer_resampling_weighted"
TV_DEFINITION = "TV_of_mean_estimated_distribution"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def schedule_control_seed_sequence(
    master_seed: int,
    namespace: int,
    instance_id: int,
    method_id: int,
    repetition_id: int,
) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [master_seed, namespace, instance_id, method_id, repetition_id]
    )


def _process_record(process: ProcessParameters) -> dict[str, Any]:
    return asdict(process)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _exact_record(exact: ExactTarget) -> dict[str, Any]:
    return _jsonable(asdict(exact))


def _method_spec(cfg: dict[str, Any], label: str) -> dict[str, Any]:
    phase = cfg["phase_a_schedule_controls_v3"]
    if label == "complete_uniform":
        return {
            "label": label,
            "method": "complete_nested",
            "method_id": int(phase["complete_uniform_method_id"]),
            "outer_k": int(phase["fixed_outer_k"]),
            "schedule": list(phase["uniform_schedule"]),
            "checkpoint": None,
        }
    if label == "complete_concentrated":
        return {
            "label": label,
            "method": "complete_nested",
            "method_id": int(phase["complete_concentrated_method_id"]),
            "outer_k": int(phase["fixed_outer_k"]),
            "schedule": list(phase["concentrated_schedule"]),
            "checkpoint": None,
        }
    if label == "outer_only":
        return {
            "label": label,
            "method": "complete_nested",
            "method_id": int(phase["outer_only_method_id"]),
            "outer_k": int(phase["outer_only_k"]),
            "schedule": list(phase["outer_only_schedule"]),
            "checkpoint": None,
        }
    raise ValueError(f"Unknown new method: {label}")


def _sampler(
    process: ProcessParameters,
    epsilon: float,
    seed: np.random.SeedSequence,
    spec: dict[str, Any],
) -> DiagnosticOutput:
    return _run_fixed_schedule(
        process,
        epsilon,
        seed,
        spec["schedule"],
        method=spec["method"],
        checkpoint="Y2",
        outer_k=spec["outer_k"],
    )


def _joint_residual(
    normalizers: np.ndarray,
    values: np.ndarray,
    target: float,
) -> dict[str, float | list[float]]:
    products = normalizers * (values - float(target))
    mean = float(products.mean())
    standard_error = float(products.std(ddof=1) / math.sqrt(len(products)))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
    }


def evaluate_cell(
    process: ProcessParameters,
    *,
    label: str,
    method_id: int,
    repetitions: int,
    master_seed: int,
    rng_namespace: int,
    sampler_spec: dict[str, Any],
    epsilon: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    exact = enumerate_target(process, epsilon)
    estimates = np.zeros((repetitions, 6), dtype=np.float64)
    normalizers = np.zeros(repetitions, dtype=np.float64)
    costs = np.zeros(repetitions, dtype=np.int64)
    schedules = np.zeros((repetitions, 5, 2), dtype=np.int16)
    returned_success = np.zeros(repetitions, dtype=np.bool_)
    pre_mass: dict[tuple[int, ...], float] = {}
    post_mass: dict[tuple[int, ...], float] = {}
    for repetition in range(repetitions):
        output = _sampler(
            process,
            epsilon,
            schedule_control_seed_sequence(
                master_seed,
                rng_namespace,
                process.instance_id,
                method_id,
                repetition,
            ),
            sampler_spec,
        )
        estimates[repetition] = _estimates(output)
        normalizers[repetition] = output.normalizer
        costs[repetition] = output.reverse_transition_evaluations
        schedules[repetition] = np.asarray(output.schedule, dtype=np.int16)
        returned_success[repetition] = bool(estimates[repetition, 0] > 0.0)
        for path, weight in zip(output.pre_paths, output.pre_weights):
            pre_mass[path] = pre_mass.get(path, 0.0) + float(weight) / repetitions
        for path in output.post_paths:
            post_mass[path] = post_mass.get(path, 0.0) + 1.0 / (
                repetitions * len(output.post_paths)
            )
        if not np.isclose(output.pre_weights.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise AssertionError("Final pre-resampling weights do not sum to one")
        if any(path not in output.pre_paths for path in output.post_paths):
            raise AssertionError("Final resampled path lost proposal ancestry")
    if set(int(value) for value in costs) != {480}:
        raise AssertionError(f"Unexpected costs for {label}: {sorted(set(costs))}")
    pre_success, pre_l, pre_r, post_success, post_l, post_r = estimates.T
    pre_success_rmse = _rmse(pre_success, exact.target_success_probability)
    pre_route_rmse = _route_rmse(pre_l, pre_r, exact)
    post_success_rmse = _rmse(post_success, exact.target_success_probability)
    post_route_rmse = _route_rmse(post_l, post_r, exact)
    summary = {
        "setting": process.name,
        "instance_id": process.instance_id,
        "label": label,
        "sampler_method": sampler_spec["method"],
        "method_id": method_id,
        "rng_entropy_layout": [
            master_seed,
            rng_namespace,
            process.instance_id,
            method_id,
            "repetition_id",
        ],
        "outer_k": sampler_spec["outer_k"],
        "inner_schedule": sampler_spec["schedule"],
        "checkpoint": sampler_spec["checkpoint"],
        "repetitions": repetitions,
        "primary_population": PRIMARY_POPULATION,
        "tv_definition": TV_DEFINITION,
        "pre": {
            "mean_terminal_success_estimate": float(pre_success.mean()),
            "terminal_success_rmse": pre_success_rmse,
            "mean_route_l_estimate": float(pre_l.mean()),
            "mean_route_r_estimate": float(pre_r.mean()),
            "route_mass_rmse": pre_route_rmse,
            "equal_weight_mean_rmse": 0.5 * (pre_success_rmse + pre_route_rmse),
            "target_tv_of_mean_estimated_distribution": _target_tv(pre_mass, exact),
        },
        "post": {
            "mean_terminal_success_estimate": float(post_success.mean()),
            "terminal_success_rmse": post_success_rmse,
            "mean_route_l_estimate": float(post_l.mean()),
            "mean_route_r_estimate": float(post_r.mean()),
            "route_mass_rmse": post_route_rmse,
            "equal_weight_mean_rmse": 0.5 * (post_success_rmse + post_route_rmse),
            "target_tv_of_mean_estimated_distribution": _target_tv(post_mass, exact),
        },
        "normalizer": _normalizer_summary(normalizers, exact.normalizer),
        "joint_residual_pre": {
            "terminal_success": _joint_residual(
                normalizers, pre_success, exact.target_success_probability
            ),
            "route_l": _joint_residual(normalizers, pre_l, exact.route_l_mass),
            "route_r": _joint_residual(normalizers, pre_r, exact.route_r_mass),
        },
        "returned_success_rate": float(returned_success.mean()),
        "returned_success_count": int(returned_success.sum()),
        "returned_success_definition": (
            "At least one hard-success trajectory in the final pre-resampling outer population"
        ),
        "reverse_transition_evaluations_per_repetition": [480],
    }
    raw = {
        "pre_terminal_success_estimate": pre_success,
        "pre_route_l_estimate": pre_l,
        "pre_route_r_estimate": pre_r,
        "post_terminal_success_estimate": post_success,
        "post_route_l_estimate": post_l,
        "post_route_r_estimate": post_r,
        "normalizer_estimate": normalizers,
        "returned_success": returned_success,
        "reverse_transition_evaluations": costs,
        "schedule": schedules,
    }
    return summary, raw


def _run_cell(
    directory: Path,
    process: ProcessParameters,
    *,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    raw_path = directory / "raw_repetitions.npz"
    if resume and summary_path.exists() and raw_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite schedule-control cell: {directory}")
    phase = cfg["phase_a_schedule_controls_v3"]
    summary, raw = evaluate_cell(
        process,
        label=spec["label"],
        method_id=spec["method_id"],
        repetitions=int(phase["repetitions"]),
        master_seed=int(phase["master_seed"]),
        rng_namespace=int(phase["rng_namespace"]),
        sampler_spec=spec,
        epsilon=float(phase["epsilon"]),
    )
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(raw_path, **raw)
    write_json(summary_path, summary)
    return summary


def _reused_paths(diagnosis_root: Path, setting: str, label: str) -> tuple[Path, Path]:
    source_label = "static_m2" if label == "duet_uniform" else "concentrated_t3"
    directory = (
        diagnosis_root
        / "temporal_budget_schedules"
        / "confirmation"
        / setting
        / source_label
    )
    return directory / "summary.json", directory / "raw_repetitions.npz"


def _expected_reused_schedule(label: str) -> list[int]:
    return [2, 2, 2, 2, 2] if label == "duet_uniform" else [1, 1, 6, 1, 1]


def audit_reused_cells(
    cfg: dict[str, Any], diagnosis_root: Path
) -> list[dict[str, Any]]:
    phase = cfg["phase_a_schedule_controls_v3"]
    rows = []
    for process in _canonical_informative():
        for label in ("duet_uniform", "duet_concentrated"):
            summary_path, raw_path = _reused_paths(diagnosis_root, process.name, label)
            if not summary_path.exists() or not raw_path.exists():
                raise FileNotFoundError(f"Missing reusable DuET cell: {summary_path.parent}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            raw = np.load(raw_path)
            schedule = np.asarray(raw["schedule"])
            expected_m = np.asarray(_expected_reused_schedule(label), dtype=np.int16)
            if summary["setting"] != process.name or summary["instance_id"] != process.instance_id:
                raise AssertionError("Reusable DuET process identity mismatch")
            if summary["repetitions"] != int(phase["repetitions"]):
                raise AssertionError("Reusable DuET repetition count mismatch")
            if summary["primary_population"] != PRIMARY_POPULATION:
                raise AssertionError("Reusable DuET primary population mismatch")
            if not np.all(schedule[:, :, 0] == int(phase["fixed_outer_k"])):
                raise AssertionError("Reusable DuET K mismatch")
            if not np.all(schedule[:, :, 1] == expected_m[None, :]):
                raise AssertionError("Reusable DuET schedule mismatch")
            if set(int(value) for value in raw["reverse_transition_evaluations"]) != {480}:
                raise AssertionError("Reusable DuET cost mismatch")
            if raw["pre_terminal_success_estimate"].shape != (int(phase["repetitions"]),):
                raise AssertionError("Reusable DuET raw repetition shape mismatch")
            rows.append(
                {
                    "setting": process.name,
                    "instance_id": process.instance_id,
                    "label": label,
                    "source_summary": str(summary_path),
                    "source_raw": str(raw_path),
                    "summary_sha256": _sha256(summary_path),
                    "raw_sha256": _sha256(raw_path),
                    "repetitions": summary["repetitions"],
                    "outer_k": int(phase["fixed_outer_k"]),
                    "inner_schedule": expected_m.tolist(),
                    "checkpoint": "Y2",
                    "cost": 480,
                    "primary_population": PRIMARY_POPULATION,
                    "returned_success_reconstructible": True,
                    "returned_success_reconstruction": (
                        "pre_terminal_success_estimate > 0; all normalized weights are positive"
                    ),
                }
            )
    return rows


def _origin_files(
    frozen_root: Path, diagnosis_root: Path, reused: list[dict[str, Any]]
) -> list[Path]:
    files = [
        frozen_root / "metrics.json",
        frozen_root / "validation.json",
        frozen_root / "resolved_config.yaml",
        diagnosis_root / "report/confirmation_metrics.json",
        diagnosis_root / "report/confirmation_plan.json",
        diagnosis_root / "reanalysis/frozen_evidence_manifest.json",
    ]
    for row in reused:
        files.extend((Path(row["source_summary"]), Path(row["source_raw"])))
    return files


def _hashes(paths: Sequence[Path], project_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(project_root)): _sha256(path)
        for path in paths
    }


def _load_method_raw(
    output: Path,
    diagnosis_root: Path,
    setting: str,
    label: str,
) -> dict[str, np.ndarray]:
    if label.startswith("duet_"):
        _, raw_path = _reused_paths(diagnosis_root, setting, label)
    else:
        raw_path = output / "cells" / setting / label / "raw_repetitions.npz"
    raw = np.load(raw_path)
    return {name: np.asarray(raw[name]) for name in raw.files}


def _unified_summary(
    output: Path,
    diagnosis_root: Path,
    process: ProcessParameters,
    label: str,
    epsilon: float,
) -> dict[str, Any]:
    exact = enumerate_target(process, epsilon)
    raw = _load_method_raw(output, diagnosis_root, process.name, label)
    success = raw["pre_terminal_success_estimate"].astype(np.float64)
    left = raw["pre_route_l_estimate"].astype(np.float64)
    right = raw["pre_route_r_estimate"].astype(np.float64)
    normalizer = raw["normalizer_estimate"].astype(np.float64)
    success_rmse = _rmse(success, exact.target_success_probability)
    route_rmse = _route_rmse(left, right, exact)
    returned = success > 0.0
    if label.startswith("duet_"):
        source_summary, _ = _reused_paths(diagnosis_root, process.name, label)
        original = json.loads(source_summary.read_text(encoding="utf-8"))
        tv = original["pre"]["target_tv"]
        normalizer_summary = original["normalizer"]
        schedule = _expected_reused_schedule(label)
        outer_k = 16
        checkpoint = "Y2"
        source = "reused_diagnosis_v2"
    else:
        source_summary = output / "cells" / process.name / label / "summary.json"
        original = json.loads(source_summary.read_text(encoding="utf-8"))
        tv = original["pre"]["target_tv_of_mean_estimated_distribution"]
        normalizer_summary = original["normalizer"]
        schedule = original["inner_schedule"]
        outer_k = original["outer_k"]
        checkpoint = original["checkpoint"]
        source = "new_v3"
    return {
        "setting": process.name,
        "instance_id": process.instance_id,
        "method": label,
        "source": source,
        "repetitions": len(success),
        "outer_k": outer_k,
        "inner_schedule": schedule,
        "checkpoint": checkpoint,
        "primary_population": PRIMARY_POPULATION,
        "mean_terminal_success_estimate": float(success.mean()),
        "terminal_success_rmse": success_rmse,
        "mean_route_l_estimate": float(left.mean()),
        "mean_route_r_estimate": float(right.mean()),
        "route_mass_rmse": route_rmse,
        "equal_weight_mean_rmse": 0.5 * (success_rmse + route_rmse),
        "target_tv_of_mean_estimated_distribution": float(tv),
        "normalizer": normalizer_summary,
        "joint_residual_pre": {
            "terminal_success": _joint_residual(
                normalizer, success, exact.target_success_probability
            ),
            "route_l": _joint_residual(normalizer, left, exact.route_l_mass),
            "route_r": _joint_residual(normalizer, right, exact.route_r_mass),
        },
        "returned_success_rate": float(returned.mean()),
        "returned_success_count": int(returned.sum()),
        "returned_success_reconstructed": label.startswith("duet_"),
        "reverse_transition_evaluations_per_repetition": 480,
    }


def _metric_triplet(
    success: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    exact: ExactTarget,
) -> dict[str, float]:
    success_rmse = _rmse(success, exact.target_success_probability)
    route_rmse = _route_rmse(left, right, exact)
    return {
        "terminal_success_rmse": success_rmse,
        "route_mass_rmse": route_rmse,
        "equal_weight_mean_rmse": 0.5 * (success_rmse + route_rmse),
    }


def _bootstrap_method_metrics(
    raw: dict[str, np.ndarray],
    exact: ExactTarget,
    repetitions: int,
    seed: np.random.SeedSequence,
) -> dict[str, np.ndarray]:
    success = raw["pre_terminal_success_estimate"].astype(np.float64)
    left = raw["pre_route_l_estimate"].astype(np.float64)
    right = raw["pre_route_r_estimate"].astype(np.float64)
    n = len(success)
    result = {
        "terminal_success_rmse": np.zeros(repetitions, dtype=np.float64),
        "route_mass_rmse": np.zeros(repetitions, dtype=np.float64),
        "equal_weight_mean_rmse": np.zeros(repetitions, dtype=np.float64),
    }
    rng = np.random.default_rng(seed)
    batch = 100
    for start in range(0, repetitions, batch):
        size = min(batch, repetitions - start)
        indices = rng.integers(0, n, size=(size, n), endpoint=False)
        success_rmse = np.sqrt(
            np.mean((success[indices] - exact.target_success_probability) ** 2, axis=1)
        )
        route_rmse = np.sqrt(
            np.mean(
                (
                    (left[indices] - exact.route_l_mass) ** 2
                    + (right[indices] - exact.route_r_mass) ** 2
                )
                / 2.0,
                axis=1,
            )
        )
        result["terminal_success_rmse"][start : start + size] = success_rmse
        result["route_mass_rmse"][start : start + size] = route_rmse
        result["equal_weight_mean_rmse"][start : start + size] = 0.5 * (
            success_rmse + route_rmse
        )
    return result


def _contrast_specs() -> list[dict[str, str]]:
    return [
        {
            "contrast_group": "scheduling_effect",
            "contrast": "complete_concentrated_vs_complete_uniform",
            "baseline": "complete_uniform",
            "candidate": "complete_concentrated",
        },
        {
            "contrast_group": "scheduling_effect",
            "contrast": "duet_concentrated_vs_duet_uniform",
            "baseline": "duet_uniform",
            "candidate": "duet_concentrated",
        },
        {
            "contrast_group": "pre_completion_steering",
            "contrast": "duet_uniform_vs_complete_uniform",
            "baseline": "complete_uniform",
            "candidate": "duet_uniform",
        },
        {
            "contrast_group": "pre_completion_steering",
            "contrast": "duet_concentrated_vs_complete_concentrated",
            "baseline": "complete_concentrated",
            "candidate": "duet_concentrated",
        },
        {
            "contrast_group": "vs_outer_only",
            "contrast": "complete_concentrated_vs_outer_only",
            "baseline": "outer_only",
            "candidate": "complete_concentrated",
        },
        {
            "contrast_group": "vs_outer_only",
            "contrast": "duet_concentrated_vs_outer_only",
            "baseline": "outer_only",
            "candidate": "duet_concentrated",
        },
    ]


def _bootstrap_contrasts(
    cfg: dict[str, Any],
    output: Path,
    diagnosis_root: Path,
    processes: Sequence[ProcessParameters],
) -> list[dict[str, Any]]:
    phase = cfg["phase_a_schedule_controls_v3"]
    bootstrap_repetitions = int(phase["bootstrap_repetitions"])
    bootstrap_namespace = int(phase["bootstrap_namespace"])
    master_seed = int(phase["master_seed"])
    epsilon = float(phase["epsilon"])
    all_rows: list[dict[str, Any]] = []
    distributions: dict[tuple[str, str, str], np.ndarray] = {}
    points: dict[tuple[str, str, str], float] = {}
    for process in processes:
        exact = enumerate_target(process, epsilon)
        for method_index, label in enumerate(METHODS):
            raw = _load_method_raw(output, diagnosis_root, process.name, label)
            point = _metric_triplet(
                raw["pre_terminal_success_estimate"],
                raw["pre_route_l_estimate"],
                raw["pre_route_r_estimate"],
                exact,
            )
            bootstrap = _bootstrap_method_metrics(
                raw,
                exact,
                bootstrap_repetitions,
                schedule_control_seed_sequence(
                    master_seed,
                    bootstrap_namespace,
                    process.instance_id,
                    method_index,
                    0,
                ),
            )
            for metric in point:
                points[(process.name, label, metric)] = point[metric]
                distributions[(process.name, label, metric)] = bootstrap[metric]
    per_setting_distributions: dict[tuple[str, str], list[np.ndarray]] = {}
    for spec in _contrast_specs():
        for process in processes:
            for metric in (
                "terminal_success_rmse",
                "route_mass_rmse",
                "equal_weight_mean_rmse",
            ):
                baseline = points[(process.name, spec["baseline"], metric)]
                candidate = points[(process.name, spec["candidate"], metric)]
                point = 1.0 - candidate / baseline
                distribution = 1.0 - (
                    distributions[(process.name, spec["candidate"], metric)]
                    / distributions[(process.name, spec["baseline"], metric)]
                )
                ci = np.quantile(distribution, [0.025, 0.975])
                all_rows.append(
                    {
                        **spec,
                        "setting": process.name,
                        "metric": metric,
                        "relative_improvement": point,
                        "bootstrap_ci95_low": float(ci[0]),
                        "bootstrap_ci95_high": float(ci[1]),
                        "bootstrap_repetitions": bootstrap_repetitions,
                        "paired": False,
                    }
                )
                per_setting_distributions.setdefault((spec["contrast"], metric), []).append(
                    distribution
                )
        for metric in (
            "terminal_success_rmse",
            "route_mass_rmse",
            "equal_weight_mean_rmse",
        ):
            setting_rows = [
                row
                for row in all_rows
                if row["contrast"] == spec["contrast"]
                and row["metric"] == metric
                and row["setting"] != "mean_4_settings"
            ]
            distribution = np.mean(
                np.stack(per_setting_distributions[(spec["contrast"], metric)]), axis=0
            )
            ci = np.quantile(distribution, [0.025, 0.975])
            all_rows.append(
                {
                    **spec,
                    "setting": "mean_4_settings",
                    "metric": metric,
                    "relative_improvement": float(
                        np.mean([row["relative_improvement"] for row in setting_rows])
                    ),
                    "bootstrap_ci95_low": float(ci[0]),
                    "bootstrap_ci95_high": float(ci[1]),
                    "bootstrap_repetitions": bootstrap_repetitions,
                    "paired": False,
                }
            )
    return all_rows


def _comparison_csv_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "setting": row["setting"],
            "method": row["method"],
            "source": row["source"],
            "K": row["outer_k"],
            "M_schedule": json.dumps(row["inner_schedule"], separators=(",", ":")),
            "repetitions": row["repetitions"],
            "reverse_transition_evaluations": row[
                "reverse_transition_evaluations_per_repetition"
            ],
            "primary_population": row["primary_population"],
            "terminal_success_mean": row["mean_terminal_success_estimate"],
            "terminal_success_rmse": row["terminal_success_rmse"],
            "route_l_mean": row["mean_route_l_estimate"],
            "route_r_mean": row["mean_route_r_estimate"],
            "route_mass_rmse": row["route_mass_rmse"],
            "combined_rmse": row["equal_weight_mean_rmse"],
            "tv_of_mean_estimated_distribution": row[
                "target_tv_of_mean_estimated_distribution"
            ],
            "normalizer_signed_relative_bias": row["normalizer"][
                "signed_relative_bias"
            ],
            "normalizer_ci95_low": row["normalizer"]["relative_bias_95pct_ci"][0],
            "normalizer_ci95_high": row["normalizer"]["relative_bias_95pct_ci"][1],
            "returned_success_rate": row["returned_success_rate"],
            "returned_success_reconstructed": row[
                "returned_success_reconstructed"
            ],
        }
        for row in summaries
    ]


def _format(value: float) -> str:
    return f"{value:.6f}"


def _percent(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def _contrast_statement(row: dict[str, Any]) -> str:
    point = float(row["relative_improvement"])
    low = float(row["bootstrap_ci95_low"])
    high = float(row["bootstrap_ci95_high"])
    if low > 0.0:
        verdict = "a clearly positive improvement"
    elif high < 0.0:
        verdict = "a clearly negative result"
    elif point > 0.0:
        verdict = "a positive point estimate with an interval crossing zero"
    elif point < 0.0:
        verdict = "a negative point estimate with an interval crossing zero"
    else:
        verdict = "no point-estimate difference"
    return (
        f"{verdict}: {_percent(point)} "
        f"(95% CI [{_percent(low)}, {_percent(high)}])"
    )


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _report(
    output: Path,
    summaries: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    summary_by_key = {(row["setting"], row["method"]): row for row in summaries}
    settings = [process.name for process in _canonical_informative()]
    combined_means = {
        spec["contrast"]: next(
            row
            for row in contrasts
            if row["contrast"] == spec["contrast"]
            and row["setting"] == "mean_4_settings"
            and row["metric"] == "equal_weight_mean_rmse"
        )
        for spec in _contrast_specs()
    }
    mean_rows = {
        (row["contrast"], row["metric"]): row
        for row in contrasts
        if row["setting"] == "mean_4_settings"
    }
    rare_settings = [setting for setting in settings if setting.startswith("rare_")]
    rare_returned = {
        method: float(
            np.mean(
                [
                    summary_by_key[(setting, method)]["returned_success_rate"]
                    for setting in rare_settings
                ]
            )
        )
        for method in METHODS
    }
    lines = [
        "# Phase A schedule controls v3",
        "",
        "## Scope and provenance",
        "",
        "This diagnostic separates physical-time budget scheduling from DuET pre-completion steering. It uses the four previously observed canonical informative settings; it is not a generalization or online-adaptation result.",
        "",
        f"- Frozen Phase A root: `{manifest['source_roots']['phase_a_cross_clock']}`",
        f"- Frozen diagnosis v2 root: `{manifest['source_roots']['phase_a_diagnosis_v2']}`",
        f"- New root: `{output}`",
        "- Existing results were verified read-only before and after the run.",
        "- All primary metrics use the weighted outer population immediately before final outer resampling.",
        "- Every method uses 480 reverse-transition evaluations per repetition; 5,000 independent repetitions per cell.",
        "- TV means TV of the mean estimated distribution, not mean per-run TV.",
        "",
        "## Method comparison",
        "",
    ]
    lines += _md_table(
        [
            "setting",
            "method",
            "success RMSE",
            "route RMSE",
            "combined RMSE",
            "TV",
            "returned success",
            "Z relative bias",
        ],
        [
            [
                setting,
                method,
                _format(summary_by_key[(setting, method)]["terminal_success_rmse"]),
                _format(summary_by_key[(setting, method)]["route_mass_rmse"]),
                _format(summary_by_key[(setting, method)]["equal_weight_mean_rmse"]),
                _format(
                    summary_by_key[(setting, method)][
                        "target_tv_of_mean_estimated_distribution"
                    ]
                ),
                _format(summary_by_key[(setting, method)]["returned_success_rate"]),
                _format(
                    summary_by_key[(setting, method)]["normalizer"][
                        "signed_relative_bias"
                    ]
                ),
            ]
            for setting in settings
            for method in METHODS
        ],
    )
    lines += [
        "",
        "`returned success` counts a repetition when at least one hard-success path is present in the final pre-resampling outer population. For reused DuET cells this is exactly reconstructed as `pre_terminal_success_estimate > 0`; discarded candidates and resampling duplicates are not counted.",
        "",
        "## Required contrasts",
        "",
        "Positive values favor the candidate named in the contrast. CIs use 2,000 independent-repetition bootstrap samples; old and new cells are not paired.",
        "",
    ]
    lines += _md_table(
        ["question", "contrast", "metric", "mean improvement", "95% CI"],
        [
            [
                row["contrast_group"],
                row["contrast"],
                row["metric"],
                _percent(row["relative_improvement"]),
                f"[{_percent(row['bootstrap_ci95_low'])}, {_percent(row['bootstrap_ci95_high'])}]",
            ]
            for row in contrasts
            if row["setting"] == "mean_4_settings"
        ],
    )
    lines += [
        "",
        "Setting-specific contrast rows are preserved in `report/contrasts.csv` and `report/metrics.json`.",
        "",
        "## Answers",
        "",
        "### 1. Does concentration help Complete Nested too?",
        "",
        (
            "Complete Nested concentrated versus uniform shows "
            + _contrast_statement(
                combined_means["complete_concentrated_vs_complete_uniform"]
            )
            + ". DuET concentrated versus uniform shows "
            + _contrast_statement(combined_means["duet_concentrated_vs_duet_uniform"])
            + ". A positive Complete result identifies a physical-time candidate-budget scheduling effect rather than a DuET-specific effect."
        ),
        "",
        "### 2. Does DuET add value at the same concentrated schedule?",
        "",
        (
            "The direct concentrated DuET-versus-Complete combined contrast shows "
            + _contrast_statement(
                combined_means["duet_concentrated_vs_complete_concentrated"]
            )
            + ". Its success contrast shows "
            + _contrast_statement(
                mean_rows[("duet_concentrated_vs_complete_concentrated", "terminal_success_rmse")]
            )
            + ", and its route contrast shows "
            + _contrast_statement(
                mean_rows[("duet_concentrated_vs_complete_concentrated", "route_mass_rmse")]
            )
            + ". Thus the additional signal is confined to terminal-success RMSE; route and combined CIs cross zero, so there is no robust overall same-schedule DuET advantage in these four settings."
        ),
        "",
        "### 3. Do concentrated methods beat Outer-only?",
        "",
        (
            "Complete concentrated versus Outer-only shows "
            + _contrast_statement(
                combined_means["complete_concentrated_vs_outer_only"]
            )
            + ". DuET concentrated versus Outer-only shows "
            + _contrast_statement(combined_means["duet_concentrated_vs_outer_only"])
            + ". Both concentrated methods are significantly worse on combined RMSE, driven by route RMSE, while terminal-success differences cross zero. Therefore neither establishes a distribution-estimation advantage over Outer-only."
        ),
        "",
        "### Path recovery versus distribution estimation",
        "",
        (
            "The ranking differs for rare-setting path recovery: mean returned-success is "
            f"{rare_returned['duet_concentrated']:.4f} for concentrated DuET, "
            f"{rare_returned['complete_concentrated']:.4f} for concentrated Complete, and "
            f"{rare_returned['outer_only']:.4f} for Outer-only. Yet Outer-only has lower combined distribution RMSE. This is an objective difference, not grounds to replace the prespecified primary metric."
        ),
        "",
        "### 4. What supports follow-up online allocation research?",
        "",
        "The evidence supports studying physical-time scheduling mechanisms in these four development/diagnostic processes. It does not currently support a DuET-specific online allocation claim: the direct same-schedule benefit is not robust for route or combined RMSE, and the strong Outer-only baseline remains better for combined distribution estimation. A later method would still need to identify the useful time location without oracle time/state input.",
        "",
        "## Correctness and limitations",
        "",
        "- Signed normalizer relative bias and 95% CI are reported for every method-setting cell.",
        "- Joint residuals `mean[Zhat * (hhat - h_exact)]` for success, route-L, and route-R are stored in `report/metrics.json`.",
        "- A nonzero finite-particle normalized-estimator bias is not treated by itself as a sampler bug, and CI inclusion of zero is not described as proof of correctness.",
        "- No new allocator, checkpoint, reward, topology, held-out family, or M schedule was searched.",
        "",
        "## Raw evidence",
        "",
        "- `manifest.json`",
        "- `cells/<setting>/<new_method>/{summary.json,raw_repetitions.npz}`",
        "- `report/method_comparison.csv`",
        "- `report/contrasts.csv`",
        "- `report/metrics.json`",
        "- `source_snapshot/` (new runner and reused sampler dependencies)",
        "",
    ]
    return "\n".join(lines)


def validate_config(cfg: dict[str, Any]) -> None:
    phase = cfg.get("phase_a_schedule_controls_v3", {})
    expected = {
        "epsilon": 0.05,
        "master_seed": 20260908,
        "rng_namespace": 3001,
        "repetitions": 5000,
        "bootstrap_repetitions": 2000,
        "uniform_schedule": [2, 2, 2, 2, 2],
        "concentrated_schedule": [1, 1, 6, 1, 1],
        "outer_only_schedule": [1, 1, 1, 1, 1],
        "fixed_outer_k": 16,
        "outer_only_k": 32,
        "checkpoint": "Y2",
    }
    errors = [
        f"{key} must be {value!r}, got {phase.get(key)!r}"
        for key, value in expected.items()
        if phase.get(key) != value
    ]
    schedules = (
        (phase.get("fixed_outer_k"), phase.get("uniform_schedule")),
        (phase.get("fixed_outer_k"), phase.get("concentrated_schedule")),
        (phase.get("outer_only_k"), phase.get("outer_only_schedule")),
    )
    for outer_k, schedule in schedules:
        if outer_k is None or schedule is None or 3 * int(outer_k) * sum(schedule) != 480:
            errors.append(f"Schedule cost must be 480: K={outer_k}, M={schedule}")
    if cfg.get("model", {}).get("dtype") != "float64":
        errors.append("dtype must remain float64")
    if cfg.get("trajectory", {}).get("horizon") != 5:
        errors.append("horizon must remain 5")
    if cfg.get("program", {}).get("potential_floor") != 0.05:
        errors.append("potential floor must remain 0.05")
    project_root = Path(cfg["project_root"]).resolve()
    output = resolve_config_path(cfg, cfg["experiment"]["output_directory"])
    frozen = resolve_config_path(cfg, phase["frozen_phase_a_root"])
    diagnosis = resolve_config_path(cfg, phase["diagnosis_v2_root"])
    if output == frozen or frozen in output.parents or output == diagnosis or diagnosis in output.parents:
        errors.append("New output must not overlap either read-only source root")
    if project_root not in output.parents:
        errors.append("Output must remain under project root")
    if errors:
        raise ValueError("Invalid Phase A schedule controls v3 config:\n- " + "\n- ".join(errors))


def run(cfg: dict[str, Any], *, resume: bool) -> Path:
    validate_config(cfg)
    project_root = Path(cfg["project_root"]).resolve()
    phase = cfg["phase_a_schedule_controls_v3"]
    output = resolve_config_path(cfg, cfg["experiment"]["output_directory"])
    frozen_root = resolve_config_path(cfg, phase["frozen_phase_a_root"])
    diagnosis_root = resolve_config_path(cfg, phase["diagnosis_v2_root"])
    diagnosis_config = resolve_config_path(cfg, phase["diagnosis_v2_config"])
    output.mkdir(parents=True, exist_ok=True)

    reused = audit_reused_cells(cfg, diagnosis_root)
    origin_files = _origin_files(frozen_root, diagnosis_root, reused)
    origin_hashes_before = _hashes(origin_files, project_root)
    exact_before = {
        process.name: _exact_record(
            enumerate_target(process, float(phase["epsilon"]))
        )
        for process in _canonical_informative()
    }
    new_summaries = []
    for process in _canonical_informative():
        for label in NEW_METHODS:
            spec = _method_spec(cfg, label)
            new_summaries.append(
                _run_cell(
                    output / "cells" / process.name / label,
                    process,
                    cfg=cfg,
                    spec=spec,
                    resume=resume,
                )
            )
            write_json(
                output / "report/run_progress.json",
                {
                    "completed_new_cells": len(new_summaries),
                    "total_new_cells": 12,
                    "last_setting": process.name,
                    "last_method": label,
                },
            )
    exact_after = {
        process.name: _exact_record(
            enumerate_target(process, float(phase["epsilon"]))
        )
        for process in _canonical_informative()
    }
    if exact_before != exact_after:
        raise RuntimeError("Base transition or exact target changed during scheduling run")
    origin_hashes_after = _hashes(origin_files, project_root)
    if origin_hashes_before != origin_hashes_after:
        raise RuntimeError("A read-only Phase A source result changed")

    summaries = [
        _unified_summary(
            output,
            diagnosis_root,
            process,
            label,
            float(phase["epsilon"]),
        )
        for process in _canonical_informative()
        for label in METHODS
    ]
    contrasts = _bootstrap_contrasts(
        cfg, output, diagnosis_root, _canonical_informative()
    )
    report_root = output / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    comparison_rows = _comparison_csv_rows(summaries)
    _write_csv(report_root / "method_comparison.csv", comparison_rows)
    _write_csv(report_root / "contrasts.csv", contrasts)
    write_json(
        report_root / "metrics.json",
        {
            "method_comparison": summaries,
            "contrasts": contrasts,
            "tv_definition": TV_DEFINITION,
            "primary_population": PRIMARY_POPULATION,
        },
    )

    original_config_text = diagnosis_config.read_text(encoding="utf-8").splitlines()
    new_config_path = project_root / "configs/duet/phase_a_schedule_controls_v3.yaml"
    new_config_text = new_config_path.read_text(encoding="utf-8").splitlines()
    config_diff = list(
        difflib.unified_diff(
            original_config_text,
            new_config_text,
            fromfile=str(diagnosis_config.relative_to(project_root)),
            tofile=str(new_config_path.relative_to(project_root)),
            lineterm="",
        )
    )
    source_snapshot_dir = output / "source_snapshot"
    source_snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_files = [
        Path(__file__),
        project_root / "src/confmh/duet/phase_a_diagnosis_v2.py",
        project_root / "src/confmh/duet/phase_a_cross_clock.py",
        project_root / "src/confmh/duet/resampling.py",
    ]
    source_snapshots = {}
    for source_file in source_files:
        destination = source_snapshot_dir / source_file.name
        shutil.copy2(source_file, destination)
        source_snapshots[source_file.name] = {
            "source": str(source_file),
            "snapshot": str(destination),
            "sha256": _sha256(destination),
        }
    save_resolved_config(cfg, output / "resolved_config.yaml")
    manifest = {
        "experiment": "phase_a_schedule_controls_v3",
        "source_roots": {
            "phase_a_cross_clock": str(frozen_root),
            "phase_a_diagnosis_v2": str(diagnosis_root),
        },
        "source_results_read_only_verified": True,
        "source_hashes_before": origin_hashes_before,
        "source_hashes_after": origin_hashes_after,
        "reused_cells": reused,
        "new_cells": [
            {
                "setting": row["setting"],
                "instance_id": row["instance_id"],
                "method": row["label"],
                "method_id": row["method_id"],
                "repetitions": row["repetitions"],
                "actual_cost": row["reverse_transition_evaluations_per_repetition"][0],
                "summary": str(
                    output / "cells" / row["setting"] / row["label"] / "summary.json"
                ),
                "raw": str(
                    output
                    / "cells"
                    / row["setting"]
                    / row["label"]
                    / "raw_repetitions.npz"
                ),
            }
            for row in new_summaries
        ],
        "rng": {
            "new_cell_entropy_layout": [
                int(phase["master_seed"]),
                int(phase["rng_namespace"]),
                "instance_id",
                "method_id",
                "repetition_id",
            ],
            "method_ids": {
                "complete_uniform": int(phase["complete_uniform_method_id"]),
                "complete_concentrated": int(
                    phase["complete_concentrated_method_id"]
                ),
                "outer_only": int(phase["outer_only_method_id"]),
                "duet_uniform_reused": int(phase["duet_uniform_method_id"]),
                "duet_concentrated_reused": int(
                    phase["duet_concentrated_method_id"]
                ),
            },
            "bootstrap_entropy_layout": [
                int(phase["master_seed"]),
                int(phase["bootstrap_namespace"]),
                "instance_id",
                "method_index",
                0,
            ],
            "python_hash_used": False,
            "old_and_new_cells_paired": False,
        },
        "processes": [_process_record(process) for process in _canonical_informative()],
        "exact_targets_before": exact_before,
        "exact_targets_after": exact_after,
        "sampler": {
            "resampling": "systematic",
            "checkpoint": "Y2 for DuET; not applicable for Complete/Outer-only",
            "primary_population": PRIMARY_POPULATION,
            "post_population_saved_as_secondary": True,
            "tv_definition": TV_DEFINITION,
            "selected_child_interface": True,
            "rao_blackwellized_extra_estimator": False,
        },
        "config": {
            "new_config": str(new_config_path),
            "diagnosis_v2_config": str(diagnosis_config),
            "unified_diff": config_diff,
        },
        "source_snapshots": source_snapshots,
    }
    report = _report(output, summaries, contrasts, manifest)
    (report_root / "REPORT.md").write_text(report, encoding="utf-8")
    write_json(output / "manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_duet_config(args.config.resolve())
    result = run(cfg, resume=args.resume)
    print(result)


if __name__ == "__main__":
    main()
