from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from confmh.duet.config import load_duet_config, resolve_config_path
from confmh.duet.evaluation_metrics import summarize_pre_resampling_population
from confmh.duet.programs import ProgressState, TemporalProgram
from confmh.duet.records import write_json


def _find_runs(inputs: list[str]) -> list[Path]:
    runs: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if (path / "metrics.json").exists():
            runs.add(path)
        elif path.is_dir():
            runs.update(item.parent for item in path.rglob("metrics.json"))
    return sorted(runs)


def _load_program(cfg: dict[str, Any], task: str) -> TemporalProgram:
    catalog_path = resolve_config_path(cfg, cfg["program"]["catalog"])
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    return TemporalProgram.from_config(catalog["tasks"][task])


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _optional_float(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    return None if value is None else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize success before final SMC output duplication."
    )
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for run in _find_runs(args.inputs):
        required = ("metrics.json", "records.json", "resolved_config.yaml")
        if any(not (run / name).exists() for name in required):
            continue
        with (run / "metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        with (run / "records.json").open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        cfg = load_duet_config(run / "resolved_config.yaml")
        task = str(metrics["task"])
        program = _load_program(cfg, task)
        horizon = int(cfg["trajectory"]["horizon"])
        final_rows = sorted(
            (row for row in records if int(row["t"]) == horizon),
            key=lambda row: int(row["particle"]),
        )
        final_success_flags = [
            bool(
                program.successful(
                    ProgressState(
                        stage=int(row["progress_stage"]),
                        failed=bool(row["progress_failed"]),
                    ),
                    row["values"],
                    horizon,
                )
            )
            for row in final_rows
        ]
        stable = summarize_pre_resampling_population(
            records,
            outer_k=int(metrics["outer_k"]),
            horizon=horizon,
            method=str(metrics["method"]),
            resampling_ess_fraction=float(
                cfg["particles"].get("outer_resampling_ess_fraction", 1.0)
            ),
            required_success_stage=(
                1 if program.kind == "terminal" else len(program.events)
            ),
            final_success_flags=final_success_flags,
        )
        summaries.append(
            {
                "run_directory": str(run),
                "method": metrics["method"],
                "task": task,
                "seed": int(metrics["seed"]),
                "outer_k": int(metrics["outer_k"]),
                "inner_m": int(metrics["inner_m"]),
                "decoder_nfe": int(metrics["decoder_nfe"]),
                "post_resampling_success_rate": float(
                    metrics["joint_program_success_rate"]
                ),
                "post_resampling_success_count": int(
                    np.sum(metrics["joint_program_success"])
                ),
                "failure_rate": _optional_float(metrics, "failure_rate"),
                "event_order_accuracy": _optional_float(metrics, "event_order_accuracy"),
                "structural_validity_rate": _optional_float(
                    metrics, "structural_validity_rate"
                ),
                "held_out_path_distance": _optional_float(
                    metrics, "held_out_path_distance"
                ),
                "held_out_endpoint_ca_rmsd_nm": _optional_float(
                    metrics, "held_out_endpoint_ca_rmsd_nm"
                ),
                "reference_intermediate_coverage": _optional_float(
                    metrics, "reference_intermediate_coverage"
                ),
                "unspecified_contact_map_similarity": _optional_float(
                    metrics, "unspecified_contact_map_similarity"
                ),
                "path_pairwise_ca_diversity_nm": _optional_float(
                    metrics, "path_pairwise_ca_diversity_nm"
                ),
                "surviving_initial_ancestors": _optional_float(
                    metrics, "surviving_initial_ancestors"
                ),
                **stable,
            }
        )

    by_group: dict[str, dict[str, Any]] = {}
    groups = sorted(
        {
            (
                str(item["task"]),
                str(item["method"]),
                int(item["outer_k"]),
                int(item["inner_m"]),
            )
            for item in summaries
        }
    )
    aggregate_metrics = (
        "pre_resampling_unique_success_rate",
        "pre_resampling_success_weight_mass",
        "post_resampling_success_rate",
        "failure_rate",
        "event_order_accuracy",
        "structural_validity_rate",
        "held_out_path_distance",
        "held_out_endpoint_ca_rmsd_nm",
        "reference_intermediate_coverage",
        "unspecified_contact_map_similarity",
        "path_pairwise_ca_diversity_nm",
        "surviving_initial_ancestors",
        "preterminal_outer_resampling_count",
    )
    for task, method, outer_k, inner_m in groups:
        rows = [
            item
            for item in summaries
            if (
                item["task"],
                item["method"],
                item["outer_k"],
                item["inner_m"],
            )
            == (task, method, outer_k, inner_m)
        ]
        label = f"{task}|{method}|K{outer_k}|M{inner_m}"
        by_group[label] = {
            "task": task,
            "method": method,
            "outer_k": outer_k,
            "inner_m": inner_m,
            "run_count": len(rows),
            "seeds": [item["seed"] for item in rows],
        }
        for metric in aggregate_metrics:
            values = [float(item[metric]) for item in rows if item.get(metric) is not None]
            by_group[label][f"mean_{metric}"] = _mean(values)
            by_group[label][f"values_{metric}"] = values
    write_json(
        args.output,
        {
            "metric_policy": (
                "Primary metrics are computed before final output resampling so one "
                "successful path cannot be counted repeatedly as independent evidence."
            ),
            "runs": summaries,
            "aggregate_by_task_method_allocation": by_group,
        },
    )
    print(Path(args.output).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
