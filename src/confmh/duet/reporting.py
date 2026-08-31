from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from confmh.duet.analysis import paired_bootstrap_ci


TASK_ORDER = {"endpoint": 0, "windowed": 1, "ordered": 2}
METHOD_ORDER = {
    "frozen": 0,
    "best_of_budget": 1,
    "outer_only": 2,
    "inner_only": 3,
    "complete_nested": 4,
    "naive_dual": 5,
    "duet": 6,
    "confrover_interp": 7,
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_run_rows(root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob("metrics.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict) or "method" not in payload:
            continue
        row = dict(payload)
        row["run_directory"] = str(path.parent)
        config_path = path.parent / "resolved_config.yaml"
        if config_path.exists():
            import yaml

            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            row["case_id"] = cfg.get("trajectory", {}).get("case_id")
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.median(values)) if values else float("nan")


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("task")), str(row.get("method")))].append(row)
    result = []
    for (task, method), items in sorted(
        grouped.items(), key=lambda item: (TASK_ORDER.get(item[0][0], 99), METHOD_ORDER.get(item[0][1], 99))
    ):
        result.append(
            {
                "task": task,
                "method": method,
                "independent_units": len(items),
                "success_mean": _mean(items, "joint_program_success_rate"),
                "success_median": _median(items, "joint_program_success_rate"),
                "success_per_million_nfe_mean": _mean(items, "success_per_million_decoder_nfe"),
                "success_per_gpu_hour_mean": _mean(items, "successful_outputs_per_gpu_hour"),
                "event_order_accuracy_mean": _mean(items, "event_order_accuracy"),
                "validity_mean": _mean(items, "structural_validity_rate"),
                "failure_rate_mean": _mean(items, "failure_rate"),
                "unresolved_rate_mean": _mean(items, "unresolved_rate"),
                "decoder_nfe_mean": _mean(items, "decoder_nfe"),
                "wall_clock_s_mean": _mean(items, "wall_clock_s"),
                "peak_gpu_memory_bytes_mean": _mean(items, "peak_gpu_memory_bytes"),
            }
        )
    return result


def paired_duet_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_unit: dict[tuple[Any, Any, Any], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row.get("case_id"), row.get("task"), row.get("seed"))
        value = row.get("joint_program_success_rate")
        if value is not None:
            by_unit[key][str(row.get("method"))] = float(value)
    methods = sorted({method for values in by_unit.values() for method in values} - {"duet"})
    result = []
    for method in methods:
        differences = [
            values["duet"] - values[method]
            for values in by_unit.values()
            if "duet" in values and method in values
        ]
        low, high = paired_bootstrap_ci(differences, seed=20260831)
        result.append(
            {
                "comparison": f"duet-minus-{method}",
                "paired_units": len(differences),
                "mean_difference": float(np.mean(differences)) if differences else float("nan"),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    return result


def _placeholder(axis, title: str, message: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def make_figures(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize_methods(rows)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    usable = [row for row in summary if np.isfinite(row["success_per_million_nfe_mean"])]
    if usable:
        for method in ("outer_only", "inner_only", "complete_nested", "naive_dual", "duet"):
            points = [row for row in usable if row["method"] == method]
            points.sort(key=lambda row: TASK_ORDER.get(row["task"], 99))
            if points:
                axis.plot(
                    [row["task"] for row in points],
                    [row["success_per_million_nfe_mean"] for row in points],
                    marker="o",
                    label=method,
                )
        axis.set_ylabel("joint success per million decoder NFE")
        axis.legend(frameon=False)
        axis.set_title("Three-regime clock separation")
    else:
        _placeholder(axis, "Three-regime clock separation", "Awaiting Phase C results")
    fig.tight_layout()
    fig.savefig(output / "figure_1_clock_separation.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.8))
    allocations = [row for row in rows if row.get("method") == "duet" and row.get("outer_k")]
    task_baselines: dict[str, dict[str, float]] = defaultdict(dict)
    for item in summary:
        task_baselines[item["task"]][item["method"]] = item["success_mean"]
    phase_points = []
    for task in sorted({str(row.get("task")) for row in allocations}):
        candidates = [row for row in allocations if row.get("task") == task]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: float(row.get("success_per_million_decoder_nfe") or -np.inf))
        baselines = task_baselines.get(task, {})
        if "outer_only" in baselines and "inner_only" in baselines:
            phase_points.append((task, best, 1.0 - baselines["outer_only"], 1.0 - baselines["inner_only"]))
    if phase_points:
        scatter = axis.scatter(
            [point[2] for point in phase_points],
            [point[3] for point in phase_points],
            c=[float(point[1].get("success_per_million_decoder_nfe") or 0.0) for point in phase_points],
            s=110,
        )
        for task, best, x_value, y_value in phase_points:
            axis.annotate(f"{task}: K{best['outer_k']}/M{best['inner_m']}", (x_value, y_value), xytext=(4, 4), textcoords="offset points")
        axis.set_xlabel("local-event rarity (1 - outer-only success)")
        axis.set_ylabel("history/order difficulty (1 - inner-only success)")
        fig.colorbar(scatter, ax=axis, label="success per million decoder NFE")
        axis.set_title("K/M allocation phase diagram")
    else:
        _placeholder(axis, "K/M allocation phase diagram", "Awaiting Phase D results")
    fig.tight_layout()
    fig.savefig(output / "figure_2_km_allocation.png", dpi=180)
    plt.close(fig)

    for number, title, required in (
        (3, "Official interpolation recovery", "held_out_path_distance"),
        (4, "Reward success versus path fidelity", "held_out_path_distance"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.8))
        usable = [row for row in rows if row.get(required) is not None]
        if usable and number == 3:
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in usable:
                grouped[str(row["method"])].append(float(row[required]))
            labels = sorted(grouped, key=lambda method: METHOD_ORDER.get(method, 99))
            axis.bar(labels, [float(np.mean(grouped[label])) for label in labels])
            axis.tick_params(axis="x", rotation=30)
            axis.set_ylabel("held-out path distance")
            axis.set_title(title)
        elif usable and number == 4:
            for method in sorted({str(row["method"]) for row in usable}):
                points = [row for row in usable if row["method"] == method]
                axis.scatter(
                    [float(row[required]) for row in points],
                    [float(row["joint_program_success_rate"]) for row in points],
                    label=method,
                )
            axis.set_xlabel("held-out path distance")
            axis.set_ylabel("joint program success")
            axis.legend(frameon=False)
            axis.set_title(title)
        else:
            _placeholder(axis, title, "Awaiting gated Phase E results")
        fig.tight_layout()
        fig.savefig(output / f"figure_{number}_{'interpolation' if number == 3 else 'success_fidelity'}.png", dpi=180)
        plt.close(fig)


def write_reports(root: str | Path, output: str | Path) -> Path:
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = collect_run_rows(root)
    summary = summarize_methods(rows)
    fields = list(summary[0]) if summary else ["task", "method", "independent_units"]
    _write_csv(output / "table_full_method_comparison.csv", summary, fields)
    _write_csv(
        output / "table_km_allocation.csv",
        rows,
        ["case_id", "task", "seed", "method", "outer_k", "inner_m", "joint_program_success_rate", "success_per_million_decoder_nfe", "successful_outputs_per_gpu_hour"],
    )
    _write_csv(
        output / "table_per_case_interpolation.csv",
        [row for row in rows if "phase_e" in row["run_directory"]],
        ["case_id", "task", "seed", "method", "joint_program_success_rate", "event_order_accuracy", "held_out_path_distance", "structural_validity_rate"],
    )
    _write_csv(
        output / "table_compute_memory.csv",
        rows,
        ["case_id", "task", "seed", "method", "decoder_nfe", "wall_clock_s", "peak_gpu_memory_bytes"],
    )
    _write_csv(
        output / "table_failure_unresolved.csv",
        rows,
        ["case_id", "task", "seed", "method", "failure_rate", "unresolved_rate", "structural_validity_rate"],
    )
    paired = paired_duet_differences(rows)
    _write_csv(
        output / "table_paired_bootstrap.csv",
        paired,
        ["comparison", "paired_units", "mean_difference", "bootstrap_95_low", "bootstrap_95_high"],
    )
    falsification = [
        {"criterion": item, "status": "not_evaluated_before_experiments"}
        for item in (
            "complete-frame nested matches or exceeds DuET-MD at equal wall clock",
            "increasing M does not improve local-event discovery",
            "increasing K does not improve order or genealogy",
            "outer-only solves locally rare ordered tasks",
            "inner-only solves long-horizon route selection",
            "naive and proper dual methods are indistinguishable",
            "success rises while held-out fidelity or validity deteriorates",
            "effect is limited to one selected protein",
        )
    ]
    _write_csv(output / "table_novelty_falsification.csv", falsification, ["criterion", "status"])
    make_figures(rows, output)
    (output / "report_manifest.json").write_text(
        json.dumps({"source_root": str(root), "run_count": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate DuET-MD outputs into required tables and figures")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(write_reports(args.root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
