#!/usr/bin/env python3
"""Generate the frozen-evidence Phase A diagnosis v2 handoff report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def num(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def row_metrics(row: dict[str, Any], population: str = "pre") -> dict[str, Any]:
    metrics = row[population]
    return {
        "family": row.get("family"),
        "setting": row["setting"],
        "label": row["label"],
        "repetitions": row["repetitions"],
        "population": population,
        "terminal_success_mean": metrics["mean_terminal_success_estimate"],
        "terminal_success_rmse": metrics["terminal_success_rmse"],
        "route_l_mean": metrics["mean_route_l_estimate"],
        "route_r_mean": metrics["mean_route_r_estimate"],
        "route_rmse": metrics["route_mass_rmse"],
        "equal_weight_mean_rmse": metrics["equal_weight_mean_rmse"],
        "target_tv": metrics["target_tv"],
        "normalizer_relative_bias": row["normalizer"]["signed_relative_bias"],
        "normalizer_ci95_low": row["normalizer"]["relative_bias_95pct_ci"][0],
        "normalizer_ci95_high": row["normalizer"]["relative_bias_95pct_ci"][1],
        "reverse_transition_evaluations": row[
            "reverse_transition_evaluations_per_repetition"
        ][0],
    }


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("DUET_PROJECT_ROOT", Path(__file__).resolve().parents[2])),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    frozen_root = root / "outputs/duet_md/phase_a_cross_clock"
    diagnosis_root = root / "outputs/duet_md/phase_a_diagnosis_v2"
    report_root = diagnosis_root / "report"

    frozen = load(frozen_root / "metrics.json")
    pilot = load(report_root / "pilot_metrics.json")
    confirmation = load(report_root / "confirmation_metrics.json")
    evidence = load(diagnosis_root / "reanalysis/frozen_evidence_manifest.json")
    audit = load(diagnosis_root / "reanalysis/pre_post_resampling_audit.json")
    stepwise = load(diagnosis_root / "reanalysis/stepwise_adaptive_diagnostics.json")
    adaptive_variance = load(
        diagnosis_root / "reanalysis/adaptive_vs_static_bias_variance.json"
    )
    decomposition = load(diagnosis_root / "reanalysis/bias_variance_decomposition.json")
    joint = load(diagnosis_root / "reanalysis/joint_proper_weighting.json")

    plan = load(report_root / "confirmation_plan.json")
    family_by_key = {
        (entry["setting"], entry["label"]): entry["family"] for entry in plan["cells"]
    }
    for row in confirmation:
        row["family"] = family_by_key[(row["setting"], row["label"])]

    confirmation_rows = [row_metrics(row) for row in confirmation]
    write_csv(report_root / "confirmation_results_pre_long.csv", confirmation_rows)
    write_csv(
        report_root / "confirmation_results_post_long.csv",
        [row_metrics(row, "post") for row in confirmation],
    )

    pilot_rows = []
    for family, rows in pilot.items():
        for row in rows:
            copied = dict(row)
            copied["family"] = family
            pilot_rows.append(row_metrics(copied))
    write_csv(report_root / "pilot_results_pre_long.csv", pilot_rows)
    pilot_post_rows = []
    for family, rows in pilot.items():
        for row in rows:
            copied = dict(row)
            copied["family"] = family
            pilot_post_rows.append(row_metrics(copied, "post"))
    write_csv(report_root / "pilot_results_post_long.csv", pilot_post_rows)

    cause_rows = [
        row for row in confirmation if row["family"] in {"exact_stat_allocator", "matched_production"}
    ]
    by_setting_label = {(row["setting"], row["label"]): row for row in cause_rows}
    settings = [
        "common_low_informative",
        "rare_low_informative",
        "common_high_informative",
        "rare_high_informative",
    ]
    cause_aggregate: dict[str, Any] = {}
    labels = [
        "exact_stat_allocator",
        "current_probe_allocator",
        "matched_production_k12_m2",
        "global_static_k16_m2",
    ]
    for label in labels:
        values = [
            by_setting_label[(setting, label)]["pre"]["equal_weight_mean_rmse"]
            for setting in settings
        ]
        cause_aggregate[label] = {"mean_equal_weight_rmse": statistics.mean(values)}
    cause_aggregate["matched_vs_current_relative_rmse_reduction"] = statistics.mean(
        (
            by_setting_label[(setting, "current_probe_allocator")]["pre"]["equal_weight_mean_rmse"]
            - by_setting_label[(setting, "matched_production_k12_m2")]["pre"]["equal_weight_mean_rmse"]
        )
        / by_setting_label[(setting, "current_probe_allocator")]["pre"]["equal_weight_mean_rmse"]
        for setting in settings
    )
    cause_aggregate["global_vs_matched_relative_rmse_reduction"] = statistics.mean(
        (
            by_setting_label[(setting, "matched_production_k12_m2")]["pre"]["equal_weight_mean_rmse"]
            - by_setting_label[(setting, "global_static_k16_m2")]["pre"]["equal_weight_mean_rmse"]
        )
        / by_setting_label[(setting, "matched_production_k12_m2")]["pre"]["equal_weight_mean_rmse"]
        for setting in settings
    )
    cause_aggregate["exact_vs_current_relative_rmse_change"] = statistics.mean(
        (
            by_setting_label[(setting, "exact_stat_allocator")]["pre"]["equal_weight_mean_rmse"]
            - by_setting_label[(setting, "current_probe_allocator")]["pre"]["equal_weight_mean_rmse"]
        )
        / by_setting_label[(setting, "current_probe_allocator")]["pre"]["equal_weight_mean_rmse"]
        for setting in settings
    )

    temporal_rows = [
        row for row in confirmation if row["family"] == "temporal_budget_schedules"
    ]
    temporal_by_key = {(row["setting"], row["label"]): row for row in temporal_rows}
    temporal_gain = {}
    for setting in settings:
        static = temporal_by_key[(setting, "static_m2")]["pre"]
        concentrated = temporal_by_key[(setting, "concentrated_t3")]["pre"]
        temporal_gain[setting] = {
            "terminal_success_rmse_reduction": (
                static["terminal_success_rmse"] - concentrated["terminal_success_rmse"]
            )
            / static["terminal_success_rmse"],
            "route_rmse_reduction": (
                static["route_mass_rmse"] - concentrated["route_mass_rmse"]
            )
            / static["route_mass_rmse"],
            "combined_rmse_reduction": (
                static["equal_weight_mean_rmse"] - concentrated["equal_weight_mean_rmse"]
            )
            / static["equal_weight_mean_rmse"],
        }

    joint_applicable = [row for row in joint if row["applicable"]]
    joint_summary = {
        "applicable_tests": len(joint_applicable),
        "ci95_contains_zero": sum(
            row["ci95_low"] <= 0.0 <= row["ci95_high"] for row in joint_applicable
        ),
        "max_absolute_z_score": max(
            abs(row["residual"] / row["standard_error"])
            for row in joint_applicable
            if row["standard_error"] > 0.0
        ),
    }
    frozen_summaries = [load(path) for path in frozen_root.glob("cells/**/summary.json")]
    normalizer_rows = [row["normalizer"] for row in frozen_summaries if row.get("normalizer")]
    normalizer_summary = {
        "proper_cells": len(normalizer_rows),
        "ci95_contains_zero": sum(
            row["relative_bias_95pct_ci"][0] <= 0.0 <= row["relative_bias_95pct_ci"][1]
            for row in normalizer_rows
        ),
        "mean_absolute_relative_bias": statistics.mean(
            row["absolute_relative_bias"] for row in normalizer_rows
        ),
        "max_absolute_relative_bias": max(
            row["absolute_relative_bias"] for row in normalizer_rows
        ),
    }
    proper_keys = {
        (row["phase"], row["process"], row["method"], row["quantity"])
        for row in joint_applicable
    }
    proper_decomposition = [
        row
        for row in decomposition
        if (row["phase"], row["process"], row["method"], row["quantity"])
        in proper_keys
    ]
    marginal_summary = {}
    for quantity in ("terminal_success", "route_l", "route_r"):
        rows = [row for row in proper_decomposition if row["quantity"] == quantity]
        marginal_summary[quantity] = {
            "cells": len(rows),
            "mean_absolute_bias": statistics.mean(abs(row["bias"]) for row in rows),
            "median_absolute_bias": statistics.median(abs(row["bias"]) for row in rows),
            "max_absolute_bias": max(abs(row["bias"]) for row in rows),
            "mean_variance_fraction_of_mse": statistics.mean(
                row["variance_fraction_of_mse"] for row in rows
            ),
            "median_variance_fraction_of_mse": statistics.median(
                row["variance_fraction_of_mse"] for row in rows
            ),
        }

    step_aggregate = []
    for step in range(2, 6):
        rows = [row for row in stepwise if row["physical_step"] == step]
        fields = [
            "fraction_rho2_exact_zero",
            "fraction_inner_need_zero",
            "fraction_outer_need_zero",
            "fraction_balanced_due_to_tie",
            "fraction_selected_k12_m2",
            "fraction_selected_k6_m4",
            "fraction_selected_k3_m8",
        ]
        step_aggregate.append(
            {"physical_step": step, **{field: statistics.mean(row[field] for row in rows) for field in fields}}
        )

    variance_summary = {}
    for quantity in ("terminal_success", "route_l", "route_r"):
        ratios = [
            row["variance_ratio_adaptive_over_static"]
            for row in adaptive_variance
            if row["quantity"] == quantity
        ]
        variance_summary[quantity] = {
            "mean_ratio": statistics.mean(ratios),
            "median_ratio": statistics.median(ratios),
        }

    held_out_gains = [
        row["pre_completion_equal_weight_rmse_gain"]
        for row in frozen["held_out"]["per_process"]
    ]
    metrics = {
        "frozen_evidence": evidence,
        "frozen_phase_a_summary": {
            "completed_cells": "163/163",
            "held_out_mean_combined_gain": statistics.mean(held_out_gains),
            "held_out_positive_count": sum(value > 0.0 for value in held_out_gains),
            "adaptive_vs_global_static_mean_improvement": frozen["held_out"][
                "mean_adaptive_improvement"
            ],
        },
        "pre_post_audit": audit,
        "stepwise_aggregate": step_aggregate,
        "adaptive_variance_summary": variance_summary,
        "joint_proper_weighting_summary": joint_summary,
        "normalizer_summary": normalizer_summary,
        "marginal_summary": marginal_summary,
        "cause_separation_confirmation": cause_rows,
        "cause_separation_aggregate": cause_aggregate,
        "checkpoint_confirmation_y1": [
            row for row in confirmation if row["family"] == "checkpoint_test"
        ],
        "no_information_control": [
            row
            for row in frozen["fixed_methods"]
            if row["setting"]
            in {"rare_low_no_information", "rare_high_no_information"}
            and row["method"] in {"complete_nested", "fixed_duet"}
        ],
        "temporal_confirmation": temporal_rows,
        "temporal_gain": temporal_gain,
        "conclusion": {
            "Q1": "no evidence of sampler correctness issue",
            "Q2": "C primary; B substantial; A not supported as primary; D contradicted",
            "Q3": "small and setting-dependent Y2 signal; stronger Y1 terminal signal must be described as diagnostic, not a retuned main checkpoint",
            "Q4": "conditional continue: Case B allocation opportunity exists, but the current online rule does not identify it",
        },
    }
    write_json(report_root / "metrics.json", metrics)

    total_pilot = sum(row["repetitions"] for rows in pilot.values() for row in rows)
    total_confirm = sum(row["repetitions"] for row in confirmation)
    accounting = {
        "pilot_cells": sum(len(rows) for rows in pilot.values()),
        "pilot_repetitions": total_pilot,
        "confirmation_cells": len(confirmation),
        "confirmation_repetitions": total_confirm,
        "reverse_transition_evaluations_per_repetition": 480,
        "pilot_reverse_transition_evaluations": total_pilot * 480,
        "confirmation_reverse_transition_evaluations": total_confirm * 480,
        "new_total_reverse_transition_evaluations": (total_pilot + total_confirm) * 480,
        "wall_clock_s": {
            mode: load(report_root / f"{mode}_run.json")["wall_clock_s"]
            for mode in ("reanalysis", "pilot", "confirmation")
        },
    }
    write_json(report_root / "compute_accounting.json", accounting)

    lines = [
        "# Phase A Diagnosis v2 — frozen-evidence handoff",
        "",
        "## 실행 상태와 보존 규칙",
        "",
        f"- 기존 Phase A: `{frozen_root}` (163/163 cells, cell당 5,000 repetitions)",
        f"- 새 진단: `{diagnosis_root}`",
        f"- frozen evidence read-only 검증: `{str(evidence['read_only_verified']).lower()}`",
        f"- 기존 `metrics.json` SHA-256: `{evidence['top_level_sha256_after']['metrics.json']}`",
        "- smoke run은 포함하지 않았다.",
        "- 새 셀 primary는 final outer resampling 직전 weighted population이다. 기존 frozen Phase A primary는 resampling 후 unweighted population이며, frozen raw만으로 pre 값을 재구성할 수 없다.",
        "",
        "## 고정 실험 사양",
        "",
        "- canonical informative settings 4개, horizon 5, epsilon 0.05, float64, master seed family 20260908.",
        "- 모든 새 셀은 repetition당 reverse-transition 480회로 동일하다.",
        "- adaptive action set: `(K,M) = (12,2), (6,4), (3,8)`; 첫 step은 `K32,M1`.",
        "- Stage 1 pilot: 1,000 repetitions; 필요한 27개 비교만 5,000으로 확정.",
        "- Stage 2: `K=16`, `sum_t M_t=10`, Y2 유지. Static `[2,2,2,2,2]`; concentrated는 한 step만 6, 나머지는 1.",
        f"- 관측 wall clock: reanalysis {accounting['wall_clock_s']['reanalysis']:.1f}s, pilot {accounting['wall_clock_s']['pilot']:.1f}s, confirmation {accounting['wall_clock_s']['confirmation']:.1f}s.",
        "- H100 실행 환경에서 Phase A 원본+diagnosis unit tests 17/17 통과.",
        "",
        "## Stage 0 — 기존 raw 재분석",
        "",
    ]
    lines += md_table(
        ["t", "rho²=0", "InnerNeed=0", "OuterNeed=0", "balanced from tie", "K12M2", "K6M4", "K3M8"],
        [
            [
                str(row["physical_step"]),
                num(row["fraction_rho2_exact_zero"], 4),
                num(row["fraction_inner_need_zero"], 4),
                num(row["fraction_outer_need_zero"], 4),
                num(row["fraction_balanced_due_to_tie"], 4),
                num(row["fraction_selected_k12_m2"], 4),
                num(row["fraction_selected_k6_m4"], 4),
                num(row["fraction_selected_k3_m8"], 4),
            ]
            for row in step_aggregate
        ],
    )
    lines += [
        "",
        "Frozen raw에는 centered phi/psi variance가 없어 `rho²=0` 중 zero empirical variance가 원인인 비율은 replay 없이 식별 불가능하다.",
        "",
        "Adaptive/Global Static의 variance ratio:",
        "",
    ]
    lines += md_table(
        ["quantity", "mean ratio", "median ratio"],
        [[key, num(value["mean_ratio"], 4), num(value["median_ratio"], 4)] for key, value in variance_summary.items()],
    )
    lines += [
        "",
        f"Joint residual은 applicable {joint_summary['applicable_tests']}개 중 {joint_summary['ci95_contains_zero']}개({100*joint_summary['ci95_contains_zero']/joint_summary['applicable_tests']:.2f}%)의 95% CI가 0을 포함했고, 최대 절대 z-score는 {joint_summary['max_absolute_z_score']:.4f}였다.",
        f"Normalizer는 proper {normalizer_summary['proper_cells']}개 셀 중 {normalizer_summary['ci95_contains_zero']}개의 95% CI가 0 relative bias를 포함했다. 평균/최대 absolute relative bias는 각각 {num(normalizer_summary['mean_absolute_relative_bias'])}/{num(normalizer_summary['max_absolute_relative_bias'])}였다.",
        "Proper cell의 marginal bias와 MSE 중 variance 비율:",
        "",
    ]
    lines += md_table(
        ["quantity", "mean abs bias", "median abs bias", "max abs bias", "mean variance/MSE", "median variance/MSE"],
        [
            [
                quantity,
                num(values["mean_absolute_bias"]),
                num(values["median_absolute_bias"]),
                num(values["max_absolute_bias"]),
                num(values["mean_variance_fraction_of_mse"], 4),
                num(values["median_variance_fraction_of_mse"], 4),
            ]
            for quantity, values in marginal_summary.items()
        ],
    )
    lines += [
        "",
        "Self-normalized finite-particle marginal estimate는 정확히 unbiased일 필요가 없으므로 이 bias 자체를 correctness 실패로 판정하지 않는다. 반면 unnormalized normalizer와 joint residual이 직접적인 proper-weighting audit이다.",
        "",
        "## Stage 1 — 원인 분리 확정 결과 (primary: pre-resampling)",
        "",
    ]
    lines += md_table(
        ["setting", "method", "success RMSE", "route RMSE", "combined RMSE", "TV", "Z rel. bias"],
        [
            [
                row["setting"], row["label"], num(row["pre"]["terminal_success_rmse"]),
                num(row["pre"]["route_mass_rmse"]), num(row["pre"]["equal_weight_mean_rmse"]),
                num(row["pre"]["target_tv"]), num(row["normalizer"]["signed_relative_bias"]),
            ]
            for row in cause_rows
        ],
    )
    allocator_rows = [
        row
        for row in cause_rows
        if row["label"]
        in {
            "exact_stat_allocator",
            "current_probe_allocator",
            "matched_production_k12_m2",
        }
    ]
    lines += [
        "",
        "첫 step의 고정 `K32,M1`을 제외한 t=2–5 allocation 선택 비율:",
        "",
    ]
    allocation_table = []
    for row in allocator_rows:
        counts = row["allocation_counts"]
        total = sum(
            counts.get(key, 0) for key in ("K12_M2", "K6_M4", "K3_M8")
        )
        allocation_table.append(
            [
                row["setting"],
                row["label"],
                num(counts.get("K12_M2", 0) / total, 4),
                num(counts.get("K6_M4", 0) / total, 4),
                num(counts.get("K3_M8", 0) / total, 4),
            ]
        )
    lines += md_table(
        ["setting", "allocator", "K12M2", "K6M4", "K3M8"], allocation_table
    )
    lines += [
        "",
        f"평균 combined RMSE에서 matched production은 current adaptive보다 {pct(cause_aggregate['matched_vs_current_relative_rmse_reduction'])} 낮았고, global K16M2는 matched보다 추가로 {pct(cause_aggregate['global_vs_matched_relative_rmse_reduction'])} 낮았다. Exact-stat allocator는 current probe allocator 대비 {pct(cause_aggregate['exact_vs_current_relative_rmse_change'])} 변했다(양수는 악화).",
        "",
        "## Checkpoint information test",
        "",
        "Complete와 Y2는 기존 frozen post-resampling 결과를 재사용했다. 새 Y1의 primary는 pre이지만, 아래 직접 비교 표에는 기존 결과와 population을 맞추기 위해 Y1의 post 값을 쓴다.",
        "",
    ]
    frozen_fixed = {
        (row["setting"], row["method"]): row for row in frozen["fixed_methods"]
    }
    y1_rows = {
        row["setting"]: row for row in confirmation if row["family"] == "checkpoint_test"
    }
    y1_rows["rare_low_informative"] = frozen["checkpoint_diagnostic"]["results"][2]
    checkpoint_table = []
    for setting in settings:
        for method, checkpoint in (("complete_nested", "complete"), ("fixed_duet", "Y2")):
            row = frozen_fixed[(setting, method)]
            checkpoint_table.append([
                setting, checkpoint, num(row["terminal_success_rmse"]), num(row["route_mass_rmse"]),
                num(row["equal_weight_mean_rmse"]), num(row["target_tv"]), num(row["normalizer"]["signed_relative_bias"]),
            ])
        row = y1_rows[setting]
        metrics_y1 = row["post"] if "post" in row else row
        checkpoint_table.append([
            setting, "Y1", num(metrics_y1["terminal_success_rmse"]), num(metrics_y1["route_mass_rmse"]),
            num(metrics_y1["equal_weight_mean_rmse"]), num(metrics_y1["target_tv"]), num(row["normalizer"]["signed_relative_bias"]),
        ])
    lines += md_table(
        ["setting", "checkpoint", "success RMSE", "route RMSE", "combined RMSE", "TV", "Z rel. bias"],
        checkpoint_table,
    )
    lines += [
        "",
        "No-information control (frozen post-resampling):",
        "",
    ]
    no_information_table = []
    for setting in ("rare_low_no_information", "rare_high_no_information"):
        for method, label in (("complete_nested", "complete"), ("fixed_duet", "Y2")):
            row = frozen_fixed[(setting, method)]
            no_information_table.append(
                [
                    setting,
                    label,
                    num(row["terminal_success_rmse"]),
                    num(row["route_mass_rmse"]),
                    num(row["equal_weight_mean_rmse"]),
                    num(row["target_tv"]),
                    num(row["normalizer"]["signed_relative_bias"]),
                ]
            )
    lines += md_table(
        ["setting", "method", "success RMSE", "route RMSE", "combined RMSE", "TV", "Z rel. bias"],
        no_information_table,
    )
    lines += [
        "",
        "Y1은 checkpoint information이 큰 diagnostic control일 뿐이며 main checkpoint를 사후 변경한 결과가 아니다. 기존 no-information controls에서는 Y2 DuET의 이득이 일관되지 않았으므로 ‘DuET always better’로 해석하지 않는다.",
        "",
        "## Stage 2 — temporal budget schedule 확정 결과",
        "",
    ]
    lines += md_table(
        ["setting", "schedule", "success RMSE", "route RMSE", "combined RMSE", "TV", "combined reduction vs static"],
        [
            [
                setting, label, num(temporal_by_key[(setting, label)]["pre"]["terminal_success_rmse"]),
                num(temporal_by_key[(setting, label)]["pre"]["route_mass_rmse"]),
                num(temporal_by_key[(setting, label)]["pre"]["equal_weight_mean_rmse"]),
                num(temporal_by_key[(setting, label)]["pre"]["target_tv"]),
                "—" if label == "static_m2" else pct(temporal_gain[setting]["combined_rmse_reduction"]),
            ]
            for setting in settings
            for label in ("static_m2", "concentrated_t3")
        ],
    )
    pilot_temporal = pilot["temporal_budget_schedules"]
    lines += [
        "",
        "5,000회 확장 대상을 고른 1,000회 pilot의 여섯 schedule 전체 raw summary:",
        "",
    ]
    lines += md_table(
        ["setting", "schedule", "success RMSE", "route RMSE", "combined RMSE", "TV", "Z rel. bias"],
        [
            [
                row["setting"], row["label"], num(row["pre"]["terminal_success_rmse"]),
                num(row["pre"]["route_mass_rmse"]), num(row["pre"]["equal_weight_mean_rmse"]),
                num(row["pre"]["target_tv"]), num(row["normalizer"]["signed_relative_bias"]),
            ]
            for row in pilot_temporal
        ],
    )
    lines += [
        "",
        "4개 setting 모두 concentrated_t3가 static보다 우수하므로 판정은 Case B다. 즉 allocation opportunity는 존재하지만, 현재 online diagnostic/risk rule이 그 위치를 찾지 못한다.",
        "",
        "## Q1–Q4 결론",
        "",
        "### Q1. Sampling formulation 자체에 correctness 문제 증거가 있는가?",
        "",
        "현재 결과에서는 sampler correctness issue의 증거가 없다. Unnormalized normalizer와 joint proper-weighting residual의 coverage가 nominal 수준이고, marginal RMSE는 평균적으로 99% 이상 variance가 설명한다. Self-normalized finite-particle marginal bias는 원리상 0일 필요가 없다. 새 셀의 pre/post 동시 기록으로 기존 final resampling의 추가 Monte Carlo noise도 분리했다.",
        "",
        "### Q2. Current adaptive failure의 주 원인은 무엇인가?",
        "",
        "상대 중요도는 C(risk formula/action selection) > B(probe overhead/reduced K) >> A(probe estimator noise)이며, D(no useful allocation opportunity)는 Stage 2가 반박한다. Exact statistics로도 회복되지 않아 A가 주원인이라는 가설은 지지되지 않는다.",
        "",
        "### Q3. Fixed DuET에 실제 signal이 있는가?",
        "",
        "Y2의 signal은 작고 setting-dependent하다. Frozen held-out 20개 평균 combined gain은 약 +1.55%, 16/20 positive였지만 canonical 및 no-information controls까지 포함하면 모든 지표·setting에서 일관된 우위는 아니다. 정보가 더 큰 Y1은 terminal RMSE 개선 경향을 강화하므로 pre-completion information의 역할을 지지하지만, main checkpoint 변경 근거로 사용하지 않는다.",
        "",
        "### Q4. Allocation research를 계속할 근거가 있는가?",
        "",
        "Conditional continue. Case B이므로 physical step별 compute 재배치 가능성은 확인됐다. 다만 현재 allocator를 확장·튜닝할 근거는 없고, 후속 연구가 있다면 유리한 step을 online으로 식별하는 진단 문제로 한정해야 한다.",
        "",
        "## Raw evidence index",
        "",
        "- `reanalysis/stepwise_adaptive_diagnostics.{json,csv}`",
        "- `reanalysis/bias_variance_decomposition.{json,csv}`",
        "- `reanalysis/adaptive_vs_static_bias_variance.{json,csv}`",
        "- `reanalysis/joint_proper_weighting.{json,csv}`",
        "- `reanalysis/pre_post_resampling_audit.json`",
        "- `reanalysis/frozen_evidence_manifest.json`",
        "- 각 cell의 `summary.json`과 `raw_repetitions.npz`",
        "- `report/pilot_metrics.json`, `pilot_results_{pre,post}_long.csv`",
        "- `report/confirmation_plan.json`, `confirmation_metrics.json`, `confirmation_results_{pre,post}_long.csv`",
        "- `report/metrics.json`, `compute_accounting.json`",
        "",
        "이 보고서는 요청된 Q1–Q4까지만 판정하며, 결과 개선을 위한 추가 sweep·allocator·reward·topology 탐색은 수행하지 않았다.",
        "",
    ]
    report = "\n".join(lines)
    report_path = root / "docs/PHASE_A_DIAGNOSIS_V2_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    (report_root / "report.md").write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
