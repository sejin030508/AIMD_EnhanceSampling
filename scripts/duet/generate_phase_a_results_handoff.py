#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}g}"


def pct(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def cell_label(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    marker = parts.index("cells")
    return parts[marker + 1], parts[marker + 2], parts[marker + 3]


def normalizer_fields(row: dict[str, Any]) -> tuple[str, str, str, str]:
    normalizer = row.get("normalizer")
    if not normalizer:
        return "—", "—", "—", "—"
    ci = normalizer["relative_bias_95pct_ci"]
    return (
        number(normalizer["mean"]),
        pct(normalizer["signed_relative_bias"]),
        f"[{pct(ci[0])}, {pct(ci[1])}]",
        number(normalizer["confidence_interval_contains_zero"]),
    )


def table(lines: list[str], headers: list[str], rows: list[list[Any]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    metrics = load_json(root / "metrics.json")
    validation = load_json(root / "validation.json")
    nfe = load_json(root / "nfe_accounting.json")
    wall = load_json(root / "wall_clock.json")
    environment = load_json(root / "environment.json")
    seeds = load_json(root / "seed_manifest.json")
    held_parameters = load_json(root / "held_out_parameters.json")
    summaries: list[dict[str, Any]] = []
    for path in sorted((root / "cells").glob("**/summary.json")):
        row = load_json(path)
        phase, setting, method_dir = cell_label(path)
        row.update(
            {
                "_path": path,
                "_relative_path": path.relative_to(root),
                "_phase": phase,
                "_setting_dir": setting,
                "_method_dir": method_dir,
            }
        )
        summaries.append(row)

    raw_paths = sorted((root / "cells").glob("**/raw_repetitions.npz"))
    if len(summaries) != validation["independently_evaluated_cells"]:
        raise RuntimeError("summary cell count does not match validation.json")
    if len(raw_paths) != len(summaries):
        raise RuntimeError("raw NPZ count does not match summary count")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in raw_paths:
            writer.writerow([path.relative_to(root), path.stat().st_size, sha256(path)])

    held = metrics["held_out"]["per_process"]
    informative_fixed = [
        row for row in metrics["fixed_duet_vs_complete"] if row["informative"]
    ]
    noinfo_fixed = [
        row for row in metrics["fixed_duet_vs_complete"] if not row["informative"]
    ]
    combined_held_gains = [row["pre_completion_equal_weight_rmse_gain"] for row in held]
    adaptive_improvements = [row["adaptive_improvement"] for row in held]
    with_normalizer = [row for row in summaries if row.get("normalizer")]
    ci_failures = [
        row
        for row in with_normalizer
        if not row["normalizer"]["confidence_interval_contains_zero"]
    ]

    lines: list[str] = [
        "# Phase A Exactly Enumerable Cross-Clock Diagnostic — execution handoff",
        "",
        "> Audience: coordinating AI/research lead. This document records the completed execution and raw numerical outputs. It does not revise the preregistered formulation.",
        "",
        "## 1. Completion status and decision-relevant result",
        "",
        "**Execution status: complete.** No Phase A process remains active. All planned independently sampled cells and repetition-level arrays are present.",
        "",
        f"- Completed cells: `{len(summaries)}/{validation['independently_evaluated_cells']}`",
        f"- Raw NPZ files: `{len(raw_paths)}/{validation['independently_evaluated_cells']}`",
        f"- Repetitions per cell: `{validation['repetitions_per_cell']:,}`",
        f"- Total independent repetitions: `{validation['total_repetitions']:,}`",
        f"- Reverse-transition evaluations: `{validation['total_reverse_transition_evaluations']:,}`",
        f"- Neural decoder calls: `{nfe['neural_decoder_calls']}`",
        f"- Wall time: `{wall['seconds']:.3f} s` = `{wall['seconds']/60:.2f} min`",
        f"- Maximum exact inner-marginal error: `{validation['maximum_inner_marginal_error']:.17g}`",
        "",
        "Outcome at a glance:",
        "",
        f"- Fixed DuET vs Complete Nested, four informative canonical settings: mean success-RMSE gain `{pct(mean(r['success_gain_pre'] for r in informative_fixed))}`, mean route-RMSE gain `{pct(mean(r['route_gain_pre'] for r in informative_fixed))}`, mean equal-weight gain `{pct(mean(r['equal_weight_gain_pre'] for r in informative_fixed))}`.",
        f"- No-information controls: mean equal-weight gain `{pct(mean(r['equal_weight_gain_pre'] for r in noinfo_fixed))}`; the two individual values have opposite signs and are near zero.",
        f"- Held-out 20 processes: Fixed DuET pre-completion equal-weight gain mean `{pct(mean(combined_held_gains))}`, median `{pct(median(combined_held_gains))}`, positive in `{sum(v > 0 for v in combined_held_gains)}/20`.",
        f"- Global static selected on development only: `{metrics['development_allocation']['selected_global_static']}`.",
        f"- Adaptive allocation improvement vs global static: mean `{pct(mean(adaptive_improvements))}`, range `{pct(min(adaptive_improvements))}` to `{pct(max(adaptive_improvements))}`, positive in `{sum(v > 0 for v in adaptive_improvements)}/20`.",
        "- Scientific readout: fixed pre-completion steering shows a small average benefit, strongest for rare informative transitions. The proposed adaptive allocator is not supported: it is worse than the development-selected static allocation on every held-out process.",
        "- Diagnostic readout is mixed: `OuterNeed` predicts success/combined benefit of large K, whereas checkpoint rho-squared does not predict pre-completion gain in this 20-process family.",
        "",
        "## 2. Frozen executed configuration",
        "",
        "- Physical process: 7 states, 5 physical transitions; hard success is open at both t=4 and t=5.",
        "- Inner process: exactly enumerable three-step binary reverse process `Y3 -> Y2 -> Y1 -> X`.",
        "- Main checkpoint: `Y2`; epsilon=0.05; beta1=0.05; beta2=0.20; beta3=0.50.",
        "- Fixed methods and budgets: Frozen `(32,1)`, Outer-only `(32,1)`, Inner-only `(1,32)`, Complete Nested `(8,4)`, Fixed DuET `(8,4)`; all `K*M=32`.",
        "- Development static candidates: `(16,2)`, `(8,4)`, `(4,8)`, `(2,16)`.",
        "- Adaptive production: first step `(32,1)`; later actions `(12,2)`, `(6,4)`, `(3,8)` after 8 discarded probes (4 parent slots x 2 probes); fixed and adaptive cost both 480 reverse transitions/repetition.",
        "- Systematic inner/outer resampling; float64; master seed `20260908`; probe and production RNG use separate spawned substreams.",
        f"- RNG derivation: `{seeds['derivation']}`.",
        "- Execution host: `sejin-h100-1-work-001-zhr2l`; computation itself was CPU-only, despite an H100 being visible.",
        f"- Python `{environment['python'].split()[0]}`, NumPy `{environment['packages']['numpy']}`, SciPy `{environment['packages']['scipy']}`.",
        "",
        "## 3. Validation and accounting raw values",
        "",
    ]

    table(
        lines,
        ["Field", "Value"],
        [[key, number(value, 17) if not isinstance(value, list) else json.dumps(value)] for key, value in validation.items()]
        + [[key, number(value, 17)] for key, value in nfe.items()],
    )

    lines += [
        "No-information Y2 look-ahead values were exactly `[0.069, 0.069]`. Of the 151 unique cells that estimate a normalizer, 143/151 (94.70%) 95% relative-bias intervals include zero, consistent with nominal Monte Carlo coverage. The eight exclusions are listed below; the largest absolute relative bias is below 2.3%.",
        "",
    ]
    table(
        lines,
        ["Cell", "Z mean", "signed rel. bias", "95% CI"],
        [
            [
                row["_relative_path"],
                number(row["normalizer"]["mean"]),
                pct(row["normalizer"]["signed_relative_bias"]),
                f"[{pct(row['normalizer']['relative_bias_95pct_ci'][0])}, {pct(row['normalizer']['relative_bias_95pct_ci'][1])}]",
            ]
            for row in ci_failures
        ],
    )

    lines += ["## 4. A1 exact-enumeration targets", ""]
    table(
        lines,
        ["Setting", "Z exact", "base success", "target success", "target route L", "target route R", "max marginal error"],
        [
            [
                setting,
                number(row["normalizer"]),
                number(row["base_terminal_success_probability"]),
                number(row["target_terminal_success_probability"]),
                number(row["route_l_target_mass"]),
                number(row["route_r_target_mass"]),
                number(row["maximum_inner_marginal_error"], 17),
            ]
            for setting, row in metrics["exact"].items()
        ],
    )

    lines += ["## 5. A2 fixed-method raw aggregates (30 cells)", ""]
    fixed_rows = []
    for row in metrics["fixed_methods"]:
        z_mean, z_bias, z_ci, z_cover = normalizer_fields(row)
        fixed_rows.append(
            [
                row["setting"], row["method"], row["outer_k"], row["inner_m"], row.get("checkpoint") or "—",
                number(row["mean_terminal_success_estimate"]), number(row["terminal_success_rmse"]),
                number(row["mean_route_l_estimate"]), number(row["route_mass_rmse"]), number(row["equal_weight_mean_rmse"]),
                number(row.get("target_tv")), z_mean, z_bias, z_ci, z_cover,
            ]
        )
    table(
        lines,
        ["Setting", "Method", "K", "M", "CP", "mean success", "success RMSE", "mean route L", "route RMSE", "mean RMSE", "TV", "Z mean", "Z bias", "Z bias 95% CI", "CI has 0"],
        fixed_rows,
    )

    lines += ["### Fixed DuET vs Complete Nested gains", ""]
    table(
        lines,
        ["Setting", "informative", "success gain", "route gain", "equal-weight gain"],
        [[r["setting"], number(r["informative"]), pct(r["success_gain_pre"]), pct(r["route_gain_pre"]), pct(r["equal_weight_gain_pre"])] for r in metrics["fixed_duet_vs_complete"]],
    )

    lines += ["## 6. A3 checkpoint diagnostic", ""]
    chi = metrics["checkpoint_diagnostic"]["chi"]
    table(lines, ["Stage", "chi"], [[stage, number(value)] for stage, value in chi.items()])
    lines.append(f"Strict information ordering `Y3 < Y2 < Y1`: `{number(metrics['checkpoint_diagnostic']['information_strictly_increases'])}`.")
    lines.append("")
    table(
        lines,
        ["Method", "CP", "success mean", "success RMSE", "route RMSE", "mean RMSE", "TV", "Z bias 95% CI"],
        [
            [
                r["method"], r.get("checkpoint") or "—", number(r["mean_terminal_success_estimate"]),
                number(r["terminal_success_rmse"]), number(r["route_mass_rmse"]), number(r["equal_weight_mean_rmse"]),
                number(r["target_tv"]), f"[{pct(r['normalizer']['relative_bias_95pct_ci'][0])}, {pct(r['normalizer']['relative_bias_95pct_ci'][1])}]",
            ]
            for r in metrics["checkpoint_diagnostic"]["results"]
        ],
    )

    lines += ["## 7. A4 development-only static allocation selection", ""]
    table(
        lines,
        ["Candidate", "development score"],
        [[label, number(score)] for label, score in metrics["development_allocation"]["candidate_scores"].items()],
    )
    lines.append(f"Selected and frozen before held-out evaluation: `{metrics['development_allocation']['selected_global_static']}`.")
    lines.append("")
    table(
        lines,
        ["Setting", "K", "M", "success RMSE", "route RMSE", "mean RMSE", "TV", "Z signed bias"],
        [
            [
                r["setting"], r["outer_k"], r["inner_m"], number(r["terminal_success_rmse"]),
                number(r["route_mass_rmse"]), number(r["equal_weight_mean_rmse"]), number(r["target_tv"]),
                pct(r["normalizer"]["signed_relative_bias"]),
            ]
            for r in metrics["development_allocation"]["results"]
        ],
    )

    lines += ["## 8. A5 held-out process parameters, exact targets, and comparisons", ""]
    parameter_by_name = {row["name"]: row for row in held_parameters}
    comparison_by_name = {row["setting"]: row for row in held}
    joined_rows = []
    for name, parameter in parameter_by_name.items():
        exact = metrics["exact"][name]
        comparison = comparison_by_name[name]
        counts = comparison["selected_schedule_counts"]
        joined_rows.append(
            [
                name, parameter["instance_id"], number(parameter["s_l"]), number(parameter["s_r"]),
                number(parameter["commitment_probability"]), number(parameter["commitment_beta2"]), parameter["higher_survival_route"],
                number(exact["normalizer"]), number(exact["base_terminal_success_probability"]), number(exact["target_terminal_success_probability"]),
                pct(comparison["pre_completion_success_rmse_gain"]), pct(comparison["pre_completion_route_rmse_gain"]), pct(comparison["pre_completion_equal_weight_rmse_gain"]),
                pct(comparison["adaptive_improvement"]), number(comparison["mean_probe_rho_squared"]), number(comparison["mean_probe_outer_need"]),
                f"{counts['K32_M1']}/{counts['K12_M2']}/{counts['K6_M4']}/{counts['K3_M8']}",
            ]
        )
    table(
        lines,
        ["Process", "id", "sL", "sR", "p", "beta2", "high route", "Z", "base success", "target success", "DuET success gain", "DuET route gain", "DuET mean gain", "adaptive improvement", "rho2", "OuterNeed", "schedule counts 32/12/6/3"],
        joined_rows,
    )

    lines += ["### Held-out method-level raw aggregates (120 cells)", ""]
    held_summaries = [row for row in summaries if row["_phase"] == "held_out"]
    held_rows = []
    for row in held_summaries:
        z_mean, z_bias, z_ci, z_cover = normalizer_fields(row)
        held_rows.append(
            [
                row["setting"], row["_method_dir"], row["outer_k"], row["inner_m"], row.get("checkpoint") or "—",
                number(row["mean_terminal_success_estimate"]), number(row["terminal_success_rmse"]),
                number(row["mean_route_l_estimate"]), number(row["route_mass_rmse"]), number(row["equal_weight_mean_rmse"]),
                number(row["target_tv"]), z_mean, z_bias, z_ci, z_cover,
            ]
        )
    table(
        lines,
        ["Process", "Method/allocation", "K", "M", "CP", "mean success", "success RMSE", "mean route L", "route RMSE", "mean RMSE", "TV", "Z mean", "Z bias", "Z bias 95% CI", "CI has 0"],
        held_rows,
    )

    lines += ["## 9. Diagnostic calibration raw values", ""]
    table(
        lines,
        ["Comparison", "Spearman rho", "p-value"],
        [[name, number(row["spearman_rho"]), number(row["p_value"])] for name, row in metrics["diagnostic_calibration"].items()],
    )

    lines += [
        "Interpretation boundary:",
        "",
        "- `OuterNeed` vs small-K minus large-K success RMSE: rho=0.9038, p=4.72e-8; strong positive calibration.",
        "- `OuterNeed` vs combined RMSE difference: rho=0.6992, p=0.000602; positive calibration.",
        "- rho-squared vs pre-completion success/equal-weight gain is not significant and has slightly negative point estimates.",
        "- Therefore diagnosis is only partially validated, and the exact adaptive risk rule should not be promoted as successful without redesign and a fresh held-out evaluation.",
        "",
        "## 10. Complete unique-cell index (163 cells)",
        "",
        "Every row below corresponds to one `summary.json` and one sibling `raw_repetitions.npz` with 5,000 repetitions.",
        "",
    ]
    all_cell_rows = []
    for row in summaries:
        z_mean, z_bias, z_ci, z_cover = normalizer_fields(row)
        all_cell_rows.append(
            [
                row["_phase"], row["setting"], row["_method_dir"], row["outer_k"], row["inner_m"], row.get("checkpoint") or "—",
                number(row["terminal_success_rmse"]), number(row["route_mass_rmse"]), number(row["equal_weight_mean_rmse"]),
                number(row.get("target_tv")), z_bias, z_cover, row["_relative_path"],
            ]
        )
    table(
        lines,
        ["Phase", "Setting", "Method dir", "K", "M", "CP", "success RMSE", "route RMSE", "mean RMSE", "TV", "Z bias", "CI has 0", "summary path"],
        all_cell_rows,
    )

    lines += [
        "## 11. Raw repetition arrays and file locations",
        "",
        "Each cell's `raw_repetitions.npz` contains:",
        "",
        "| Array | Shape | dtype | Meaning |",
        "| --- | --- | --- | --- |",
        "| `terminal_success_estimate` | `(5000,)` | float64 | per-repetition terminal-success estimate |",
        "| `route_l_estimate` | `(5000,)` | float64 | per-repetition L-route mass estimate |",
        "| `route_r_estimate` | `(5000,)` | float64 | per-repetition R-route mass estimate |",
        "| `normalizer_estimate` | `(5000,)` | float64 | per-repetition global normalizer estimate; NaN where not applicable |",
        "| `reverse_transition_evaluations` | `(5000,)` | int64 | exactly 480 for every repetition |",
        "| `schedule` | `(5000,5,2)` | int16 | selected `(K,M)` at each physical step |",
        "| `probe_diagnostics_rho2_inner_outer` | `(5000,4,3)` | float64 | adaptive diagnostics for t>=2; NaN for non-adaptive cells |",
        "",
        f"Local raw-output root: `{root}`",
        "",
        f"Three-panel result figure: `{root / 'phase_a_three_panel.png'}`",
        "",
        "Original H100 root: `/workspace/sejin/AI_MD_NSMC_v31_stage/outputs/duet_md/phase_a_cross_clock`",
        "",
        f"Raw-file SHA256 manifest: `{args.manifest.resolve()}`",
        "",
        "Primary source files:",
        "",
        "- `metrics.json`: complete A1-A5 aggregate result object",
        "- `validation.json`: frozen-spec validation and workload counts",
        "- `held_out_parameters.json`: all 20 held-out process parameters",
        "- `exact/*.json`: enumerated physical/inner paths and exact targets",
        "- `cells/**/summary.json`: 163 per-cell aggregate summaries",
        "- `cells/**/raw_repetitions.npz`: 163 repetition-level result arrays",
        "- `resolved_config.yaml`, `seed_manifest.json`, `nfe_accounting.json`, `environment.json`: reproducibility metadata",
        "",
        "## 12. Caveats for the coordinating AI",
        "",
        "1. This is an exactly enumerable diagnostic, not a biological protein result.",
        "2. Fixed DuET's measured gains are small. The strongest canonical success gain is in rare/high informative (+4.13%); the mean held-out combined gain is +1.55%.",
        "3. The adaptive allocator is a negative result under this frozen rule. Do not summarize Phase A as validating adaptive K/M allocation.",
        "4. The output contains only one fixed-budget target-TV measurement per method/process, not a population-size convergence series. The requested qualitative convergence criterion cannot be independently claimed from this run alone.",
        "5. Eight of 151 normalizer 95% intervals exclude zero, which is approximately nominal 5% behavior across many independent intervals; no multiplicity-adjusted anomaly test was preregistered.",
        "6. `git_state.json` could not capture a commit because git was unavailable in the runtime. Reproducibility is anchored by the resolved config, config SHA256, copied source checkout, seeds, and raw arrays.",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output.resolve())
    print(args.manifest.resolve())


if __name__ == "__main__":
    main()
