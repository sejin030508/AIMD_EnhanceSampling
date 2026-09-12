from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "frozen": "Frozen",
    "outer_only": "Outer-only",
    "inner_only": "Inner-only",
    "complete_nested": "Complete Nested",
    "fixed_duet": "Fixed DuET",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the specified Phase A three-panel figure")
    parser.add_argument(
        "--input",
        default="outputs/duet_md/phase_a_cross_clock/metrics.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/duet_md/phase_a_cross_clock/phase_a_three_panel.png",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    metrics_path = Path(args.input).expanduser().resolve()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    proper = [
        row
        for row in metrics["fixed_methods"]
        if row["method"] in {"outer_only", "complete_nested", "fixed_duet"}
        and row["setting"].endswith("informative")
    ]
    for method in ("outer_only", "complete_nested", "fixed_duet"):
        rows = [row for row in proper if row["method"] == method]
        axes[0].scatter(
            [row["target_tv"] for row in rows],
            [row["normalizer"]["absolute_relative_bias"] for row in rows],
            label=METHOD_LABELS[method],
        )
    axes[0].set_title("A. Correctness")
    axes[0].set_xlabel("Full-path total variation")
    axes[0].set_ylabel("Absolute relative normalizer bias")
    axes[0].legend(frameon=False, fontsize=8)

    regimes = [
        "common_low_informative",
        "rare_low_informative",
        "common_high_informative",
        "rare_high_informative",
    ]
    methods = ["frozen", "outer_only", "inner_only", "complete_nested", "fixed_duet"]
    width = 0.16
    x = np.arange(len(regimes))
    for index, method in enumerate(methods):
        values = []
        for regime in regimes:
            row = next(
                item
                for item in metrics["fixed_methods"]
                if item["setting"] == regime and item["method"] == method
            )
            values.append(row["equal_weight_mean_rmse"])
        axes[1].bar(
            x + (index - 2) * width,
            values,
            width,
            label=METHOD_LABELS[method],
        )
    axes[1].set_title("B. Method regimes")
    axes[1].set_xticks(x, ["C/L", "R/L", "C/H", "R/H"])
    axes[1].set_ylabel("Equal-weight mean RMSE")
    axes[1].legend(frameon=False, fontsize=7)

    held_out = metrics["held_out"]["per_process"]
    rho = np.asarray([row["mean_probe_rho_squared"] for row in held_out])
    gain = np.asarray([row["pre_completion_success_rmse_gain"] for row in held_out])
    improvement = np.asarray([row["adaptive_improvement"] for row in held_out])
    points = axes[2].scatter(rho, gain, c=improvement, cmap="coolwarm", edgecolor="black")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("C. Diagnosis and allocation")
    axes[2].set_xlabel(r"Mean probe $\rho^2$")
    axes[2].set_ylabel("DuET pre-completion RMSE gain")
    colorbar = figure.colorbar(points, ax=axes[2])
    colorbar.set_label("Adaptive improvement vs global static")

    figure.savefig(output, dpi=300)
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
