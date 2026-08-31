from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from confmh.analysis import (
    _common_edges,
    _fes,
    _fes_rmse,
    _hist,
    _jsd,
    _normalized,
    _timeseries_ess,
    _weighted_ess,
)
from confmh.config import load_config, resolve_path
from confmh.level11.frozen_bias import FrozenGridBias, HarmonicPotential
from confmh.level11.metrics import rejection_run_lengths
from confmh.utils import kbt_kj_mol, read_json, write_json


def check_pilot_gate(config_path: str | Path, summary_path: str | Path) -> Path:
    cfg = load_config(config_path)
    summary_path = Path(summary_path).resolve()
    summary = read_json(summary_path)
    passing = [float(value) for value in summary["selection"]["passing_ratios"] if float(value) > 0]
    if not passing:
        raise RuntimeError("Pilot gate failed: no positive guidance ratio passed all pilot criteria")
    selected = float(summary["selection"]["selected_clip_ratio"])
    configured = float(cfg["level11"]["guidance"]["clip_ratio"])
    if not np.isclose(selected, configured):
        raise RuntimeError(
            f"Pilot selected clip_ratio={selected}, but umbrella config uses {configured}. "
            "Review the pilot and update the config explicitly before Stage 2."
        )
    return summary_path


def evaluate_stage2_gate(root_dir: str | Path) -> Path:
    root = Path(root_dir).resolve()
    comparison_paths = sorted((root / "comparison").glob("window_*/comparison.json"))
    if len(comparison_paths) != 5:
        raise FileNotFoundError(f"Expected five window comparisons under {root / 'comparison'}")
    comparisons = [read_json(path) for path in comparison_paths]
    unsteered_wham = read_json(root / "unsteered" / "combined" / "metrics.json")
    guided_wham = read_json(root / "guided" / "combined" / "metrics.json")
    ess_ratios = np.asarray([item["ratios"]["ess_per_wallclock"] for item in comparisons])
    esjd_ratios = np.asarray([item["ratios"]["pc1_esjd_per_proposal"] for item in comparisons])
    best_ratios = np.maximum(ess_ratios, esjd_ratios)
    geometric_ess = float(np.exp(np.mean(np.log(np.maximum(ess_ratios, 1e-12)))))
    geometric_esjd = float(np.exp(np.mean(np.log(np.maximum(esjd_ratios, 1e-12)))))
    jsd_change = float(guided_wham["wham_jsd"] - unsteered_wham["wham_jsd"])
    fes_change = float(
        guided_wham["wham_fes_rmse_kj_mol"] - unsteered_wham["wham_fes_rmse_kj_mol"]
    )
    geometry_changes = [item["changes"]["geometry_failure_rate"] for item in comparisons]
    payload = {
        "window_comparisons": [str(path) for path in comparison_paths],
        "geometric_mean_ess_per_wallclock_ratio": geometric_ess,
        "geometric_mean_esjd_per_proposal_ratio": geometric_esjd,
        "windows_with_any_efficiency_improvement": int(np.sum(best_ratios > 1.0)),
        "hardest_window_efficiency_improved": bool(best_ratios[-1] > 1.0),
        "wham_jsd_change": jsd_change,
        "wham_fes_rmse_change_kj_mol": fes_change,
        "maximum_geometry_failure_rate_change": float(max(geometry_changes)),
    }
    payload["passed"] = bool(
        (geometric_ess >= 1.2 or geometric_esjd >= 1.2)
        and payload["windows_with_any_efficiency_improvement"] >= 3
        and payload["hardest_window_efficiency_improved"]
        and jsd_change <= 0.03
        and fes_change <= 0.5
        and payload["maximum_geometry_failure_rate_change"] <= 0.0
    )
    output = root / "stage2_gate.json"
    write_json(output, payload)
    if not payload["passed"]:
        raise RuntimeError(f"Stage 2 gate failed; inspect {output}")
    return output


def _load_potential(run_dir: Path, cfg: dict[str, Any]):
    mode = str(cfg["experiment"]["mode"]).lower()
    if mode == "umbrella":
        return HarmonicPotential(
            float(cfg["experiment"]["center"]), float(cfg["experiment"]["kappa_kj_mol"])
        )
    if mode == "frozen_opes":
        return FrozenGridBias.load(run_dir / "bias.npz")
    raise ValueError(f"Unsupported Level 1.1 analysis mode={mode!r}")


def analyze_level11_run(run_dir: str | Path, bins: int = 40) -> Path:
    import matplotlib.pyplot as plt
    from scipy.stats import wasserstein_distance

    run_dir = Path(run_dir).resolve()
    cfg = load_config(run_dir / "resolved_config.yaml")
    metadata = read_json(run_dir / "metadata.json")
    chain = np.load(run_dir / "chain.npz")
    burn_in = int(metadata.get("burn_in", cfg.get("level11", {}).get("stopping", {}).get("burn_in", 50)))
    values = np.asarray(chain["pc1_accepted"], dtype=float)[burn_in:]
    if len(values) == 0:
        raise ValueError("No production samples to analyze")
    accepted = np.asarray(chain["accepted"], dtype=bool)[burn_in:]
    walltimes = np.asarray(chain["total_walltime_s"], dtype=float)[burn_in:]
    potential = _load_potential(run_dir, cfg)
    reference = np.load(resolve_path(cfg, cfg["reference"]["reference_cv"]))["cv"][:, 0]
    temperature = float(cfg["system"].get("temperature_k", 300.0))
    kbt = kbt_kj_mol(temperature)
    edges = _common_edges(reference, bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    target_weights = _normalized(-potential.energy(reference) / kbt)
    generated_probability = _hist(values, edges)
    target_probability = _hist(reference, edges, target_weights)
    ess = _timeseries_ess(values)
    total_walltime = float(np.sum(walltimes))
    rejection_runs = rejection_run_lengths(accepted)

    pc1_esjd = np.asarray(chain["pc1_squared_jump_accepted"], dtype=float)[burn_in:]
    ca_esjd = np.asarray(chain["ca_esjd_accepted_nm2"], dtype=float)[burn_in:]
    geometry_valid = np.asarray(chain["geometry_valid"], dtype=bool)[burn_in:]
    hashes = np.asarray(chain["structure_hash"])[burn_in:]
    metrics: dict[str, Any] = {
        "method": metadata["method"],
        "mode": metadata["mode"],
        "production_samples": len(values),
        "acceptance_rate": float(np.mean(accepted)),
        "mean_rejection_run": float(np.mean(rejection_runs)) if rejection_runs else 0.0,
        "max_rejection_run": int(max(rejection_runs, default=0)),
        "ess": ess,
        "ess_per_proposal": ess / len(values),
        "ess_per_wallclock_second": ess / max(total_walltime, 1e-12),
        "accepted_pc1_esjd_per_proposal": float(np.mean(pc1_esjd)),
        "accepted_pc1_esjd_per_wallclock_second": float(np.sum(pc1_esjd) / max(total_walltime, 1e-12)),
        "accepted_ca_esjd_nm2_per_proposal": float(np.mean(ca_esjd)),
        "accepted_ca_esjd_nm2_per_wallclock_second": float(np.sum(ca_esjd) / max(total_walltime, 1e-12)),
        "unique_accepted_structures": int(len(np.unique(hashes[accepted]))),
        "mean_proposal_walltime_s": float(np.mean(walltimes)),
        "geometry_failure_rate": float(1.0 - np.mean(geometry_valid)),
        "target_jsd": _jsd(generated_probability, target_probability),
        "target_wasserstein": float(
            wasserstein_distance(values, reference, v_weights=target_weights)
        ),
        "target_fes_rmse_kj_mol": _fes_rmse(generated_probability, target_probability, kbt),
        "support_recall": float(
            np.mean(
                (values >= np.quantile(reference, 0.005))
                & (values <= np.quantile(reference, 0.995))
            )
        ),
    }
    if str(cfg["experiment"]["mode"]).lower() == "frozen_opes":
        unbias_weights = _normalized(potential.energy(values) / kbt)
        unbiased_probability = _hist(values, edges, unbias_weights)
        base_probability = _hist(reference, edges)
        metrics.update(
            {
                "reweighted_jsd": _jsd(unbiased_probability, base_probability),
                "reweighted_fes_rmse_kj_mol": _fes_rmse(
                    unbiased_probability, base_probability, kbt
                ),
                "reweighting_ess": _weighted_ess(unbias_weights),
            }
        )

    output = run_dir / "level11_analysis"
    output.mkdir(exist_ok=True)
    write_json(output / "metrics.json", metrics)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(centers, _fes(target_probability, kbt), label="fixed target")
    ax.plot(centers, _fes(generated_probability, kbt), label="generated")
    ax.set_xlabel("standardized PC1")
    ax.set_ylabel("free energy (kJ/mol)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "target_fes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(values, linewidth=0.65)
    axes[0].set_ylabel("standardized PC1")
    ratios = np.asarray(chain["guidance_to_score_ratio_mean"], dtype=float)[burn_in:]
    axes[1].plot(ratios, linewidth=0.65)
    axes[1].set_ylabel("guidance/base RMS")
    axes[1].set_xlabel("production proposal")
    fig.tight_layout()
    fig.savefig(output / "trace_and_guidance.png", dpi=180)
    plt.close(fig)
    return output


def compare_level11_runs(
    unsteered_dir: str | Path,
    guided_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    unsteered_dir = Path(unsteered_dir).resolve()
    guided_dir = Path(guided_dir).resolve()
    for path in (unsteered_dir, guided_dir):
        metrics_path = path / "level11_analysis" / "metrics.json"
        if not metrics_path.exists():
            analyze_level11_run(path)
    unsteered = read_json(unsteered_dir / "level11_analysis" / "metrics.json")
    guided = read_json(guided_dir / "level11_analysis" / "metrics.json")

    def ratio(key: str) -> float:
        return float(guided[key]) / max(float(unsteered[key]), 1e-12)

    comparison = {
        "unsteered": unsteered,
        "guided": guided,
        "ratios": {
            "ess_per_wallclock": ratio("ess_per_wallclock_second"),
            "pc1_esjd_per_proposal": ratio("accepted_pc1_esjd_per_proposal"),
            "pc1_esjd_per_wallclock": ratio("accepted_pc1_esjd_per_wallclock_second"),
        },
        "changes": {
            "target_jsd": float(guided["target_jsd"]) - float(unsteered["target_jsd"]),
            "target_fes_rmse_kj_mol": float(guided["target_fes_rmse_kj_mol"])
            - float(unsteered["target_fes_rmse_kj_mol"]),
            "support_recall": float(guided["support_recall"])
            - float(unsteered["support_recall"]),
            "geometry_failure_rate": float(guided["geometry_failure_rate"])
            - float(unsteered["geometry_failure_rate"]),
        },
    }
    comparison["engineering_gate_passed"] = bool(
        comparison["ratios"]["ess_per_wallclock"] >= 1.2
        and comparison["changes"]["target_jsd"] <= 0.03
        and comparison["changes"]["target_fes_rmse_kj_mol"] <= 0.5
        and comparison["changes"]["support_recall"] >= -0.05
        and comparison["changes"]["geometry_failure_rate"] <= 0.0
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", comparison)
    return output_dir
