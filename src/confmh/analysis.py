from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

from confmh.bias import HarmonicBias, OPES1D
from confmh.config import load_config, resolve_path
from confmh.utils import kbt_kj_mol, read_json, write_json


def _normalized(log_weights: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=float)
    return np.exp(log_weights - logsumexp(log_weights))


def _weighted_ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights**2))


def _timeseries_ess(values: np.ndarray) -> float:
    """Estimate ESS by truncating the autocorrelation sum at its first non-positive lag."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return float(n)
    centered = values - values.mean()
    variance = float(np.dot(centered, centered))
    if variance == 0:
        return 1.0
    autocorrelation = np.correlate(centered, centered, mode="full")[n - 1 :] / variance
    tau = 1.0
    for value in autocorrelation[1:]:
        if value <= 0:
            break
        tau += 2.0 * float(value)
    return float(max(1.0, min(n, n / tau)))


def _hist(values: np.ndarray, edges: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    hist, _ = np.histogram(values, bins=edges, weights=weights)
    hist = hist.astype(float) + 1e-12
    return hist / hist.sum()


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _fes(probability: np.ndarray, kbt: float) -> np.ndarray:
    free_energy = -kbt * np.log(np.maximum(probability, 1e-12))
    return free_energy - free_energy.min()


def _fes_rmse(p: np.ndarray, q: np.ndarray, kbt: float) -> float:
    mask = (p > 1e-5) & (q > 1e-5)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((_fes(p, kbt)[mask] - _fes(q, kbt)[mask]) ** 2)))


def _common_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    lower, upper = np.quantile(reference, [0.001, 0.999])
    margin = max(0.1, 0.05 * (upper - lower))
    return np.linspace(lower - margin, upper + margin, bins + 1)


def analyze_run(run_dir: str | Path, bins: int = 80) -> Path:
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    cfg = load_config(run_dir / "resolved_config.yaml")
    metadata = read_json(run_dir / "metadata.json")
    chain = np.load(run_dir / "chain.npz")
    reference = np.load(resolve_path(cfg, cfg["reference"]["reference_cv"]))["cv"][:, 0]
    all_values = np.asarray(chain["cv"], dtype=float)
    all_phases = np.asarray(chain["phase"]).astype(str)
    burn_in = int(cfg.get("analysis", {}).get("burn_in", 0))
    values = all_values[burn_in:]
    phases = all_phases[burn_in:]
    kbt = float(metadata["kbt_kj_mol"])
    mode = str(metadata["mode"])
    edges = _common_edges(reference, bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)

    metrics: dict[str, Any] = {
        "mode": mode,
        "acceptance_rate": float(np.mean(chain["accepted"])),
        "mean_proposal_walltime_s": float(np.mean(chain["proposal_walltime_s"])),
        "timeseries_ess": _timeseries_ess(values),
        "ess_fraction": _timeseries_ess(values) / len(values),
    }

    if mode == "base":
        generated = values
        generated_weights = None
        target_weights = None
        target_name = "reference base"
    elif mode == "umbrella":
        experiment = cfg["experiment"]
        bias = HarmonicBias(float(experiment["center"]), float(experiment["kappa_kj_mol"]))
        generated = values
        generated_weights = None
        target_weights = _normalized(-bias.energy(reference) / kbt)
        target_name = "reference reweighted to umbrella target"
    elif mode == "opes":
        production = phases == "production"
        generated = values[production]
        production_accepted = np.asarray(chain["accepted"], dtype=bool)[burn_in:][production]
        production_burn_in = int(cfg.get("analysis", {}).get("production_burn_in", 0))
        generated = generated[production_burn_in:]
        production_accepted = production_accepted[production_burn_in:]
        if len(generated) == 0:
            raise ValueError("No OPES production samples found")
        bias = OPES1D.load(run_dir / "bias.npz")
        generated_weights = None
        target_weights = _normalized(-bias.energy(reference) / kbt)
        target_name = "reference reweighted to frozen OPES target"
        production_ess = _timeseries_ess(generated)
        metrics.update(
            {
                "production_samples": int(len(generated)),
                "production_acceptance_rate": float(np.mean(production_accepted)),
                "production_timeseries_ess": production_ess,
                "production_ess_fraction": production_ess / len(generated),
            }
        )
    else:
        raise ValueError(mode)

    p_generated = _hist(generated, edges, generated_weights)
    p_target = _hist(reference, edges, target_weights)
    metrics.update(
        {
            "target": target_name,
            "target_jsd": _jsd(p_generated, p_target),
            "target_wasserstein": float(
                wasserstein_distance(
                    generated,
                    reference,
                    u_weights=generated_weights,
                    v_weights=target_weights,
                )
            ),
            "target_fes_rmse_kj_mol": _fes_rmse(p_generated, p_target, kbt),
            "support_recall": float(
                np.mean(
                    (generated >= np.quantile(reference, 0.005))
                    & (generated <= np.quantile(reference, 0.995))
                )
            ),
        }
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(centers, _fes(p_target, kbt), label=target_name)
    ax.plot(centers, _fes(p_generated, kbt), label="generated")
    ax.set_xlabel("standardized PC1")
    ax.set_ylabel("free energy (kJ/mol)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "target_fes.png", dpi=180)
    plt.close(fig)

    if mode == "opes":
        bias = OPES1D.load(run_dir / "bias.npz")
        unbias_weights = _normalized(bias.energy(generated) / kbt)
        p_unbiased = _hist(generated, edges, unbias_weights)
        p_base = _hist(reference, edges)
        metrics.update(
            {
                "unbiased_jsd": _jsd(p_unbiased, p_base),
                "unbiased_fes_rmse_kj_mol": _fes_rmse(p_unbiased, p_base, kbt),
                "reweighting_ess": _weighted_ess(unbias_weights),
            }
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(centers, _fes(p_base, kbt), label="reference base")
        ax.plot(centers, _fes(p_unbiased, kbt), label="OPES reweighted")
        ax.set_xlabel("standardized PC1")
        ax.set_ylabel("free energy (kJ/mol)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "unbiased_fes.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(bias.grid, bias.bias_grid)
        ax.set_xlabel("standardized PC1")
        ax.set_ylabel("frozen bias (kJ/mol)")
        fig.tight_layout()
        fig.savefig(output / "frozen_bias.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(values, linewidth=0.6)
    if mode == "opes":
        adapt_steps = int(cfg["experiment"].get("adapt_steps", 0))
        ax.axvline(adapt_steps, linestyle="--", label="bias frozen")
        ax.legend()
    ax.set_xlabel("MCMC step")
    ax.set_ylabel("standardized PC1")
    fig.tight_layout()
    fig.savefig(output / "trace.png", dpi=180)
    plt.close(fig)

    write_json(output / "metrics.json", metrics)
    return output


def analyze_baseline_comparison(
    full_history_dirs: list[str | Path],
    one_step_dirs: list[str | Path],
    output_dir: str | Path,
    bins: int = 20,
) -> Path:
    """Aggregate matched 100 ns replicates and compare both baselines to ATLAS."""
    import matplotlib.pyplot as plt

    groups = {
        "full_history": [Path(path).resolve() for path in full_history_dirs],
        "one_step": [Path(path).resolve() for path in one_step_dirs],
    }
    for name, paths in groups.items():
        if not paths or any(not (path / "chain.npz").exists() for path in paths):
            raise FileNotFoundError(f"Incomplete {name} baseline directories: {paths}")
    first_cfg = load_config(groups["full_history"][0] / "resolved_config.yaml")
    reference_data = np.load(resolve_path(first_cfg, first_cfg["reference"]["reference_cv"]))
    reference = np.asarray(reference_data["cv"][:, 0], dtype=float)
    edges = _common_edges(reference, bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p_reference = _hist(reference, edges)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "reference": {
            "replicates": int(len(np.unique(reference_data["source_trajectory"]))),
            "samples": int(len(reference)),
        }
    }
    fig_fes, ax_fes = plt.subplots(figsize=(7, 4.5))
    ax_fes.plot(centers, _fes(p_reference, kbt_kj_mol(300.0)), label="ATLAS")
    fig_trace, axes_trace = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for axis, (name, paths) in zip(axes_trace, groups.items()):
        replicas = [np.asarray(np.load(path / "chain.npz")["cv"], dtype=float) for path in paths]
        combined = np.concatenate(replicas)
        probability = _hist(combined, edges)
        ess = float(sum(_timeseries_ess(values) for values in replicas))
        metrics[name] = {
            "replicates": len(replicas),
            "samples": int(len(combined)),
            "timeseries_ess": ess,
            "ess_fraction": ess / len(combined),
            "target_jsd": _jsd(probability, p_reference),
            "target_wasserstein": float(wasserstein_distance(combined, reference)),
            "support_recall": float(
                np.mean(
                    (combined >= np.quantile(reference, 0.005))
                    & (combined <= np.quantile(reference, 0.995))
                )
            ),
        }
        ax_fes.plot(centers, _fes(probability, kbt_kj_mol(300.0)), label=name)
        for index, values in enumerate(replicas):
            axis.plot(np.arange(1, len(values) + 1) * 2.56, values, label=f"seed {index}")
        axis.set_ylabel(f"{name}\nstandardized PC1")
        axis.legend(ncol=len(replicas), fontsize=8)

    ax_fes.set_xlabel("standardized PC1")
    ax_fes.set_ylabel("free energy (kJ/mol)")
    ax_fes.legend()
    fig_fes.tight_layout()
    fig_fes.savefig(output_dir / "target_fes.png", dpi=180)
    plt.close(fig_fes)
    axes_trace[-1].set_xlabel("nominal ConfRover time (ns)")
    fig_trace.tight_layout()
    fig_trace.savefig(output_dir / "traces.png", dpi=180)
    plt.close(fig_trace)
    write_json(output_dir / "metrics.json", metrics)
    return output_dir


def wham_weights(reduced_bias_kn: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reduced_bias_kn = np.asarray(reduced_bias_kn, dtype=float)
    counts = np.asarray(counts, dtype=float)
    log_counts = np.log(counts)
    f = np.zeros(len(counts))
    for _ in range(100000):
        log_denom = logsumexp(log_counts[:, None] + f[:, None] - reduced_bias_kn, axis=0)
        new_f = -logsumexp(-reduced_bias_kn - log_denom[None, :], axis=1)
        new_f -= new_f[0]
        if np.max(np.abs(new_f - f)) < 1e-10:
            f = new_f
            break
        f = new_f
    else:
        raise RuntimeError("WHAM failed to converge")
    log_denom = logsumexp(log_counts[:, None] + f[:, None] - reduced_bias_kn, axis=0)
    return _normalized(-log_denom), f


def analyze_umbrella_suite(run_dirs: list[str | Path], output_dir: str | Path, bins: int = 80) -> Path:
    import matplotlib.pyplot as plt

    run_dirs = [Path(path).resolve() for path in run_dirs]
    run_dirs = [path for path in run_dirs if (path / "chain.npz").exists()]
    if len(run_dirs) < 2:
        raise ValueError("At least two umbrella windows are required")
    all_values = []
    counts = []
    biases = []
    window_metrics = []
    reference = None
    kbt = None
    for run_dir in run_dirs:
        cfg = load_config(run_dir / "resolved_config.yaml")
        metadata = read_json(run_dir / "metadata.json")
        chain = np.load(run_dir / "chain.npz")
        burn_in = int(cfg.get("analysis", {}).get("burn_in", 0))
        values = np.asarray(chain["cv"], dtype=float)[burn_in:]
        all_values.append(values)
        counts.append(len(values))
        center = float(cfg["experiment"]["center"])
        biases.append(HarmonicBias(center, float(cfg["experiment"]["kappa_kj_mol"])))
        ess = _timeseries_ess(values)
        window_metrics.append(
            {
                "run_dir": str(run_dir),
                "center": center,
                "samples": len(values),
                "acceptance_rate": float(np.mean(chain["accepted"][burn_in:])),
                "timeseries_ess": ess,
                "ess_fraction": ess / len(values),
            }
        )
        if reference is None:
            reference = np.load(resolve_path(cfg, cfg["reference"]["reference_cv"]))["cv"][:, 0]
            kbt = float(metadata["kbt_kj_mol"])
    assert reference is not None and kbt is not None
    samples = np.concatenate(all_values)
    reduced = np.stack([bias.energy(samples) / kbt for bias in biases])
    weights, window_f = wham_weights(reduced, np.asarray(counts))
    edges = _common_edges(reference, bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p_wham = _hist(samples, edges, weights)
    p_reference = _hist(reference, edges)
    window_histograms = np.stack([_hist(values, edges) for values in all_values])
    overlap = np.minimum(window_histograms[:, None, :], window_histograms[None, :, :]).sum(axis=2)
    neighbor_overlap = [float(overlap[index, index + 1]) for index in range(len(all_values) - 1)]
    metrics = {
        "num_windows": len(run_dirs),
        "wham_jsd": _jsd(p_wham, p_reference),
        "wham_fes_rmse_kj_mol": _fes_rmse(p_wham, p_reference, kbt),
        "wham_weight_ess": _weighted_ess(weights),
        "windows": window_metrics,
        "neighbor_overlap": neighbor_overlap,
        "minimum_neighbor_overlap": min(neighbor_overlap),
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "wham.npz",
        centers=centers,
        probability=p_wham,
        reference_probability=p_reference,
        sample_weights=weights,
        window_free_energies=window_f,
        overlap=overlap,
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(centers, _fes(p_reference, kbt), label="reference")
    ax.plot(centers, _fes(p_wham, kbt), label="umbrella WHAM")
    ax.set_xlabel("standardized PC1")
    ax.set_ylabel("free energy (kJ/mol)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wham_fes.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(overlap, vmin=0.0, vmax=1.0, origin="lower")
    fig.colorbar(image, ax=ax, label="histogram overlap")
    ax.set_xlabel("window")
    ax.set_ylabel("window")
    fig.tight_layout()
    fig.savefig(output_dir / "window_overlap.png", dpi=180)
    plt.close(fig)
    write_json(output_dir / "metrics.json", metrics)
    return output_dir
