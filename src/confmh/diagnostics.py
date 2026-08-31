from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from confmh.config import resolve_path, save_config
from confmh.experiment import _build_kernel
from confmh.pca_cv import PCACV
from confmh.utils import seed_everything, write_json


def _pc1_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    lower, upper = np.quantile(np.asarray(reference, dtype=float), [0.005, 0.995])
    return np.linspace(lower, upper, int(n_bins) + 1)


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2).astype(int)


def _transition_residual(stationary: np.ndarray, transition: np.ndarray) -> float:
    flux = np.asarray(stationary)[:, None] * np.asarray(transition)
    upper = np.triu_indices(len(stationary), k=1)
    numerator = np.sum(np.abs(flux[upper] - flux.T[upper]))
    denominator = np.sum(flux[upper] + flux.T[upper])
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _residual_batch(stationary: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    flux = stationary[None, :, None] * transitions
    upper = np.triu_indices(len(stationary), k=1)
    direct = flux[:, upper[0], upper[1]]
    reverse = flux[:, upper[1], upper[0]]
    numerator = np.sum(np.abs(direct - reverse), axis=1)
    denominator = np.sum(direct + reverse, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def _fit_reversible_null(
    stationary: np.ndarray, transition: np.ndarray, smoothing: float = 1.0e-4
) -> np.ndarray:
    """Build a reversible finite-sample null with the requested stationary marginal."""

    stationary = np.asarray(stationary, dtype=float)
    flux = stationary[:, None] * np.asarray(transition, dtype=float)
    kernel = 0.5 * (flux + flux.T) + float(smoothing) * np.outer(stationary, stationary)
    scale = np.ones_like(stationary)
    for _ in range(200_000):
        updated = np.sqrt(
            scale * stationary / np.maximum(kernel @ scale, np.finfo(float).tiny)
        )
        if np.max(np.abs(updated - scale)) < 1.0e-13:
            scale = updated
            break
        scale = updated
    reversible_flux = scale[:, None] * kernel * scale[None, :]
    return reversible_flux / stationary[:, None]


def _cluster_bootstrap_residuals(
    *,
    stationary: np.ndarray,
    source_bins: np.ndarray,
    destination_bins: np.ndarray,
    group_ids: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_bins = len(stationary)
    unique_groups = np.unique(group_ids)
    group_source = np.empty(len(unique_groups), dtype=int)
    group_counts = np.zeros((len(unique_groups), n_bins), dtype=int)
    for index, group in enumerate(unique_groups):
        selected = group_ids == group
        bins = np.unique(source_bins[selected])
        if len(bins) != 1:
            raise ValueError(f"Bootstrap group {group} spans multiple source bins")
        group_source[index] = bins[0]
        group_counts[index] = np.bincount(destination_bins[selected], minlength=n_bins)

    boot_counts = np.zeros((replicates, n_bins, n_bins), dtype=float)
    for source_bin in range(n_bins):
        members = np.flatnonzero(group_source == source_bin)
        if len(members) == 0:
            continue
        sampled = rng.integers(0, len(members), size=(replicates, len(members)))
        boot_counts[:, source_bin, :] = group_counts[members[sampled]].sum(axis=1)
    row_sums = boot_counts.sum(axis=2, keepdims=True)
    transitions = np.divide(
        boot_counts,
        row_sums,
        out=np.zeros_like(boot_counts),
        where=row_sums > 0,
    )
    return _residual_batch(stationary, transitions)


def _null_bootstrap_residuals(
    *,
    stationary: np.ndarray,
    transition: np.ndarray,
    row_counts: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    null_transition = _fit_reversible_null(stationary, transition)
    simulated = np.zeros((replicates, len(stationary), len(stationary)), dtype=float)
    for source_bin, count in enumerate(np.asarray(row_counts, dtype=int)):
        if count:
            probabilities = null_transition[source_bin]
            probabilities = probabilities / probabilities.sum()
            simulated[:, source_bin, :] = rng.multinomial(
                int(count), probabilities, size=replicates
            )
    row_sums = simulated.sum(axis=2, keepdims=True)
    transitions = np.divide(
        simulated,
        row_sums,
        out=np.zeros_like(simulated),
        where=row_sums > 0,
    )
    return null_transition, _residual_batch(stationary, transitions)


def _projection_summary(
    *,
    reference_pc1: np.ndarray,
    source_pc1: np.ndarray,
    destination_pc1: np.ndarray,
    group_ids: np.ndarray,
    n_bins: int,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    edges = _pc1_edges(reference_pc1, n_bins)
    stationary = np.histogram(reference_pc1, bins=edges)[0].astype(float)
    stationary /= stationary.sum()
    source_bins = _digitize(source_pc1, edges)
    destination_bins = _digitize(destination_pc1, edges)
    counts = np.zeros((n_bins, n_bins), dtype=float)
    np.add.at(counts, (source_bins, destination_bins), 1)
    row_counts = counts.sum(axis=1)
    transition = np.divide(
        counts,
        row_counts[:, None],
        out=np.zeros_like(counts),
        where=row_counts[:, None] > 0,
    )
    flux = stationary[:, None] * transition
    residual = _transition_residual(stationary, transition)

    rng = np.random.default_rng(int(seed) + n_bins * 10_007)
    cluster = _cluster_bootstrap_residuals(
        stationary=stationary,
        source_bins=source_bins,
        destination_bins=destination_bins,
        group_ids=group_ids,
        replicates=bootstrap_replicates,
        rng=rng,
    )
    null_transition, null = _null_bootstrap_residuals(
        stationary=stationary,
        transition=transition,
        row_counts=row_counts,
        replicates=bootstrap_replicates,
        rng=rng,
    )
    cluster_quantiles = np.nanquantile(cluster, [0.025, 0.5, 0.975])
    null_quantiles = np.nanquantile(null, [0.5, 0.95, 0.975])
    basic_interval = np.clip(
        [2 * residual - cluster_quantiles[2], 2 * residual - cluster_quantiles[0]],
        0.0,
        1.0,
    )
    summary = {
        "n_bins": int(n_bins),
        "num_transitions": int(len(source_pc1)),
        "num_source_structures": int(len(np.unique(group_ids))),
        "row_transition_counts": row_counts.astype(int).tolist(),
        "projected_detailed_balance_residual": residual,
        "cluster_bootstrap_percentile_interval_95": [
            float(cluster_quantiles[0]),
            float(cluster_quantiles[2]),
        ],
        "cluster_bootstrap_basic_interval_95": [
            float(basic_interval[0]),
            float(basic_interval[1]),
        ],
        "cluster_bootstrap_median": float(cluster_quantiles[1]),
        "reversible_null_median": float(null_quantiles[0]),
        "reversible_null_percentile_95": float(null_quantiles[1]),
        "reversible_null_percentile_97_5": float(null_quantiles[2]),
        "reversible_null_tail_probability": float(np.mean(null >= residual)),
        "bootstrap_replicates": int(bootstrap_replicates),
    }
    arrays = {
        "edges": edges,
        "stationary": stationary,
        "counts": counts,
        "transition": transition,
        "flux": flux,
        "null_transition": null_transition,
        "cluster_bootstrap_residual": cluster,
        "null_bootstrap_residual": null,
    }
    return summary, arrays


def _swap_classifier_test(
    *,
    source_cv: np.ndarray,
    destination_cv: np.ndarray,
    source_rg_nm: np.ndarray,
    destination_rg_nm: np.ndarray,
    group_ids: np.ndarray,
    pair_weights: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Test whether ordered projected transition pairs are distinguishable from swaps."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    source_features = np.column_stack([source_cv[:, :2], source_rg_nm])
    destination_features = np.column_stack([destination_cv[:, :2], destination_rg_nm])
    forward = np.column_stack([source_features, destination_features])
    reverse = np.column_stack([destination_features, source_features])
    features = np.concatenate([forward, reverse], axis=0)
    labels = np.concatenate(
        [np.ones(len(forward), dtype=int), np.zeros(len(reverse), dtype=int)]
    )
    groups = np.concatenate([group_ids, group_ids])
    weights = np.concatenate([pair_weights, pair_weights])
    unique_groups = np.unique(group_ids)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Pair-swap test requires at least two source-structure groups")
    folds = list(GroupKFold(n_splits=n_splits).split(features, labels, groups))

    def cross_validated_auc(target: np.ndarray) -> float:
        predictions = np.zeros(len(target), dtype=float)
        for train, test in folds:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2_000, random_state=int(seed)),
            )
            model.fit(
                features[train],
                target[train],
                logisticregression__sample_weight=weights[train],
            )
            predictions[test] = model.predict_proba(features[test])[:, 1]
        return float(roc_auc_score(target, predictions, sample_weight=weights))

    observed = cross_validated_auc(labels)
    rng = np.random.default_rng(int(seed) + 90_001)
    null_auc = np.empty(permutations, dtype=float)
    group_lookup = np.searchsorted(unique_groups, groups)
    for index in range(permutations):
        flips = rng.integers(0, 2, size=len(unique_groups), dtype=int)
        permuted = np.bitwise_xor(labels, flips[group_lookup])
        null_auc[index] = cross_validated_auc(permuted)
    return {
        "features": "ordered [PC1, PC2, C-alpha radius-of-gyration] pairs",
        "cross_validated_auc": observed,
        "permutation_tail_probability": float(
            (1 + np.sum(null_auc >= observed)) / (permutations + 1)
        ),
        "permutations": int(permutations),
        "null_auc_median": float(np.median(null_auc)),
        "null_auc_percentile_95": float(np.quantile(null_auc, 0.95)),
        "warning": "This tests projected pair orientation, not full-state reversibility.",
    }


def _radius_of_gyration(ca_nm: np.ndarray) -> np.ndarray:
    centered = ca_nm - ca_nm.mean(axis=-2, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=-1))


def _prepare_balanced_seed_manifest(cfg: dict[str, Any]) -> Path:
    import mdtraj as md

    reference_cfg = cfg["reference"]
    diagnostic = cfg["diagnostic"]
    manifest = resolve_path(cfg, reference_cfg["seed_manifest"])
    seeds_per_bin = int(diagnostic["balanced_seeds_per_bin"])
    n_bins = int(diagnostic.get("n_bins", 12))
    expected = seeds_per_bin * n_bins
    if manifest.exists() and not bool(diagnostic.get("regenerate_seed_manifest", False)):
        with manifest.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) == expected and all(Path(row["pdb"]).exists() for row in rows):
            return manifest

    reference_path = resolve_path(cfg, reference_cfg["reference_cv"])
    reference = np.load(reference_path)
    scores = np.asarray(reference["cv"], dtype=float)
    source_trajectory = np.asarray(reference["source_trajectory"], dtype=int)
    source_frame = np.asarray(reference["source_frame"], dtype=int)
    trajectories = [Path(str(item)) for item in reference["trajectory_paths"]]
    topology_path = resolve_path(cfg, reference_cfg["topology_pdb"])
    topology = md.load(str(topology_path))
    protein_selection = str(reference_cfg.get("protein_selection", "protein and chainid 0"))
    protein_indices = topology.topology.select(protein_selection)
    if len(protein_indices) == 0:
        raise ValueError(f"Selection matched no atoms: {protein_selection}")

    edges = _pc1_edges(scores[:, 0], n_bins)
    bins = _digitize(scores[:, 0], edges)
    rng = np.random.default_rng(int(diagnostic.get("seed", 20260827)))
    seed_dir = manifest.parent / "seed_frames"
    seed_dir.mkdir(parents=True, exist_ok=True)
    selected: list[int] = []
    for source_bin in range(n_bins):
        candidates = np.flatnonzero(bins == source_bin)
        if len(candidates) < seeds_per_bin:
            raise ValueError(
                f"PC1 bin {source_bin} has {len(candidates)} reference frames; "
                f"need {seeds_per_bin}"
            )
        choices: list[int] = []
        trajectory_ids = np.unique(source_trajectory[candidates])
        base, remainder = divmod(seeds_per_bin, len(trajectory_ids))
        for offset, trajectory_id in enumerate(trajectory_ids):
            pool = candidates[source_trajectory[candidates] == trajectory_id]
            quota = min(base + int(offset < remainder), len(pool))
            choices.extend(rng.choice(pool, size=quota, replace=False).tolist())
        if len(choices) < seeds_per_bin:
            remaining = np.setdiff1d(candidates, np.asarray(choices, dtype=int))
            choices.extend(
                rng.choice(remaining, size=seeds_per_bin - len(choices), replace=False).tolist()
            )
        rng.shuffle(choices)
        selected.extend(choices[:seeds_per_bin])

    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed_index, sample_index in enumerate(selected):
        pdb_path = seed_dir / f"seed_{seed_index:04d}.pdb"
        trajectory_index = int(source_trajectory[sample_index])
        frame_index = int(source_frame[sample_index])
        if not pdb_path.exists():
            frame = md.load_frame(
                str(trajectories[trajectory_index]),
                frame_index,
                top=str(topology_path),
                atom_indices=protein_indices,
            )
            frame.save_pdb(str(pdb_path))
        rows.append(
            {
                "seed_index": seed_index,
                "source_bin": int(bins[sample_index]),
                "pc1": float(scores[sample_index, 0]),
                "pc2": float(scores[sample_index, 1]),
                "source_trajectory": trajectory_index,
                "source_frame": frame_index,
                "pdb": str(pdb_path.resolve()),
            }
        )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _save_seed_record(
    *,
    path: Path,
    seed_index: int,
    condition: Path,
    proposals,
    pca: PCACV,
) -> None:
    import mdtraj as md

    source_trajectory = md.load(str(condition))
    source_ca_indices = source_trajectory.topology.select("name CA")
    source_ca = source_trajectory.xyz[-1, source_ca_indices, :]
    source_cv = pca.project_trajectory(source_trajectory[-1])[0]
    destination_cv: list[np.ndarray] = []
    destination_ca: list[np.ndarray] = []
    for proposal in proposals:
        trajectory = proposal.load()
        endpoint = trajectory[-1]
        ca_indices = endpoint.topology.select("name CA")
        destination_cv.append(pca.project_trajectory(endpoint)[0])
        destination_ca.append(endpoint.xyz[-1, ca_indices, :])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            seed_index=np.asarray(seed_index, dtype=int),
            source_cv=np.asarray(source_cv, dtype=np.float32),
            destination_cv=np.asarray(destination_cv, dtype=np.float32),
            source_ca_nm=np.asarray(source_ca, dtype=np.float32),
            destination_ca_nm=np.asarray(destination_ca, dtype=np.float32),
        )
    temporary.replace(path)


def _load_seed_records(record_paths: list[Path]) -> dict[str, np.ndarray]:
    source_cv: list[np.ndarray] = []
    destination_cv: list[np.ndarray] = []
    source_ca: list[np.ndarray] = []
    destination_ca: list[np.ndarray] = []
    groups: list[int] = []
    replicate_index: list[int] = []
    for path in record_paths:
        record = np.load(path)
        destinations = np.asarray(record["destination_cv"])
        n_replicates = len(destinations)
        source_cv.extend([np.asarray(record["source_cv"])] * n_replicates)
        destination_cv.extend(destinations)
        source_ca.extend([np.asarray(record["source_ca_nm"])] * n_replicates)
        destination_ca.extend(np.asarray(record["destination_ca_nm"]))
        groups.extend([int(record["seed_index"])] * n_replicates)
        replicate_index.extend(range(n_replicates))
    return {
        "source_cv": np.asarray(source_cv),
        "destination_cv": np.asarray(destination_cv),
        "source_ca_nm": np.asarray(source_ca),
        "destination_ca_nm": np.asarray(destination_ca),
        "group_id": np.asarray(groups, dtype=int),
        "replicate_index": np.asarray(replicate_index, dtype=int),
    }


def diagnose_reversibility(cfg: dict[str, Any]) -> Path:
    import matplotlib.pyplot as plt

    diagnostic = cfg["diagnostic"]
    seed = int(diagnostic.get("seed", 20260827))
    seed_everything(seed)
    output_dir = resolve_path(cfg, diagnostic.get("output_dir", "outputs/reversibility"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")

    if diagnostic.get("balanced_seeds_per_bin") is not None:
        manifest = _prepare_balanced_seed_manifest(cfg)
    else:
        manifest = resolve_path(cfg, cfg["reference"]["seed_manifest"])
    with manifest.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    max_seeds = diagnostic.get("max_seeds")
    if max_seeds is not None:
        rows = rows[: int(max_seeds)]
    proposals_per_seed = int(diagnostic.get("proposals_per_seed", 4))
    n_bins = int(diagnostic.get("n_bins", 12))

    pca = PCACV.load(resolve_path(cfg, cfg["reference"]["pca_model"]))
    reference_data = np.load(resolve_path(cfg, cfg["reference"]["reference_cv"]))
    reference_cv = np.asarray(reference_data["cv"], dtype=float)
    record_root = output_dir / "records"
    proposal_root = output_dir / "proposals"
    record_root.mkdir(exist_ok=True)
    proposal_root.mkdir(exist_ok=True)

    pending = [
        (index, row)
        for index, row in enumerate(rows)
        if not (record_root / f"seed_{index:04d}.npz").exists()
    ]
    kernel = _build_kernel(cfg, seed) if pending else None
    for seed_index, row in pending:
        condition = Path(row["pdb"])
        if not condition.is_absolute():
            condition = resolve_path(cfg, condition)
        step_dir = proposal_root / f"seed_{seed_index:04d}"
        if step_dir.exists():
            shutil.rmtree(step_dir)
        proposals = kernel.propose(
            condition_pdb=condition,
            output_dir=step_dir,
            step=seed_index,
            n_replicates=proposals_per_seed,
        )
        _save_seed_record(
            path=record_root / f"seed_{seed_index:04d}.npz",
            seed_index=seed_index,
            condition=condition,
            proposals=proposals,
            pca=pca,
        )
        if bool(diagnostic.get("cleanup_proposals", True)):
            shutil.rmtree(step_dir, ignore_errors=True)

    record_paths = [record_root / f"seed_{index:04d}.npz" for index in range(len(rows))]
    missing = [path for path in record_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} seed records; rerun to resume")
    records = _load_seed_records(record_paths)
    np.savez_compressed(output_dir / "transition_records.npz", **records)

    bootstrap_replicates = int(diagnostic.get("bootstrap_replicates", 30_000))
    sensitivity_bins = sorted(
        set(int(item) for item in diagnostic.get("sensitivity_bins", [6, 8, n_bins]))
    )
    sensitivity: dict[str, dict[str, Any]] = {}
    primary_arrays: dict[str, np.ndarray] | None = None
    for bins in sensitivity_bins:
        summary, arrays = _projection_summary(
            reference_pc1=reference_cv[:, 0],
            source_pc1=records["source_cv"][:, 0],
            destination_pc1=records["destination_cv"][:, 0],
            group_ids=records["group_id"],
            n_bins=bins,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        sensitivity[str(bins)] = summary
        np.savez_compressed(output_dir / f"reversibility_{bins}bins.npz", **arrays)
        if bins == n_bins:
            primary_arrays = arrays
    if primary_arrays is None:
        raise ValueError("diagnostic.n_bins must be included in sensitivity_bins")
    np.savez_compressed(output_dir / "reversibility.npz", **primary_arrays)

    primary_edges = primary_arrays["edges"]
    primary_stationary = primary_arrays["stationary"]
    source_bins = _digitize(records["source_cv"][:, 0], primary_edges)
    pairs_per_bin = np.bincount(source_bins, minlength=n_bins).astype(float)
    pair_weights = primary_stationary[source_bins] / pairs_per_bin[source_bins]
    swap = _swap_classifier_test(
        source_cv=records["source_cv"],
        destination_cv=records["destination_cv"],
        source_rg_nm=_radius_of_gyration(records["source_ca_nm"]),
        destination_rg_nm=_radius_of_gyration(records["destination_ca_nm"]),
        group_ids=records["group_id"],
        pair_weights=pair_weights,
        permutations=int(diagnostic.get("swap_permutations", 200)),
        seed=seed,
    )
    primary = sensitivity[str(n_bins)]
    metrics = {
        **primary,
        "sensitivity": sensitivity,
        "pair_swap_test": swap,
        "num_seed_frames": len(rows),
        "proposals_per_seed": proposals_per_seed,
        "balanced_seeds_per_primary_bin": diagnostic.get("balanced_seeds_per_bin"),
        "warning": (
            "Projected and pair-feature diagnostics can falsify, but cannot prove, "
            "full-state reversibility."
        ),
    }
    write_json(output_dir / "metrics.json", metrics)

    flux = primary_arrays["flux"]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(flux - flux.T, origin="lower", aspect="auto", cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="stationary flux asymmetry")
    ax.set_xlabel("destination PC1 bin")
    ax.set_ylabel("source PC1 bin")
    fig.tight_layout()
    fig.savefig(output_dir / "flux_asymmetry.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.asarray(sensitivity_bins)
    observed = [sensitivity[str(item)]["projected_detailed_balance_residual"] for item in bins]
    null_95 = [sensitivity[str(item)]["reversible_null_percentile_95"] for item in bins]
    ax.plot(bins, observed, marker="o", label="observed residual")
    ax.plot(bins, null_95, marker="s", label="reversible-null 95th percentile")
    ax.set_xlabel("number of PC1 bins")
    ax.set_ylabel("projected DB residual")
    ax.set_xticks(bins)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bin_sensitivity.png", dpi=180)
    plt.close(fig)
    return output_dir
