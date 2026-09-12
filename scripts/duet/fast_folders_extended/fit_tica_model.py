#!/usr/bin/env python
"""Fit a TPS-DPS-compatible TICA model for a fast-folding protein.

The benchmark ships fitted ``tica_model.pkl`` objects but not the script that
produced them, so the settings here were read back off the published pickles
(``scripts/fast_folders_extended/inspect_tica_model.py``):

    dim = 2, scaling = "kinetic_map", epsilon = 1e-6

The lag time is the one setting that is genuinely per protein in the published
models (chignolin 500, trpcage 100, bba 100, bbl 1 frames), so it has to be
chosen here rather than copied.  ``--validate-against`` refits a protein that
already has a published model and reports how closely the refit reproduces it;
run that first, because a refit that cannot reproduce BBA cannot be trusted to
put a new protein on the same scale.

Features follow ``tps-dps/src/utils/utils.py`` exactly: a PyEMMA featurizer
built on ``folded.pdb`` with ``add_backbone_torsions(cossin=True)``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

PMF_BINS = 50  # xs.npy / ys.npy in the published bundles are length-50 grids.


def build_featurizer(folded_pdb: Path):
    from pyemma import coordinates as coor

    featurizer = coor.featurizer(str(folded_pdb))
    featurizer.add_backbone_torsions(cossin=True)
    return featurizer


def load_features(
    folded_pdb: Path,
    topology: Path,
    trajectories: list[Path],
    stride: int,
) -> list[np.ndarray]:
    """Featurise reference trajectories on the ``folded.pdb`` feature layout.

    The reference simulations carry their own topology, which need not be
    atom-for-atom identical to the benchmark's ``folded.pdb``.  Backbone
    torsions only need N/CA/C, so the two agree as long as the residue count
    does; that is asserted rather than assumed.
    """
    import mdtraj as md
    from pyemma import coordinates as coor

    reference = build_featurizer(folded_pdb)
    expected = reference.dimension()
    working = coor.featurizer(str(topology))
    working.add_backbone_torsions(cossin=True)
    if working.dimension() != expected:
        raise SystemExit(
            f"feature dimension mismatch: reference topology gives "
            f"{working.dimension()} but {folded_pdb.name} gives {expected}. "
            "Align the residue selection before fitting."
        )
    features = []
    for path in trajectories:
        trajectory = md.load(str(path), top=str(topology), stride=stride)
        features.append(np.asarray(working.transform(trajectory), dtype=np.float64))
    return features


def fit(features: list[np.ndarray], lag: int, dim: int = 2):
    from pyemma import coordinates as coor

    return coor.tica(
        features, lag=int(lag), dim=int(dim), kinetic_map=True, epsilon=1e-6
    )


def implied_timescale_scan(
    features: list[np.ndarray], lags: list[int], dim: int = 2
) -> list[dict]:
    """Report the slowest timescales per candidate lag.

    A lag is defensible when the recovered timescales have stopped drifting;
    picking one by eye from a single lag is what makes a refit arbitrary.
    """
    rows = []
    for lag in lags:
        try:
            model = fit(features, lag, dim)
            timescales = np.asarray(model.timescales)[:dim]
            rows.append(
                {
                    "lag": int(lag),
                    "timescales": np.round(timescales, 4).tolist(),
                    "eigenvalues": np.round(
                        np.asarray(model.eigenvalues)[:dim], 6
                    ).tolist(),
                }
            )
        except Exception as error:
            rows.append({"lag": int(lag), "error": repr(error)})
    return rows


def projected_pmf(projection: np.ndarray) -> dict[str, np.ndarray]:
    counts, x_edges, y_edges = np.histogram2d(
        projection[:, 0], projection[:, 1], bins=PMF_BINS
    )
    xs = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys = 0.5 * (y_edges[:-1] + y_edges[1:])
    density = counts / counts.sum()
    with np.errstate(divide="ignore"):
        free_energy = -np.log(np.where(density > 0, density, np.nan))
    free_energy -= np.nanmin(free_energy)
    return {"xs": xs, "ys": ys, "pmf": free_energy}


def compare_to_published(
    published: Path, refit, features: list[np.ndarray]
) -> dict:
    """Quantify agreement between a refit and a published TICA model.

    Sign and ordering of TICA components are arbitrary, so agreement is
    measured by the absolute correlation of each projected component.
    """
    reference = joblib.load(published)
    stacked = np.concatenate(features)
    ours = np.asarray(refit.transform(stacked))
    theirs = np.asarray(reference.transform(stacked))
    dimensions = min(ours.shape[1], theirs.shape[1])
    correlations = [
        float(abs(np.corrcoef(ours[:, index], theirs[:, index])[0, 1]))
        for index in range(dimensions)
    ]
    return {
        "published": str(published),
        "published_lag": int(getattr(reference, "_lagtime", -1)),
        "refit_lag": int(getattr(refit, "_lagtime", -1)),
        "per_component_abs_correlation": correlations,
        "frames_compared": int(len(stacked)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folded-pdb", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--trajectories", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lag", type=int, required=True)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--scan-lags",
        nargs="*",
        type=int,
        default=[],
        help="Optional implied-timescale scan used to justify --lag.",
    )
    parser.add_argument(
        "--validate-against",
        type=Path,
        help="A published tica_model.pkl to compare the refit against.",
    )
    args = parser.parse_args()

    features = load_features(
        args.folded_pdb, args.topology, args.trajectories, args.stride
    )
    report: dict = {
        "folded_pdb": str(args.folded_pdb),
        "topology": str(args.topology),
        "trajectory_count": len(features),
        "frames_per_trajectory": [int(len(item)) for item in features],
        "feature_dimension": int(features[0].shape[1]),
        "stride": int(args.stride),
        "settings": {
            "dim": int(args.dim),
            "lag": int(args.lag),
            "scaling": "kinetic_map",
            "epsilon": 1e-6,
        },
    }
    if args.scan_lags:
        report["implied_timescale_scan"] = implied_timescale_scan(
            features, args.scan_lags, args.dim
        )

    model = fit(features, args.lag, args.dim)
    projection = np.asarray(model.transform(np.concatenate(features)))
    grids = projected_pmf(projection)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "tica_model.pkl")
    for name, array in grids.items():
        np.save(args.output_dir / f"{name}.npy", array)
    report["timescales"] = np.round(
        np.asarray(model.timescales)[: args.dim], 4
    ).tolist()
    report["projection_bounds"] = {
        "tic1": [float(projection[:, 0].min()), float(projection[:, 0].max())],
        "tic2": [float(projection[:, 1].min()), float(projection[:, 1].max())],
    }
    if args.validate_against:
        report["validation"] = compare_to_published(
            args.validate_against, model, features
        )

    (args.output_dir / "tica_fit_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
