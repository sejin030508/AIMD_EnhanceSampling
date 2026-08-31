from __future__ import annotations

import importlib.util
import sys
from typing import Any

from confmh.config import resolve_path
from confmh.utils import expand_globs


def doctor(cfg: dict[str, Any]) -> int:
    failures = 0

    def check(condition: bool, message: str) -> None:
        nonlocal failures
        print(f"[{'OK' if condition else 'FAIL'}] {message}")
        failures += int(not condition)

    for package in ("numpy", "scipy", "sklearn", "yaml", "mdtraj", "matplotlib"):
        check(importlib.util.find_spec(package) is not None, f"Python package: {package}")

    model = cfg.get("model", {})
    backend = str(model.get("backend", "confrover")).lower()
    if backend == "confrover":
        repo = resolve_path(cfg, model.get("repo", "external/ConfRover"))
        check((repo / "src" / "confrover").exists(), f"ConfRover repository: {repo}")
        check(importlib.util.find_spec("confrover") is not None, "Installed/importable confrover")
    elif backend == "proar":
        repo = resolve_path(cfg, model.get("repo", "external/ProAR"))
        check((repo / "run.py").exists(), f"ProAR repository: {repo}")
        for key in (
            "forecaster_checkpoint",
            "interpolator_checkpoint",
            "interpolator_config",
        ):
            check(resolve_path(cfg, model[key]).exists(), f"ProAR asset: {key}")
        input_case = resolve_path(cfg, model.get("input_data_dir", "data/proar")) / str(
            cfg["system"]["case_id"]
        )
        for filename in ("init.pdb", "esm_seq.npy", "esm_pair.npy"):
            check((input_case / filename).exists(), f"ProAR input: {filename}")
        sys.path.insert(0, str(repo))
        for package in ("torch", "hydra", "omegaconf", "openfold", "esm"):
            check(importlib.util.find_spec(package) is not None, f"ProAR package: {package}")
    else:
        check(False, f"Supported model.backend: {backend}")

    start_pdb = resolve_path(cfg, cfg["system"]["start_pdb"])
    check(start_pdb.exists(), f"Starting PDB: {start_pdb}")

    reference = cfg.get("reference", {})
    if "pca_model" in reference:
        check(resolve_path(cfg, reference["pca_model"]).exists(), "Fitted PCA model")
    if "reference_cv" in reference:
        check(resolve_path(cfg, reference["reference_cv"]).exists(), "Reference CV oracle")
    if "topology_pdb" in reference:
        topology = resolve_path(cfg, reference["topology_pdb"])
        check(topology.exists(), f"ATLAS topology: {topology}")
    if "trajectory_globs" in reference:
        paths = expand_globs(reference["trajectory_globs"], root=resolve_path(cfg, "."))
        check(len(paths) > 0, f"ATLAS trajectories matched: {len(paths)}")

    print(f"Doctor completed with {failures} failure(s).")
    return 1 if failures else 0
