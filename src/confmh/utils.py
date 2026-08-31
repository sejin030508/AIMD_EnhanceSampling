from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

R_KJ_MOL_K = 0.00831446261815324


def kbt_kj_mol(temperature_k: float) -> float:
    return R_KJ_MOL_K * float(temperature_k)


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def add_repo_to_path(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    for candidate in (repo, repo / "src", repo.parent):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), str(repo), os.environ.get("PYTHONPATH", "")]
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_globs(patterns: str | Iterable[str], root: Path | None = None) -> list[Path]:
    import glob

    if isinstance(patterns, str):
        patterns = [patterns]
    results: list[Path] = []
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if root is not None and not path.is_absolute():
            path = root / path
        results.extend(Path(item).resolve() for item in glob.glob(str(path), recursive=True))
    return sorted(dict.fromkeys(results))


def safe_unlink_tree(path: Path) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path)
