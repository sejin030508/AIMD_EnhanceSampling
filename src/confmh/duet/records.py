from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from confmh.duet.config import resolve_config_path, save_resolved_config


def _json_default(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "packages": {},
    }
    for distribution in (
        "numpy", "scipy", "scikit-learn", "PyYAML", "mdtraj", "matplotlib",
        "torch", "lightning", "transformers",
    ):
        try:
            metadata["packages"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            metadata["packages"][distribution] = None
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        metadata["torch"] = None
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_metadata(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Any]] = [("config", cfg.get("_config_path"))]
    for section, keys in (
        ("model", ("checkpoint", "interpolator_checkpoint")),
        ("trajectory", ("initial_structure", "initial_history")),
        ("program", ("catalog",)),
        ("reference", ("topology", "pca_model", "reference_cv", "held_out_trajectory", "case_manifest")),
    ):
        values = cfg.get(section, {})
        for key in keys:
            if values.get(key):
                candidates.append((f"{section}.{key}", values[key]))
    hash_limit = int(os.environ.get("DUET_HASH_MAX_BYTES", str(256 * 1024 * 1024)))
    result = []
    for label, value in candidates:
        path = resolve_config_path(cfg, value)
        row: dict[str, Any] = {"label": label, "path": str(path), "exists": path.exists()}
        if path.is_file():
            stat = path.stat()
            row.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            row["sha256"] = _sha256(path) if stat.st_size <= hash_limit else None
            if row["sha256"] is None:
                row["checksum_note"] = f"not hashed because size exceeds DUET_HASH_MAX_BYTES={hash_limit}"
        result.append(row)
    return result


def git_metadata(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return {"is_git_repository": False, "error": probe.stderr.strip()}
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short"], text=True, capture_output=True, check=False
    )
    diffstat = subprocess.run(
        ["git", "-C", str(root), "diff", "--stat"], text=True, capture_output=True, check=False
    )
    return {
        "is_git_repository": True,
        "root": probe.stdout.strip(),
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "status_short": status.stdout.splitlines(),
        "diff_stat": diffstat.stdout.splitlines(),
    }


def initialize_run_directory(
    output_dir: str | Path,
    cfg: dict[str, Any],
    *,
    command: str,
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not (overwrite or resume):
        raise FileExistsError(f"Run exists; pass --resume or --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output / "resolved_config.yaml")
    write_json(output / "environment.json", environment_metadata())
    write_json(output / "git_state.json", git_metadata(cfg.get("project_root", ".")))
    write_json(output / "assets.json", asset_metadata(cfg))
    (output / "command.txt").write_text(command.rstrip() + "\n", encoding="utf-8")
    for name in ("stdout.log", "stderr.log"):
        (output / name).touch(exist_ok=True)
    return output
