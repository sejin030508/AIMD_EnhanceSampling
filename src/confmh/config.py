from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    cleaned = {key: value for key, value in copy.deepcopy(cfg).items() if not key.startswith("_")}
    # Resolved configs are relocated under outputs/. Preserve the original project root
    # instead of carrying a relative path whose meaning changes at the new location.
    if "_config_dir" in cfg:
        cleaned["project_root"] = str(project_root(cfg))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cleaned, handle, sort_keys=False)


def project_root(cfg: dict[str, Any]) -> Path:
    root = cfg.get("project_root")
    if root:
        path = Path(root).expanduser()
        if not path.is_absolute():
            path = Path(cfg["_config_dir"]) / path
        return path.resolve()
    return Path(cfg["_config_dir"]).parent.resolve()


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root(cfg) / path).resolve()


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key '{key}' in {context}")
    return mapping[key]
