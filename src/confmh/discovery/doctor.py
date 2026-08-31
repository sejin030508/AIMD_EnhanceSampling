from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from confmh.config import resolve_path
from confmh.discovery.runner import _build_control_cv


def discovery_doctor(cfg: dict[str, Any], output_path: str | Path | None = None) -> Path:
    checks: dict[str, Any] = {}

    def path_check(name: str, path: Path) -> None:
        checks[name] = {"path": str(path), "exists": path.exists()}

    path_check("start_pdb", resolve_path(cfg, cfg["system"]["start_pdb"]))
    path_check("alternate_state_for_analysis", resolve_path(cfg, cfg["benchmark"]["target_pdb"]))
    backend = str(cfg["model"].get("backend", "confrover"))
    path_check("model_repo", resolve_path(cfg, cfg["model"]["repo"]))
    if backend == "proar":
        case_dir = resolve_path(cfg, cfg["model"]["input_data_dir"]) / cfg["system"]["case_id"]
        for filename in ("init.pdb", "esm_seq.npy", "esm_pair.npy"):
            path_check(f"proar_{filename}", case_dir / filename)
        for key in ("forecaster_checkpoint", "interpolator_checkpoint", "interpolator_config"):
            path_check(key, resolve_path(cfg, cfg["model"][key]))
    try:
        import torch

        checks["torch"] = {"available": True, "version": torch.__version__}
        checks["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except ImportError as exc:
        checks["torch"] = {"available": False, "error": repr(exc)}
        checks["cuda"] = {"available": False}
    cv = _build_control_cv(cfg)
    checks["control_cv"] = cv.describe()
    checks["target_blind_guard"] = {
        "passed": not (
            cfg["protocol"]["name"] == "target_blind" and checks["control_cv"]["target_loaded"]
        )
    }
    checks["budget"] = {
        "total_model_calls": int(cfg["run"]["total_model_calls"]),
        "candidates_per_step": int(cfg["run"]["candidates_per_step"]),
        "lanes": len(cfg["controller"]["centers"]),
    }
    required_path_checks = [value["exists"] for value in checks.values() if isinstance(value, dict) and "exists" in value]
    checks["ready"] = bool(
        all(required_path_checks)
        and checks["cuda"]["available"]
        and checks["target_blind_guard"]["passed"]
    )
    checks["mh_used"] = False
    if output_path is None:
        output_path = resolve_path(cfg, cfg["run"]["output_dir"]) / "doctor.json"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    return output_path
