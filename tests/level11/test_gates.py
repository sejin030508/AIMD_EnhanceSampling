import json
from pathlib import Path

from confmh.level11.analysis import check_pilot_gate, evaluate_stage2_gate


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pilot_gate_accepts_matching_selected_ratio(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("level11:\n  guidance:\n    clip_ratio: 0.25\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    _write(summary, {"selection": {"passing_ratios": [0.25], "selected_clip_ratio": 0.25}})
    assert check_pilot_gate(config, summary) == summary.resolve()


def test_stage2_gate_with_synthetic_passing_metrics(tmp_path: Path):
    root = tmp_path / "standard"
    for index in range(5):
        _write(
            root / "comparison" / f"window_{index:02d}" / "comparison.json",
            {
                "ratios": {"ess_per_wallclock": 1.3, "pc1_esjd_per_proposal": 1.1},
                "changes": {"geometry_failure_rate": 0.0},
            },
        )
    _write(
        root / "unsteered" / "combined" / "metrics.json",
        {"wham_jsd": 0.10, "wham_fes_rmse_kj_mol": 3.0},
    )
    _write(
        root / "guided" / "combined" / "metrics.json",
        {"wham_jsd": 0.11, "wham_fes_rmse_kj_mol": 3.2},
    )
    assert evaluate_stage2_gate(root) == (root / "stage2_gate.json").resolve()
