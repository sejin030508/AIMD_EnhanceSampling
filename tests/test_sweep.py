from pathlib import Path

import numpy as np

from confmh.config import load_config
from confmh.sweep import make_umbrella_configs


def test_umbrella_sweep_preserves_template_output_suite(tmp_path: Path):
    root = tmp_path / "project"
    configs = root / "configs"
    reference = root / "data" / "reference"
    configs.mkdir(parents=True)
    reference.mkdir(parents=True)
    np.savez_compressed(reference / "reference_cv.npz", cv=np.arange(20)[:, None])
    template = configs / "template.yaml"
    template.write_text(
        "project_root: ..\n"
        "reference:\n"
        "  reference_cv: data/reference/reference_cv.npz\n"
        "experiment:\n"
        "  mode: umbrella\n"
        "run:\n"
        "  seed: 10\n"
        "  output_dir: outputs/umbrella_350/template\n",
        encoding="utf-8",
    )

    paths = make_umbrella_configs(template, configs / "generated")

    assert len(paths) == 5
    first = load_config(paths[0])
    last = load_config(paths[-1])
    assert first["run"]["output_dir"] == "outputs/umbrella_350/window_00"
    assert last["run"]["output_dir"] == "outputs/umbrella_350/window_04"
    assert first["run"]["seed"] == 10
    assert last["run"]["seed"] == 14
