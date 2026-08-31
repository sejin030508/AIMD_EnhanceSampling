from pathlib import Path

from confmh.config import load_config, project_root, resolve_path, save_config


def test_relative_project_root(tmp_path: Path):
    root = tmp_path / "project"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    path = config_dir / "x.yaml"
    path.write_text("project_root: ..\n")
    cfg = load_config(path)
    assert project_root(cfg) == root.resolve()
    assert resolve_path(cfg, "data/x") == (root / "data/x").resolve()

    relocated = root / "outputs" / "run" / "resolved_config.yaml"
    save_config(cfg, relocated)
    reloaded = load_config(relocated)
    assert project_root(reloaded) == root.resolve()
    assert resolve_path(reloaded, "data/x") == (root / "data/x").resolve()
