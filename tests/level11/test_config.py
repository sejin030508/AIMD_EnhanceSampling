import pytest

from confmh.level11.config import GuidanceConfig, level11_enabled, validate_level11_config


def test_missing_level11_defaults_to_disabled():
    assert not level11_enabled({})


def test_guidance_config_validation():
    GuidanceConfig.from_mapping({"clip_ratio": 0.0, "active_fraction": 0.25})
    with pytest.raises(ValueError, match="clip_ratio"):
        GuidanceConfig.from_mapping({"clip_ratio": -0.1})
    with pytest.raises(ValueError, match="translation"):
        GuidanceConfig.from_mapping({"degrees_of_freedom": "rotation"})


def test_proar_is_explicitly_rejected_for_level11():
    cfg = {
        "level11": {"enabled": True},
        "model": {"backend": "proar"},
    }
    with pytest.raises(ValueError, match="ConfRover-only"):
        validate_level11_config(cfg, require_paths=False)
