import copy

import pytest

from confmh.duet.config import guidance_update_steps, validate_duet_config


def _config():
    return {
        "model": {
            "backend": "pvb", "sampler_mode": "sde", "reverse_steps": 20,
            "sde_step": 20, "decoder_microbatch_size": 1,
        },
        "trajectory": {"horizon": 4},
        "particles": {
            "outer_k": 2, "inner_m": 2,
            "inner_checkpoint_progresses": [0.5, 0.75],
            "outer_resampling": "systematic",
            "outer_resampling_ess_fraction": 0.5,
        },
        "program": {
            "type": "terminal", "potential": "tica",
            "potential_floor": 1.0e-300, "tica": {"dimensions": 2},
        },
        "experiment": {"methods": ["guided_duet"]},
        "guidance": {
            "enabled": True, "strength": 0.25,
            "schedule": "all_stochastic", "inner_resampling": True,
        },
    }


def test_guided_config_accepts_all_stochastic_schedule():
    cfg = _config()
    validate_duet_config(cfg)
    assert guidance_update_steps(cfg) is None


def test_guided_config_accepts_explicit_stochastic_steps():
    cfg = _config()
    cfg["guidance"].update({"schedule": "explicit", "update_steps": [0, 4, 18]})
    validate_duet_config(cfg)
    assert guidance_update_steps(cfg) == (0, 4, 18)


@pytest.mark.parametrize(
    "change",
    [
        ("model", "backend", "confrover"),
        ("program", "potential", "rmsd"),
        ("guidance", "strength", -0.1),
        ("guidance", "schedule", "unknown"),
    ],
)
def test_guided_config_rejects_invalid_contract(change):
    cfg = copy.deepcopy(_config())
    section, key, value = change
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_duet_config(cfg)
