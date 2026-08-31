import numpy as np

from confmh.adapters.mock import MockIterativeFrameAdapter
from confmh.duet.config import estimate_decoder_nfe


def test_decoder_and_frame_counts_are_exact():
    adapter = MockIterativeFrameAdapter(reverse_steps=8, residues=4)
    history = [np.zeros((4, 3))]
    state = adapter.prepare_history(history)
    frames = adapter.sample_complete_frames(state, 5, range(5), checkpoint_progress=0.75)
    assert len(frames) == 5
    assert adapter.accounting.reverse_decoder_evaluations == 5 * 8
    assert adapter.accounting.temporal_encoder_evaluations == 1
    assert adapter.accounting.generated_complete_frames == 5


def test_method_specific_allocations_match_decoder_budget():
    cfg = {
        "model": {"reverse_steps": 200},
        "trajectory": {"horizon": 8},
        "particles": {"outer_k": 4, "inner_m": 4},
        "experiment": {
            "seeds": [1, 2, 3],
            "task_count": 2,
            "decoder_population_budget": 16,
            "method_settings": {
                "outer_only": {"outer_k": 16, "inner_m": 1},
                "inner_only": {"outer_k": 1, "inner_m": 16},
                "duet": {"outer_k": 4, "inner_m": 4},
            },
        },
    }
    expected = 8 * 200 * 16 * 3 * 2
    assert estimate_decoder_nfe(cfg, "outer_only") == expected
    assert estimate_decoder_nfe(cfg, "inner_only") == expected
    assert estimate_decoder_nfe(cfg, "duet") == expected
