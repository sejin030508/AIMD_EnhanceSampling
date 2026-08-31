import pytest

torch = pytest.importorskip("torch")

from confmh.level11.guidance import apply_translation_guidance, guidance_is_active


def test_guidance_schedule_uses_only_final_quarter():
    active = [guidance_is_active(step, 200, 0.25) for step in range(200)]
    assert sum(active) == 50
    assert not active[149]
    assert active[150]


def test_zero_ratio_is_exactly_base_score():
    base = torch.randn(2, 5, 3)
    gradient = torch.randn_like(base)
    mask = torch.ones(2, 5)
    guided, diagnostics = apply_translation_guidance(
        base,
        gradient,
        mask,
        beta=0.4,
        eta=1.0,
        clip_ratio=0.0,
        clip_eps=1e-8,
        remove_global_translation=True,
    )
    assert torch.equal(base, guided)
    assert torch.equal(diagnostics.clipped_guidance_rms, torch.zeros(2))


def test_guidance_is_clipped_and_translation_centered():
    base = torch.ones(1, 6, 3)
    gradient = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.float32)
    guided, diagnostics = apply_translation_guidance(
        base,
        gradient,
        mask,
        beta=1.0,
        eta=1.0,
        clip_ratio=0.25,
        clip_eps=1e-8,
        remove_global_translation=True,
    )
    delta = guided - base
    assert diagnostics.guidance_to_score_ratio.item() <= 0.25 + 1e-6
    assert torch.allclose(delta[:, :4].sum(dim=1), torch.zeros(1, 3), atol=1e-6)
    assert torch.allclose(delta[:, 4:], torch.zeros(1, 2, 3))


def test_negative_bias_gradient_moves_downhill():
    base = torch.ones(1, 2, 3)
    coords = torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0]]])
    gradient = 2 * coords
    mask = torch.ones(1, 2)
    guided, _ = apply_translation_guidance(
        base,
        gradient,
        mask,
        beta=1.0,
        eta=1.0,
        clip_ratio=10.0,
        clip_eps=1e-8,
        remove_global_translation=True,
    )
    assert torch.sum((guided - base) * gradient) < 0
