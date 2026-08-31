from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuidanceStepDiagnostics:
    base_score_rms: object
    raw_guidance_rms: object
    clipped_guidance_rms: object
    guidance_to_score_ratio: object
    clip_scale: object


def guidance_is_active(step: int, total_steps: int, active_fraction: float) -> bool:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 < active_fraction <= 1:
        raise ValueError("active_fraction must be in (0, 1]")
    active_steps = max(1, int(round(total_steps * active_fraction)))
    return int(step) >= total_steps - active_steps


def _masked_rms(value, mask, eps: float):
    import torch

    weights = mask.to(dtype=value.dtype).unsqueeze(-1)
    numerator = (value.square() * weights).sum(dim=(-2, -1))
    denominator = (weights.sum(dim=(-2, -1)) * value.shape[-1]).clamp_min(eps)
    return torch.sqrt(numerator / denominator)


def apply_translation_guidance(
    base_score,
    bias_gradient,
    mask,
    *,
    beta: float,
    eta: float,
    clip_ratio: float,
    clip_eps: float,
    remove_global_translation: bool,
):
    """Add clipped ``-eta * beta * grad(bias)`` to a translation score."""
    import torch

    if base_score.shape != bias_gradient.shape:
        raise ValueError("base_score and bias_gradient shapes must match")
    if mask.shape != base_score.shape[:-1]:
        raise ValueError("mask shape must match score residue dimensions")
    weights = mask.to(dtype=base_score.dtype).unsqueeze(-1)
    raw = (-float(eta) * float(beta) * bias_gradient.to(base_score.dtype)) * weights
    if remove_global_translation:
        count = weights.sum(dim=-2, keepdim=True).clamp_min(1.0)
        raw = (raw - (raw * weights).sum(dim=-2, keepdim=True) / count) * weights

    base_rms = _masked_rms(base_score, mask, clip_eps)
    raw_rms = _masked_rms(raw, mask, clip_eps)
    allowed = float(clip_ratio) * base_rms
    scale = torch.minimum(
        torch.ones_like(raw_rms),
        allowed / raw_rms.clamp_min(float(clip_eps)),
    )
    if clip_ratio == 0 or eta == 0:
        scale = torch.zeros_like(scale)
    clipped = raw * scale[..., None, None]
    clipped_rms = _masked_rms(clipped, mask, clip_eps)
    ratio = clipped_rms / base_rms.clamp_min(float(clip_eps))
    diagnostics = GuidanceStepDiagnostics(
        base_score_rms=base_rms.detach(),
        raw_guidance_rms=raw_rms.detach(),
        clipped_guidance_rms=clipped_rms.detach(),
        guidance_to_score_ratio=ratio.detach(),
        clip_scale=scale.detach(),
    )
    return base_score + clipped, diagnostics
