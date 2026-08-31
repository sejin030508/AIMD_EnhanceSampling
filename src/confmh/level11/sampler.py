from __future__ import annotations

import time
from typing import Any

import numpy as np

from confmh.level11.config import GuidanceConfig
from confmh.level11.guidance import apply_translation_guidance, guidance_is_active


class GuidedEulerSampler:
    """Local drop-in replacement for ConfRover's Euler sampler.

    ConfRover's reverse ODE/SDE updates the scaled translation as
    ``x_(t-dt) = x_t - f_t dt + c g_t^2 score_t dt + noise``.  Adding
    ``-beta * grad(b)`` therefore moves the reverse update downhill in the
    clean-endpoint bias.  Treating a clean-endpoint gradient as a noisy-state
    score correction remains a heuristic approximation.
    """

    def __init__(self, base_sampler: Any, torch_pc1, potential, config: GuidanceConfig, beta: float):
        self.diffusion_steps = int(base_sampler.diffusion_steps)
        self.early_stop = int(base_sampler.early_stop)
        self.mode = str(base_sampler.mode)
        self.tmin = float(base_sampler.tmin)
        self.tmax = float(base_sampler.tmax)
        self.torch_pc1 = torch_pc1
        self.potential = potential
        self.config = config
        self.beta = float(beta)
        self.guidance_enabled = bool(config.enabled)
        self._records: list[dict[str, float]] = []
        self._guidance_walltime_s = 0.0
        self._last_clean_pc1 = float("nan")
        self._last_clean_bias = float("nan")

    def set_guidance(self, *, enabled: bool | None = None, clip_ratio: float | None = None) -> None:
        if enabled is not None:
            self.guidance_enabled = bool(enabled)
        if clip_ratio is not None:
            values = dict(self.config.__dict__)
            values["clip_ratio"] = float(clip_ratio)
            self.config = GuidanceConfig(**values)
            self.config.validate()

    def reset_diagnostics(self) -> None:
        self._records = []
        self._guidance_walltime_s = 0.0
        self._last_clean_pc1 = float("nan")
        self._last_clean_bias = float("nan")

    def diagnostics(self) -> dict[str, float | int | bool]:
        def aggregate(key: str, reducer, default: float = 0.0) -> float:
            values = [record[key] for record in self._records if np.isfinite(record[key])]
            return float(reducer(values)) if values else float(default)

        return {
            "guidance_enabled": self.guidance_enabled,
            "guidance_clip_ratio": float(self.config.clip_ratio),
            "guidance_active_fraction": float(self.config.active_fraction),
            "base_translation_score_rms_mean": aggregate("base_score_rms", np.mean),
            "base_translation_score_rms_max": aggregate("base_score_rms", np.max),
            "raw_guidance_rms_mean": aggregate("raw_guidance_rms", np.mean),
            "clipped_guidance_rms_mean": aggregate("clipped_guidance_rms", np.mean),
            "guidance_to_score_ratio_mean": aggregate("guidance_to_score_ratio", np.mean),
            "guidance_to_score_ratio_max": aggregate("guidance_to_score_ratio", np.max),
            "clip_scale_mean": aggregate("clip_scale", np.mean),
            "guided_step_count": len(self._records),
            "total_denoising_steps": int(self.diffusion_steps - self.early_stop),
            "guidance_walltime_s": float(self._guidance_walltime_s),
            "last_clean_pc1": float(self._last_clean_pc1),
            "last_clean_bias_kj_mol": float(self._last_clean_bias),
        }

    def _guided_score(self, base_score, pred_rigids_0, rigids_mask, se3_diffuser):
        import torch

        started = time.perf_counter()
        r3 = se3_diffuser._r3_diffuser
        coordinate_scaling = float(r3._r3_conf.coordinate_scaling)
        if not np.isfinite(coordinate_scaling) or coordinate_scaling <= 0:
            raise ValueError(f"Invalid ConfRover coordinate_scaling={coordinate_scaling}")

        # decoder.sample() is decorated with inference_mode.  Temporarily leave
        # it only for the detached clean endpoint; model parameters stay frozen.
        with torch.inference_mode(False), torch.enable_grad():
            clean_scaled = (
                pred_rigids_0.get_trans().detach().to(dtype=torch.float32).clone()
                * coordinate_scaling
            )
            clean_scaled.requires_grad_(True)
            clean_pc1 = self.torch_pc1.project(clean_scaled)
            clean_bias = self.potential.torch_energy(clean_pc1)
            bias_gradient = torch.autograd.grad(clean_bias.sum(), clean_scaled, create_graph=False)[0]
            guided_score, diag = apply_translation_guidance(
                base_score.to(dtype=torch.float32),
                bias_gradient,
                rigids_mask,
                beta=self.beta,
                eta=self.config.eta,
                clip_ratio=self.config.clip_ratio,
                clip_eps=self.config.clip_eps,
                remove_global_translation=self.config.remove_global_translation,
            )

        def scalar_mean(value) -> float:
            return float(value.detach().float().mean().cpu())

        self._records.append(
            {
                "base_score_rms": scalar_mean(diag.base_score_rms),
                "raw_guidance_rms": scalar_mean(diag.raw_guidance_rms),
                "clipped_guidance_rms": scalar_mean(diag.clipped_guidance_rms),
                "guidance_to_score_ratio": scalar_mean(diag.guidance_to_score_ratio),
                "clip_scale": scalar_mean(diag.clip_scale),
            }
        )
        self._last_clean_pc1 = scalar_mean(clean_pc1)
        self._last_clean_bias = scalar_mean(clean_bias)
        self._guidance_walltime_s += time.perf_counter() - started
        return guided_score.to(device=base_score.device, dtype=base_score.dtype)

    def reverse_sample(
        self,
        model_nn,
        se3_diffuser,
        rigids_t,
        s,
        z,
        **feats,
    ):
        import torch

        assert not model_nn.training
        rigids_mask = feats["rigids_mask"]
        aatype = feats["aatype"]
        batch_size = aatype.shape[0]
        dt = torch.full(
            (batch_size,),
            1.0 / self.diffusion_steps,
            device=aatype.device,
            dtype=s.dtype,
        )
        t = torch.ones((batch_size,), device=aatype.device, dtype=s.dtype)
        pred_atom14 = None
        pred_rigids_0 = None

        total_steps = self.diffusion_steps - self.early_stop
        for step_t in range(total_steps):
            output = model_nn(t=t, s=s, z=z, rigids_t=rigids_t, **feats)
            pred_rigids_0 = output["pred_rigids_0"]
            pred_atom14 = output["pred_atom14"]
            pred_rot_score = se3_diffuser.calc_rot_score(
                rigids_t.get_rots(),
                pred_rigids_0.get_rots(),
                t,
                use_cached_score=True,
            )
            pred_rot_score = pred_rot_score * rigids_mask[..., None]
            pred_trans_score = se3_diffuser.calc_trans_score(
                rigids_t.get_trans(),
                pred_rigids_0.get_trans(),
                t[:, None, None],
                use_torch=True,
            )
            pred_trans_score = pred_trans_score * rigids_mask[..., None]

            if self.guidance_enabled and guidance_is_active(
                step_t, total_steps, self.config.active_fraction
            ):
                pred_trans_score = self._guided_score(
                    pred_trans_score, pred_rigids_0, rigids_mask, se3_diffuser
                )

            rigids_t = se3_diffuser.reverse(
                rigids_t=rigids_t,
                rot_score=pred_rot_score,
                trans_score=pred_trans_score,
                t=float(t[0]),
                dt=float(dt[0]),
                mode=self.mode,
            )
            t -= dt
            if t.min() < self.tmin:
                break

        if pred_atom14 is None or pred_rigids_0 is None:
            raise RuntimeError("ConfRover reverse sampler completed no denoising steps")
        return pred_atom14, pred_rigids_0
