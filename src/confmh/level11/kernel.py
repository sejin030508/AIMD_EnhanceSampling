from __future__ import annotations

from pathlib import Path

from confmh.kernel import ConfRoverKernel
from confmh.level11.config import GuidanceConfig
from confmh.level11.sampler import GuidedEulerSampler
from confmh.level11.torch_pc1 import TorchPC1


class GuidedConfRoverKernel(ConfRoverKernel):
    """ConfRover one-step proposal with an opt-in local guided sampler."""

    def __init__(
        self,
        *,
        pca_model: str | Path,
        potential,
        guidance: GuidanceConfig,
        beta: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        decoder = self.model.decoder
        base_sampler = decoder.sampler
        if base_sampler is None:
            from confrover.model.decoder.confdiff.sampler.euler import EulerSampler

            base_sampler = EulerSampler(diffusion_steps=self.diffusion_steps)
        self._decoder = decoder
        self.base_sampler = base_sampler
        self.guided_sampler = GuidedEulerSampler(
            base_sampler,
            TorchPC1.load(pca_model),
            potential,
            guidance,
            beta,
        )
        decoder.sampler = self.guided_sampler
        self.last_diagnostics: dict = {}

    def set_guidance(self, *, enabled: bool | None = None, clip_ratio: float | None = None) -> None:
        self.guided_sampler.set_guidance(enabled=enabled, clip_ratio=clip_ratio)

    def propose(self, **kwargs):
        self.guided_sampler.reset_diagnostics()
        proposals = super().propose(**kwargs)
        self.last_diagnostics = self.guided_sampler.diagnostics()
        return proposals

    def propose_seeded(
        self,
        *,
        condition_pdb: str | Path,
        output_dir: str | Path,
        proposal_seed: int,
        n_replicates: int = 1,
    ):
        """Generate a proposal with an explicit seed for paired experiments."""
        self.guided_sampler.reset_diagnostics()
        proposals = self.generate_forward(
            condition_pdb=condition_pdb,
            output_dir=output_dir,
            n_frames=2,
            n_replicates=n_replicates,
            seed=int(proposal_seed),
        )
        self.last_diagnostics = self.guided_sampler.diagnostics()
        return proposals

    def propose_upstream_seeded(
        self,
        *,
        condition_pdb: str | Path,
        output_dir: str | Path,
        proposal_seed: int,
        n_replicates: int = 1,
    ):
        """Use the untouched upstream sampler for zero-equivalence checks."""
        self._decoder.sampler = self.base_sampler
        try:
            return self.generate_forward(
                condition_pdb=condition_pdb,
                output_dir=output_dir,
                n_frames=2,
                n_replicates=n_replicates,
                seed=int(proposal_seed),
            )
        finally:
            self._decoder.sampler = self.guided_sampler
