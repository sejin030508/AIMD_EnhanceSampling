from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from confmh.utils import add_repo_to_path


@dataclass
class Proposal:
    topology_pdb: Path
    trajectory_xtc: Path

    def load(self):
        import mdtraj as md

        return md.load(str(self.trajectory_xtc), top=str(self.topology_pdb))


class ConfRoverKernel:
    """Thin wrapper around the public ConfRover.generate() API."""

    proposal_horizon = 1

    def __init__(
        self,
        *,
        repo: str | Path,
        model_name: str,
        device: str,
        case_id: str,
        seqres: str,
        stride_in_10ps: int,
        diffusion_steps: int = 200,
        seed: int = 42,
        cache_dir: str | Path = "confrover_cache",
        ckpt_dir: str | Path | None = None,
        kv_cache_type: str = "offloaded",
        use_deepspeed_evo_attention: bool = False,
    ):
        repo = Path(repo).resolve()
        add_repo_to_path(repo)
        try:
            from confrover.model import ConfRover
        except ImportError as exc:
            raise ImportError(
                "ConfRover is not importable. Clone/install the baseline first; see README.md."
            ) from exc

        self.case_id = case_id
        self.seqres = seqres
        self.stride_in_10ps = int(stride_in_10ps)
        self.diffusion_steps = int(diffusion_steps)
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "seed": self.seed,
            "kv_cache_type": kv_cache_type,
            "use_deepspeed_evo_attention": use_deepspeed_evo_attention,
            "ckpt_dir": str(
                Path(ckpt_dir).expanduser().resolve()
                if ckpt_dir is not None
                else self.cache_dir / "confrover_ckpts"
            ),
        }
        self.model = ConfRover.from_pretrained(model_name, **kwargs)
        self.model.to(device)
        self.model.eval()

    def propose(
        self,
        *,
        condition_pdb: str | Path,
        output_dir: str | Path,
        step: int,
        n_replicates: int = 1,
    ) -> list[Proposal]:
        step_seed = self.seed + int(step) * 1009
        return self.generate_forward(
            condition_pdb=condition_pdb,
            output_dir=output_dir,
            n_frames=2,
            n_replicates=n_replicates,
            seed=step_seed,
        )

    def generate_forward(
        self,
        *,
        condition_pdb: str | Path,
        output_dir: str | Path,
        n_frames: int,
        n_replicates: int = 1,
        seed: int | None = None,
    ) -> list[Proposal]:
        """Generate a full-history forward trajectory in one ConfRover call."""
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if int(n_frames) < 2:
            raise ValueError("A forward trajectory needs at least two frames")
        self.model.generate(
            case_id=self.case_id,
            seqres=self.seqres,
            task_mode="forward",
            output_dir=str(output_dir),
            n_replicates=int(n_replicates),
            n_frames=int(n_frames),
            stride_in_10ps=self.stride_in_10ps,
            conditions=str(Path(condition_pdb).resolve()),
            cache_dir=str(self.cache_dir),
            msa_root=str(self.cache_dir / "msa"),
            folding_repr=str(self.cache_dir / "folding_repr"),
            seed=self.seed if seed is None else int(seed),
            diffusion_steps=self.diffusion_steps,
        )
        case_dir = output_dir / self.case_id
        proposals: list[Proposal] = []
        for replicate in range(int(n_replicates)):
            prefix = case_dir / f"{self.case_id}_sample{replicate}"
            pdb_path = prefix.with_suffix(".pdb")
            xtc_path = prefix.with_suffix(".xtc")
            if not pdb_path.exists() or not xtc_path.exists():
                raise FileNotFoundError(
                    f"Expected ConfRover output {pdb_path} and {xtc_path}; inspect {output_dir}"
                )
            proposals.append(Proposal(pdb_path, xtc_path))
        return proposals
