from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from confmh.kernel import Proposal
from confmh.utils import add_repo_to_path, seed_everything


class ProARKernel:
    """In-process adapter for the official ProAR inference implementation.

    ProAR emits six 400 ps frames per forecasting block.  The steering kernel
    uses the block endpoint, so one proposal represents a nominal 2.4 ns model
    interval.  ``generate_forward`` retains all 400 ps intermediate frames.
    """

    stride_in_10ps = 40

    def __init__(
        self,
        *,
        repo: str | Path,
        device: str,
        case_id: str,
        input_data_dir: str | Path,
        forecaster_checkpoint: str | Path,
        interpolator_checkpoint: str | Path,
        interpolator_config: str | Path,
        cache_dir: str | Path,
        horizon: int = 6,
        sampling_type: str = "naive",
        refine_intermediate_predictions: bool = True,
        seed: int = 42,
    ):
        self.repo = Path(repo).expanduser().resolve()
        self.device_name = str(device)
        self.case_id = str(case_id)
        self.input_data_dir = Path(input_data_dir).expanduser().resolve()
        self.forecaster_checkpoint = Path(forecaster_checkpoint).expanduser().resolve()
        self.interpolator_checkpoint = Path(interpolator_checkpoint).expanduser().resolve()
        self.interpolator_config = Path(interpolator_config).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.horizon = int(horizon)
        self.proposal_horizon = self.horizon
        self.sampling_type = str(sampling_type)
        self.refine_intermediate_predictions = bool(refine_intermediate_predictions)
        self.seed = int(seed)

        if self.horizon < 2:
            raise ValueError("ProAR horizon must be at least 2")
        if self.sampling_type not in {"naive", "cold"}:
            raise ValueError("ProAR sampling_type must be 'naive' or 'cold'")
        required = [
            self.repo / "run.py",
            self.repo / "src" / "configs" / "main_config.yaml",
            self.input_data_dir / self.case_id / "init.pdb",
            self.input_data_dir / self.case_id / "esm_seq.npy",
            self.input_data_dir / self.case_id / "esm_pair.npy",
            self.forecaster_checkpoint,
            self.interpolator_checkpoint,
            self.interpolator_config,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing ProAR asset(s):\n" + "\n".join(missing))

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_root = self.cache_dir / f"runtime_{os.getpid()}"
        runtime_case = self.runtime_root / self.case_id
        runtime_case.mkdir(parents=True, exist_ok=True)
        for filename in ("esm_seq.npy", "esm_pair.npy"):
            source = self.input_data_dir / self.case_id / filename
            target = runtime_case / filename
            if not target.exists():
                try:
                    target.symlink_to(source)
                except OSError:
                    shutil.copy2(source, target)
        self.runtime_pdb = runtime_case / "init.pdb"
        shutil.copy2(self.input_data_dir / self.case_id / "init.pdb", self.runtime_pdb)

        add_repo_to_path(self.repo)
        self._initialize_model()

    def _initialize_model(self) -> None:
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf, open_dict
        from src.interface import get_lightning_module

        config_dir = self.repo / "src" / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="main_config.yaml", overrides=["experiment=atlas"])
        with open_dict(cfg):
            cfg.datamodule.data_dir = str(self.runtime_root)
            cfg.datamodule.horizon = self.horizon
            cfg.datamodule.prediction_horizon = self.horizon
            cfg.module.autoregressive_steps = 0
            cfg.module.num_predictions = 1
            cfg.module.verbose = False
            cfg.module.save_dir = str(self.cache_dir / "pdb")
            cfg.diffusion.sampling_type = self.sampling_type
            cfg.diffusion.refine_intermediate_predictions = self.refine_intermediate_predictions
            cfg.diffusion.interpolator_local_checkpoint_path = str(
                self.interpolator_checkpoint
            )
            cfg.diffusion.hydra_local_config_path = str(self.interpolator_config)
            cfg.ckpt_path = str(self.forecaster_checkpoint)
            cfg.work_dir = str(self.cache_dir / "work")
            cfg.verbose = False
            cfg.seed = self.seed
        self.config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

        seed_everything(self.seed)
        model = get_lightning_module(self.config)
        model = model.__class__.load_from_checkpoint(
            str(self.forecaster_checkpoint), **model.hparams
        )
        self.device = torch.device(self.device_name)
        self.model = model.to(self.device).eval()

    def _move_to_device(self, value: Any):
        import torch

        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move_to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        return value

    def _load_batch(self):
        from src.datamodules.datasets.mdtrajectory import InferenceDataset
        from src.datamodules.molecular_dynamics_simulation import length_batching, shape_schema

        dataset = InferenceDataset(str(self.runtime_root))
        if len(dataset) != 1:
            raise RuntimeError(
                f"Expected one ProAR runtime case in {self.runtime_root}, found {len(dataset)}"
            )
        item = dataset[0]
        batch = length_batching(
            [item],
            shape_schema={key: [None] + value for key, value in shape_schema.items()},
            use_length_batching=False,
        )
        return self._move_to_device(batch)

    @staticmethod
    def _strip_singletons(value: Any, ndim: int) -> np.ndarray:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
        except ImportError:
            pass
        array = np.asarray(value)
        while array.ndim > ndim and array.shape[0] == 1:
            array = array[0]
        if array.ndim != ndim:
            raise RuntimeError(f"Unexpected ProAR array shape {array.shape}; expected rank {ndim}")
        return array

    def _atom_mask(self, features: dict[str, Any]) -> np.ndarray:
        mask_key = (
            "atom37_atom_exists"
            if "atom37_atom_exists" in features
            else "all_atom_mask"
        )
        return self._strip_singletons(features[mask_key], 2)

    def _write_pdb(
        self,
        features: dict[str, Any],
        path: Path,
        *,
        atom_mask_override: np.ndarray | None = None,
    ) -> None:
        from openfold.np.protein import Protein, to_pdb

        positions = self._strip_singletons(features["all_atom_positions"], 3)
        atom_mask = self._atom_mask(features)
        if atom_mask_override is not None:
            if atom_mask_override.shape != atom_mask.shape:
                raise RuntimeError(
                    "ProAR atom-mask shape mismatch: "
                    f"{atom_mask_override.shape} != {atom_mask.shape}"
                )
            atom_mask = atom_mask * atom_mask_override
        aatype = self._strip_singletons(features["aatype"], 1).astype(np.int64)
        residue_index = self._strip_singletons(features["residue_index"], 1).astype(np.int64)
        protein = Protein(
            aatype=aatype,
            atom_positions=positions,
            atom_mask=atom_mask,
            residue_index=residue_index + 1,
            b_factors=np.zeros_like(atom_mask),
            chain_index=np.zeros_like(aatype),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_pdb(protein), encoding="utf-8")

    def _sample_steps(self, *, condition_pdb: Path, n_steps: int, seed: int) -> tuple[Any, dict]:
        import torch

        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        shutil.copy2(condition_pdb, self.runtime_pdb)
        seed_everything(seed)
        batch = self._load_batch()
        self.model.hparams.autoregressive_steps = 0
        self.model.datamodule_config.prediction_horizon = int(n_steps)
        with torch.inference_mode():
            results = self.model._evaluation_step(
                batch,
                batch_idx=0,
                split="predict",
                return_outputs=True,
            )
        return batch, results

    def _materialize_proposal(
        self,
        *,
        condition_pdb: Path,
        output_dir: Path,
        prefix: str,
        n_steps: int,
        seed: int,
    ) -> Proposal:
        import mdtraj as md

        batch, results = self._sample_steps(
            condition_pdb=condition_pdb,
            n_steps=n_steps,
            seed=seed,
        )
        frame_dir = output_dir / f"{prefix}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        condition_features = {
            "all_atom_positions": batch["all_atom_positions"],
            "all_atom_mask": batch["all_atom_mask"],
            "aatype": batch["aatype"],
            "residue_index": batch["residue_index"],
        }
        frame_features = [condition_features]
        for index in range(1, n_steps + 1):
            key = f"t{index}_preds"
            if key not in results:
                raise RuntimeError(f"ProAR did not return {key}; keys={sorted(results)}")
            frame_features.append(results[key])

        common_atom_mask = np.logical_and.reduce(
            [self._atom_mask(features) > 0.5 for features in frame_features]
        ).astype(np.float32)
        pdb_paths = []
        for index, features in enumerate(frame_features):
            path = frame_dir / f"frame_{index:06d}.pdb"
            self._write_pdb(
                features,
                path,
                atom_mask_override=common_atom_mask,
            )
            pdb_paths.append(path)

        trajectory = md.load([str(path) for path in pdb_paths])
        topology_pdb = output_dir / f"{prefix}.pdb"
        trajectory[0].save_pdb(str(topology_pdb))
        trajectory_xtc = output_dir / f"{prefix}.xtc"
        trajectory.save_xtc(str(trajectory_xtc))
        return Proposal(topology_pdb=topology_pdb, trajectory_xtc=trajectory_xtc)

    def propose(
        self,
        *,
        condition_pdb: str | Path,
        output_dir: str | Path,
        step: int,
        n_replicates: int = 1,
    ) -> list[Proposal]:
        return self.generate_forward(
            condition_pdb=condition_pdb,
            output_dir=output_dir,
            n_frames=self.horizon + 1,
            n_replicates=n_replicates,
            seed=self.seed + int(step) * 1009,
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
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        condition_pdb = Path(condition_pdb).expanduser().resolve()
        n_steps = int(n_frames) - 1
        base_seed = self.seed if seed is None else int(seed)
        proposals = []
        for replicate in range(int(n_replicates)):
            proposals.append(
                self._materialize_proposal(
                    condition_pdb=condition_pdb,
                    output_dir=output_dir,
                    prefix=f"{self.case_id}_sample{replicate}",
                    n_steps=n_steps,
                    seed=base_seed + replicate,
                )
            )
        return proposals
