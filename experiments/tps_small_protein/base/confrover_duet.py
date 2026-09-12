from __future__ import annotations

"""ConfRover adapter for full-history DuET-MD sampling.

The reverse loop mirrors
``confrover/model/decoder/confdiff/sampler/euler.py`` at upstream commit
``30af5c3bfaee8497f8dfdf7f1c16097e843a0246``.  It is kept local so the
vendor checkout remains unmodified and so checkpoint state can be returned,
resampled, and continued with independent SDE noise.
"""

from dataclasses import dataclass
from contextlib import contextmanager
import gc
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from confmh.adapters.base_iterative_frame import IterativeFrameAdapter
from confmh.duet.resampling import gather_state
from confmh.utils import add_repo_to_path


@contextmanager
def seeded_numpy(seed: int):
    """Temporarily seed legacy NumPy RNG calls made inside ConfRover.

    ConfRover's SE(3) prior and reverse-SDE implementation use the module-level
    ``np.random`` API.  Torch-only seeding therefore does not control particle
    continuations.  Saving and restoring the global state keeps each serial
    particle/step stream deterministic without mutating the vendor checkout.
    """

    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        np.random.set_state(state)


@dataclass
class ConfRoverFrame:
    atom37_a: np.ndarray
    atom37_mask: np.ndarray
    aatype: np.ndarray

    @property
    def ca_nm(self) -> np.ndarray:
        # OpenFold atom37 index 1 is CA; ConfRover coordinates are in Angstrom.
        return np.asarray(self.atom37_a[:, 1, :], dtype=float) / 10.0


@dataclass
class ConfRoverHistoryState:
    s: Any
    z: Any
    aatype: Any
    padding_mask: Any
    pretrained_single: Any
    pretrained_pair: Any
    frame_count: int


@dataclass
class ConfRoverParticleState:
    rigids_t7: Any
    s: Any
    z: Any
    aatype: Any
    padding_mask: Any
    rigids_mask: Any
    pretrained_single: Any
    pretrained_pair: Any
    noise_seeds: np.ndarray
    conditioning_ids: np.ndarray
    ancestor_metadata: np.ndarray
    step: int
    time_value: float
    cached_rot_score: Any = None
    cached_trans_score: Any = None
    cached_pred_atom14: Any = None
    cached_pred_rigids_0_7: Any = None
    final_pred_atom14: Any = None
    final_pred_rigids_0_7: Any = None


class ConfRoverDuETAdapter(IterativeFrameAdapter):
    def __init__(
        self,
        *,
        repository_path: str | Path,
        checkpoint: str | Path,
        initial_structure: str | Path,
        case_id: str,
        seqres: str,
        cache_dir: str | Path,
        device: str = "cuda:0",
        dtype: str = "float32",
        stride_in_10ps: int = 256,
        reverse_steps: int = 200,
        sampler_mode: str = "sde",
        kv_cache_type: str = "offloaded",
        decoder_microbatch_size: int | None = None,
        pairformer_chunk_size: int | None = None,
        tmin: float = 0.01,
        ca_adjacent_quality_threshold_a: float = 4.5,
        ca_adjacent_hard_threshold_a: float = 5.5,
        ca_adjacent_hard_tolerance_a: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if sampler_mode not in {"ode", "sde"}:
            raise ValueError("ConfRover sampler_mode must be ode or sde")
        self.repository_path = Path(repository_path).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.initial_structure = Path(initial_structure).expanduser().resolve()
        self.case_id = str(case_id)
        self.seqres = str(seqres)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.device_name = str(device)
        self.dtype_name = str(dtype)
        self.stride_in_10ps = int(stride_in_10ps)
        self.reverse_steps = int(reverse_steps)
        self.sampler_mode = str(sampler_mode)
        self.kv_cache_type = str(kv_cache_type)
        self.decoder_microbatch_size = (
            None
            if decoder_microbatch_size is None
            else int(decoder_microbatch_size)
        )
        if (
            self.decoder_microbatch_size is not None
            and self.decoder_microbatch_size < 1
        ):
            raise ValueError("decoder_microbatch_size must be positive")
        self.pairformer_chunk_size = (
            None if pairformer_chunk_size is None else int(pairformer_chunk_size)
        )
        if self.pairformer_chunk_size is not None and self.pairformer_chunk_size < 1:
            raise ValueError("pairformer_chunk_size must be positive")
        self.tmin = float(tmin)
        self.ca_adjacent_quality_threshold_a = float(
            ca_adjacent_quality_threshold_a
        )
        self.ca_adjacent_hard_threshold_a = float(ca_adjacent_hard_threshold_a)
        self.ca_adjacent_hard_tolerance_a = float(ca_adjacent_hard_tolerance_a)
        self.model: Any = None
        self.dataset: Any = None
        self.static_batch: dict[str, Any] | None = None
        self._initial_frame: ConfRoverFrame | None = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        if not self.repository_path.exists():
            raise FileNotFoundError(self.repository_path)
        if not self.checkpoint.exists():
            raise FileNotFoundError(self.checkpoint)
        if not self.initial_structure.exists():
            raise FileNotFoundError(self.initial_structure)
        add_repo_to_path(self.repository_path)
        import torch
        from lightning.pytorch.utilities import move_data_to_device
        from confrover.data.infer import GenCaseConfig, GenDataset, GenDatasetConfig, PDBConditions
        from confrover.data.pretrain_repr import OpenFoldReprLoader
        from confrover.model import ConfRover

        self.model = ConfRover.from_pretrained(
            str(self.checkpoint),
            seed=0,
            kv_cache_type=self.kv_cache_type,
            use_deepspeed_evo_attention=False,
        )
        if self.pairformer_chunk_size is not None:
            # Exact inference-time sub-batching for the O(L^3) triangular
            # attention workspace.  This does not alter model weights, the
            # sampler, rewards, populations, checkpoints, or physical horizon.
            self.model.temporal.pairformer_config.chunk_size = (
                self.pairformer_chunk_size
            )
            if hasattr(self.model.encoder, "chunk_size"):
                self.model.encoder.chunk_size = self.pairformer_chunk_size
        self.model.to(self.device_name)
        self.model.eval()
        conditions = PDBConditions.from_list(str(self.initial_structure), "forward")
        repr_loader = OpenFoldReprLoader(repr_root=self.cache_dir / "folding_repr")
        # Required representations already exist in the shared Level-1 cache.
        # A missing entry is reported rather than silently downloading/recomputing it.
        repr_loader.load(seqres=self.seqres)
        case = GenCaseConfig(
            case_id=self.case_id,
            seqres=self.seqres,
            seqlen=len(self.seqres),
            task_mode="forward",
            n_replicates=1,
            rep_id=0,
            n_frames=1,
            stride_in_10ps=self.stride_in_10ps,
            conditions=conditions,
        )
        self.dataset = GenDataset(
            GenDatasetConfig(
                name="duet_adapter",
                cases=[case],
                task_mode="forward",
                n_replicates=1,
                n_frames=1,
                stride_in_10ps=self.stride_in_10ps,
            ),
            repr_loader=repr_loader,
        )
        self.static_batch = move_data_to_device(
            self.dataset.collate([self.dataset[0]]), self.device_name
        )
        coords = conditions.load_coords(seqlen=len(self.seqres))[0]
        mask = np.all(np.isfinite(coords), axis=-1)
        self._initial_frame = ConfRoverFrame(
            atom37_a=np.nan_to_num(coords, nan=0.0).astype(np.float32),
            atom37_mask=mask.astype(bool),
            aatype=self.static_batch["aatype"][0].detach().cpu().numpy(),
        )
        # Keep model parameters in their checkpoint dtype.  State tensors follow s.dtype.
        if self.dtype_name not in {"float32", "bfloat16", "float16"}:
            raise ValueError(f"Unsupported dtype: {self.dtype_name}")
        torch.set_grad_enabled(False)

    def initial_history(self) -> list[ConfRoverFrame]:
        self.load_model()
        assert self._initial_frame is not None
        return [self._initial_frame]

    def release_transient_memory(self) -> None:
        """Release decoder/history workspaces between outer-parent updates.

        Full-history conditioning grows the temporary encoder shapes with
        physical time.  On long K=16 runs, retaining inactive CUDA allocator
        blocks from earlier parents can exhaust a 48 GiB device even though no
        live particle needs those blocks.  This only releases unreachable
        cached storage; model/static tensors and all RNG state are unchanged.
        """
        import torch

        gc.collect()
        if self.device_name.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_history(self, history: Sequence[Any]) -> ConfRoverHistoryState:
        self.load_model()
        if not all(isinstance(frame, ConfRoverFrame) for frame in history):
            raise TypeError("ConfRover history must contain ConfRoverFrame objects")
        import torch
        from lightning.pytorch.utilities import move_data_to_device
        from openfold.np import residue_constants as rc
        from openfold.utils import rigid_utils as ru

        assert self.static_batch is not None and self.dataset is not None
        frames = list(history)
        count = len(frames)
        coords = np.stack([frame.atom37_a for frame in frames]).astype(np.float64)
        ca = rc.atom_order["CA"]
        coords -= np.mean(coords[..., ca : ca + 1, :], axis=1, keepdims=True)
        static_aatype = self.static_batch["aatype"][0]
        repeated_aatype = static_aatype.repeat(count)
        processed = self.dataset.process_coords(
            coords.reshape(-1, 37, 3), repeated_aatype.detach().cpu()
        )
        processed = move_data_to_device(processed, self.device_name)
        length = len(self.seqres)
        rigids = ru.Rigid.from_tensor_4x4(processed["rigidgroups_gt_frames"])[:, 0]
        rigids7 = rigids.to_tensor_7().reshape(count, length, 7)
        pseudo_beta = processed["pseudo_beta"].float().reshape(count, length, 3)
        pseudo_beta_mask = processed["pseudo_beta_mask"].float().reshape(count, length)

        begin_rigids = self.model.mask_token_rigids.expand(1, length, -1)
        begin_beta = self.model.mask_token_pseudo_beta.expand(1, length, -1)
        begin_beta_mask = self.model.mask_token_pseudo_beta_mask.expand(1, length)
        src_rigids = torch.cat([begin_rigids, rigids7], dim=0)
        src_beta = torch.cat([begin_beta, pseudo_beta], dim=0)
        src_beta_mask = torch.cat([begin_beta_mask, pseudo_beta_mask], dim=0)
        source_frames = count + 1
        aatype = static_aatype[None].expand(source_frames, -1)
        padding = self.static_batch["padding_mask"][0][None].expand(source_frames, -1)
        pretrained_single = self.static_batch["pretrained_single"][0][None].expand(
            source_frames, -1, -1
        )
        pretrained_pair = self.static_batch["pretrained_pair"][0][None].expand(
            source_frames, -1, -1, -1
        )
        single, pair = self.model.encoder(
            aatype=aatype,
            padding_mask=padding,
            rigids_0=src_rigids,
            batch_size=1,
            struct_mask=torch.ones(source_frames, device=src_rigids.device, dtype=single_dtype(self.static_batch)),
            pseudo_beta=src_beta,
            pseudo_beta_mask=src_beta_mask,
            pretrained_single=pretrained_single,
            pretrained_pair=pretrained_pair,
        )
        fused = self.model._fuse_single_pair((single, pair))
        inputs_embeds = fused.permute(1, 0, 2)  # (residue+pair tokens, physical history, channel)
        token_count = inputs_embeds.shape[0]
        position_ids = (
            torch.arange(source_frames + 1, device=inputs_embeds.device)
            * self.stride_in_10ps
        )[None].expand(token_count, -1)
        generation = self.model.temporal.prepare_configs_for_generation(
            inputs=inputs_embeds,
            max_length=source_frames + 1,
            position_ids=position_ids,
            use_cache=True,
        )
        model_kwargs = self.model.temporal._get_initial_cache_position(
            inputs_embeds[..., 0], generation["model_kwargs"]
        )
        model_inputs = self.model.temporal.prepare_inputs_for_generation(
            inputs_embeds=inputs_embeds, **model_kwargs
        )
        output = self.model.temporal(
            **model_inputs,
            return_dict=True,
            batch_size=1,
            rigids_mask=padding,
        )
        hidden = output.last_hidden_state[:, -1, :][None]
        s, z = self.model._split_single_pair(hidden, seqlen=length)
        self.accounting.temporal_encoder_evaluations += source_frames
        return ConfRoverHistoryState(
            s=s,
            z=z,
            aatype=static_aatype[None],
            padding_mask=self.static_batch["padding_mask"][0][None],
            pretrained_single=self.static_batch["pretrained_single"][0][None],
            pretrained_pair=self.static_batch["pretrained_pair"][0][None],
            frame_count=count,
        )

    def initialize_inner_particles(
        self, history_state: ConfRoverHistoryState, count: int, seeds: Sequence[int]
    ) -> ConfRoverParticleState:
        self.load_model()
        import torch
        from openfold.utils import rigid_utils as ru

        if len(seeds) != count:
            raise ValueError("Expected one deterministic seed per inner particle")
        samples = []
        cuda_devices = (
            [torch.device(self.device_name).index or 0]
            if self.device_name.startswith("cuda")
            else []
        )
        for seed in seeds:
            initial_seed = int(
                np.random.SeedSequence([int(seed), 0]).generate_state(1, dtype=np.uint32)[0]
            )
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(initial_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(initial_seed)
                with seeded_numpy(initial_seed):
                    rigid = self.model.decoder.diffuser.sample_ref(
                        n_samples=1,
                        num_frames=1,
                        seq_len=len(self.seqres),
                        device=self.device_name,
                    )
                samples.append(rigid.to_tensor_7())
        rigids_t7 = torch.cat(samples, dim=0)
        expand = lambda tensor: tensor.expand((count,) + tuple(tensor.shape[1:])).clone()
        return ConfRoverParticleState(
            rigids_t7=rigids_t7,
            s=expand(history_state.s),
            z=expand(history_state.z),
            aatype=expand(history_state.aatype),
            padding_mask=expand(history_state.padding_mask),
            rigids_mask=expand(history_state.padding_mask).to(history_state.s.dtype),
            pretrained_single=expand(history_state.pretrained_single),
            pretrained_pair=expand(history_state.pretrained_pair),
            noise_seeds=np.asarray(seeds, dtype=np.int64),
            conditioning_ids=np.full(count, history_state.frame_count, dtype=np.int64),
            ancestor_metadata=np.arange(count, dtype=np.int64),
            step=0,
            time_value=1.0,
        )

    def _compute_prediction(self, state: ConfRoverParticleState) -> None:
        import torch
        from openfold.utils import rigid_utils as ru

        batch = state.aatype.shape[0]
        chunk_size = min(self.decoder_microbatch_size or batch, batch)
        rot_scores = []
        trans_scores = []
        pred_atom14 = []
        pred_rigids_0_7 = []
        for start in range(0, batch, chunk_size):
            stop = min(start + chunk_size, batch)
            t = torch.full(
                (stop - start,),
                state.time_value,
                device=state.aatype.device,
                dtype=state.s.dtype,
            )
            rigids_t = ru.Rigid.from_tensor_7(state.rigids_t7[start:stop])
            output = self.model.decoder.model_nn(
                t=t,
                s=state.s[start:stop],
                z=state.z[start:stop],
                rigids_t=rigids_t,
                aatype=state.aatype[start:stop],
                padding_mask=state.padding_mask[start:stop],
                rigids_mask=state.rigids_mask[start:stop],
                pretrained_single=state.pretrained_single[start:stop],
                pretrained_pair=state.pretrained_pair[start:stop],
            )
            chunk_pred_rigids = output["pred_rigids_0"]
            chunk_mask = state.rigids_mask[start:stop]
            rot_scores.append(
                self.model.decoder.diffuser.calc_rot_score(
                    rigids_t.get_rots(),
                    chunk_pred_rigids.get_rots(),
                    t,
                    use_cached_score=True,
                )
                * chunk_mask[..., None]
            )
            trans_scores.append(
                self.model.decoder.diffuser.calc_trans_score(
                    rigids_t.get_trans(),
                    chunk_pred_rigids.get_trans(),
                    t[:, None, None],
                    use_torch=True,
                )
                * chunk_mask[..., None]
            )
            pred_atom14.append(output["pred_atom14"])
            pred_rigids_0_7.append(chunk_pred_rigids.to_tensor_7())
        state.cached_rot_score = torch.cat(rot_scores, dim=0)
        state.cached_trans_score = torch.cat(trans_scores, dim=0)
        state.cached_pred_atom14 = torch.cat(pred_atom14, dim=0)
        state.cached_pred_rigids_0_7 = torch.cat(pred_rigids_0_7, dim=0)
        self.accounting.reverse_decoder_evaluations += batch

    def _reverse_cached(self, state: ConfRoverParticleState) -> None:
        import torch
        from openfold.utils import rigid_utils as ru

        if state.cached_rot_score is None:
            self._compute_prediction(state)
        batch = state.aatype.shape[0]
        dt = 1.0 / self.reverse_steps
        next_rigids = []
        cuda_devices = (
            [torch.device(self.device_name).index or 0]
            if self.device_name.startswith("cuda")
            else []
        )
        for index in range(batch):
            seed = int(
                np.random.SeedSequence(
                    [int(state.noise_seeds[index]), 1, int(state.step)]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                current = ru.Rigid.from_tensor_7(state.rigids_t7[index : index + 1])
                with seeded_numpy(seed):
                    updated = self.model.decoder.diffuser.reverse(
                        rigids_t=current,
                        rot_score=state.cached_rot_score[index : index + 1],
                        trans_score=state.cached_trans_score[index : index + 1],
                        t=float(state.time_value),
                        dt=float(dt),
                        mode=self.sampler_mode,
                    )
                next_rigids.append(updated.to_tensor_7())
        state.rigids_t7 = torch.cat(next_rigids, dim=0)
        state.final_pred_atom14 = state.cached_pred_atom14
        state.final_pred_rigids_0_7 = state.cached_pred_rigids_0_7
        state.cached_rot_score = None
        state.cached_trans_score = None
        state.cached_pred_atom14 = None
        state.cached_pred_rigids_0_7 = None
        state.step += 1
        state.time_value = 1.0 - state.step / self.reverse_steps

    def _advance(self, state: ConfRoverParticleState, target_step: int) -> ConfRoverParticleState:
        target_step = min(int(target_step), self.reverse_steps)
        while state.step < target_step and state.time_value >= self.tmin:
            self._reverse_cached(state)
        return state

    def denoise_to_checkpoint(
        self, particle_state: ConfRoverParticleState, checkpoint_progress: float
    ) -> ConfRoverParticleState:
        target = int(round(self.reverse_steps * float(checkpoint_progress)))
        state = self._advance(particle_state, target)
        if state.step < self.reverse_steps and state.time_value >= self.tmin:
            self._compute_prediction(state)
        return state

    def _frames_from_atom14(self, atom14: Any, aatype: Any) -> list[ConfRoverFrame]:
        from confrover.model.utils.all_atom import atom14_to_atom37

        atom37, mask = atom14_to_atom37(atom14, aatype)
        coords = atom37.detach().float().cpu().numpy()
        masks = mask.detach().cpu().numpy().astype(bool)
        types = aatype.detach().cpu().numpy()
        return [ConfRoverFrame(coords[i], masks[i], types[i]) for i in range(len(coords))]

    def predict_clean(self, particle_state: ConfRoverParticleState) -> list[Any]:
        if particle_state.cached_pred_atom14 is None:
            self._compute_prediction(particle_state)
        self.accounting.predicted_clean_evaluations += particle_state.aatype.shape[0]
        return self._frames_from_atom14(
            particle_state.cached_pred_atom14, particle_state.aatype
        )

    def resample_particle_state(
        self, particle_state: ConfRoverParticleState, ancestor_indices: Any
    ) -> ConfRoverParticleState:
        return gather_state(particle_state, np.asarray(ancestor_indices, dtype=int))

    def reseed_particle_state(
        self,
        particle_state: ConfRoverParticleState,
        independent_seeds: Sequence[int],
    ) -> ConfRoverParticleState:
        if len(independent_seeds) != particle_state.aatype.shape[0]:
            raise ValueError("Expected one independent continuation seed per particle")
        particle_state.noise_seeds = np.asarray(independent_seeds, dtype=np.int64)
        return particle_state

    def denoise_to_end(
        self, particle_state: ConfRoverParticleState, independent_seeds: Sequence[int]
    ) -> ConfRoverParticleState:
        particle_state = self.reseed_particle_state(
            particle_state, independent_seeds
        )
        return self._advance(particle_state, self.reverse_steps)

    def finalize_frames(self, particle_state: ConfRoverParticleState) -> list[Any]:
        if particle_state.final_pred_atom14 is None:
            raise RuntimeError("ConfRover particles did not reach a clean prediction")
        frames = self._frames_from_atom14(
            particle_state.final_pred_atom14, particle_state.aatype
        )
        self.accounting.generated_complete_frames += len(frames)
        return frames

    def sample_complete_frames_direct(
        self, history_state: ConfRoverHistoryState, count: int, seeds: Sequence[int]
    ) -> list[ConfRoverFrame]:
        """Unintercepted local SDE/ODE path used by the Phase-B marginal control."""
        state = self.initialize_inner_particles(history_state, count, seeds)
        state = self._advance(state, self.reverse_steps)
        return self.finalize_frames(state)

    def validate_frames(self, frames: Sequence[Any]) -> list[dict[str, Any]]:
        results = []
        for frame in frames:
            ca_a = np.asarray(frame.ca_nm) * 10.0
            adjacent = np.linalg.norm(np.diff(ca_a, axis=0), axis=-1)
            quality_violations = adjacent >= self.ca_adjacent_quality_threshold_a
            nonfinite = int(np.size(frame.atom37_a) - np.isfinite(frame.atom37_a).sum())
            pairwise = np.linalg.norm(ca_a[:, None, :] - ca_a[None, :, :], axis=-1)
            nonneighbor_pairs = np.triu(np.ones(pairwise.shape, dtype=bool), k=2)
            clash_count = int(np.sum((pairwise < 1.0) & nonneighbor_pairs))
            results.append(
                {
                    "valid": bool(
                        nonfinite == 0
                        and clash_count == 0
                        and np.all(
                            adjacent
                            <= self.ca_adjacent_hard_threshold_a
                            + self.ca_adjacent_hard_tolerance_a
                        )
                    ),
                    "nonfinite_coordinate_count": nonfinite,
                    "ca_adjacent_max_a": float(np.max(adjacent)) if len(adjacent) else 0.0,
                    "ca_adjacent_count_ge_4_5a": int(np.sum(quality_violations)),
                    "ca_adjacent_fraction_ge_4_5a": (
                        float(np.mean(quality_violations)) if len(adjacent) else 0.0
                    ),
                    "ca_adjacent_quality_threshold_a": self.ca_adjacent_quality_threshold_a,
                    "ca_adjacent_hard_threshold_a": self.ca_adjacent_hard_threshold_a,
                    "ca_adjacent_hard_tolerance_a": self.ca_adjacent_hard_tolerance_a,
                    "ca_clash_count_lt_1a": int(clash_count),
                }
            )
        return results


def single_dtype(batch: dict[str, Any]):
    return batch["pretrained_single"].dtype
