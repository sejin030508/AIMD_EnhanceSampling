#!/usr/bin/env python3
"""Run the 2 x 3 x 4 x 4 short ConfRover sampler contrast for Bundle A.

The three arms deliberately separate the upstream autoregressive pipeline,
the upstream EulerSampler called behind our history adapter, and the current
locally-interceptable SDE loop.  All outputs are raw atom37 tensors; no reward,
resampling, minimization, or long rollout is used.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from confrover_duet import ConfRoverDuETAdapter, ConfRoverFrame, seeded_numpy


ATOM37 = {name: i for i, name in enumerate([
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
])}


@contextmanager
def deterministic(seed: int, device: str):
    devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with seeded_numpy(seed):
            yield


def geometry(trajectories: np.ndarray, masks: np.ndarray) -> dict:
    ca = ATOM37["CA"]
    maxima, invalid_frames, nan_frames, clash_frames = [], 0, 0, 0
    for path, path_mask in zip(trajectories, masks):
        row = []
        for frame, mask in zip(path, path_mask):
            xyz = frame[:, ca]
            adjacent = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
            maximum = float(np.nanmax(adjacent)) if len(adjacent) else 0.0
            row.append(maximum)
            nonfinite = not bool(np.isfinite(frame[mask]).all())
            pair = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
            clash = bool(np.any((pair < 1.0) & np.triu(np.ones_like(pair, bool), 2)))
            invalid_frames += int(nonfinite or maximum > 5.501 or clash)
            nan_frames += int(nonfinite)
            clash_frames += int(clash)
        maxima.append(row)
    return {
        "ca_adjacent_max_a": maxima,
        "global_ca_adjacent_max_a": float(np.max(maxima)),
        "invalid_frame_count": invalid_frames,
        "frame_count": int(trajectories.shape[0] * trajectories.shape[1]),
        "nan_frame_count": nan_frames,
        "ca_clash_frame_count": clash_frames,
    }


def official_forward(adapter: ConfRoverDuETAdapter, trajectories: int, transitions: int,
                     seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lightning.pytorch.utilities import move_data_to_device
    from confrover.data.infer import GenCaseConfig, GenDataset, GenDatasetConfig, PDBConditions
    from confrover.data.pretrain_repr import OpenFoldReprLoader
    from confrover.model.decoder.confdiff.sampler.euler import EulerSampler

    conditions = PDBConditions.from_list(str(adapter.initial_structure), "forward")
    cases = [GenCaseConfig(
        case_id=f"{adapter.case_id}_official", seqres=adapter.seqres,
        seqlen=len(adapter.seqres), task_mode="forward", n_replicates=trajectories,
        rep_id=i, n_frames=transitions + 1, stride_in_10ps=adapter.stride_in_10ps,
        conditions=conditions,
    ) for i in range(trajectories)]
    repr_loader = OpenFoldReprLoader(repr_root=adapter.cache_dir / "folding_repr")
    dataset = GenDataset(GenDatasetConfig(
        name="bundle_a_official", cases=cases, task_mode="forward",
        n_replicates=trajectories, n_frames=transitions + 1,
        stride_in_10ps=adapter.stride_in_10ps,
    ), repr_loader=repr_loader)
    adapter.model.decoder.sampler = EulerSampler(diffusion_steps=adapter.reverse_steps, mode="sde")
    coords_rows, mask_rows, aa_rows = [], [], []
    # Official CLI inference uses batch_size=1.  Preserve that memory and RNG
    # behavior rather than batching the four replicates into a different job.
    for i in range(len(dataset)):
        batch = move_data_to_device(dataset.collate([dataset[i]]), adapter.device_name)
        with deterministic(seed + i, adapter.device_name):
            output = adapter.model._ar_sample(**batch)
        coord = output["atom37"].detach().float().cpu().numpy()[0]
        mask0 = output["atom37_mask"].detach().cpu().numpy().astype(bool)[0]
        aa0 = output["aatype"].detach().cpu().numpy()[0]
        coords_rows.append(coord)
        mask_rows.append(np.repeat(mask0[None], coord.shape[0], axis=0))
        aa_rows.append(np.repeat(aa0[None], coord.shape[0], axis=0))
    coords = np.asarray(coords_rows)
    masks = np.asarray(mask_rows)
    aatype = np.asarray(aa_rows)
    return coords, masks, aatype


def adapter_rollout(adapter: ConfRoverDuETAdapter, trajectories: int, transitions: int,
                    seed: int, upstream_sampler: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from confrover.model.decoder.confdiff.sampler.euler import EulerSampler
    from confrover.model.utils.all_atom import atom14_to_atom37

    histories = [[adapter.initial_history()[0]] for _ in range(trajectories)]
    if upstream_sampler:
        adapter.model.decoder.sampler = EulerSampler(diffusion_steps=adapter.reverse_steps, mode="sde")
    for step in range(transitions):
        for path_index, history in enumerate(histories):
            state = adapter.prepare_history(history)
            stream_seed = int(np.random.SeedSequence([seed, path_index, step]).generate_state(1)[0])
            if upstream_sampler:
                with deterministic(stream_seed, adapter.device_name):
                    atom14, _ = adapter.model.decoder.sample(
                        aatype=state.aatype, s=state.s, z=state.z,
                        padding_mask=state.padding_mask, num_frames=1,
                        pretrained_single=state.pretrained_single,
                        pretrained_pair=state.pretrained_pair,
                    )
                atom37, mask = atom14_to_atom37(atom14, state.aatype)
                frame = ConfRoverFrame(
                    atom37[0].detach().float().cpu().numpy(),
                    mask[0].detach().cpu().numpy().astype(bool),
                    state.aatype[0].detach().cpu().numpy(),
                )
            else:
                frame = adapter.sample_complete_frames_direct(state, 1, [stream_seed])[0]
            history.append(frame)
    coords = np.asarray([[f.atom37_a for f in h] for h in histories], dtype=np.float32)
    masks = np.asarray([[f.atom37_mask for f in h] for h in histories], dtype=bool)
    aatype = np.asarray([[f.aatype for f in h] for h in histories], dtype=np.int64)
    return coords, masks, aatype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", choices=["trpcage", "prmt6"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=["official_forward", "official_sampler_adapter", "current_custom_sde"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=307)
    args = parser.parse_args()

    common = dict(
        repository_path="/workspace/sejin/confrover_mh_steering/external/ConfRover",
        checkpoint="/workspace/sejin/confrover_mh_steering/data/confrover_cache/confrover_ckpts/confrover_base_20m_v1_0.pt",
        cache_dir="/workspace/sejin/confrover_mh_steering/data/confrover_cache",
        device=args.device, reverse_steps=200, sampler_mode="sde", kv_cache_type="offloaded",
        decoder_microbatch_size=1, pairformer_chunk_size=32,
    )
    if args.protein == "trpcage":
        common.update(
            initial_structure="/workspace/sejin/AI_MD_NSMC_phase_b_recovery/data/small_protein_transition_pilot/prepared/trpcage/unfolded_minimized_confrover_heavy.pdb",
            case_id="bundle_a_trpcage", seqres="DAYAQWLKDGGPSSGRPPPS", stride_in_10ps=128,
        )
    else:
        common.update(
            initial_structure="/workspace/sejin/phase_b_pockets_recovery/prepared/prmt6/prmt6_start_model_numbering.pdb",
            case_id="bundle_a_prmt6",
            seqres="DVSVHEEMIADRVRTDAYRLGILRNWAALRGKTVLDVGAGTGILSIFCAQAGARRVYAVEASAIWQQAREVVRFNGLEDRVHVLPGPVETVELPEQVDAIVSEWMGYGLLHESMLSSVLHARTKWLKEGGLLLPASAELFIAPISDQMLEWRLGFWSQVKQHYGVDMSCLEGFATRCLMGHSEIVVQGLSGEDVLARPQRFAQLELSRAGLEQELEAGVGGRFRCSCYGSAPMHGFAIWFQVTFPGGESEKPLVLSTSPFHPATHWKQALLYLNEPVQVEQDTDVSGEITLLPSRDNPRRLRVLLRYKVGDQEEKTKDFAMED",
            stride_in_10ps=256,
        )
    adapter = ConfRoverDuETAdapter(**common)
    adapter.load_model()
    started = perf_counter()
    if args.arm == "official_forward":
        arrays = official_forward(adapter, 4, 4, args.seed)
    else:
        arrays = adapter_rollout(adapter, 4, 4, args.seed, args.arm == "official_sampler_adapter")
    elapsed = perf_counter() - started
    out = args.output_root / args.protein / args.arm
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trajectories_atom37.npz", trajectories_atom37_a=arrays[0], trajectories_atom37_mask=arrays[1], trajectories_aatype=arrays[2])
    meta = {
        "protein": args.protein, "arm": args.arm, "seed": args.seed,
        "trajectory_count": 4, "transitions_per_trajectory": 4,
        "physical_transition_count": 16, "stride_in_10ps": common["stride_in_10ps"],
        "reverse_steps": 200, "sampler_mode": "sde", "reward": None,
        "outer_resampling": False, "inner_resampling": False,
        "wall_clock_s": elapsed, "geometry": geometry(arrays[0], arrays[1]),
        "definitions": {
            "official_forward": "upstream ConfRover._ar_sample with upstream EulerSampler(mode=sde)",
            "official_sampler_adapter": "our full-history adapter, next frame decoded by upstream Decoder.sample/EulerSampler(mode=sde)",
            "current_custom_sde": "our current interceptable reverse loop using diffuser.reverse(mode=sde)",
        },
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
