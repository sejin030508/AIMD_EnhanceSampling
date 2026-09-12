# Session handoff — 2026-09-13 (before the 08:00 KST power cut)

All work is committed and pushed to branch `fast-folders-extended` of
`sejin030508/AIMD_EnhanceSampling`, HEAD `ba18fe6`. Nothing was running at
shutdown. The workspace itself is the user's to hand back.

## What is on the cluster

Everything below is under `/workspace`, which is the only tree that survives a
restart.

| Path | Size | Contents |
| --- | ---: | --- |
| `…/outputs/small_protein_transition_pilot` | 88 MB | the original 24-cell pilot, untouched |
| `…/outputs/small_protein_timing_variation` | 23 MB | 12 checkpoint-timing cells, evaluated |
| `…/outputs/pvb_small_protein` | 34 MB | 8 PVB cells, evaluated |
| `…/data/fast_folders_reference` | 33 GB | BBA / Homeodomain / Protein B reference MD + sha256 |
| `/workspace/sejin/pvb_assets` | 314 MB | PVB source @ `c08e5e3c`, 5 checkpoints, isolated site-packages |

`DUET_ASSET_ROOT` is `/workspace/sejin/confrover_mh_steering`; the ConfRover
checkpoint and the `folding_repr` cache live there and were not modified.

## Results

**Checkpoint timing (ConfRover, DuET, stride 128, T=32, seeds 211/223).**
(0.75, 0.90) is the completed pilot, not re-run.

| protein | timing | wValidTHP (211 / 223) |
| --- | --- | --- |
| Trp-cage | 0.25+0.50 | 0.000 / 0.000 |
| Trp-cage | **0.50+0.75** | **0.637 / 0.604** |
| Trp-cage | 0.75+0.90 | 0.333 / 0.120 |
| Trp-cage | 0.25+0.50+0.75 | 0.552 / 0.000 |
| BBA | 0.25+0.50 | 0.000 / 0.000 |
| BBA | 0.50+0.75 | 0.000 / 0.000 |
| BBA | **0.75+0.90** | **0.656** / 0.000 |
| BBA | 0.25+0.50+0.75 | 0.000 / 0.000 |

(0.50, 0.75) is the only setting where both seeds succeeded, and only on
Trp-cage. Early intervention (0.25+0.50) produced no hits anywhere. Two seeds
cannot support a superiority claim, and the peptide-geometry limitation from the
Bundle A audit still applies to every ConfRover hit.

**PVB minimal set (T=32, sde_step 20, checkpoints 0.75/0.90, seeds 211/223).**
DuET lowers backbone RMSD in all four conditions (BBA 7.5→4.7 A, Trp-cage
6.1→4.0 A) and reaches whole-path validity 1.00 *with the peptide C-N check
enforced* — the check Bundle A showed ConfRover fails in every frame. All DuET
TICA hits are 0, which is expected: PVB's lag is 100 ps, so T=32 spans 3.2 ns
against ConfRover's 40.96 ns.

> The two backends are matched on transition count, never on physical time.
> `stride_in_10ps` is recorded as 10 for PVB so this cannot be misread.

## Per-cell cost

| Configuration | seconds/cell |
| --- | ---: |
| ConfRover DuET, K4xM4, reverse 200 | 2372-2709 (~40-45 min) |
| PVB DuET, K4xM4, sde_step 20 | 48-64 |
| PVB Frozen, K16xM1, Trp-cage | 107-108 |
| PVB Frozen, K16xM1, BBA | 749-1056 |

The ConfRover figure is from the arm that had a GPU to itself. Two other arms
recorded 3100-4700 s because a duplicate launcher put two cells on one GPU;
those numbers are contention, not cost.

PVB is roughly 40x cheaper per DuET cell. Frozen is slower than DuET at equal
NFE because K=16 runs sequentially while DuET batches M=4.

## Traps found, and what they cost

1. **Config stage vs runner stage.** The cell runner derives its output stage
   from molecule and stride; the sampler uses the stage named in the config.
   Naming them differently sends evaluation to a directory that does not exist,
   and all twelve timing cells reported `evaluation_failed` after their sampling
   had already succeeded. Recovered by re-running evaluation only
   (`evaluate_timing_runs.py`, seconds per cell). Keep the config stage equal to
   `stride<N>_t32` and isolate runs with `SMALL_PROTEIN_OUTPUT_ROOT`.
2. **`small_protein_env.sh` assigns `PYTHONPATH` outright** and is re-sourced by
   the cell runner, which dropped PVB's isolated dependencies. It now appends
   `PVB_SITE` when set.
3. **`phase_b_runner.py` on the cluster has diverged from this repo** — it reads
   checkpoint progresses from the config, the repo copy does not. It was patched
   in place. Do not deploy the repo copy over it.
4. **`pip install --target` without `--no-deps`** pulls torch 2.14 and a CUDA 13
   stack into the target directory, shadowing the environment's torch 2.1.2 and
   breaking the `torch_scatter` built against it.

## Left running, deliberately

Two ABL1 wrappers on the A6000 have been in `T` (stopped) state since
2026-09-03: PID 1663861 (GPU2, `duet`) and PID 1663984 (GPU3,
`complete_nested`). Their Python children are already zombies, so they hold no
GPU. They were parked pending the whole-path validity decision and were left
alone rather than killed. The power cut ends them either way; resuming that work
means relaunching `resume_abl1_a6000.sh`, not `kill -CONT`.

A6000 GPUs 0 and 2 were busy with another user's work and were not touched.

## Next steps

- Fit TICA for Protein B and Homeodomain from the downloaded reference MD.
  Compatibility is judged within a protein (sequence, per-torsion atoms, feature
  order, lag against trajectory length, trajectory boundaries), not against the
  published BBA model. The reference is adaptive sampling — 7,297 runs of 500
  frames for BBA — so pass trajectories to PyEMMA as a list.
- PVB at a matched physical horizon needs T≈410, about 16 min/cell for DuET.
  Outer genealogy length changes, so treat it as its own condition.
- Whether PVB's clean peptide geometry survives an all-atom energy check, the
  way Bundle A tested ConfRover.
