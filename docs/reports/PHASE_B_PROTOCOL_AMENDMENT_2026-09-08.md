# Phase B protocol amendment and execution record

Timestamp: 2026-09-08T22:46:00+09:00

## Scope and preservation rule

This amendment was applied before any B1 scientific run existed.  It does not
modify or overwrite the following immutable artifacts:

- PRMT5 and PRMT6 preflight outputs;
- the original `5.5 Å` adjacent-CA hard validity threshold;
- the PRMT6 strict B0 output at
  `/mnt/ssd0/sejin/phase_b_pockets/outputs/prmt6/b0/frozen/seed_17`.

The original PRMT6 strict B0 remains `FAIL`: 3/4 valid paths (one invalid
generated frame), and the original global audit remains as historical
provenance only.

## Protein-local readiness

The global `all_ready_for_b1` gate is not used by the amended launcher.

- PRMT5 uses its existing strict `ready_for_b1=true` result and starts B1.
- PRMT6 first receives a separately named expanded diagnostic.  Its strict B0
  result is never reclassified; the new result only makes an independent
  exploratory-pilot decision.

## Original PRMT6 strict-B0 geometry diagnosis (read-only)

The invalid event is path 0, generated frame 4.  The maximum adjacent CA pair
is model indices 156--157, author/UniProt residues 209--210 (`SER--GLN`),
prepared-PDB residues 157--158.

| quantity | Å |
|---|---:|
| current invalid-frame distance | 5.722644 |
| previous-frame distance | 4.401737 |
| prepared start distance | 3.887893 |
| 6W6D holo distance | 3.899169 |

The pair is consecutive in both mapping systems, not at a construct boundary,
and does not involve a missing start/holo backbone residue.  It is therefore
not currently diagnosed as a numbering, chain-boundary, or preprocessing
artifact.  The separate machine-readable report is
`outputs/prmt6/diagnostics/strict_b0_seed_17_geometry.json`.

## Expanded PRMT6 diagnostic

If no preprocessing artifact is found (the present diagnosis satisfies that
condition), run exactly one separate Frozen diagnostic:

- K=16, M=1, T=4, seed=41;
- same frozen ConfRover, SDE sampler, lag, start structure, and no steering;
- original `5.5 Å` threshold;
- output stage `b0_expanded_diagnostic`, so it cannot overwrite B0.

The report records valid-path fraction, invalid-frame fraction (both including
the start and generated-only forms), per-path maximum adjacent-CA distance,
invalid-pair frequencies, NaN rate, and CA-clash rate.  A repeated same-site
break is operationally recorded as the same exact adjacent model-index pair
being invalid in at least two frames.

Readiness is: >=15/16 valid = `proceed`; 13--14/16 =
`exploratory_proceed`; <=12/16 or a repeated same-site break = `hold_b1`;
nonfinite coordinates, a <1 Å CA clash, or preprocessing/mapping artifact =
`hard_stop`.  The first two decisions alone permit PRMT6 B1.

## Amended B1 design

For each protein, B1 has 4 methods x 2 fixed seeds (`17`, `29`) = 8 runs:

| method | K | M | decoder population budget |
|---|---:|---:|---:|
| Frozen | 16 | 1 | 16 |
| Outer-only | 16 | 1 | 16 |
| Complete Nested | 4 | 4 | 16 |
| Fixed DuET | 4 | 4 | 16 |

All protein-specific reward/reference/checkpoint/lag/sampler/start settings
are copied unchanged into separately named amended configs.  Outer-only is
added as M=1 with the maximum K at the same nominal budget.

For each B1 run, the primary result is now explicitly
`valid_open_like_fraction = (# valid AND open-like trajectories) / (# all
trajectories)`.  Invalid trajectories remain in the denominator.  The old
SMC-weighted form is retained under the separate key
`weighted_valid_open_like_fraction`; likewise `weighted_valid_path_fraction`
is retained separately.  This semantic change applies to new B1 outputs only;
already-written preflight/B0 JSON files are unchanged.

## Execution queue

- PRMT5 amended B1 was launched at the timestamp above.
- PRMT6 expanded diagnostic is queued after PRMT5 releases the GPU allocation;
  it does not use PRMT5 outcome to alter any PRMT6 setting.
- If the independent PRMT6 readiness permits it, the queue launches its B1;
  otherwise it stops and preserves the diagnostic evidence.

Relevant implementation artifacts on A6000:

- `configs/phase_b_pockets/prmt5_b1_amended.yaml`
- `configs/phase_b_pockets/prmt6_b1_amended.yaml`
- `scripts/duet/diagnose_phase_b_prmt6_b0.py`
- `scripts/duet/run_phase_b_amended_b1_a6000.sh`
- `scripts/duet/run_phase_b_prmt6_diagnostic_queue_a6000.sh`

## OOM recovery amendment

Timestamp: 2026-09-09T08:20:14+09:00

The first amended B1 attempt stopped after five CUDA OOM failures.  Three
PRMT6 seed-17 cells completed, but PRMT5 produced no complete cell.  All first
attempt outputs remain untouched.

A complete 16-cell re-execution was authorized under a new stage named
`b1_recovery_mb1`.  Scientific settings remain fixed: K/M, T=16, seeds 17/29,
reward, checkpoint, sampler, lag, reverse steps, and nominal decoder NFE.  The
only execution changes are `decoder_microbatch_size=1` and
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.

- PRMT6 recovery runs on A6000 GPUs 0 and 3, two cells concurrently.
- PRMT5 recovery is deployed to a separate H100 project/output directory.  The
  H100 was found to contain a pre-existing 7P46 Complete Nested process using
  the GPU, so a non-interfering queue waits until the GPU has no compute process
  before launching PRMT5 sequentially.
- Existing B1, strict B0, expanded B0, and readiness artifacts are not
  overwritten.  The H100 received read-only copies of prepared inputs and
  readiness provenance.

Recovery configs and launchers:

- `configs/phase_b_pockets/prmt5_b1_recovery_mb1.yaml`
- `configs/phase_b_pockets/prmt6_b1_recovery_mb1.yaml`
- `scripts/duet/run_phase_b_recovery_mb1.sh`
- `scripts/duet/queue_phase_b_prmt5_h100.sh`

## OOM recovery amendment

Timestamp: 2026-09-09T08:45:00+09:00

The first T=16 B1 attempt stopped after five cells exhausted 48 GiB A6000
devices.  Completed and failed artifacts were retained.  A separately named
`b1_recovery_mb1` stage was created for all 2 proteins x 4 methods x 2 seeds.

The scientific settings remain fixed.  Recovery changes only execution-memory
handling:

- `decoder_microbatch_size=1` for every recovery cell;
- CUDA allocator `max_split_size_mb=128,garbage_collection_threshold=0.8`;
- per-parent release of unreachable full-history/decoder workspaces followed
  by `torch.cuda.empty_cache()`; model/static tensors and RNG state are kept.

The pre-cache-release recovery attempts were stopped before expected OOM and
moved, not deleted, into explicitly named failed-attempt directories.  A6000
now runs PRMT6 on physical GPUs 0 and 3, two cells at a time.  H100 runs PRMT5
sequentially on its single GPU.  The pre-existing H100 7P46 Complete Nested
process and its global `h100_1` launcher were terminated at user request;
their on-disk outputs were not deleted.

H100 initially lacked the PRMT5 OpenFold representation.  That launch failed
before model sampling; all resulting empty run directories/logs were archived
as `b1_recovery_mb1_failed_missing_repr_attempt1`.  The exact precomputed cache
from A6000 was copied with matching SHA-256 hashes and its single sequence-index
entry was merged without replacing H100's existing index.  Both servers then
started the same recovery code; initial allocated device use was approximately
11 GiB per active process rather than the previous near-48-GiB accumulation.

## Cross-server recovery reallocation

Timestamp: 2026-09-09T11:27:35+09:00

At user request, the A6000 pending queue was rebalanced toward the faster H100
without interrupting already advanced cells or duplicating a protein/method/
seed result.

- The A6000 scheduler parent was stopped while its already-running PRMT6
  seed-17 Frozen and Outer-only child processes were left intact.
- A separate A6000 coordinator waits for those two cells to finish and, only
  if both metrics files exist, runs seed-17 Complete Nested and DuET on GPUs 0
  and 3.  It does not launch PRMT6 seed 29.
- The H100 continues all eight PRMT5 cells.  After all eight metrics exist, a
  separate coordinator runs all four PRMT6 seed-29 methods sequentially on the
  H100.
- The exact PRMT6 OpenFold representation was copied from A6000 to H100 and
  verified by SHA-256 before the reallocated cells were queued.

At this timestamp, PRMT5 seed-17 Frozen, Outer-only, and Complete Nested had
completed on H100; DuET was active.  PRMT6 seed-17 Frozen and Outer-only were
active on A6000.  Recovery completion was 3/16, with 3 active and 10 queued.

## B2 two-checkpoint DuET follow-up (prepared, not launched)

Timestamp: 2026-09-09T15:31:00+09:00

The follow-up isolates the effect of adding a second diffusion-time steering
intervention.  It is a separately named stage, `b2_duet_075_090`, and does not
overwrite B0, B1, or B1 recovery artifacts.

- Scope: PRMT5 and PRMT6; only Complete Nested and DuET-MD.
- New shared seed: 43.  Together with preserved B1 seeds 17 and 29, this
  yields three seeds per method/protein for the combined pilot evidence.
- DuET-MD: two Feynman--Kac inner resampling checkpoints at completed reverse
  fractions 0.75 and 0.90 (steps 150 and 180 of 200).
- Complete Nested: no intermediate resampling/steering; it retains its
  existing 0.75 generation split solely as the matched complete-frame control.
- Fixed settings: horizon 16 (17 physical frames including start), lag 256 x
  10 ps, SDE sampler, 200 reverse steps, K=4/M=4, decoder population budget
  16, endpoint reward, structural validity, Open-like definition, and the
  microbatch-1 memory-recovery execution setting.

The Phase-B runner was corrected to pass the config's
`inner_checkpoint_progresses` to `OuterSMC`; it had previously hard-coded the
single 0.75 value despite the inner sampler already supporting multiple
checkpoints.  New metrics explicitly record both the full configured checkpoint
list and the list of actual DuET steering checkpoints.

Prepared H100 artifacts (validated, but not launched):

- `configs/phase_b_pockets/prmt5_b2_complete_nested_control.yaml`
- `configs/phase_b_pockets/prmt5_b2_duet_075_090.yaml`
- `configs/phase_b_pockets/prmt6_b2_complete_nested_control.yaml`
- `configs/phase_b_pockets/prmt6_b2_duet_075_090.yaml`
- `scripts/duet/run_phase_b2_duet_075_090_cells.sh`
- `scripts/duet/run_phase_b2_duet_075_090_3h_plan.sh`

The prepared three-hour plan is four serial H100 cells, after all 16 B1
recovery metrics exist: PRMT5 Complete Nested and DuET seed 43, followed by
PRMT6 Complete Nested and DuET seed 43.  No B2 output directory existed at
validation time, and no B2 trajectory had started.

At user authorization on 2026-09-09, the B2 plan was queued on H100.  Its
server-local prerequisite is eight PRMT5 B1 recovery metrics plus four PRMT6
seed-29 B1 recovery metrics: the four PRMT6 seed-17 metrics reside on A6000
and are already complete.  This corrected a launcher-only gate that had
incorrectly expected all 16 distributed B1 metrics under the H100 output root.
The queue does not alter or overlap B1; it starts B2 only after the four H100
PRMT6 seed-29 cells complete.

## B2 multi-GPU reallocation

Timestamp: 2026-09-09T17:34:00+09:00

The B2 cells are scientifically independent of the pending B1 metrics.  On
user instruction, the formerly serial B2 plan was split across the two
available Sejin H100 pods rather than waiting for the B1 H100 to become idle.

- `sejin-h100-1-work-002-84brf` was verified idle and started PRMT5 seed-43
  Complete Nested, followed serially by PRMT5 DuET.
- `sejin-h100-1-work-001-zhr2l` continues the in-flight PRMT6 B1 seed-29
  recovery.  When it finishes, its existing B2 queue runs only PRMT6 seed-43
  Complete Nested and DuET.
- The output filesystem is shared across these H100 pods.  The work-001 B2
  launcher was narrowed to PRMT6 before PRMT5 was dispatched, preventing
  duplicate cells.
- The available A6000 devices were not used for these four cells because their
  observed recovery-cell runtime is materially slower than H100; the two-H100
  split yields the earliest expected completion without adding a hardware
  confound.

## B3 DuET development experiment

Timestamp: 2026-09-10T00:12:00+09:00

B3 tests steering feasibility rather than method superiority.  Existing B1/B2
artifacts and scientific definitions remain unchanged.

B3-0 first evaluates unsteered candidate signal.  For each protein, the first
four fully valid T=16 Frozen population paths (deterministic path-index order)
are used as full-history parents.  Eight unsteered next-frame candidates per
parent are generated once, with predicted-clean structures recorded at 50%,
75%, and 90% reverse progress and again at completion.  No checkpoint
resampling is performed.  Candidate distance, validity, parent-relative
distance change, checkpoint/final rank correlation, and offline weights/ESS
for a=4,16,64 are saved.  PRMT5 uses Frozen seed 17; PRMT6 uses Frozen seed 29,
because these complete valid-prefix populations are available on H100.

B3-1 consists of exactly four Fixed-DuET runs: PRMT5/PRMT6 crossed with reward
coefficient a=4/16, all at new seed 101.  K=4, M=4, one 75% checkpoint, T=48,
2.56 ns nominal lag, SDE, 200 reverse steps, and microbatch 1 are fixed.
Pre-resampling populations and weights are retained at transitions 16, 32,
and 48 without extra resampling, restart, or history reset.

The implementation adds config-driven reward coefficient and optional
pre-resampling snapshot capture.  Focused DuET tests (12 tests) and a separate
mock snapshot assertion passed before GPU launch.  B3-0 completed with 6,368
decoder NFE per protein and saved all 32 candidates at all four observation
stages.  Offline a=64 weights were uniformly clipped at log floor -30 for
these near-d0 candidates, producing ESS=8; no a=64 rollout was added.

GPU allocation:

- work-001: PRMT5 a=4, then PRMT6 a=16;
- work-002: PRMT5 a=16;
- work-004: PRMT6 a=4 queued only after its pre-existing 7P46 route launcher
  exits; the existing workload is not interrupted.

An hourly thread heartbeat monitors only recoverable execution failures and
may not alter any scientific B3 setting or start B3-2 automatically.
