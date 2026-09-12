# Phase B B3 cryptic-pocket development handoff

Date: 2026-09-10  
Audience: theory/coordination GPT agent  
Status: **B3-0 and all four prespecified B3-1 cells complete; B3-2 not started.**

## 1. Why B3 was run

B3 is a **DuET-only development experiment**, not a method-superiority claim.
The prior B1/B2 pilot (PRMT5 and PRMT6) produced zero Open-like trajectories
for every Frozen, Outer-only, Complete Nested, and DuET cell.  B3 therefore
asked the narrower question:

> With the frozen ConfRover surrogate and this endpoint task, can a stronger
> diffusion-time steering signal create a valid population-level movement
> toward the holo-pocket loop?

The original B3 instruction explicitly required that, if a usable DuET setting
were found, it later be compared again against matched Complete Nested and
Outer-only controls.  Those controls were intentionally **not** run in B3-1.

## 2. Immutable scientific setup

| Item | Fixed value |
|---|---|
| Proteins | PRMT5, PRMT6 |
| Model | frozen `confrover_base_20m_v1_0` checkpoint |
| Sampler | SDE; 200 reverse diffusion steps/frame |
| Physical lag | 256 x 10 ps = 2.56 ns/generated frame |
| B3-1 horizon | T=48 transitions; 49 structures including start; nominal 122.88 ns, **not a kinetic claim** |
| DuET population | K=4 outer histories, M=4 inner candidates; nominal decoder population KxM=16 |
| Diffusion steering | exactly one inner Feynman--Kac resampling checkpoint at 75% completed reverse diffusion |
| Outer algorithm | systematic resampling when ESS <= 0.5K; proper normalizer interface |
| Seed | 101 for every B3-1 cell |
| Structural validity | original adjacent-CA hard threshold 5.5 Angstrom; unchanged |
| Intermediate snapshots | pre-resampling populations/weights at transitions 16, 32, 48; no extra selection, restart, or history reset |

### Reward and outcome definitions

For each generated frame, `d(x)` is the fixed-core-aligned pocket-loop N/CA/C
RMSD to the holo reference(s), in Angstrom.  PRMT5 takes the minimum to 6UXY
and 6UXX; PRMT6 uses 6W6D.

```text
log psi_a(x) = max(-30, -a * (d(x)/d0)^2)
```

| Protein | d0 | Holo/open-like threshold |
|---|---:|---:|
| PRMT5 | 6.143322 Angstrom | d <= 1.5 Angstrom |
| PRMT6 | 4.668808 Angstrom | d <= 1.5 Angstrom |

The checkpoint score is the same potential evaluated on predicted-clean
coordinates.  Inner selection uses this score during the 75% diffusion-time
intervention; outer selection uses the corresponding proper normalizer rather
than double-counting the selected frame score.

Primary yield is `valid_open_like_fraction = #(valid AND Open-like paths) / #(all paths)`;
invalid trajectories stay in the denominator.  Open-like requires
a fully valid path and both final generated frames within 1.5 Angstrom.  The
hidden-contact observables remain evaluation-only.

## 3. B3-0: unresampled candidate diagnostic

Purpose: determine whether unsteered next-frame candidates contain a small
but usable distance signal, and whether early predicted-clean scores rank it.

- Parents: first four fully valid T=16 Frozen population paths.
  - PRMT5 source: B1-recovery Frozen seed 17, source indices [0,1,2,3].
  - PRMT6 source: B1-recovery Frozen seed 29, source indices [0,2,3,6].
- Per parent: eight unsteered next-frame candidates, no checkpoint resampling.
- Observed predicted-clean progress: 50%, 75%, 90%, and final.
- Each protein: 32 candidates, all valid, 6,368 decoder NFE.

### B3-0 raw summaries

`min / median d/d0` among valid candidates:

| Protein | 50% | 75% | 90% | Final |
|---|---|---|---|---|
| PRMT5 | 0.9665 / 1.0321 | 0.9579 / 1.0363 | 0.9840 / 1.0424 | 0.9706 / 1.0627 |
| PRMT6 | 0.9466 / 1.0429 | 0.9365 / 1.0465 | 0.9221 / 1.0352 | 0.9472 / 1.0425 |

Checkpoint-vs-final Spearman rank correlations, one value per parent:

| Protein | 50% | 75% | 90% |
|---|---|---|---|
| PRMT5 | -0.119, -0.071, -0.595, 0.524 | -0.024, 0.452, -0.476, 0.310 | 0.429, 0.524, 0.071, 0.143 |
| PRMT6 | 0.310, 0.310, 0.500, 0.690 | 0.857, 0.548, 0.976, 0.905 | 0.929, 0.643, 0.929, 0.929 |

Offline mean ESS (eight candidates/parent):

| Protein | a=4: 50/75/90/final | a=16: 50/75/90/final |
|---|---|---|
| PRMT5 | 7.899 / 7.824 / 7.726 / 7.600 | 6.773 / 5.881 / 5.330 / 4.519 |
| PRMT6 | 7.541 / 7.366 / 6.905 / 6.871 | 4.400 / 3.881 / 2.972 / 3.158 |

`a=64` was evaluated only offline.  The `-30` log-potential floor clipped all
near-start candidates, making its weights uniform (ESS=8); **no a=64 rollout
was added.**

Interpretation of B3-0: PRMT6 had a strong 75%/90% checkpoint-to-final ranking
signal and a=16 created substantial selection pressure.  PRMT5 ranking was
weak/inconsistent, although a=16 still reduced ESS.  This motivated exactly
the prespecified a=4 vs a=16 B3-1 comparison, not a further parameter sweep.

## 4. B3-1: four prespecified runs

Matrix: `2 proteins x 2 reward coefficients = 4` Fixed-DuET runs.  Every cell
has 152,832 decoder NFE (`48 x 16 x 199`), 1,537 potential evaluations, zero
potential-floor clipping, and the same physical/diffusion settings above.

| Protein | a | Final valid paths | Valid Open-like | Anytime hit | Weighted final d/d0 | Best valid final d | Outer resamples | Min outer ESS | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PRMT5 | 4 | 0/4 | 0/4 | 0/4 | 1.0197 | — | 2 | 1.974 | 123.7 min |
| PRMT5 | 16 | 4/4 | 0/4 | 0/4 | 0.9517 | 5.7986 A | 9 | 1.102 | 156.3 min |
| PRMT6 | 4 | 1/4 | 0/4 | 0/4 | 1.0225 | 4.9139 A | 2 | 1.977 | 146.6 min |
| PRMT6 | 16 | 4/4 | 0/4 | 0/4 | 0.9401 | 4.3434 A | 14 | 1.127 | 156.9 min |

`Weighted final d/d0` is computed before the final outer resampling.  Lower is
closer to the holo loop.  The best-valid value excludes invalid paths.

### Per-path final distances (Angstrom)

| Cell | Final distances | Path validity |
|---|---|---|
| PRMT5 a=4 | 6.2931, 6.3422, 6.1963, 6.3155 | F, F, F, F |
| PRMT5 a=16 | 5.7986, 5.8576, 6.0814, 5.8600 | T, T, T, T |
| PRMT6 a=4 | 4.9139, 4.8874, 4.7494, 4.5922 | T, F, F, F |
| PRMT6 a=16 | 4.3843, 4.3434, 4.3551, 4.4766 | T, T, T, T |

### Intermediate saved-population results

Numbers are the pre-resampling weighted population mean `d/d0`; the parentheses
give valid paths at that transition.

| Protein | a | T=16 | T=32 | T=48 |
|---|---:|---:|---:|---:|
| PRMT5 | 4 | 1.0394 (3/4) | 1.0727 (3/4) | 1.0197 (0/4) |
| PRMT5 | 16 | 0.9431 (4/4) | 0.9379 (4/4) | 0.9517 (4/4) |
| PRMT6 | 4 | 1.0176 (3/4) | 1.0197 (2/4) | 1.0225 (1/4) |
| PRMT6 | 16 | 0.9329 (3/4) | 0.9706 (3/4) | 0.9401 (4/4) |

## 5. Execution provenance and deviations from the scientific protocol

The first unchunked PRMT5 B3-1 attempts OOMed in full-history Pairformer
triangular attention.  They produced no `metrics.json` and were moved (not
deleted) to explicitly named `*_failed_oom_unchunked_attempt1` directories.

Recovery added only `pairformer_chunk_size=32` alongside the already-fixed
microbatch-1/offloaded-KV execution settings.  This sub-batches tensor
evaluation to bound memory; it does **not** alter checkpoint weights, model
parameters, reward, K/M, seed, horizon, sampler, 75% checkpoint, validity
rule, or output selection policy.  It can change floating-point operation
ordering, so it is recorded as an execution amendment.  All four recovered
scientific cells completed.

Focused test coverage before launch: 12 DuET tests plus a mock snapshot test
passed.  The run-level metrics retain resolved configuration, command,
environment, checkpoint/manifest SHA-256, records, ancestry, curves,
pre/post-resampling populations, and intermediate populations at T=16/32/48.

## 6. Interpretation boundary and decision requested

What B3 establishes:

1. At `a=16`, both proteins maintain a valid final population and end roughly
   4.8% (PRMT5) and 6.0% (PRMT6) closer to the holo-loop RMSD target than the
   respective start structures.
2. At `a=4`, neither protein shows useful average approach, and final validity
   deteriorates substantially.
3. Hence there is a **distance-only, single-seed steering signal** at a=16.

What B3 does **not** establish:

1. No path reached the 1.5 A Open-like threshold, even transiently.  This is
   not cryptic-pocket restoration.
2. The approach is not monotonic over T=16/32/48, and the present data do not
   yet verify that core geometry and hidden contacts change plausibly.
3. B3 has no matched Complete Nested or Outer-only control, so it makes no
   claim that diffusion-time steering is superior to complete-frame selection
   or physical-time SMC.
4. One seed per cell is development evidence, not an uncertainty estimate.

**Requested theory/coordination decision:** determine whether the a=16 signal
is sufficient to authorize the single prespecified B3-2 repeat on a new seed,
after inspecting core-geometry/hidden-contact trajectories.  If it is, repeat
only the qualifying a=16 protein/configuration with unchanged reward and
thresholds.  If that repeat remains positive, the next confirmatory stage must
run matched Fixed DuET, Complete Nested, and Outer-only comparisons rather than
opening another steering hyperparameter sweep.

## 7. Artifact locations

H100 project:

```text
/workspace/sejin/AI_MD_NSMC_phase_b_recovery
```

Shared output root:

```text
/workspace/sejin/phase_b_pockets_recovery/outputs
```

B3-0:

```text
prmt5/b3_0_checkpoint_probe/source_frozen_seed17/probe_seed101/
prmt6/b3_0_checkpoint_probe/source_frozen_seed29/probe_seed101/
```

B3-1 (each contains `metrics.json`, `intermediate_metrics.json`, curves,
records, validity, weights, and atom37 populations):

```text
prmt5/b3_duet_a4_t48/duet/seed_101/
prmt5/b3_duet_a16_t48/duet/seed_101/
prmt6/b3_duet_a4_t48/duet/seed_101/
prmt6/b3_duet_a16_t48/duet/seed_101/
```

Earlier B1/B2/protocol context is retained in:

- `reports/PHASE_B_CRYPTIC_POCKET_HANDOFF_2026-09-09.md`
- `reports/PHASE_B_PROTOCOL_AMENDMENT_2026-09-08.md`
