# Phase B cryptic-pocket pilot — execution-status handoff

## Status at 2026-09-08 21:52 KST

**The Phase B B1 protein-method comparison has not started.** The supervisor queue stopped before B1 because PRMT6 failed one required structural-validity check in its B0 Frozen preflight. This is a preflight stop, not a DuET-vs-Complete result.

| Item | State |
| --- | --- |
| PRMT5 preflight | Passed |
| PRMT6 preflight | Failed one hard frame-validity condition in B0 |
| B1 runs completed | `0 / 12` |
| B1 runs started | `0 / 12` |
| Running Phase B processes | None |
| GPU activity | All four A6000 GPUs idle |
| `preflight_status.json` | `all_ready_for_b1=false`, `b1_started=false` |

The queue was intentionally guarded so that **neither PRMT5 nor PRMT6 B1 starts unless both proteins pass preflight**. Therefore, the passing PRMT5 B1 cells were not run either.

## 1. Frozen Phase B scope

Phase B is an exploratory endpoint-steering pilot under a frozen ConfRover-base checkpoint. It is separate from Phase A and makes no kinetic or physical-opening-probability claim.

| Protein | Start | Holo target reference(s) | Construct | Reward loop | d0 |
| --- | --- | --- | ---: | ---: | ---: |
| PRMT5 | 7KIC-A pseudo-apo | 6UXY-A, 6UXX-A | UniProt 294–637 | 435–445 | 6.143322 Å |
| PRMT6 | AF-Q96LA8-F1-model_v4 | 6W6D-A | UniProt 53–375 | 155–165 | 4.668808 Å |

The endpoint potential is shared across methods:

```text
log psi(x) = max(-30, -4 * (d(x) / d0)^2)
```

`d(x)` is the minimum fixed-core-aligned loop N/CA/C RMSD to the holo reference set. The core excludes the reward loop, five residues on either side, construct ends, and non-shared backbone positions.

### Planned B1 matrix — not executed

Each protein was scheduled for these six cells, for 12 total:

| Method | K | M | Horizon | Seeds | Status |
| --- | ---: | ---: | ---: | --- | --- |
| Frozen | 16 | 1 | 16 | 17, 29 | Not started |
| Complete Nested | 4 | 4 | 16 | 17, 29 | Not started |
| Fixed DuET | 4 | 4 | 16 | 17, 29 | Not started |

All B1 cells use stochastic SDE, 200 reverse steps, physical stride 256 × 10 ps = 2.56 ns/frame, decoder microbatch 2, offloaded KV cache, systematic outer resampling at ESS <= 0.5K, and equal decoder budget `K*M=16`. Fixed DuET uses the completed reverse fraction `0.75` checkpoint.

## 2. Required preflight policy

Per protein, B1 requires all of the following:

1. `K=1,M=4,T=1` DuET memory gate complete with at least one valid path.
2. `K=4,M=4,T=2` full-history DuET memory gate complete with at least one valid path.
3. `K=4,M=1,T=4` Frozen B0 complete with at least one valid path.
4. B0 has no nonfinite coordinates, no CA clashes under 1 Å, and **no invalid frames**.

B0 is a structural-stability gate, not a pocket-opening success screen. `valid_open_like_fraction=0` in B0 is therefore not the reason for the stop.

## 3. Raw preflight results

### PRMT5 — all checks passed

| Stage | Method | K | M | T | NFE | Wall time | Peak GPU memory | Valid-path fraction | Final weighted d/d0 | Best valid final d | Clipping |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Memory gate 1 | DuET | 1 | 4 | 1 | 796 | 107.210 s | 5,529,746,944 B (5.15 GiB) | 1.00 | 0.964354 | 5.924335 Å | 0 |
| Memory gate 2 | DuET | 4 | 4 | 2 | 6,368 | 860.948 s | 8,402,875,904 B (7.83 GiB) | 1.00 | 0.974135 | 5.553317 Å | 0 |
| B0 | Frozen | 4 | 1 | 4 | 3,184 | 494.800 s | 13,908,715,520 B (12.95 GiB) | 1.00 | 1.028801 | 6.157251 Å | 0 |

PRMT5 B0 had no nonfinite coordinates, no <1 Å CA clashes, and no invalid frames. The preflight audit therefore reports `prmt5 ready=true`.

### PRMT6 — memory gates passed; B0 did not

| Stage | Method | K | M | T | NFE | Wall time | Peak GPU memory | Valid-path fraction | Final weighted d/d0 | Best valid final d | Clipping |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Memory gate 1 | DuET | 1 | 4 | 1 | 796 | 95.680 s | 4,672,880,640 B (4.35 GiB) | 1.00 | 0.963712 | 4.499385 Å | 0 |
| Memory gate 2 | DuET | 4 | 4 | 2 | 6,368 | 766.586 s | 7,102,007,296 B (6.61 GiB) | 1.00 | 1.021742 | 4.521918 Å | 0 |
| B0 | Frozen | 4 | 1 | 4 | 3,184 | 438.359 s | 11,748,153,344 B (10.94 GiB) | **0.75** | 1.119799 | 4.772883 Å | 0 |

The two PRMT6 DuET gates both had valid path fraction 1.0. The B0 failure is isolated to structural validity, not memory, NaN, potential clipping, or steric clash.

## 4. Exact stopping event: PRMT6 B0

The B0 result contains four outer paths, each containing the initial frame plus four generated frames. One generated final frame in the first outer path is invalid:

| Check | Observed value | Hard rule | Outcome |
| --- | ---: | ---: | --- |
| Nonfinite coordinate count | 0 | must be 0 | Passed |
| CA clashes <1 Å | 0 | must be 0 | Passed |
| Maximum adjacent-CA distance | **5.722644 Å** | <= 5.5 Å + 0.001 Å tolerance | **Failed** |
| Invalid frames in B0 | 1 | must be 0 | **Failed** |

That frame has 24 adjacent CA pairs at or above the softer 4.5 Å quality threshold (7.45% of adjacent pairs). Other generated PRMT6 frames also have some soft quality-threshold exceedances, but all stay below the 5.5 Å hard cutoff and are marked valid.

The audit uses `all(frame.valid)` over every B0 frame. Thus the single 5.722644 Å adjacent-CA separation yields:

```json
"b0_no_invalid_frames": false,
"ready_for_b1": false,
"all_ready_for_b1": false
```

The B1 supervisor has `set -e` and checks:

```text
.all_ready_for_b1 == true and .b1_started == false
```

The check failed, so it exited before the B1 launcher. No B1 log files or B1 metrics files exist.

## 5. Interpretation boundary

What is established:

- PRMT5 can run the prescribed preflight configuration without an observed structural-validity failure.
- PRMT6 can load the representation, execute both DuET memory gates, and produce finite, non-clashing structures.
- One PRMT6 unsteered B0 trajectory contains a hard adjacent-CA stretching violation at its last generated frame.

What is **not** established:

- There is no PRMT5/PRMT6 B1 result, so there is no valid Frozen-vs-Complete-Nested-vs-DuET comparison.
- The PRMT6 event does not show that endpoint steering fails, because it occurred in the unsteered B0 gate before B1.
- The B0 `valid_open_like_fraction=0` is not a negative pocket result; B0 is not scored as a pocket-opening experiment.
- The hard validity threshold should not be relaxed after observing this event without explicitly revising the protocol and rerunning the relevant preflight under that revised policy.

## 6. Decision required before any restart

The execution agent has made **no** configuration change and has not launched a replacement run. A coordinating agent should choose one of the following protocol-level responses:

1. **Keep the current hard validity rule.** Diagnose PRMT6's structural instability (affected residue span, representation/start structure, and model dynamics), then rerun a predeclared preflight only after a justified technical correction.
2. **Amend the preflight policy prospectively.** For example, define a priori whether a specified number of independent B0 seeds, rather than one seed, determines feasibility. This requires a documented protocol revision; it cannot silently convert the current failed gate into a pass.
3. **Revise the PRMT6 task or remove/replace it from Phase B.** This changes the pilot scope and should be recorded as such.

Re-running seed 17 unchanged can serve only as a reproducibility/debug check; it does not add independent evidence for feasibility.

## 7. Reproducibility and artifact locations

### A6000 output root

```text
/mnt/ssd0/sejin/phase_b_pockets/outputs
```

### Key raw artifacts

```text
PRMT5 preflight audit:
/mnt/ssd0/sejin/phase_b_pockets/outputs/preflight_status.json

PRMT5 B0 metrics:
/mnt/ssd0/sejin/phase_b_pockets/outputs/prmt5/b0/frozen/seed_17/metrics.json

PRMT6 B0 metrics:
/mnt/ssd0/sejin/phase_b_pockets/outputs/prmt6/b0/frozen/seed_17/metrics.json

PRMT6 B0 per-frame validity, paths, weights, and curves:
/mnt/ssd0/sejin/phase_b_pockets/outputs/prmt6/b0/frozen/seed_17/

Full supervisor log:
/mnt/ssd0/sejin/phase_b_pockets/outputs/_launcher/full_pipeline.log
```

The PRMT6 B0 directory contains `metrics.json`, `records.json`, `population_curves.json`, `outer_ancestry.json`, pre/post-resampling atom37 trajectory NPZ files, resolved config, environment, and checkpoint/manifest provenance.

### Config and provenance SHA256

| Artifact | SHA256 |
| --- | --- |
| `prmt5_b1.yaml` | `204f1c81efaa19b8a8fba044d35c87813ca070055c09d8baac84045632c946d5` |
| `prmt6_b1.yaml` | `a7dfbbd0df872f8f943d373535012a651bef023f0da4cb1057a7147c4d742375` |
| `prmt6_b0.yaml` | `8aeb42edfe7948476168208956838489777cda3e0254ca7aef4788eaa24dc7b5` |
| PRMT5 manifest | `8503000c2e55613cfec3814e90690f7d5ffdeca97891d7ebe74f6636ac8c9c9b` |
| PRMT6 manifest | `829cc35d5c41248221c2fa294fc3e8d0e6836767578ee8a01284e5b20e40cf7a` |
| PRMT6 B0 metrics | `d4cbe431716339a2a37e615802253ebbc1f9fd22908fc3a7dc9313103c470fa6` |

## Bottom line for ChatGPT

Phase B was correctly stopped before scientific B1 comparison. PRMT5 is technically ready. PRMT6 has one hard CA-adjacency geometry violation in a single unsteered B0 path, despite passing memory gates and having no NaNs or CA clashes. The next action is a protocol/design decision, not a result interpretation or a silent threshold change.
