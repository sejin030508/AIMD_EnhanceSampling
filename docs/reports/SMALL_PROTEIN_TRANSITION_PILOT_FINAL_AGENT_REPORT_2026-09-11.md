# Small-protein transition recovery pilot — final agent report

Date: 2026-09-11 (KST)  
Status: **COMPLETE — 24/24 production cells sampled and evaluated**

## Executive conclusion

The supplied fixed small-protein pilot is complete. No further experiment is
specified by the original prompt: it explicitly says to report after the full
matrix and not automatically launch further exploration.

This pilot supplies **mixed diagnostic evidence**, not a publishable claim of
physical transition recovery.

- **Chignolin:** DuET consistently lowers final whole-backbone RMSD relative to
  Frozen, but obtains no valid final official-TICA target hit in any of four
  cells. Two DuET cells have zero whole-path validity. This is endpoint approach,
  not recovered folding.
- **Trp-cage:** only the 1.28-ns DuET cells have nonzero weighted valid THP
  (`0.333`, `0.120`); corresponding Frozen cells are `0.0625`, `0.125`. This is
  the clearest positive DuET signal in the pilot, but it is based on two seeds.
- **BBA:** Frozen has valid final hits in all four cells (`0.125–0.250`), while
  DuET has high weighted valid THP in two cells (`0.656`, `0.599`) and zero in
  two others. DuET has much lower final backbone/heavy RMSD but high seed/lag
  variability. It suggests selection can concentrate probability on a small
  number of hits; it does not establish consistently higher recovery.
- **Physical plausibility warning:** conditional sampled-frame ETS values for
  target-hit paths are roughly `1e11–1e17 kJ/mol`, vastly above minimized
  endpoint energies (order `-1e3` to `-7e3 kJ/mol`). The current adjacent-CA
  validity gate can therefore admit endpoint/TICA hits with severe atomistic
  strain or clashes. Do not describe these as physically valid folded
  transitions without a stronger geometry/energy gate and an independently
  justified follow-up protocol.

## Scope, provenance, and frozen configuration

- Namespace: `/workspace/sejin/AI_MD_NSMC_phase_b_recovery/outputs/small_protein_transition_pilot`
- Official TPS-DPS source: `https://github.com/kiyoung98/tps-dps.git`
- Pinned commit: `61fd65ad2e2f110d65c176a8c8f5c2fe8bdab034`
- Proteins: Chignolin (10 aa), Trp-cage (20 aa), BBA (28 aa)
- Start/target: official unfolded/folded PDBs, each minimized once using the
  official BaseDynamics force field; generated frames were not repaired.
- Model: frozen `confrover_base_20m_v1_0`, SDE, 200 reverse steps, full history.
- Reward: `log psi(x) = -16 * (d(x)/d0)^2`, where `d` is whole-protein common
  N/CA/C Kabsch RMSD to the prepared folded target in Å. No floor/clipping.
- T=32 generated transitions (33 structures including x0); seeds 211 and 223.
- Lags: stride 16 (160 ps nominal) and stride 128 (1.28 ns nominal).
- Frozen: K=16, M=1, no inner/outer steering. DuET: K=4, M=4, two actual inner
  resampling checkpoints at completed-reverse fractions 0.75 and 0.90 (steps
  150 and 180 of 200).
- All cells use `101,888 = 32 × 16 × 199` decoder NFE. Total matrix NFE:
  `2,445,312`.

The production ordering was amended at user request to complete the remaining
DuET cells before remaining Frozen cells. This changed only resource priority:
no config, reward, seed, checkpoint, metric, or result-dependent selection was
changed. See the preflight report for the exact amendment.

## Metric semantics

- **wBB / wHeavy:** final pre-resampling-weighted whole-backbone / common-heavy
  Kabsch RMSD in Å. Common-heavy is explicitly used because ConfRover omits
  terminal OXT.
- **wValidTHP:** sum of final outer weights over paths that are both whole-path
  valid and within official TICA distance `<0.75` of the folded target.
- **hits:** number of valid final target paths in the returned population.
  Frozen populations have 16 paths; DuET populations have 4 paths.
- **valid:** whole-path valid fraction in that returned population. Invalid paths
  remain in the weighted denominator; they are never removed and renormalized.
- **ETS:** mean, over valid final target-reaching paths, of the maximum official
  potential across saved frames; it is a conditional diagnostic, not a physical
  transition-state energy. `NA` means no valid final target-reaching path.

## Raw production results

### Chignolin

| Lag | Method | Seed | wBB Å | wHeavy Å | wValidTHP | hits | valid | anytime valid hits | ETS paths | sampler min |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | Frozen | 211 | 3.756 | 5.495 | 0.000 | 0/16 | 0.563 | 6 | 0 | 44.2 |
| 16 | Frozen | 223 | 3.566 | 5.262 | 0.000 | 0/16 | 0.438 | 6 | 0 | 43.8 |
| 16 | DuET | 211 | 2.106 | 3.281 | 0.000 | 0/4 | 0.000 | 0 | 0 | 38.6 |
| 16 | DuET | 223 | 2.389 | 4.672 | 0.000 | 0/4 | 0.750 | 3 | 0 | 42.8 |
| 128 | Frozen | 211 | 4.007 | 5.850 | 0.000 | 0/16 | 0.625 | 7 | 0 | 39.9 |
| 128 | Frozen | 223 | 3.858 | 5.286 | 0.000 | 0/16 | 0.875 | 13 | 0 | 44.0 |
| 128 | DuET | 211 | 2.798 | 4.504 | 0.000 | 0/4 | 0.000 | 0 | 0 | 41.5 |
| 128 | DuET | 223 | 1.762 | 2.888 | 0.000 | 0/4 | 0.000 | 0 | 0 | 37.8 |

### Trp-cage

| Lag | Method | Seed | wBB Å | wHeavy Å | wValidTHP | hits | valid | anytime valid hits | ETS paths | sampler min |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | Frozen | 211 | 4.899 | 6.468 | 0.000 | 0/16 | 0.063 | 0 | 0 | 46.6 |
| 16 | Frozen | 223 | 5.108 | 6.516 | 0.063 | 1/16 | 0.250 | 2 | 1 | 41.7 |
| 16 | DuET | 211 | 4.329 | 6.285 | 0.000 | 0/4 | 1.000 | 0 | 0 | 40.0 |
| 16 | DuET | 223 | 4.062 | 5.795 | 0.000 | 0/4 | 1.000 | 0 | 0 | 40.3 |
| 128 | Frozen | 211 | 4.837 | 6.420 | 0.063 | 1/16 | 0.438 | 6 | 1 | 46.3 |
| 128 | Frozen | 223 | 4.959 | 6.460 | 0.125 | 2/16 | 0.188 | 3 | 2 | 40.9 |
| 128 | DuET | 211 | 2.719 | 4.294 | 0.333 | 1/4 | 0.500 | 2 | 1 | 40.5 |
| 128 | DuET | 223 | 3.317 | 4.832 | 0.120 | 1/4 | 1.000 | 4 | 1 | 40.2 |

### BBA

| Lag | Method | Seed | wBB Å | wHeavy Å | wValidTHP | hits | valid | anytime valid hits | ETS paths | sampler min |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | Frozen | 211 | 7.334 | 8.999 | 0.250 | 4/16 | 0.813 | 8 | 4 | 46.9 |
| 16 | Frozen | 223 | 6.671 | 8.316 | 0.250 | 4/16 | 0.500 | 7 | 4 | 45.2 |
| 16 | DuET | 211 | 3.240 | 5.513 | 0.000 | 0/4 | 0.250 | 0 | 0 | 43.4 |
| 16 | DuET | 223 | 3.262 | 5.200 | 0.599 | 1/4 | 1.000 | 4 | 1 | 43.5 |
| 128 | Frozen | 211 | 6.629 | 8.484 | 0.250 | 4/16 | 0.313 | 5 | 3 | 45.9 |
| 128 | Frozen | 223 | 6.105 | 7.706 | 0.125 | 2/16 | 0.375 | 3 | 1 | 46.8 |
| 128 | DuET | 211 | 3.236 | 5.267 | 0.656 | 2/4 | 1.000 | 4 | 2 | 44.2 |
| 128 | DuET | 223 | 3.597 | 6.035 | 0.000 | 0/4 | 0.750 | 0 | 0 | 43.6 |

## Aggregate facts, not significance tests

| Method | Cells | Mean wValidTHP | Cells with >=1 valid final hit | Mean sampler minutes |
|---|---:|---:|---:|---:|
| Frozen | 12 | 0.0938 | 7/12 | 44.4 |
| DuET | 12 | 0.1423 | 4/12 | 41.4 |

This simple pooled mean is not an estimator of method superiority: proteins,
lags, returned population sizes (16 vs 4), outer weights, and only two seeds
are heterogeneous. It is included solely as a compact audit of the completed
matrix.

## Interpretation by question

### Does the pilot show that diffusion-time steering can help?

It shows a **task-dependent signal**, not a general result. At Trp-cage stride
128, both DuET seeds exceed the corresponding Frozen weighted valid THP and
have substantially lower final RMSD. At BBA, DuET produces high weighted mass
in two cells, with lower RMSD, but fails in the other two; Frozen generates
more diffuse but nonzero hits in all four BBA cells. Chignolin has no valid
endpoint recovery under either method.

### Does lower RMSD imply valid folding recovery?

No. Chignolin is the direct counterexample: DuET lowers RMSD but has no valid
final TICA hit. BBA further shows that endpoint/TICA hits must be considered
with geometry diagnostics. RMSD is an approach diagnostic, not sufficient proof
of a physically meaningful transition.

### What does ETS add?

It is a decisive caution. Target-reaching paths have very large conditional
potential diagnostics: e.g. Trp-cage values from approximately `1.5e11` to
`3.5e16 kJ/mol`, BBA values from approximately `3.7e14` to `2.8e17 kJ/mol`.
These are incompatible with calling the frames relaxed physical transitions.
They may reflect severe local clashes/strain under the official force field,
while still passing the broad adjacent-CA validity gate. ETS was deliberately
not included in reward, so this does not invalidate the sampling arithmetic; it
limits the biological/physical claim.

### Timing and ETS overhead

T=32 sampling took roughly 38–47 minutes per cell. ETS cost was negligible for
cells with no valid final target path. For BBA Frozen stride16/seed211, four
qualifying paths required `4 × 33 = 132` H-only minimizations and added 23.98
minutes after 46.88 minutes of sampling: 34% of end-to-end wall time for that
cell. This conditional cost explains the earlier apparent slowdown.

## Artifacts

Remote raw outputs:

- `/workspace/sejin/AI_MD_NSMC_phase_b_recovery/outputs/small_protein_transition_pilot/summary_all.csv`
- `/workspace/sejin/AI_MD_NSMC_phase_b_recovery/outputs/small_protein_transition_pilot/summary_all.json`
- `/workspace/sejin/AI_MD_NSMC_phase_b_recovery/outputs/small_protein_transition_pilot/report.md`
- Per-cell: `metrics.json`, `small_protein_metrics.json`,
  `small_protein_frame_diagnostics.json`, `tica_projection.npz`, and
  `tica_pmf_overlay.png` under the corresponding protein/stride/method/seed
  directory.

Setup, hash, and scheduler amendment record:

- `reports/SMALL_PROTEIN_TRANSITION_PILOT_PREFLIGHT_2026-09-10.md`

## Do not automatically run more

The original protocol's terminal condition has been met. A follow-up must be
explicitly chosen rather than automatically sweeping settings. Scientifically,
the most useful follow-up decision is whether to strengthen an evaluation-only
validity/geometry audit before interpreting endpoint hits, and then decide
whether a fresh-seed confirmation is justified for the Trp-cage stride128 and
BBA candidate settings. Do not change rewards, targets, or thresholds after
examining these outputs.
