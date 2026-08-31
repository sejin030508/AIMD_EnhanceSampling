# DuET-MD experiment protocol

## Task construction and split

- R1 and R2 are design references. They determine the DuET-specific PCA basis,
  endpoint interval, intermediate interval, contact thresholds, event windows,
  and reward scale.
- R3 is held out for reference-fidelity evaluation.
- The endpoint basin is the design PC1 tail farthest from the starting frame.
- The intermediate interval is centered halfway between the start and endpoint
  PC1 centers and is active at physical frames 3–5.
- Ordered events are two stable C-alpha contact threshold crossings selected
  from R1/R2 transition segments with a configured minimum sequence separation,
  distance change, stability, and physical-frame separation.
- If two valid ordered events cannot be selected, Task C is unresolved; no
  hand-picked fallback is substituted.

For the present 6J56-A assets, the deterministic far-tail endpoint has one
R1/R2 start-to-target transition segment. This support count is saved in the
selection report and must be reported with Task C; it is not inflated by
treating overlapping frames as independent transitions.

The sampler receives only observable definitions, intervals, event order, and
windows. Held-out intermediate coordinates are never model inputs.

## Baselines and compute matching

The core decoder population per physical step is 16:

| Method | K | M | Weighting |
|---|---:|---:|---|
| Frozen trajectories | 16 independent K=1 paths | 1 | none |
| Best of budget | same 16 paths, post-hoc ranking | 1 | search baseline only |
| Outer-only | 16 | 1 | `psi_t/psi_{t-1}` |
| Inner-only | 1 | 16 | selected full history; `Zhat` diagnostic |
| Naive dual | 4 | 4 | selected `psi_t/psi_{t-1}` |
| Complete-frame nested | 4 | 4 | `Zhat_complete/psi_{t-1}` |
| DuET-MD | 4 | 4 | `Zhat/psi_{t-1}` |

All controlled particle methods use the same SDE base sampler. The official ODE
rollout appears only in the Phase-B feasibility comparison.

## Sequential gates

1. Exact toy: telescoping, local-normalizer, joint proper-weighting, constant
   potential, and intended-target convergence must pass.
2. ConfRover preflight: genuine independent post-resampling SDE continuation,
   full-history preservation, constant-potential marginal agreement, and
   structural feasibility must pass. Structural feasibility uses zero
   nonfinite/clashing outputs, a 5.5 A hard adjacent-CA limit, and at most 10%
   degradation in mean maximum adjacent-CA distance versus official ConfRover
   ODE. The original 4.5 A exceedance rate remains a reported quality metric.
   Conformational displacements use the official ConfRover writer convention:
   every frame is Kabsch-aligned to frame 0, while raw displacements are kept
   as coordinate-frame diagnostics.
3. Core 6J56-A: three tasks, seven methods, three paired seeds.
4. K/M grid: `(16,1),(8,2),(4,4),(2,8),(1,16)` on windowed and ordered tasks.
5. Official interpolation: eight-case outcome-blind pilot, frozen protocol,
   then the official full manifest only.
6. ProAR transfer: only after a positive ConfRover two-clock gate and a stable
   stochastic ProAR refinement checkpoint are verified.

## Metrics and statistical unit

Primary metrics are joint program success per decoder NFE and per GPU-hour plus
event-order accuracy. Fidelity metrics use held-out intermediate visitation,
DTW in orthogonal PCA/contact features, milestone timing, intermediate
coverage, and held-out motion-direction alignment. Diversity and validity
include route clusters, path entropy, pairwise diversity, surviving ancestors,
inner/outer ESS, `Zhat` coefficient of variation, geometry/clash failures,
frame displacement, and filtering rate.

The independent unit is a protein case or temporal-program task, not individual
trajectories from one task. Report per-unit results, mean, median, paired
differences, paired bootstrap 95% intervals, and unresolved rates. ATLAS is a
held-out MD reference, not exact physical ground truth.

## Negative-result policy

Negative gates and unresolved cases are retained with logs and diagnostics.
DuET-MD is weakened if complete-frame nested matches it at equal wall clock,
if `M` does not improve local discovery, if `K` does not preserve event order or
genealogy, or if reward success rises while held-out fidelity or validity falls.
No result is described as exact physical dynamics.

## Automated reporting

`scripts/duet/generate_reports.sh` scans completed runs and writes the required
method, K/M, interpolation, compute/memory, failure, paired-bootstrap, and
falsification tables under `outputs/duet_md/reports/`. It also renders the four
requested figure panels. Panels whose gated phase has not run are marked as
awaiting data rather than populated with synthetic values.

Every scientific run additionally records `assets.json` with paths, sizes,
timestamps, and SHA-256 checksums for files up to `DUET_HASH_MAX_BYTES`.
