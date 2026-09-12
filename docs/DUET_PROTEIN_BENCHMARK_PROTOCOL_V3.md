# DuET-MD protein benchmark construction protocol v3.1

**Status:** acceleration-aware, role-based specification for all new
cross-protein experiments. It supersedes the original route-v3 timing and
literal R1/R2/R3 split. Completed 6J56 and 7LP1 experiments retain their
historical definitions and are not relabelled.

## 1. Purpose and scope

This document defines one repeatable benchmark family for the DuET-MD paper.
The same construction algorithm is intended to be applied to roughly ten
proteins, after which performance is summarized across proteins.  Protein-level
benchmarks are separate from later mechanism-specific case studies such as
pocket opening or cryptic-pocket formation.

The benchmark asks:

> Given one transition route actually observed in reference MD, can a frozen
> trajectory generator produce the ordered structural program A then B and
> reach the target basin within a predeclared nominal-time deadline, and does
> DuET-MD complete it earlier or more often than matched baselines at equal
> model, K/M allocation, seed family, and decoder NFE?

It does **not** assume that the selected route is the unique, dominant, or
biologically causal mechanism.  A route seen in only one reference replicate
is a valid route-steering target, but must be labelled `single_reference_route`.
Reference crossing times define route geometry and diagnostics; generated A/B
events are not required to reproduce the same narrow time windows.

## 2. Terms

- **Reference trajectory**: an existing atomistic MD trajectory. No new MD is
  run during benchmark construction.
- **Initial structure**: the single PDB structure used as physical frame 0 for
  every generated path.
- **Start state**: a set of reference structures whose PC1 is near the initial
  structure and, in route v3, whose C-alpha RMSD to it is below a fixed limit.
- **Target state**: the far PC1 state selected using design references only.
- **Transition path**: the segment from the last stable visit to the start state
  to stable entry into the target state.
- **D1 (route-discovery replicate)**: the original ATLAS replicate named by the
  predeclared official ConfRover interpolation case. It defines the reference
  route and A/B milestones.
- **D2 (design-support replicate)**: the lowest-index remaining ATLAS replicate.
  It supports PCA/state construction and measures independent route support.
- **H (held-out replicate)**: the one remaining ATLAS replicate. It is never
  used to define the task and provides optional secondary fidelity evidence.
- **Route support**: occurrence of the same A-before-B definition in another
  time-compatible design transition path. Support strengthens the claim but is
  not required to define a route-specific benchmark.
- **A and B**: persistent C-alpha distance milestones on one discovery
  transition path. They are not automatically physical contacts or causal
  events.
- **Terminal**: the first persistent target-state hit by the frozen generated
  deadline; it need not occur only at the final frame.
- **Nominal surrogate time**: generated frame index multiplied by the configured
  ConfRover lag. It is useful for matched method comparisons but is not an
  unbiased physical kinetic clock after reward-based path selection.

## 3. Data split

The minimum input is three independently seeded MD replicates under the same
physical conditions and with matching topology. Their original labels are data
identifiers, not permanent statistical roles. Assign roles before any Frozen,
Complete Nested, or DuET outcome is inspected:

1. `D1` is the replicate encoded in the predeclared official ConfRover
   interpolation case;
2. `D2` is the lowest-index replicate not used as D1;
3. `H` is the remaining replicate.

D1 and D2 may determine PCA, basins, residue pairs, thresholds, deadlines, and
reward scale. H may determine none of them. If H does not contain a comparable
transition, held-out route fidelity is recorded as unavailable rather than a
benchmark failure. If H contains a different valid route, that indicates route
heterogeneity rather than proving the generated route wrong.

Splitting one long trajectory into R1/R2/R3 does not create independent
replicates. Non-overlapping paths from the same trajectory may be useful
descriptively but are not counted as independent seeds.

### 3.1 Status of the earlier 6J56 and 7LP1 cases

The earlier proteins were not sampled randomly from all ATLAS entries, but they
also do not constitute an outcome-blind representative cohort:

- **6J56-A** was the already available ConfRover example/development case, with
  compatible checkpoint, cache, initial structure, and ATLAS R1/R2/R3 assets.
  It was therefore an engineering anchor selected mainly by availability and
  protocol compatibility, not by a systematic protein-wide biological screen.
- **7LP1-A** was pre-specified from the official ConfRover interpolation cases
  as a small, inexpensive replication target with a stride-256 transition and
  compatible ATLAS references. It was selected to test the method beyond 6J56
  at manageable cost, not after seeing its DuET result.

Both proteins have nevertheless influenced task, allocation, and checkpoint
development. They must be labelled **development proteins** and must not be
presented as two randomly selected or independent confirmatory members of the
approximately ten-protein generalization cohort. New confirmatory proteins are
selected by the frozen eligibility gates in this document without inspecting
Complete Nested or DuET outcomes; the full candidate list and gate attrition
must be retained.

## 4. Protocol frozen before method comparison

For each protein, freeze the following before inspecting Complete Nested or
DuET outcomes:

1. topology, sequence, initial PDB, D1/D2/H assignment, and file hashes;
2. model physical lag, generated horizon, and nominal-time deadline;
3. PCA/start/target definitions;
4. route-discovery segment in D1;
5. A/B residue pairs, directions, thresholds, order, and persistence;
6. frozen reachability seeds and acceptance rules;
7. K/M allocation, diffusion checkpoint, evaluation seeds, and metrics.

If a frozen gate fails, record the protein-task as `unresolved`. Do not replace
events after looking at method outcomes.

## 5. Transition-path extraction

### 5.1 State construction

Fit C-alpha PCA on D1/D2 only. The start PC1 interval remains centered on the
actual initial PDB. The target is the design PC1 tail farthest from the initial
score. Route v3 additionally requires start frames to be structurally close to
the initial PDB by Kabsch-aligned C-alpha RMSD.

### 5.2 Stable last exit and stable target entry

For every design replicate:

1. find contiguous start-state visits lasting at least the configured start
   dwell time;
2. find contiguous target-state visits lasting at least the configured target
   dwell time;
3. for a target entry, choose the final qualifying start visit before it;
4. define the transition-path start as the last frame of that start visit;
5. define target entry as the first frame of the stable target visit.

Target visits are scanned chronologically and the first path passing all frozen
eligibility gates is retained. The builder does not inspect method outcomes or
manually choose among several paths.

This removes the v2 behavior that started at the earliest convenient frame in
the start PC1 interval and therefore mixed arbitrary basin waiting time with the
structural transition.

### 5.3 Reference geometry and nominal-time challenge

Let

```text
model_step_ps = 10 * physical_lag_in_10ps
model_duration_ps = horizon * model_step_ps
```

Keep two reference times separate:

- `reference_first_passage_ps`: from the exact D1 initial structure to the first
  stable target entry, including basin waiting;
- `reference_transition_path_ps`: from the last stable start-state exit to the
  same target entry, excluding basin waiting.

The last-exit transition path defines structural route geometry and A/B order.
It is analyzed on existing MD frames; atomic coordinates are never linearly
interpolated. Generated frames are not required to map one-to-one onto those
reference timestamps.

The generated deadline is an enhanced-discovery challenge. The original
`[0.80, 1.10]` duration-matching gate is removed. When a matching D1
first-passage observation exists, predeclare

```text
nominal_compression = reference_first_passage_ps / model_duration_ps
```

in the range `[1, 10]`: generated nominal time may be up to tenfold shorter than
the observed reference first passage. This ratio sets challenge difficulty; one
observed first passage is not a kinetic-rate estimator. If the discrete
ConfRover lag/horizon grid cannot meet the frozen challenge without implausible
frame jumps, mark the protein unresolved rather than tuning after method
outcomes.

Lag and horizon must still pass unsteered one-step calibration against reference
MD. Whole-path continuity, aligned per-step displacement, clashes, and geometry
are required so nominal acceleration cannot be achieved by structural
teleportation.

## 6. Route-specific A/B construction

### 6.1 Discovery does not require recurrence in every replicate

A and B must both occur, in order, on the same D1 transition path. They do not
have to occur in D2. This separates two questions:

- `route exists`: at least one observed reference path contains A then B;
- `route is independently supported`: another independent design path also
  contains the same A then B definition.

The builder prefers a pair with greater independent support when several valid
pairs exist, but `min_route_support_count: 1` permits a route defined by D1
alone. The output records:

- `route_support_count`;
- `route_support_fraction` among retained design transition paths;
- `single_reference_route` or `independently_supported_route`;
- the supporting segment indices and A-to-B gaps.

Absence of A/B from a replicate that never reaches the target is not evidence
against the route. A target-reaching replicate with another order is evidence
of route heterogeneity and prevents a universal-mechanism claim.

### 6.2 Distance-milestone selection

Enumerate C-alpha residue-pair distances on the discovery transition path.
Initial defaults require:

- sequence separation of at least 5 residues;
- absolute start-to-end change of at least 0.15 nm;
- the actual initial PDB to be at least 0.15 nm on the pre-event side;
- a crossing persistent for at least 2 sampled reference frames;
- distinct persistent A and B episodes in strict temporal order on D1.

Direction and threshold are defined from the discovery path. In the current
minimal implementation, the threshold is the midpoint between that path's
sampled start and end distances. The same fixed threshold is then evaluated on
the other design paths to measure support. A future multi-path definition may
use a median, but that is not claimed for a route defined from one path.

The term **distance milestone** is preferred over **contact** unless the
threshold has the physical meaning of a contact cutoff. A large C-alpha
distance threshold is not a residue contact merely because it is stored by the
legacy code under `contact_selection`.

### 6.3 Reference timing is diagnostic, not a generated window

Record raw and fractional D1 crossing times for A and B, but do not center
generated event windows on them. For generated trajectories, A and B are active
throughout frames `1..horizon`; the state machine enforces order and the common
deadline. This deliberately permits an enhanced sampler to complete a route
substantially earlier than the conventional reference observation.

Reference crossing fractions and equal-thirds summaries may be reported only
as route-shape diagnostics. They are not success constraints.

## 7. Ordered success semantics

The ordered program is:

```text
A persists for at least two generated frames
then B persists for at least two generated frames
then the target persists for at least two generated frames
all completed by the frozen horizon deadline
```

The same frame cannot complete two successive stages. An earlier stable target
hit is a success; target membership is not restricted to the final frame. The
completion frame of the second persistent target hit defines
`time_to_success_frames`. Paths not completing by the horizon are right
censored at the horizon.

Reference and generated persistence both suppress one-frame fluctuations. As
of 2026-09-05, the runtime implements generated multi-frame persistence,
strict ordered stage completion, first-stable target success, deadline
censoring, and time-to-success recording. The DuET unit suite passes before
any confirmatory method outcome has been generated.

## 8. Sequential eligibility gates

Apply these gates per protein in order:

1. **Asset gate**: three independent, topology-consistent references exist.
2. **Role gate**: D1/D2/H are assigned deterministically from the predeclared
   official case before method outcomes.
3. **Transition gate**: D1 has a stable, initial-structure-matched transition
   path; its duration is recorded but need not match generated duration.
4. **Nominal challenge gate**: lag/horizon and the `[1,10]` first-passage
   compression range are frozen, with acceptable one-step calibration.
5. **Route gate**: one persistent A-before-B pair exists on D1.
6. **D2 support status**: report recurrence, heterogeneity, or absence; D2
   recurrence strengthens the claim but is not required.
7. **H diagnostic status**: report whether H contains a comparable path;
   absence is not a failure and only makes held-out fidelity unavailable.
8. **Frozen support gate**: unsteered generation reaches A, A-then-B, and full
   success at nonzero but nonsaturated rates under predeclared bounds.
9. **Checkpoint gate**: predicted-clean checkpoint scores have positive
   continuation selection value at event-relevant steps.
10. **Validity gate**: generated structures and whole paths satisfy geometry,
    clash, and per-step continuity checks.

Only after these gates are frozen may Complete Nested and DuET be compared.

## 9. Baseline suite for all protein benchmarks

Use the following five-method suite for future main experiments. Citation
numbers refer to the manuscript bibliography.

| Method | Outer physical-time SMC | Inner diffusion-time steering | Cross-clock weighting | Prior-work interpretation |
|---|---:|---:|---|---|
| Frozen full-history rollout | No | No | - | Default MD-surrogate sampling [1] |
| Outer-only trajectory SMC | Yes | No | - | Path-time twisting related to [7] |
| Inner-only FKC/TDS | No | Yes | - | Diffusion-time steering based on [3,4,11] |
| Complete-frame Nested IS/SMC | Yes | After complete next-frame generation | Proper normalizer | Controls whether internal diffusion-time allocation is needed |
| **DuET-MD** | **Yes** | **Yes** | **Uses** $\widehat Z_t/\psi_{t-1}$ | Properly weighted NSMC interface [5,6] across two clocks |

Within a protein, all methods must use the same frozen model, physical lag,
horizon, task, evaluation seeds, and total decoder-population budget. With a
budget $B=K M$, use $B$ independent rollouts for Frozen and Outer-only,
`K=1, M=B` for Inner-only, and the same factorization `K x M` for Complete
Nested and DuET. Report measured decoder NFE and GPU-hour in addition to the
nominal budget.

The default main-experiment budget is **`B=64`**. Use `K=64, M=1` for Frozen
and Outer-only, `K=1, M=64` for Inner-only, and **`K=16, M=4`** for Complete
Nested and DuET. This choice replaces the exploratory budget-16 setting, whose
small outer population gave poor power and high miss variance for rare ordered
events. It is frozen before running new confirmatory proteins.

Budget 64 is a resource and power choice, not a claim that a larger total
population automatically makes DuET superior. On the 7LP1 development task,
performance depended strongly on the `K/M` allocation: `K=16, M=4` was the
most favorable tested allocation, while `K=8, M=8` at the same total budget was
not. Accordingly, allocation sensitivity must be disclosed and this default
must not be retuned separately for each confirmatory protein after outcomes are
seen.

The primary mechanistic comparison is DuET versus Complete Nested: they have
the same `K`, `M`, complete-frame selection rule, and decoder NFE, while
differing in whether allocation occurs inside diffusion time. Methods not in
the table, such as `naive_dual`, are diagnostic ablations rather than members
of the primary baseline suite.

## 10. Cross-protein evaluation for the paper

The independent statistical unit is the **protein benchmark**, not an output
trajectory. The frozen candidate and reserve order is recorded in
`docs/DUET_ATLAS_CONFIRMATORY_COHORT_SCREEN_V1.md`. For approximately ten
proteins:

1. report each protein separately;
2. compute the paired DuET-minus-Complete-Nested difference per protein;
3. report macro mean and median across proteins;
4. bootstrap proteins, not individual output paths, for confidence intervals;
5. report how many candidates were unresolved at each gate;
6. stratify or annotate `single_reference_route` versus
   `independently_supported_route`;
7. keep decoder NFE, K/M, seeds, sampler, and checkpoint matched within each
   protein.

The primary acceleration-aware metrics are:

1. weighted cumulative success probability
   $P(T_{success} \leq t)$ over the common nominal-time grid;
2. cumulative count or rate of unique pre-resampling successful lineages;
3. right-censored time to success, summarized per protein by a fixed-horizon
   restricted mean time to success or another preregistered survival summary;
4. final-deadline success probability mass and unique success rate;
5. success per measured decoder NFE and per GPU-hour.

Frozen versus each steering method addresses whether biased path selection
finds the frozen route earlier or more often. DuET versus Complete Nested is
the primary mechanistic comparison at matched `K`, `M`, and decoder NFE.

Structural validity, maximum aligned frame-to-frame displacement, clashes,
diversity, ancestry, and ESS remain required secondary/falsification metrics.
Path distance and final-frame fidelity against H are optional secondary
metrics: report them only when H contains a comparable transition, and never
turn their absence or disagreement into an automatic success/failure gate.

Do not pool all trajectories from all proteins and treat them as independent
samples. A small protein producing 80 paths must not outweigh a large protein
producing fewer paths in the cross-protein result.

## 11. Separation from case studies

This benchmark family measures average algorithmic performance on automatically
constructed route programs. Later pocket-opening, cryptic-pocket, allosteric,
or mutation case studies answer different biological questions and may use
expert-defined observables. They must be reported in a separate section and
must not be pooled into the ten-protein benchmark average.

These time-to-success results compare methods under the same frozen surrogate
and nominal frame clock. They do not estimate unbiased physical kinetics,
MFPT, transition rates, or free-energy differences.

## 12. Implementation and template

- Builder: `src/confmh/duet/atlas_programs.py`
- Runtime program semantics: `src/confmh/duet/programs.py`
- Optional reference-path comparison: `src/confmh/duet/reference_eval.py`
- Copyable configuration:
  `configs/duet/templates/prepare_atlas_route_benchmark_v3.yaml`

The legacy runtime remains available so existing v2/original-v3 catalogs and
exploratory outputs remain reproducible. Route v3.1 now implements:

- resolve original R1/R2/R3 identifiers into frozen D1/D2/H roles;
- remove the reference/generated duration-matching gate and record the
  preregistered nominal-compression challenge instead;
- use broad generated A/B availability with strict ordered state transitions;
- require multi-frame persistence for generated A, B, and target membership;
- accept the first stable target hit by the deadline and record time to
  success;
- treat H comparison as optional secondary evidence.

The state machine and builder changes above were completed on 2026-09-05 and
the DuET unit suite passed (`61 passed`). The copyable v3.1 template is now an
executable preparation template. Materialize per-protein paths and run:

```bash
duet-md prepare-programs \
  --config configs/duet/<protein>_route_v3_1.yaml
```

Inspect `selection_report.yaml` before any model comparison. It contains the
original replicate-to-role mapping, source frame indices, raw transition and
first-passage times, nominal-compression factor, route support status, residue
pairs, thresholds, persistent crossings, deadline, and H diagnostic status
needed for audit and preregistration.

## 13. Differences from earlier protocols

| Component | 7LP1 v2 / original route v3 | Route v3.1 |
|---|---|---|
| Reference roles | Literal R1/R2 design, R3 held out | D1 is official-case replicate, D2 lowest remaining, H final remaining |
| Path start | Earliest eligible frame / stable last exit | Stable last exit for route geometry |
| Reference time use | Resampling or narrow fixed-lag matching | Record first passage and transition-path time; no one-to-one generated mapping |
| Challenge duration | Match observed duration closely | Predeclare nominal compression in `[1,10]` when first passage is available |
| A/B timing | Narrow reference-derived windows | Broad availability; strict A then B by common deadline |
| Persistence | Reference crossing only | Persistent generated A, B, and target episodes |
| Terminal | Late-window or final-frame target | First persistent target hit by the deadline |
| Held-out replicate | Required R3 fidelity/gate | H is optional secondary fidelity evidence |
| Scientific claim | Normalized/time-matched route steering | Earlier/more frequent biased route discovery under a frozen surrogate |

Existing 6J56, 7LP1 v2, and original route-v3 results must not be silently
relabelled as v3.1. A v3.1 benchmark requires rebuilding the task and using
fresh evaluation seeds.
