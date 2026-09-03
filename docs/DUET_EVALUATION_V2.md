# DuET-MD evaluation v2

## Purpose

This protocol tests a conditional claim: diffusion-time resampling is useful
when a locally rare next-frame event is predictable before the reverse process
finishes, while physical-time particles remain useful for chronological event
completion and route diversity. It does not assume that DuET must beat
complete-frame nested SMC on every task.

## Evidence that motivated the redesign

- In the 6J56-A horizon-24 run at seed 20260913, Complete nested and DuET each
  contained one successful path before final outer resampling. Their success
  weight masses were 0.4385 and 0.4167. The reported 4/8 versus 3/8 output gap
  was mainly the number of copies created by the final resampling operation.
- At seed 20260914, K8 x M4 Complete nested contained one successful path with
  success mass 0.1947; DuET contained none. DuET failed at event A, before B or
  the terminal event, while its inner ESS stayed close to 4/4. The checkpoint
  therefore had little selection signal for this rare event in that seed.
- Across 64 frozen 6J56-A design paths, event A was reached by 3/64 and A then B
  by 2/64. The task is rare but not outside the observed frozen-surrogate
  support. Increasing inner discovery budget is tested before relaxing the
  reference-derived threshold.
- The fixed 6J56-A v3 event B target is already satisfied by the initial
  structure, and only one R1/R2 transition segment supports the task. Its
  current runs are therefore allocation diagnostics, not formal evidence for
  a two-event molecular transition.
- The original 7LP1-A endpoint task was saturated by particle methods and its
  windowed task was unresolved. On the ordered task, K4 x M4 DuET succeeded in
  one of three seeds, while the DuET allocation K2 x M8 succeeded in all three.
  This points to insufficient inner discovery at M=4, but K2 x M8 retained only
  one surviving outer ancestor on average and is not by itself evidence for a
  two-clock advantage.
- The original 7LP1-A ordered contact selector compared median crossing times.
  One selected design replicate had B before A, and generated windows excluded
  some observed crossings. Those results remain exploratory. The v2 selector
  now requires per-design-segment A-before-B support and records its stability.
- The corrected 7LP1-A v2 pair is an opening of distance (0, 7) in frames 1--5
  followed by a closing of distance (25, 36) in frames 19--23. The two R1/R2
  ordered gaps are 22 and 20 frames; initial-to-threshold margins are 0.957 and
  0.182 nm. The A event was reached by 9/48 old frozen design paths, making it
  nontrivial but observable. A fresh horizon-24 frozen gate remains required
  before the method pilot.

## Frozen protocol selection

Protocol selection uses only ATLAS R1/R2, frozen-surrogate design rollouts, and
checkpoint diagnostics. R3 remains held out for path fidelity. Method outcomes
from evaluation seeds must not be used to modify task thresholds.

An ordered task is eligible only if:

1. at least two R1/R2 transition segments are available;
2. the selected A-before-B pair satisfies the configured minimum gap in every
   required design segment;
3. its event windows include the observed R1/R2 crossings;
4. frozen design rollouts show nonzero but nonsaturated reachability;
5. predicted-clean checkpoint scores have positive continuation-marginal
   selection gain and a median Spearman rank correlation of at least 0.5 at
   event-relevant physical steps.

Tasks that fail these checks are reported as unresolved, not replaced after
looking at DuET results.

For the fresh 7LP1-A v2 gate, two predeclared frozen seeds with 48 independent
paths each are pooled only after retaining the per-seed counts.  The task is
eligible for a method pilot when all of the following hold:

1. event A is reached by 5--50% of frozen paths;
2. at least two independent paths reach A and then B in the required order,
   while the ordered A-to-B rate remains at or below 25%;
3. at least one independent path also reaches the terminal basin, while full
   joint success remains at or below 20%;
4. both frozen seeds retain structural validity and neither seed alone
   accounts for every A-to-B discovery.

If A-to-B is supported but the terminal event alone fails, a terminal-only
redesign may use R1/R2 and these frozen design paths before evaluation seeds
are launched.  Contact thresholds and event windows are never changed after
Complete nested or DuET outcomes are observed.

## Controlled methods and compute

The main comparison retains:

- frozen full-history rollouts;
- outer-only trajectory SMC;
- inner-only diffusion-time steering;
- complete-frame nested SMC;
- DuET-MD.

Best-of-budget and Naive Dual are omitted from the new main evaluation because
they are not needed for the central empirical claim. Complete nested is the
primary strong baseline.

Within each protein-task unit, every comparison uses the same decoder NFE,
checkpoint, SDE base sampler, temporal program, and seed family. Complete
nested and DuET must also be compared at the same K/M allocation. The first
allocation tests are:

- 6J56-A, population budget 32: K8 x M4 and K4 x M8;
- 7LP1-A, population budget 32: K8 x M4 and K4 x M8.

The original 7LP1-A budget-16 grid is retained only as exploratory history.
With a frozen event-A rate near 10%, K4 x M4 has high miss variance, while
K2 x M8 cannot retain enough physical-time lineages to support the two-clock
claim. K4 x M8 is therefore the inner-discovery pilot and K8 x M4 is its
outer-diversity control. If either allocation passes the checkpoint and
preterminal-resampling gates, K8 x M8 at population budget 64 is the
predeclared confirmation setting.

The primary 7LP1-A horizon remains 24 frames. The frozen paths place event A
at frames 3--5, event B at 19--20, and successful terminal entries at 21--24,
so this horizon already exposes a long physical-time dependency rather than
truncating it. Increasing the horizon after seeing method outcomes is not
allowed. A 48-frame, half-lag temporal-resolution ablation may be run only
after the primary K/M setting is frozen; it must rederive windows from R1/R2,
hold R3 out, and match decoder NFE across methods.

Checkpoint progress remains 0.90 for the 6J56-A development task. For 7LP1-A
v2, two successful frozen histories were diagnosed at progress 0.90 and 0.95
before any method outcome was generated. Progress 0.90 produced a negative
rank correlation and negative selection gain at one event-B entry, whereas
0.95 restored positive rank correlation and yielded positive selection gain
at every diagnosed event-relevant step. The fixed 7LP1-A v2 method checkpoint
is therefore 0.95.

## Primary and secondary metrics

The primary program metrics are measured before final output resampling:

1. unique successful paths per outer population;
2. total normalized weight mass on successful paths;
3. weighted and unweighted event-stage curves over physical time;
4. success per decoder NFE and GPU-hour.

Post-resampling success rate is retained as generated output yield, not as the
number of independent successful discoveries. One successful path copied four
times is one discovery, not four independent observations.

Secondary gates are:

- held-out R3 path distance and endpoint CA RMSD;
- intermediate coverage and unspecified-contact similarity;
- structural validity;
- path diversity and surviving initial ancestors;
- inner and outer ESS;
- the number and timing of outer resampling events before the terminal frame;
- checkpoint-to-continuation rank correlation and selection gain.

A success increase does not support DuET if validity or held-out fidelity
degrades materially, or if all outputs descend from one path when route
multiplicity is part of the claim.
Runs with no preterminal outer resampling are not treated as evidence that the
physical-time SMC clock improved sampling, even if they use `K > 1`.

## Statistical design

Use at least five predeclared evaluation seeds per protein-task-method-allocation
cell after development is frozen. A trajectory copy is not an independent
statistical unit. Report raw seed values, per-protein-task paired differences,
mean and median differences, and paired bootstrap intervals across eligible
protein-task units. 6J56-A has one R1/R2 transition segment and remains a
development/falsification case rather than standalone generalization evidence.

## Decision rule

DuET supports the intended empirical claim only if it improves stable success
mass or unique discovery over Complete nested at matched K/M and NFE on more
than one eligible protein-task unit, while preserving fidelity and at least two
meaningful outer lineages where the task is intended to test both clocks. If
Complete nested matches or exceeds it, the result supports complete-frame
nested steering but not a diffusion-time resampling advantage.

## Final checkpoint refinement

The five-seed 7LP1-A population study selected K16 x M4 as the last development
allocation. Before final output resampling, DuET reached a mean unique-success
rate of 0.600 versus 0.4625 for Complete nested, and success weight mass of
0.8908 versus 0.8536. The paired five-seed bootstrap intervals still included
zero, so these values identify a promising mechanism setting rather than a
finished superiority claim. Both methods retained 100% endpoint structural
validity. DuET's held-out path distance was about 5% larger, which must not be
hidden by the success comparison.

The original implementation performs one predicted-clean inner resampling per
generated physical frame, at reverse-diffusion progress 0.95. This operation is
repeated for every physical step and outer particle; it is not a single global
operation for the full trajectory. Because only about ten of 199 executed
reverse steps remain after the 0.95 checkpoint, selected copies have little
time to develop independent SDE continuations.

The final, bounded refinement adds an optional two-checkpoint schedule. Frozen
successful histories were used to compare progress 0.75, 0.85, 0.90, and 0.95.
Progress 0.75 had event-relevant rank correlations as low as 0.26--0.31, while
0.85 retained positive selection gain at every diagnosed step and generally
had rank correlation at or above 0.67. The fixed development schedule is
therefore [0.85, 0.95]. Each checkpoint uses an incremental potential ratio,
resamples the complete diffusion state, and assigns independent continuation
noise. The endpoint correction closes the potential telescope. A ConfRover
smoke run confirmed identical decoder NFE, zero telescoping error, and 100%
structural validity.

No further checkpoint or K/M search is allowed on this 7LP1-A task. The final
three-hour evaluation consists of:

1. fresh seeds 20260925--20260926 comparing K16 x M4 Complete nested against
   two-checkpoint DuET at matched decoder NFE;
2. seeds 20260920--20260923 running only two-checkpoint DuET, paired with the
   already completed one-checkpoint DuET and Complete nested outputs;
3. reporting pre-resampling unique success, success weight mass, validity,
   held-out fidelity, genealogy, decoder NFE, and wall clock.

The two-checkpoint variant is retained only if it improves the pre-resampling
success metrics over one-checkpoint DuET without material validity or held-out
fidelity loss, and the fresh-seed comparison does not reverse that direction.
Otherwise the one-checkpoint implementation remains the final method and the
negative refinement result is reported.
