# Phase B4 cryptic-pocket failure analysis

Date: 2026-09-10  
Scope: PRMT6, SMARCA2, PI3Kalpha B4 DuET-only production runs

## Bottom line

B4 failure is unlikely to be explained by one bad reward coefficient or a
broken Feynman--Kac implementation. Runtime checkpoint/telescoping audits pass,
and increasing `a=16` to `a=32` lowers the target RMSD in all three proteins.
The dominant problem is that the algorithm selects partial or geometrically
degraded local-RMSD improvements that are not equivalent to a physically open
pocket. Because DuET is resampling-based rather than gradient-based, it cannot
create a transition that is absent from the small proposal population.

## Priority-ranked causes

### 1. Generated-frame geometry is already degraded, while the sampling reward does not penalize it

Confidence: high.

- Every generated path-frame in all six cells contains at least one adjacent
  CA distance of at least 4.5 A.
- The median generated frame contains approximately 15--17 such CA gaps.
- Approximately 6.6--7.8% of consecutive peptide C--N distances exceed 2.0 A.
- The input structures have normal geometry: maximum adjacent CA is only
  3.90--3.93 A and C--N is approximately 1.34 A.
- These deviations appear from the first generated transition and occur across
  hundreds of residue pairs, rather than only around the rewarded pocket loop.

The current `5.5 A` hard validity test catches only extreme breaks. Thus a path
can be classified as valid despite many 4.5--5.4 A CA gaps. Validity is also
evaluation-only, not part of inner/outer selection. Strong reward can therefore
select a locally closer but physically worse shortcut.

PRMT6 a=32 is the clearest example: it reaches a lower local RMSD, but a
frame-4 CA gap of 5.559 A is copied through outer ancestry into all four retained
paths. This invalid population cannot count as success.

This issue is present at similar rates for a=16 and a=32, so its origin is more
likely the unrelaxed ConfRover proposal/output interface than reward strength
alone. An isolated Frozen one-step audit is still required to distinguish an
upstream decoder property from an adapter/reconstruction issue.

### 2. Local backbone RMSD is not an identifiable proxy for physical pocket opening

Confidence: high.

The reward uses only N/CA/C coordinates from one selected loop. It does not
reward ligand-space clearance, side-chain rearrangement, distal gate movement,
or a physically valid path. B4 shows that local RMSD and pocket opening separate.

- SMARCA2 a=32 gives the strongest valid local approach (`d/d0=0.5335`) but
  the best path still has a minimum protein--J7G distance of 0.790 A. The holo
  value is 2.676 A.
- In SMARCA2, the rewarded 852--860 loop partially leaves the ligand space,
  while the strongest final clashes shift to residues 884, 892, and 894 outside
  the reward region. Across retained path-frames, local RMSD and the number of
  ligand atoms within 1.5 A are essentially uncorrelated (Spearman about -0.06
  for a=32).
- PI3Kalpha a=32 retains sub-Angstrom ligand overlaps at residues 937--939 and
  additional close contacts at residues 1021--1022 outside the reward region.

The reward is therefore underdetermined: many conformations can reduce loop
RMSD without producing the desired sterically open pocket.

### 3. The desired transition lies near or outside the effective proposal support

Confidence: high for the present budget; exact attribution to the model versus
task requires unresampled candidate logging.

For the best final paths, projection onto the direct apo-to-holo moving-region
displacement is only partial:

| Protein/cell | Projection progress | Direction cosine | Interpretation |
|---|---:|---:|---|
| PRMT6 a=16 valid | 0.30 | 0.54 | weak and substantially off-direction |
| PRMT6 a=32 invalid | 0.60 | 0.74 | stronger direction, but invalid shortcut |
| SMARCA2 a=32 valid | 0.53 | 0.93 | coherent target-directed half-transition |
| PI3Kalpha a=32 valid | 0.30 | 0.61 | small, noisy partial transition |

SMARCA2 a=32 reaches projection 0.58 by T=12 and returns to 0.53 at T=24;
simply extending the same trajectory did not continue toward the endpoint.
PI3Kalpha similarly plateaus around 0.26--0.30 after T=8. This is more
consistent with proposal saturation than with a horizon that is merely a few
frames too short.

DuET here performs weight-based resampling only. It does not apply a reward
gradient to coordinates. Increasing `a` changes which existing candidate is
replicated, but cannot synthesize an opening motion that none of the four inner
candidates proposed.

### 4. K=4 outer SMC rapidly loses trajectory diversity

Confidence: high.

The four reported final paths are not four independent histories. Coordinate
genealogy shows that retained trajectories share one exact prefix for most of
the horizon:

| Cell | Final retained paths remain one identical lineage through |
|---|---:|
| PRMT6 a=16 | T=14 |
| PRMT6 a=32 | T=20 |
| SMARCA2 a=16 | T=19 |
| SMARCA2 a=32 | T=19 |
| PI3Kalpha a=32 | T=17 |

At a=32 there are 9--12 outer resampling events. PI3Kalpha ends with weight
`[0.0034, 0.0104, 0.9860, 0.0003]` and ESS 1.0286. Thus effective population
size is close to one precisely where multiple opening mechanisms should be
maintained. The same mechanism also copies PRMT6's early invalid frame into
the entire final population.

### 5. Some apo/holo task definitions are not clean local transitions

Confidence: high for SMARCA2 and PI3Kalpha; medium as a cross-protein cause.

The starting structures are AlphaFold predictions, not experimentally verified
apo basins.

| Protein | Start-to-holo fixed-core RMSD | Main task-definition issue |
|---|---:|---|
| PRMT6 | 1.15 A | reasonably matched core; failure is mainly proposal/reward/validity |
| SMARCA2 | 5.03 A | the supposed fixed core itself differs extensively between AF start and 6EG3 |
| PI3Kalpha | 0.79 A | core is matched, but holo lacks residues 943--950 in the moving loop |

SMARCA2 is therefore not a simple nine-residue loop-opening task under the
current alignment: most of the construct changes by several Angstrom. Stronger
selection improves the loop while the best a=32 path drifts 3.20 A from the
starting core and remains 4.94 A from the holo core. Local-RMSD improvement is
strongly associated with global core drift (Spearman about -0.80).

PI3Kalpha defines its reward on only 19 of the requested 27 residues, split
into 931--942 and 951--957. The unobserved central 943--950 segment is critical
to the physical loop but unconstrained by the holo RMSD, so the endpoint is not
fully specified.

### 6. The proposal population is small for a rare transition

Confidence: medium-high.

Although each run uses 76,416 decoder NFE, the structural choice at one parent
is among only `M=4` candidates, with `K=4` histories. Two late resampling steps
can reallocate those four branches but cannot explore a broad transition tube.
This is a severe diversity bottleneck for a cryptic-pocket event, especially
after repeated outer resampling.

The 75%/90% checkpoint implementation itself is verified. PRMT6's earlier
unresampled diagnostic also showed that these checkpoints can rank final scores.
There is currently no equivalent unresampled checkpoint-predictiveness audit
for SMARCA2/PI3Kalpha, so blaming the exact checkpoint times would be premature.

### 7. Strict binary success and one seed make the outcome look uniformly zero, but are not the main failure

Confidence: high.

The final-two-frame 1.5-A rule is strict, but the closest valid endpoints remain
3.93 A (PRMT6), 4.82 A (SMARCA2), and 4.05 A (PI3Kalpha observed mask), and no
valid anytime hit occurs. Ligand-space clearance also fails. Relaxing only the
persistence rule would therefore not convert these into convincing openings.

One seed prevents uncertainty estimation and could miss a rare success, but
the repeated geometry, reward, and diversity failures across three proteins
make pure bad luck an insufficient explanation.

## Protein-specific diagnosis

### PRMT6

- The endpoint pair is the cleanest of the three at the core level.
- a=16 produces only about 30% target-direction progress and large orthogonal
  motion; residues 158, 163, and 164 become farther from the holo geometry in
  the best valid path.
- a=32 reveals that stronger selection can find more target-directed motion,
  but does so through a frame-4 structural defect that is never penalized and
  is propagated to all descendants.
- Primary issue: proposal geometry plus post-hoc-only validity, followed by
  low diversity.

### SMARCA2

- There is a real coherent local signal: a=32 moves approximately halfway in
  the apo-to-holo loop direction.
- The signal saturates by T=12 and does not reach the endpoint by T=24.
- The fixed core is not fixed (`5.03 A` start-to-holo RMSD), and lower loop RMSD
  accompanies greater global deformation.
- Ligand occlusion is controlled by residues outside 852--860, so the current
  reward cannot uniquely specify opening.
- Primary issue: endpoint/core construction and reward incompleteness, not lack
  of any steering signal.

### PI3Kalpha

- a=32 produces only about 30% target progress and almost total terminal
  ancestry/weight collapse.
- The reward omits the unobserved central eight residues, so it cannot constrain
  the complete opening loop.
- Severe contacts remain at rewarded residues 937--939 and appear at distal
  residues 1021--1022.
- Primary issue: incomplete endpoint definition plus insufficient proposal and
  population diversity.

## Attribution boundary

B4 has no matched Frozen, Complete Nested, or Outer-only cells, and it did not
store all unselected final candidate coordinates for SMARCA2/PI3Kalpha. It is
therefore not possible to assign a numerical fraction of failure to ConfRover,
DuET, reward design, or sampling budget.

What is supported is narrower: DuET's weighting machinery operated as designed
and amplified lower local RMSD, but the current model/task interface did not
produce a diverse population of physically valid, fully open pocket paths.

## Most informative next diagnostics

1. Run a one-transition, unsteered Frozen candidate audit and compare raw
   peptide geometry before any selection. This identifies decoder versus
   selection-induced structural defects.
2. Rebuild the pocket endpoint as a preregistered composite evaluation target:
   local backbone state plus side-chain/ligand-space clearance and core
   preservation. Decide separately whether validity should gate sampling.
3. Redefine SMARCA2 using a genuinely stable local core or a better matched
   experimental apo/holo pair; do not keep the present 5-A core mismatch.
4. Treat PI3Kalpha as exploratory until a complete experimental moving-region
   reference or a justified non-RMSD endpoint is available.
5. Before a large rollout, log all unresampled candidates at 50/75/90/final to
   test whether target-like valid proposals exist and whether checkpoint ranks
   predict final pocket clearance, not only local RMSD.
6. Only after the target and structural gate are frozen should population
   allocation and matched Complete Nested/Outer-only comparisons be revisited.

Coordinate-level raw diagnostics are saved at:

`outputs/phase_b_b4_pymol_2026-09-10/failure_diagnostics.json`
