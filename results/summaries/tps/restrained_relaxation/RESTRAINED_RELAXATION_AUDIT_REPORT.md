# Restrained OpenMM relaxation audit — corrected atom37 mapping

Date: 2026-09-11  
Status: complete; post-hoc diagnostic only.

## Scope and immutability

This audit does not edit or reclassify any Bundle A sampling result.  It
evaluates selected stored coordinates after a target-independent local
relaxation.  Relaxed coordinates were not fed back to ConfRover, the physical
history, the diffusion checkpoints, the reward, or NSMC weights.

The valid result namespace is this directory.  An earlier audit attempt is
preserved remotely but **must not be interpreted**: its atom37-to-topology
mapping exchanged `CB` (index 3) and `O` (index 4).  The corrected mapping is
the same order as the official evaluator: `N, CA, C, CB, O, ...`.

## Fixed procedure

- Force field / solvent / topology: existing Bundle A setup,
  `protein.ff14SBonlysc + implicit/gbn2`, no cutoff.
- All present atoms were movable.  Standard hydrogens were completed only to
  construct the OpenMM topology; they were unrestrained and movable.
- Every topology heavy atom was harmonically restrained to the *same frame's*
  completed generated coordinates, strength 10 kcal mol^-1 A^-2.
  No folded target, holo, ligand, TICA coordinate, or reward was used as a
  restraint reference.
- OpenMM LocalEnergyMinimizer (L-BFGS), force tolerance 10 kJ mol^-1 nm^-1,
  maximum 2,000 iterations.  Energies below exclude restraint energy.
- GPU backend: the H100-compatible OpenMM 8.2 CUDA runtime.  The OpenMM 8.6
  plugin in the prior evaluation environment failed with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`; it was not used for the final audit.

## Frozen path selection

For each of Trp-cage and BBA, and separately for Frozen and DuET, selection
was made before any relaxation: the lexicographically first `(stride, seed,
run, population index)` raw-valid final TICA-hit path and the first raw-valid
path with no generated-frame TICA hit.  Each selected path has 33 frames.
The exact frozen choices are recorded in `trpcage/frozen_path_selection.json`
and `bba/frozen_path_selection.json`.

The small-protein scope therefore contains 8 paths / 264 frame appearances.
Three appearances per protein were exact duplicate ancestry frames and were
computed once and reused: 129 unique Trp-cage frames and 129 unique BBA
frames.  PRMT6 contains every stored generated frame from the Bundle A
official-forward and custom-SDE arms: 16 + 16 = 32 frames.

## Main results

| System | Unique frames | Median physical energy, raw -> relaxed (kJ/mol) | Gross peptide C-N defects, total raw -> relaxed | Heavy clashes <1 A, total raw -> relaxed | Median / max atom displacement (A) | Force tolerance met |
|---|---:|---:|---:|---:|---:|---:|
| Trp-cage | 129 | 7.52e10 -> -1.75e3 | 802 -> 1 | 223 -> 5 | 2.78 / 8.79 | 0 / 129 |
| BBA | 129 | 4.50e11 -> -5.73e3 | 1309 -> 14 | 375 -> 0 | 3.45 / 11.61 | 0 / 129 |
| PRMT6 | 32 | 4.29e13 -> -4.41e4 | 4115 -> 120 | 150 -> 6 | 2.59 / 7.07 | 0 / 32 |

The relaxation substantially lowers the physical potential and removes most
gross C-N and heavy-clash defects.  It is not a small, fully converged repair:
none of the 290 unique generated frames met the 10 kJ mol^-1 nm^-1 force
tolerance by the 2,000-iteration cap.  Some residual bond/angle defects
remain, especially for PRMT6 (bond deviations >0.5 A: 52,256 -> 1,414;
angle deviations >20 degrees: 123,728 -> 3,643).

## Target-state preservation (small proteins)

| System | Raw TICA-hit frames | Relaxed TICA-hit frames | Retained raw hits | Lost raw hits | Newly gained hits |
|---|---:|---:|---:|---:|---:|
| Trp-cage | 42 | 38 | 32 | 10 | 6 |
| BBA | 33 | 25 | 18 | 15 | 7 |

The preselected raw-hit endpoints show the same mixed behavior:

| Protein | Method | Raw final hit -> relaxed final hit | TICA distance raw -> relaxed | Raw-to-relaxed backbone RMSD (A) |
|---|---|---|---:|---:|
| Trp-cage | Frozen | yes -> no | 0.522 -> 1.570 | 0.584 |
| Trp-cage | DuET | yes -> yes | 0.464 -> 0.510 | 0.391 |
| BBA | Frozen | yes -> no | 0.710 -> 0.848 | 0.551 |
| BBA | DuET | yes -> no | 0.703 -> 0.919 | 0.586 |

Normal folded-target controls retain their TICA state after relaxation
(Trp-cage distance 0.0000 -> 0.0009; BBA 0.0000 -> 0.0005; both backbone
RMSD 0.0008 A).  This verifies that the corrected topology/atom37 conversion
and TICA reassessment do not by themselves move a physically valid target out
of its basin.

## Important structural limitation

Raw structures have no chirality inversions under the template-sign check.
After this unconstrained heavy-atom minimization, chirality inversions appear
in 124/129 Trp-cage frames, 128/129 BBA frames, and 31/32 PRMT6 frames.  Thus
the present force field + restraint + 2,000-iteration protocol cannot be used
as a production structural-repair step, even where it removes clashes and
retains a TICA hit.  This is an outcome of the audit, not a post-hoc failure
filter.

## Interpretation

1. The raw surrogate outputs are often repairable enough to remove enormous
   local strain and clashes under a common, target-independent relaxation.
2. That repair is not reliably path-preserving: 10/42 Trp-cage and 15/33 BBA
   raw TICA-hit frames lose the target classification, while others gain it.
   It therefore cannot validate raw path success or be used to compare
   Frozen and DuET success rates.
3. The all-frame failure to reach the force tolerance and widespread chirality
   change make the current relaxation setting a **diagnostic**, not an
   acceptable downstream proposal kernel or an evidence of physical
   transition-path recovery.
4. PRMT6 establishes that local strain can be reduced at larger scale, but it
   has no TICA/holo-success reassessment in this audit; it is geometry and
   computational-feasibility evidence only.

The next scientifically meaningful step, if pursued, is to diagnose why this
force-field/minimizer setup crosses chirality (topology/protonation/constraints
and relaxation implementation) using normal controls and a short, controlled
restraint/constraint ablation.  Do not add relaxation to sampling or change
the existing Bundle A scientific labels based on this audit.

## Raw outputs

- `trpcage/frames.json`
- `bba/frames.json`
- `prmt6/frames.json`
- `trpcage/frozen_path_selection.json`
- `bba/frozen_path_selection.json`

