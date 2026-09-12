# DuET-MD ATLAS confirmatory cohort screen v1.1

Frozen on 2026-09-05, before running Complete Nested or DuET on any protein in
this list.

## Purpose

This document pre-registers the protein screening order for the approximately
ten-protein DuET-MD route-v3.1 benchmark. It is a screening cohort, not a claim
that all ten primary candidates already pass the transition, route, frozen
support, checkpoint, and validity gates.

The earlier 6J56-A and 7LP1-A systems are excluded because they are development
proteins that have already influenced task and hyperparameter choices.

## Outcome-blind selection rule

Starting from the official ConfRover conformation-interpolation case table:

1. require a matching ATLAS metadata entry and downloadable 10,000-frame
   protein-only archive;
2. exclude 6J56-A and 7LP1-A;
3. exclude peptide-scale systems by requiring at least 70 ATLAS residues;
4. retain official stride 128 or 256 cases, avoiding stride 512/1024 cases so
   the physical-time grid stays closer to the existing 6J56/7LP1 setup and the
   100 ns ATLAS reference limit;
5. rank by ATLAS residue count, then by case ID.

The resulting first ten cases are the primary screening cohort. Reserves may
replace a primary candidate only after a pre-method eligibility-gate failure.
The failed candidate and reason remain in the attrition table. No replacement
is allowed because of Complete Nested or DuET performance.

## Primary screening cohort and frozen replicate roles

| Priority | Official ConfRover case | ATLAS chain | Residues | Stride | D1 | D2 | H | Archive |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | `6JV8-A-R3F1000S256` | `6jv8_A` | 76 | 256 | R3 | R1 | R2 | available |
| 2 | `6IN7-A-R3F5000S128` | `6in7_A` | 81 | 128 | R3 | R1 | R2 | available |
| 3 | `7BWF-B-R3F3000S128` | `7bwf_B` | 92 | 128 | R3 | R1 | R2 | available |
| 4 | `6TLY-A-R2F1000S256` | `6tly_A` | 99 | 256 | R2 | R1 | R3 | available |
| 5 | `7S86-A-R3F5000S256` | `7s86_A` | 99 | 256 | R3 | R1 | R2 | available |
| 6 | `6RRV-A-R1F1000S256` | `6rrv_A` | 127 | 256 | R1 | R2 | R3 | available |
| 7 | `6LUS-A-R2F3000S128` | `6lus_A` | 133 | 128 | R2 | R1 | R3 | available |
| 8 | `6GUS-A-R2F7000S256` | `6gus_A` | 155 | 256 | R2 | R1 | R3 | available |
| 9 | `6Q9C-A-R3F3000S256` | `6q9c_A` | 155 | 256 | R3 | R1 | R2 | available |
| 10 | `6OVK-R-R2F7000S128` | `6ovk_R` | 219 | 128 | R2 | R1 | R3 | available |

## Predeclared reserves

| Reserve order | Official ConfRover case | ATLAS chain | Residues | Stride | D1 | D2 | H | Archive |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | `7RM7-A-R3F3000S128` | `7rm7_A` | 228 | 128 | R3 | R1 | R2 | available |
| 2 | `7AEX-A-R3F5000S256` | `7aex_A` | 275 | 256 | R3 | R1 | R2 | available |
| 3 | `7P46-A-R3F5000S256` | `7p46_A` | 282 | 256 | R3 | R1 | R2 | available |
| 4 | `6QJ0-A-R1F1000S256` | `6qj0_A` | 412 | 256 | R1 | R2 | R3 | available |
| 5 | `7P41-D-R2F3000S256` | `7p41_D` | 448 | 256 | R2 | R1 | R3 | available |
| 6 | `6P5X-B-R1F3000S256` | `6p5x_B` | 457 | 256 | R1 | R2 | R3 | available |

## Replicate-role rule

The original R1/R2/R3 labels identify simulation seeds; they are not fixed
statistical roles. For every case, role assignment is frozen before any method
outcome is inspected:

1. `D1` is the replicate encoded in the official ConfRover case ID;
2. `D2` is the lowest-index remaining replicate;
3. `H` is the final remaining replicate.

For example, `6JV8-A-R3F1000S256` makes original R3 the D1 route-discovery
trajectory, original R1 D2, and original R2 H. In the ConfRover interpolation
benchmark, `R3F1000S256` means that structures were drawn from original R3 at
frames 1000, 1256, and so on through frame 3048 to define a nine-frame
interpolation case. For DuET-MD, that case identifies the initial structure,
endpoint challenge, and an observed reference route; the intervening MD
coordinates are not supplied to the generator during rollout.

D1/D2 may define PCA, basins, A/B, reward scale, and deadline. H defines none
of them. If H has a comparable transition, it provides optional secondary
path-fidelity evidence. If it does not transition, fidelity is `unavailable`,
not failed; if it follows another valid route, report route heterogeneity.

## Availability verification

On the freeze date, every primary and reserve chain returned ATLAS metadata and
its `/ATLAS/protein/{pdb_chain}` endpoint returned HTTP 206 for a one-byte range
request with `application/octet-stream`. This verifies remote archive presence
without downloading the full archives.

Relevant primary sources:

- ConfRover paper: https://arxiv.org/abs/2505.17478
- ConfRover implementation: https://github.com/ByteDance-Seed/ConfRover
- ATLAS paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10767941/
- ATLAS API: https://www.dsimb.inserm.fr/ATLAS/api/docs

## Preparation status

The v3.1 runtime changes were implemented and the DuET unit suite passed on
2026-09-05 (`61 passed`). Static construction then produced 10 frozen tasks:
seven primary candidates plus reserves 1--3. `6in7_A` failed the accelerated
nominal-challenge gate; `6lus_A` and `6ovk_R` failed the predeclared D1
last-exit transition gate. These failures remain in the attrition report and
no method outcome was inspected. See
[DUET_ATLAS_COHORT_PREPARATION_2026-09-05.md](DUET_ATLAS_COHORT_PREPARATION_2026-09-05.md)
for the final cohort, event definitions, and launch matrix.

This preparation does **not** by itself authorize a production launch. Each
frozen task still requires its unsteered reachability, checkpoint-predictivity,
and generated whole-path validity support gates on the experiment GPU before
Complete Nested or DuET outcomes are inspected.
