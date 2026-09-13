# PVB fast-folding TPS — first-pass root cause analysis

54/54 cells complete. PVB-ATLAS frozen, `sde_step=20`, T=32, K·M=64,
Trp-cage / BBA / BBL, seeds 199/211/223. Four DuET checkpoint timings plus
Frozen (K64,M1) and Complete Nested (K8,M8) at (0.75, 0.90).

## What the matrix shows

Means over three seeds.

**DuET timing sweep** — wBB (A) / whole-path validity / native-contact Q:

| protein | 0.25+0.50 | 0.50+0.75 | 0.75+0.90 | 0.25+0.50+0.75 |
| --- | --- | --- | --- | --- |
| Trp-cage | 3.92 / 1.00 / 0.244 | 3.99 / 1.00 / 0.256 | 3.76 / 1.00 / 0.256 | 3.95 / 1.00 / 0.295 |
| BBA | 5.09 / 1.00 / 0.415 | 5.48 / 1.00 / 0.430 | 4.88 / 1.00 / 0.430 | 4.82 / 1.00 / 0.444 |
| BBL | 16.01 / 0.04 / 0.286 | 16.32 / 0.38 / 0.230 | 15.88 / 0.21 / 0.262 | 15.40 / 0.25 / 0.274 |

**Method comparison at (0.75, 0.90)** — wBB / validity / Q / anytime hits:

| protein | Frozen (K64) | Complete Nested (K8x8) | DuET (K8x8) |
| --- | --- | --- | --- |
| Trp-cage | 6.09 / 0.95 / 0.333 / 0.0 | 3.88 / 1.00 / 0.269 / 0.3 | 3.76 / 1.00 / 0.256 / 0.0 |
| BBA | 7.67 / 1.00 / 0.437 / 5.0 | 5.42 / 1.00 / 0.393 / 0.0 | 4.88 / 1.00 / 0.430 / 0.0 |
| BBL | 18.69 / 0.14 / 0.306 / 0.7 | 16.76 / 0.17 / 0.266 / 0.0 | 15.88 / 0.21 / 0.262 / 0.3 |

TICA hits are near-absent: 6 of 54 cells have any, 14 hits in total.

## Three findings

### 1. Steering works on its own reward and nothing else

DuET lowers backbone RMSD against Frozen in every protein, by 2.3 A on
Trp-cage, 2.8 A on BBA, 2.8 A on BBL, consistently across seeds. That is the
reward it optimises, so it is the expected result and not evidence of recovery.

Native-contact Q does **not** follow: Frozen reaches 0.333 / 0.437 / 0.306
against DuET's 0.256 / 0.430 / 0.262. Lower RMSD with no gain in native contacts
means the population is being pulled toward a low-RMSD region that is not more
native-like — RMSD and contact formation have come apart.

### 2. Frozen finds more TICA hits than DuET, and the population sizes explain
only part of it

BBA: Frozen 5.0 anytime hits from 64 paths (7.8%), DuET 0 from 8 (0%).
Trp-cage and BBL are the same direction at smaller counts.

Frozen has 8x the paths, so a per-path comparison is what matters, and even
per-path Frozen is ahead. The reward concentrates K=8 particles on one low-RMSD
basin, while 64 independent rollouts cover more of the space; the TICA target is
defined on backbone torsions, not RMSD, so breadth finds it and depth does not.

**This is the central negative result: steering toward whole-protein RMSD moves
the population away from the TICA target, not toward it.**

### 3. BBL's validity collapse is real, not a gate artifact

The first hypothesis was arithmetic: whole-path validity fails if any single
peptide bond in any frame leaves [1.0, 1.7] A, and BBL offers 46 bonds x 33
frames against Trp-cage's 19 x 33, so a size gradient could appear with no
difference in per-bond quality. Measuring the per-bond rate refutes that:

| protein | residues | bonds measured | violating | per-bond rate | per-path | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trp-cage | 20 | 195,624 | 9 | 0.00005 | 0.029 | 1.91 A |
| BBA | 28 | 277,992 | **0** | 0.00000 | 0.000 | — |
| BBL | 47 | 473,616 | 627 | **0.00132** | 0.811 | **5.76 A** |

BBL's rate is 26x Trp-cage's, far beyond the 2.4x chain-length ratio, and a
5.76 A C-N distance is a severed backbone, not a stretched bond. BBA, in the
middle by length, has zero violations — so this does not track chain length.

What does track it is how far the start structure sits from anything PVB was
trained on:

| protein | Rg start | Rg folded | ratio | d0 |
| --- | ---: | ---: | ---: | ---: |
| Trp-cage | 10.19 | 7.01 | 1.45 | 7.20 A |
| BBA | 13.13 | 9.09 | 1.44 | 8.07 A |
| **BBL** | **23.45** | **10.35** | **2.27** | **18.07 A** |

Trp-cage and BBA start 1.45x expanded; BBL starts 2.27x expanded, at Rg 23.45 A
for a protein whose folded Rg is 10.35 A. PVB-ATLAS is fine-tuned on ATLAS
equilibrium MD, which contains folded and near-folded states — an essentially
extended chain is outside that distribution, and the bridge has no reason to
keep a backbone intact there. The single-transition smoke test passed because it
starts from that structure but only takes one step; the damage accumulates over
32.

## Hypotheses and how to test them

**H1 — BBL fails because its start structure is out of distribution, not
because of its length.**
Predicts: starting BBL from a less expanded structure restores validity to
Trp-cage/BBA levels, with no change to the model.
Test: rerun BBL Frozen from a start structure at Rg ~14 A (interpolated toward
the folded state, or an intermediate frame), same seeds. 3 cells.
Falsified if validity stays near 0.2 at matched Rg ratio.

**H2 — RMSD steering actively moves the population away from the TICA basin.**
Predicts: TICA distance to target increases with steering strength while RMSD
decreases; Frozen's per-path hit rate exceeds DuET's at matched path count.
Test: subsample Frozen to 8 paths x 8 resamples to match DuET's population, and
compare per-path hit rate; separately, run DuET at reward coefficient 4 and 16
on BBA and plot RMSD against TICA distance. 6 cells, no new code.
Falsified if the hit-rate gap closes once populations are matched.

**H3 — Checkpoint timing cannot be discriminated at this horizon.**
The four timings differ by less than the seed scatter on every protein
(Trp-cage 3.76-3.99, BBA 4.82-5.48, BBL 15.40-16.32). Either timing genuinely
does not matter for PVB, or 3.2 ns leaves too little signal to separate them.
Test: repeat the sweep on BBA at a longer horizon only if H2 shows the metric
can move at all. Deferred behind H2 — sweeping a metric that cannot respond
measures nothing.

## Order

H2 first: it decides whether the primary metric is usable, and H3 is meaningless
until it is answered. H1 in parallel, since it is independent and cheap.
