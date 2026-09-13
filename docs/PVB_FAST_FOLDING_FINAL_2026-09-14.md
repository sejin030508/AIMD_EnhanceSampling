# PVB fast-folding TPS — final report

66 cells, all complete, no failures outstanding. PVB-ATLAS frozen,
`sde_step=20`, T=32, budget K·M=64, Trp-cage / BBA / BBL, seeds 199/211/223.

> **One sentence.** DuET optimises its reward exactly as intended and beats
> Frozen on backbone RMSD by 2.3–2.8 Å in every protein and every seed, but that
> reward is not aligned with the benchmark's success condition: native-contact Q
> does not follow it, and only the unsteered Frozen arm ever reaches the TICA
> target basin.

## Matrix

| Block | Cells | Content |
| --- | ---: | --- |
| Timing sweep | 36 | DuET, 4 checkpoint timings × 3 proteins × 3 seeds |
| Controls | 18 | Frozen (K64,M1) and Complete Nested (K8,M8) at (0.75, 0.90) |
| K/M gradient | 12 | DuET at K2×M32 and K32×M2, BBA and Trp-cage |

Per-cell cost: DuET K8×M8 2–3.5 min, Frozen K64×M1 6–7.5 min. PVB is roughly
40× cheaper per DuET cell than ConfRover.

## Three results

### 1. Steering works, on its own reward only

At (0.75, 0.90), means over three seeds:

| protein | Frozen | Complete Nested | DuET |
| --- | --- | --- | --- |
| Trp-cage wBB | 6.09 | 3.88 | **3.76** |
| BBA wBB | 7.67 | 5.42 | **4.88** |
| BBL wBB | 18.69 | 16.76 | **15.88** |

Consistent across every protein and seed. Native-contact Q does not follow:
Frozen 0.333 / 0.437 / 0.306 against DuET 0.256 / 0.430 / 0.262. The population
moves into a lower-RMSD region that is not more native-like.

### 2. Only the unsteered arm reaches the target, and K does not explain it

The first reading was that DuET is simply too narrow — 8 paths against Frozen's
64 — to produce the tail event a hit requires. The K/M gradient refutes that.
Budget held at 64, BBA:

| K | M | paths | TICA min | RMSD | hit/path |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 32 | 2 | 1.426 | 4.79 | 0.000 |
| 8 | 8 | 8 | 0.984 | 4.96 | 0.000 |
| 32 | 2 | 32 | 1.050 | 4.98 | 0.000 |
| **64** | **1** | 64 | **0.222** | 7.67 | **0.078** |

Raising K sixteen-fold leaves the closest approach flat at ~1.0 and the hit rate
at zero. Frozen, which differs by having no steering at all, reaches 0.222.
Trp-cage behaves the same way (DuET 1.07–1.30, Frozen 0.808).

**What separates the arms is whether steering is on, not how the budget is
split.** RMSD steering concentrates the population in a basin the TICA target is
not in, and that is what removes the tail.

### 3. BBL's validity collapse is a start-structure problem

Whole-path validity: Trp-cage 1.00, BBA 1.00, BBL 0.04–0.38.

The first hypothesis was arithmetic — whole-path validity fails on any single
bad peptide bond, and BBL offers 46 bonds × 33 frames against Trp-cage's 19 × 33.
Per-bond rates refute it:

| protein | residues | bonds | violating | per-bond | worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trp-cage | 20 | 195,624 | 9 | 0.00005 | 1.91 Å |
| BBA | 28 | 277,992 | **0** | 0.00000 | — |
| BBL | 47 | 473,616 | 627 | **0.00132** | **5.76 Å** |

26× Trp-cage's rate, against a 2.4× chain-length ratio, with a worst C–N
distance of 5.76 Å — a severed backbone. BBA, in the middle by length, is
perfect, so length does not order these.

Start-structure expansion does:

| protein | Rg start | Rg folded | ratio |
| --- | ---: | ---: | ---: |
| Trp-cage | 10.19 | 7.01 | 1.45 |
| BBA | 13.13 | 9.09 | 1.44 |
| **BBL** | **23.45** | **10.35** | **2.27** |

PVB-ATLAS is fine-tuned on equilibrium MD, which contains folded and near-folded
states. BBL starts essentially extended, outside that distribution. The
single-transition smoke test passed because damage accumulates over 32 steps,
not one.

## What this does not say

- Nothing here is evidence that DuET recovers folding. Every TICA hit came from
  the arm with no steering.
- Timing was not discriminated. The four checkpoint timings differ by less than
  seed scatter on every protein (Trp-cage 3.76–3.99, BBA 4.82–5.48). That is a
  non-result, not a finding that timing does not matter: a sweep over a metric
  that does not respond measures nothing.
- Three seeds cannot support a superiority claim in either direction.
- T=32 at PVB's 100 ps lag is 3.2 ns. Fast folders fold on microseconds. Zero
  hits is the expected outcome at this horizon and is not itself informative.

## What to do next

The blocker is reward alignment, not the sampler. `log ψ = −16·(d_RMSD/d₀)²`
pulls toward low whole-protein RMSD while success is defined on the first two
backbone-torsion TICA coordinates, and the two disagree — visible three ways
(RMSD falls, Q flat, TICA distance rises).

**H4 — a reward defined on the TICA coordinate closes the gap.**
Replace the reward with −λ·(TICA distance)² and rerun DuET K8×M8 on BBA, three
seeds, against the existing Frozen baseline. This needs a new observable, not a
config change: the TICA projection has to become available inside the sampling
loop.
Falsified if TICA-reward DuET still reaches zero hits, which would put the limit
on the 3.2 ns horizon rather than on the reward.

Until H4 is answered, further timing or K/M sweeps are not worth running.

## Artifacts

- `scripts/duet/fast_folders_extended/results_pvb_all_66.json` — all 66 cells
- `outputs/pvb_full/<tag>/…/native_contact_q.json` — per-run Q sidecars
- `compute_native_contacts.py`, `diagnose_validity.py`,
  `analyze_gate_sensitivity.py`, `test_h2_tica_vs_rmsd.py` — the analyses above
