# Phase B cryptic-pocket pilot handoff

Date: 2026-09-09  
Purpose: concise handoff from the execution agent to the theory/coordination agent.

## 1. Objective and current status

The pilot tests whether DuET-MD can steer frozen ConfRover trajectories from a
closed/start structure toward a holo-like cryptic-pocket loop conformation.

- Proteins: PRMT5 and PRMT6.
- B1 recovery: 2 proteins × 4 methods × 2 seeds = **16/16 complete**.
- B2 two-checkpoint follow-up: 2 proteins × 2 methods × 1 new seed = **4/4 complete**.
- No active Phase-B process remains; both H100s used for B2 are idle.
- Across all completed B1 and B2 cells, **Open-like success = 0** and
  **anytime threshold hit = 0**.

## 2. Fixed scientific setup

- Frozen ConfRover checkpoint; SDE sampler; 200 reverse-diffusion steps.
- Physical horizon `T=16`: start plus 16 generated frames (17 structures).
- Lag `256 × 10 ps = 2.56 ns` per generated transition; nominal horizon
  `40.96 ns`.
- Decoder population budget: `K × M = 16` for every method.
- B1 DuET inner steering: one checkpoint at completed reverse fraction 0.75.
- OOM recovery changed execution only: `decoder_microbatch_size=1`, allocator
  tuning, and release of unreachable transient GPU cache between parents.
- Structural-validity threshold remains fixed at adjacent-CA distance 5.5 Å.

Methods in B1:

| Method | K | M | Outer SMC | Inner steering |
|---|---:|---:|---|---|
| Frozen | 16 | 1 | No | No |
| Outer-only | 16 | 1 | Yes | No |
| Complete Nested | 4 | 4 | Yes, proper normalizer | No; select after completion |
| DuET-MD | 4 | 4 | Yes, proper normalizer | Yes |

## 3. Reward and success definition

For a generated frame `x`, rigidly align its fixed core backbone to the holo
reference, then compute pocket-loop N/CA/C backbone RMSD `d(x)` in Å. PRMT5
uses the minimum over holo references 6UXY and 6UXX; PRMT6 uses 6W6D.

```text
log psi(x) = max(-30, -4 * (d(x) / d0)^2)
```

- PRMT5: `d0 = 6.143322 Å`.
- PRMT6: `d0 = 4.668808 Å`.
- The same `psi` is used at diffusion and physical time; its role differs.
- DuET inner weighting uses incremental potential ratios between checkpoints.
- DuET/Complete Nested outer weighting uses `log Zhat_t - log psi_(t-1)`.

Open-like is an evaluation criterion, not a discontinuous reward:

```text
threshold = min(1.5 Å, 0.5*d0) = 1.5 Å for both proteins
success = full path valid AND both final two generated frames have d <= 1.5 Å
```

Primary success denominator includes every trajectory, including invalid ones.
Hidden contact observables are recorded for evaluation only and never affect
sampling.

## 4. Readiness history

- PRMT5 passed its original protein-level preflight and proceeded unchanged.
- PRMT6's original strict B0 remains **FAIL** and was not overwritten.
- The separately authorized expanded frozen diagnostic (`K=16, M=1, T=4`,
  seed 41) produced 15/16 valid paths, one invalid generated frame out of 64,
  no NaN, no major clash, and no repeated same-site break. It therefore
  received `exploratory_proceed`; this does not relabel the strict B0 result.

## 5. B1 recovery raw summary

Seeds: 17 and 29. `Final d/d0` is the final pre-resampling population's
SMC-weighted mean. Lower is closer to the holo loop.

| Protein | Seed | Method | Valid-path fraction | Weighted final d/d0 | Open-like | Anytime hit |
|---|---:|---|---:|---:|---:|---:|
| PRMT5 | 17 | Frozen | 0.8750 | 1.0509 | 0 | 0 |
| PRMT5 | 17 | Outer-only | 0.8750 | 1.0288 | 0 | 0 |
| PRMT5 | 17 | Complete Nested | 0.5000 | 1.0544 | 0 | 0 |
| PRMT5 | 17 | DuET-MD | 0.5000 | 1.0011 | 0 | 0 |
| PRMT5 | 29 | Frozen | 0.8125 | 1.0430 | 0 | 0 |
| PRMT5 | 29 | Outer-only | 0.8125 | 1.0039 | 0 | 0 |
| PRMT5 | 29 | Complete Nested | 1.0000 | 1.0236 | 0 | 0 |
| PRMT5 | 29 | DuET-MD | 0.7500 | 1.0256 | 0 | 0 |
| PRMT6 | 17 | Frozen | 0.6250 | 1.0613 | 0 | 0 |
| PRMT6 | 17 | Outer-only | 0.6250 | 1.0105 | 0 | 0 |
| PRMT6 | 17 | Complete Nested | 0.7500 | 1.0442 | 0 | 0 |
| PRMT6 | 17 | DuET-MD | 0.7500 | 1.0278 | 0 | 0 |
| PRMT6 | 29 | Frozen | 0.6875 | 1.0614 | 0 | 0 |
| PRMT6 | 29 | Outer-only | 0.6875 | 1.0255 | 0 | 0 |
| PRMT6 | 29 | Complete Nested | 1.0000 | 0.9654 | 0 | 0 |
| PRMT6 | 29 | DuET-MD | 0.5000 | 1.0356 | 0 | 0 |

Every B1 cell used `decoder_nfe=50,944`.

## 6. B2: 75% + 90% diffusion steering

B2 used new seed 43 and retained all other scientific settings. DuET performs
two actual Feynman--Kac resampling interventions at reverse steps 150/200 and
180/200. Complete Nested remains a no-inner-steering control and selects only
after complete frame generation.

| Protein | Method | Checkpoints | Valid path | Weighted final d/d0 | Open-like | Anytime hit | Wall time |
|---|---|---|---:|---:|---:|---:|---:|
| PRMT5 | Complete Nested | none | 1.00 | 0.9910 | 0 | 0 | 47.8 min |
| PRMT5 | DuET-MD | 0.75, 0.90 | 1.00 | 1.0720 | 0 | 0 | 47.6 min |
| PRMT6 | Complete Nested | none | 1.00 | 1.0090 | 0 | 0 | 47.8 min |
| PRMT6 | DuET-MD | 0.75, 0.90 | 0.75 | 0.9995 | 0 | 0 | 35.1 min |

- All B2 cells also used `decoder_nfe=50,944`.
- Mean DuET inner ESS at 75%/90%: PRMT5 `3.900/3.913`, PRMT6
  `3.790/3.845`, out of M=4.
- Maximum telescoping error was exactly `0.0` for both DuET runs.
- Thus multi-checkpoint weighting operated correctly, but checkpoint weights
  were weakly discriminative because ESS remained close to 4.

## 7. Current interpretation and decision needed

The implementation and cross-clock weighting checks passed, but the present
pilot does **not** demonstrate cryptic-pocket recovery or DuET superiority.
Adding 90% steering did not produce even a transient 1.5 Å hit. More seeds with
the identical setup are therefore unlikely to be the most informative next
use of compute.

The next protocol decision should distinguish at least these possibilities:

1. the frozen ConfRover transition kernel does not place enough mass near the
   holo pocket within T=16;
2. holo-loop RMSD is too weak or too late as the sole checkpoint potential;
3. M=4 provides too little candidate diversity for inner selection;
4. the 1.5 Å endpoint criterion is substantially stricter than the reachable
   candidate distribution.

Before a larger run, inspect the full per-frame `d/d0` distributions and the
75%/90% per-candidate checkpoint potentials, then preregister one controlled
change (reward shaping, horizon, or K/M allocation) rather than changing
several axes simultaneously.

## 8. Artifact locations

Execution roots:

- H100 project: `/workspace/sejin/AI_MD_NSMC_phase_b_recovery`
- H100/shared outputs: `/workspace/sejin/phase_b_pockets_recovery/outputs`
- A6000 PRMT6 seed-17 outputs: `/mnt/ssd0/sejin/phase_b_pockets/outputs`

Each completed run contains `metrics.json`, `records.json`,
`population_curves.json`, `outer_ancestry.json`, normalized weights, and saved
pre/post-resampling atom37 populations.

Detailed amendment history is in
`reports/PHASE_B_PROTOCOL_AMENDMENT_2026-09-08.md`.
