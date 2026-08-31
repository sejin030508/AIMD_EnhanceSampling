# Stage-1 pre-experiment readiness

Verified on 2026-08-31. No ConfRover reverse-diffusion protein experiment was
run during this stage.

## Completed gates

- The complete test suite passes: 55 tests total, including 24 DuET-specific
  tests.
- The finite-state Phase-A correctness run passes all configured gates.
  Maximum proper-weighting L1 residual over `M={1,2,4,8,16}` is 0.0341.
- At convergence scale 16, mean intermediate-marginal TV is 0.0234 for DuET,
  0.0416 for complete-frame nested, and 0.0933 for naive dual. Naive dual has
  normalizer relative bias 0.157, while DuET is 0.0135.
- Both `sejin-h100-1-work-001-zhr2l` and
  `sejin-h100-1-work-004-96566` expose an NVIDIA H100 NVL and load the frozen
  ConfRover-base checkpoint plus the cached OpenFold representation.
- The concrete full-history path accepts one- and two-frame histories. Both
  produce `s=[1,129,128]` and `z=[1,129,129,128]` conditioning tensors.
- R1/R2-only task construction succeeds. R3 is not read during construction.
  The selected ordered events are contact `(29,127)` at frame window 3–5,
  contact `(1,126)` at frame window 5–7, then the PC1 terminal basin at frame
  8. The reference crossings are at frames 4 and 6.
- All Stage-A through Stage-D configs pass dry-run with equal decoder population
  budgets. In the core factorial, every method is estimated at 230,400 decoder
  calls across three tasks and three seeds.
- Required result tables and figure-generation scripts complete a no-data
  smoke test and mark gated panels as awaiting results.

## Next gate

The next command is:

```bash
bash /workspace/sejin/AI_MD_NSMC/scripts/duet/run_confrover_preflight.sh
```

This is Phase B and is the first real ConfRover ODE/SDE feasibility experiment.
It must pass before Phase C is started. The GPU wall-time ceiling remains
`null` by the user's instruction and should be fixed after observing this
initial preflight.

## Deliberately gated scope

- Phase E cannot run from the currently available assets. The official
  interpolation case manifest, per-case initial structures, and
  `confrover_interp_20m_v1_0.pt` are absent. This is an asset gate, not a silent
  fallback to a hand-built case list.
- Phase F remains disabled until the ConfRover two-clock claim gate is positive.
  Although the ProAR clone and checkpoint exist, a stable stochastic
  intermediate refinement state has not been verified; complete-frame nested
  remains the declared fallback.
- The 6J56-A far-tail endpoint is supported by one R1/R2 start-to-target segment.
  The selection report preserves this limited design support, so later claims
  must not describe it as multi-replicate crossing stability.

## Claim boundary

All outputs are bias-conditioned path candidates under a frozen surrogate
prior. They are not exact `U+b` dynamics, equilibrium samples, MFPTs,
committors, or kinetic estimates.
