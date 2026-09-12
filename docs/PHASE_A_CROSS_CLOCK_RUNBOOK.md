# Phase A Cross-Clock Runbook

This runbook covers the exactly enumerable Phase A experiment specified on
2026-09-08. It is independent of the earlier exact-toy backend and does not
modify or consume protein-benchmark outputs.

## Frozen scope

- Seven physical states and five physical transitions
- Three exact binary reverse-diffusion transitions per physical transition
- Six fixed settings: four informative canonical regimes and two rare
  no-information controls
- Five fixed methods with KM = 32
- Main DuET checkpoint fixed at Y2
- Four development-only static allocations
- Twenty held-out processes generated with master seed 20260908
- Four total probe-parent slots with two discarded probes per slot
- Three adaptive production actions with KM = 24
- 5,000 independent repetitions per evaluated cell
- float64 computation and systematic outer/inner resampling

The executable config rejects changes to these frozen values:

    configs/duet/phase_a_cross_clock.yaml

## Pre-run validation

    bash scripts/duet/run_phase_a_cross_clock.sh --validate-only

Validation checks the complete frozen configuration, verifies that every
reverse kernel marginalizes to its physical transition with error below
1e-12, and verifies constant Y2 look-ahead potential in the no-information
control.

## Full experiment

    bash scripts/duet/run_phase_a_cross_clock.sh

The run uses 163 independently evaluated cells. Two A3 cells and four
development cells are exact reuses of matching A2 cells rather than duplicate
Monte Carlo runs. The resulting sampled workload is:

    163 cells x 5,000 repetitions x 480 reverse transitions
    = 391,200,000 reverse-transition evaluations

This backend is CPU-only and makes no neural decoder calls.

If interrupted, resume without repeating completed cells:

    bash scripts/duet/run_phase_a_cross_clock.sh --resume

Each cell writes its compressed raw repetition arrays before writing
summary.json; the summary is the completion marker used by --resume.

## Outputs

Default output root:

    outputs/duet_md/phase_a_cross_clock

Important files:

- validation.json: frozen-spec and exact-kernel checks
- held_out_parameters.json: all 20 held-out transition tables
- exact/*.json: physical paths, inner diffusion paths, exact target,
  normalizer, success probability, and symmetric L/R route masses
- cells/*/*/*/raw_repetitions.npz: repetition-level estimates, normalizers,
  adaptive schedules, and probe diagnostics
- cells/*/*/*/summary.json: per-cell RMSE, TV, normalizer confidence interval,
  and compute accounting
- metrics.json: A1-A5 aggregate metrics and comparisons
- seed_manifest.json: deterministic method IDs and RNG derivation
- nfe_accounting.json: matched reverse-transition cost

## Three-panel result figure

After metrics.json has been produced:

    python scripts/duet/plot_phase_a_cross_clock.py

This writes:

    outputs/duet_md/phase_a_cross_clock/phase_a_three_panel.png
