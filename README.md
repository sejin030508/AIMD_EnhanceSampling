# AI-MD NSMC / DuET-MD

This repository implements **DuET-MD**, a training-free, dual-time particle
steering pipeline for frozen full-history molecular-dynamics emulators.

The output is described as:

> bias-conditioned transition paths under the frozen surrogate prior

It is not exact dynamics under a new potential, does not estimate kinetics,
and cannot recover transitions outside the support of the frozen emulator.

## Repository relationship

This is a clean implementation repository. It reuses the existing Level-1
project and server assets without changing them:

```text
/workspace/sejin/AI_MD_NSMC                    # this repository
/workspace/sejin/confrover_mh_steering         # read-only code/data/checkpoints
```

No files under the previous `outputs/` tree are modified. New results are
written only below `outputs/duet_md/`.

## Implemented scope

- terminal, windowed-intermediate, and ordered two-event programs;
- PC1, C-alpha RMSD, C-alpha residue-pair distance, and binary contact observables;
- full-history outer SMC;
- one-checkpoint reverse-diffusion Feynman–Kac sampler;
- coupled predictive-normalizer weighting;
- frozen, best-of-budget, outer-only, inner-only, naive-dual,
  complete-frame nested, and DuET-MD baselines;
- exact finite-state two-clock benchmark and proper-weighting tests;
- ConfRover ODE/SDE preflight and fixed-compute K/M grid runners;
- deterministic R1/R2 task construction with R3 held out;
- run metadata, seed, NFE, wall-clock, peak-memory, resume, and overwrite guards.
- held-out R3 PCA/DTW/RMSD/contact-map evaluation and route-JSD suppression rules;
- paired-bootstrap tables plus scripts for the four requested figures.

## Setup and dry run

```bash
export DUET_PROJECT_ROOT=/workspace/sejin/AI_MD_NSMC
export DUET_ASSET_ROOT=/workspace/sejin/confrover_mh_steering
export PYTHONPATH="$DUET_PROJECT_ROOT/src"

bash scripts/duet/setup_and_validate.sh
bash scripts/duet/run_stage1_dry_runs.sh
```

Construct the 6J56-A programs from R1/R2 before protein experiments:

```bash
python -m confmh.duet.cli prepare-programs \
  --config configs/duet/prepare_6j56_programs.yaml
```

The scientific phases are intentionally gated:

```bash
bash scripts/duet/run_correctness.sh
bash scripts/duet/run_confrover_preflight.sh
bash scripts/duet/run_6j56_core.sh
bash scripts/duet/run_6j56_km_grid.sh
bash scripts/duet/run_interp_pilot.sh
bash scripts/duet/run_interp_full.sh
bash scripts/duet/run_proar_transfer.sh
bash scripts/duet/generate_reports.sh
```

See [DUET.md](docs/DUET.md), [the formulation](docs/DUET_FORMULATION.md),
[the integration map](docs/DUET_INTEGRATION_MAP.md), and
[the experiment protocol](docs/DUET_EXPERIMENT_PROTOCOL.md). The concrete
pre-experiment verification status is recorded in
[STAGE1_READINESS.md](docs/STAGE1_READINESS.md).
