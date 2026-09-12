# AI-MD NSMC / DuET-MD

This repository implements **DuET-MD**, a training-free, dual-time particle
steering pipeline for frozen full-history molecular-dynamics emulators.

The output is described as:

> bias-conditioned transition paths under the frozen surrogate prior

It is not exact dynamics under a new potential, does not estimate kinetics,
and cannot recover transitions outside the support of the frozen emulator.

## Current status

The repository contains the core DuET-MD implementation, preregistered
experiment configurations, frozen execution payloads, and compact result
summaries. Large model checkpoints, downloaded reference trajectories, and
full generated trajectories are intentionally not versioned.

The current evidence should be read as method development, not a final
biophysical claim:

- exact finite-state tests validate the cross-clock weighting arithmetic;
- ATLAS and cryptic-pocket studies are exploratory surrogate-sampling studies;
- the small-protein TPS pilot shows reward-directed endpoint approach, but a
  later all-atom audit found that the raw generated paths are not yet
  atomistically valid transition paths.

See [the project layout](docs/PROJECT_LAYOUT.md),
[experiment snapshots](experiments/README.md),
[result summaries](results/README.md), and
[experiment reports](docs/reports/README.md) for the shortest route through
the repository.

## Repository relationship

This is a clean implementation repository. It reuses the existing Level-1
project and server assets without changing them:

```text
/workspace/sejin/AI_MD_NSMC                    # this repository
/workspace/sejin/confrover_mh_steering         # read-only code/data/checkpoints
```

No files under the previous `outputs/` tree are modified. New results are
written only below `outputs/duet_md/`.

The later recovery pilots were run in an isolated server namespace. Their
portable launch payloads and compact reports have been copied into this
repository without merging experimental snapshots into the canonical
`src/` package.

## Implemented scope

- terminal, windowed-intermediate, and ordered two-event programs;
- PC1, C-alpha RMSD, C-alpha residue-pair distance, and binary contact observables;
- full-history outer SMC;
- one- and multi-checkpoint reverse-diffusion Feynman–Kac samplers;
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
[the experiment protocol](docs/DUET_EXPERIMENT_PROTOCOL.md). The reusable
cross-protein route benchmark is specified separately in
[DUET_PROTEIN_BENCHMARK_PROTOCOL_V3.md](docs/DUET_PROTEIN_BENCHMARK_PROTOCOL_V3.md),
with its preregistered protein order in
[DUET_ATLAS_CONFIRMATORY_COHORT_SCREEN_V1.md](docs/DUET_ATLAS_CONFIRMATORY_COHORT_SCREEN_V1.md).
The frozen 10-protein cohort, attrition, event definitions, and launch matrix
are summarized in
[DUET_ATLAS_COHORT_PREPARATION_2026-09-05.md](docs/DUET_ATLAS_COHORT_PREPARATION_2026-09-05.md).
The concrete pre-experiment verification status is recorded in
[STAGE1_READINESS.md](docs/STAGE1_READINESS.md).
