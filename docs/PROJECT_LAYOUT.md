# Project layout

`AI_MD_NSMC` is the canonical Git repository for DuET-MD. The repository is
organized so that reusable code, fixed experiment specifications, reports,
and large runtime artifacts remain distinguishable.

| Path | Purpose |
|---|---|
| `src/confmh/duet/` | Canonical DuET-MD algorithms, observables, weighting, runners, and evaluation code |
| `src/confmh/adapters/` | Interfaces to frozen full-history MD surrogate models |
| `configs/` | Reproducible YAML/JSON experiment configurations |
| `scripts/duet/` | Preparation, launch, audit, and report-generation entry points |
| `tests/` | Unit and integration tests for weighting, program semantics, adapters, and evaluation |
| `experiments/` | Frozen payloads used by later isolated server experiments; retained for exact provenance |
| `results/summaries/` | Compact machine-readable tables and audit results suitable for Git |
| `docs/reports/` | Human-readable experiment handoffs, results, limitations, and post-hoc audits |
| `data/` | Small prepared inputs and mappings that cannot be reconstructed from configuration alone |
| `outputs/` | Full runtime outputs; intentionally ignored by Git |

## Canonical code versus frozen payloads

New development should target `src/`, `configs/`, `scripts/`, and `tests/`.
Files below `experiments/` are snapshots of the exact launch payloads used for
specific Phase B, TPS, and geometry-audit runs. They may contain duplicated
adapter code because reproducibility requires preserving the historical
execution state. They are not a second canonical package.

## External and large assets

The following are deliberately excluded from Git:

- ConfRover and other upstream repository clones;
- model checkpoints and representation caches;
- ATLAS/reference XTC, TRR, DCD, and H5 trajectories;
- full generated atom-coordinate populations and visualization packages;
- temporary GPU-container caches.

Downloadable assets should be reconstructed from the recorded source URLs,
commits, manifests, and hashes. Generated outputs must be kept in persistent
workspace storage and in the dated local backup described outside the public
repository.

## Reading order

1. `README.md`
2. `docs/DUET_FORMULATION.md`
3. `docs/DUET_EXPERIMENT_PROTOCOL.md`
4. `docs/reports/README.md`
5. The relevant config and frozen payload for the experiment of interest

For the latest small-protein transition pilot, start with
`docs/reports/TPS_SMALL_PROTEIN_TRANSITION_NOTION_REPORT_2026-09-12.md`.
