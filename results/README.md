# Compact result summaries

This directory contains small, machine-readable summaries that support the
human-readable reports in `docs/reports/`.

- `summaries/7lp1/`: compact 7LP1 evaluation summaries salvaged from server temporary storage
- `summaries/tps/`: the complete 24-cell TPS table and subsequent geometry,
  energy, relaxation, and chirality audit summaries

These are not the full output trees. Full coordinate populations, trajectories,
per-frame arrays, checkpoints, and caches are excluded from Git and retained in
persistent server storage plus a dated local backup.

Reported metrics retain their original experimental semantics. In particular,
the TPS `weighted_valid_thp` field uses the original coarse C-alpha validity
gate; the later geometry audit must be consulted before making an atomistic
transition-path claim.
