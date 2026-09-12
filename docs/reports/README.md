# Experiment report index

This directory collects frozen reports from completed or interrupted DuET-MD
experiments. Reports preserve the configuration and interpretation that were
current when each experiment was run; later audits do not overwrite earlier
raw results.

## Recommended entry points

- `TPS_SMALL_PROTEIN_TRANSITION_NOTION_REPORT_2026-09-12.md`: self-contained
  account of the Chignolin/Trp-cage/BBA pilot, all 24 production results, and
  the final physical-validity interpretation.
- `PHASE_B4_CRYPTIC_POCKET_AGENT_HANDOFF_2026-09-10.md`: detailed PRMT6,
  SMARCA2, and PI3K-alpha B4 settings and results.
- `PHASE_B4_CRYPTIC_POCKET_NOTION_SUMMARY_2026-09-10.md`: shorter Phase B4
  overview.
- `BUNDLE_A_CROSS_CLOCK_AUDIT_2026-09-11/BUNDLE_A_FINAL_REPORT_2026-09-11.md`:
  raw-output geometry, energy, TICA, and official-sampler audit.
- `restrained_relaxation_audit_2026-09-11_atom37_corrected/RESTRAINED_RELAXATION_AUDIT_REPORT.md`:
  corrected post-hoc OpenMM relaxation diagnostic.
- `chirality_recheck_2026-09-12_v2/CHIRALITY_RECHECK_REPORT.md`: independent
  chirality recheck of the relaxation audit.

## Interpretation rule

Read newer audit reports together with the original experiment report. A
coarse target hit or lower RMSD is evidence of reward-directed sampling, not
by itself evidence of an atomistically valid transition or physical kinetics.
