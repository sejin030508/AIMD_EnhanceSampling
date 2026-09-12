# Phase B cryptic-pocket pilot runbook

## Scope

Phase B is isolated from Phase A and uses endpoint-only steering with the frozen
ConfRover-base checkpoint. B1 must not start until `preflight_status.json` reports
both proteins as ready.

## Frozen task definitions

| Protein | Start | Holo reference(s) | Construct | Reward loop | d0 |
|---|---|---|---:|---:|---:|
| PRMT5 | 7KIC-A pseudo-apo | 6UXY-A, 6UXX-A | 294–637 | 435–445 | 6.143322 Å |
| PRMT6 | AF-Q96LA8-F1-model_v4 | 6W6D-A | 53–375 | 155–165 | 4.668808 Å |

The reward is the minimum fixed-core-aligned loop N/CA/C RMSD to the listed holo
references. The same atom37 function and Å units are used for start, predicted-clean,
and final frames:

`log psi(x) = max(-30, -4 * (d(x) / d0)^2)`.

The core excludes the reward loop plus five residues on both sides, five residues at
both construct ends, and backbone positions not shared by start and every reference.
Author/UniProt numbering, prepared PDB numbering, and zero-based model indices are
stored in each `residue_mapping.csv` and `manifest.json`.

## B1 settings (not yet executed)

| Method | K | M | T | Checkpoint |
|---|---:|---:|---:|---:|
| Frozen | 16 | 1 | 16 | none used |
| Complete Nested | 4 | 4 | 16 | completed candidates |
| Fixed DuET | 4 | 4 | 16 | 0.75 completed reverse fraction |

All use stochastic SDE, 200-step reverse schedule, stride 256 (nominal 2.56 ns),
offloaded cache, decoder microbatch 2, systematic outer resampling at ESS ≤ 0.5K,
and seeds 17 and 29. `K*M=16` for every method. B1 order is PRMT5 then PRMT6.

## Preflight

Per protein, preflight consists of:

1. reward/mapping/invariance and history-fork tests;
2. DuET K=1, M=4, T=1 memory gate;
3. DuET K=4, M=4, T=2 full-history memory gate;
4. Frozen K=4, M=1, T=4 B0 structural stability run.

B0 is not screened on pocket opening. Its seed is 17, the first preregistered B1 seed,
because the supplied B0 protocol fixed the population and horizon but did not assign a
separate RNG seed. B0 outputs remain outside the B1 table.

The B1 launcher is `scripts/duet/run_phase_b_b1_a6000.sh`. It only runs if the audit
file reports both proteins ready and does not overwrite Phase A or B0 outputs.
