# Phase B4 protocol amendment: PI3Kα observed-residue RMSD

- Recorded: 2026-09-10 13:34 KST
- Namespace: `B4_pocket_search`
- Parent task retained: `pi3ka` with status `hold_missing_target_reward_backbone`
- Amended task: `pi3ka_observed_mask`

## Reason

The 8TSB holo reference lacks N/CA/C coordinates for UniProt residues
943–950 inside the originally proposed moving region 931–957. The full-region
RMSD therefore remains undefined and its original BLOCKED record is preserved.

## Approved metric amendment

- Construct generated without deletion: AF-P42336 residues 765–1051.
- Holo reference: 8TSB chain A.
- Original moving region: residues 931–957 (27 residues).
- Observed reward/evaluation mask: 931–942 and 951–957 (19 residues).
- Excluded from RMSD only: 943–950 (8 residues).
- Reward atoms: N/CA/C, all required in both start and holo for every retained residue.
- Fixed alignment core continues to exclude the full original moving region
  931–957 and the existing five-residue guard band. Mask reduction does not
  return any part of 931–957 to the alignment core.
- `d0` is recomputed with this same alignment and observed mask. The task stops
  as `endpoint_not_separated` if `d0 <= 1.5 Å`.

The metric is named `observed_loop_rmsd`. Its 1.5 Å/two-final-frame endpoint
fraction is named `observed_loop_holo_like_fraction`. Neither is interpreted as
full-loop restoration, ligand binding, or pocket-opening success.

## Unchanged run settings

- DuET only; strengths 16 and 32; seed 103.
- K=4, M=4, T=24; SDE; lag 2.56 ns; full history.
- Reverse schedule 200; inner resampling after 150 and 180 completed updates
  (75% and 90%).
- No log-potential floor; existing incremental multi-checkpoint FKC and outer
  `Zhat / psi(parent)` correction.
- T=16 and T=24 pre-resampling population and weights are retained.
- No automatic seed, reward, checkpoint, horizon, allocator, or baseline sweep.

## Evaluation firewall

The generated 943–950 segment remains present in validity and ligand-space
evaluation. Holo ligand coordinates are evaluation-only. Multi-cutoff distance
counts are stored without a binary ligand-space success threshold. Missing holo
coordinates are not treated as evidence that the physical pocket is empty.

AlphaFold DB v6 P42336 is frozen by SHA256 and shared by the a=16 and a=32
runs. It is recorded as the same UniProt/construct used for this pilot, not as
a file-identical reconstruction of the original SLICE input.
