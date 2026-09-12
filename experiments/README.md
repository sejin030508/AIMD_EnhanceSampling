# Frozen experiment payloads

This directory preserves the portable code and configuration bundles used by
later experiments that ran in isolated server workspaces.

| Directory | Contents |
|---|---|
| `phase_b_cryptic_pocket/` | PRMT5/PRMT6 amendments and B4 PRMT6/SMARCA2/PI3K-alpha launch material |
| `tps_small_protein/` | Chignolin, Trp-cage, and BBA preparation, sampling, evaluation, and scheduling payload |
| `geometry_audits/bundle_a/` | TPS-hit and official-sampler geometry/energy audit |
| `geometry_audits/restrained_relaxation/` | Restrained OpenMM relaxation and independent chirality checks |

These files document what was executed. The canonical reusable implementation
remains under `src/`. Do not silently copy an older snapshot back into `src/`;
port a change deliberately and run the main test suite.

Compiled caches, model checkpoints, upstream repositories, and generated
trajectories are not included here.
