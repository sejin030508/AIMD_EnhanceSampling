# PVB preflight

Checks whether the published PVB checkpoint can serve as a second backend for
DuET-MD, before any adapter is written.

**Model**: `yaledeus/PVB` @ `c08e5e3c`, checkpoint `pvb_atlas.ckpt` (Google Drive
link from the repository README). Paper: *Unified Biomolecular Trajectory
Generation via Pretrained Variational Bridge*, ICLR 2026, arXiv:2602.07588.

**Input**: `7lp1_A_R2F5000_start.pdb` — the same start structure the DuET ATLAS
runs use, and in-distribution for the ATLAS checkpoint.

## What was checked and why

| Script | Question |
| --- | --- |
| `pvb_load_test.py` | Does the checkpoint load, and what are its settings? |
| `pvb_preflight.py` | Does the rollout run, is its geometry usable, can the inner loop be split and branched, and is the clean endpoint recoverable? |
| `pvb_determinism.py` | What is the official loop's own reproducibility floor? |

The determinism script exists because the split-loop check is only meaningful
against that floor: re-running `model.inference` under the same seed already
differs by ~1.4e-4 A on CUDA, so a split loop agreeing to 1.1e-4 A is exact
agreement, not a discrepancy.

## Results (7LP1_A, 4 particles, sde_step 10, 8 rollout steps)

**Loads**: `module.model.dyVAE`, 9,965,317 parameters, `sigma=0.2`,
`using_ode=False` (so the sampler is a genuine SDE), backbone `torchmdnet`.

**Runs**: 8 rollout steps x 4 particles in 2.14 s. Hydrogens are dropped by the
model's own preprocessing (647 -> 329 atoms), all 39 residues retained.

**Geometry** — first and last generated frame, compared against what the Bundle A
audit found for ConfRover:

| | ConfRover (Bundle A) | PVB (here) |
| --- | --- | --- |
| peptide C-N violations | every one of 96 frames | **0 of 38 bonds, both frames** |
| peptide C-N range | 0.073 - 3.006 A | **1.265 - 1.422 A** |
| adjacent CA | up to 5.578 A | 3.64 - 4.08 A |
| CA clashes < 3 A | present | 0 |
| non-finite coordinates | present | 0 |

**Split loop**: reproduces `model.inference` to 1.094e-4 A, below the official
loop's own 1.411e-4 A reproducibility floor.

**Branching**: after copying all four particles from particle 0 at step 7 and
resuming with fresh noise, the resumed branches separate by 0.16-0.17 A. Fresh
SDE noise does decorrelate duplicates, which is what the inner filter needs.

**Clean endpoint**: the drift head is trained against `(x1 - xt) / (1 - t)`, so
`x1_hat = xt + (1 - t) * drf_pred` inverts it. Measured against the realised
endpoint of the same trajectory:

| t | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| RMSD to endpoint (A) | 0.859 | 0.488 | 0.302 | 0.206 | 0.087 | 0.000 |

Monotone, and exactly zero at the last step because that step is deterministic.
This is the model's own endpoint prediction recovered algebraically, not an
externally defined forecast, so it has the same standing as ConfRover's
`pred_atom14` and can be described as clean-prediction steering.

## Environment

`confrover-mh` (torch 2.1.2+cu121) plus an isolated `--target` directory so the
working ConfRover environment is left untouched:

```bash
pip install --no-deps --target /workspace/sejin/pvb_assets/site \
  torch_scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install --no-deps --target /workspace/sejin/pvb_assets/site \
  e3nn easydict opt_einsum opt_einsum_fx emcee
export PYTHONPATH=/workspace/sejin/pvb_assets/site
```

`--no-deps` matters: a plain `--target` install pulls torch 2.14 and a full
CUDA 13 stack into the directory, which then shadows the environment's torch
2.1.2 and breaks the `torch_scatter` build compiled against it.

Assets live under `/workspace/sejin/pvb_assets/` (checkpoints, source, results)
so they survive a host restart.

## What this does not establish

This is a frozen-rollout preflight: no reward, no selection, no outer filter,
one protein, 8 steps. It shows the model connects and that the mechanics DuET
needs are present. It says nothing about whether PVB supports the transitions
the benchmark asks for, which is what a proper support gate would test.

The physical lag per rollout step is still unresolved and has to be pinned down
before PVB results can be compared against ConfRover at matched physical time.
