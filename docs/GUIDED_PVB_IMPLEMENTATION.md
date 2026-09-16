# Guided-PVB integration contract

`guided_duet` adds Cartesian score guidance to the existing PVB–DuET inner
bridge. It does not replace DuET selection, alter Frozen/CN/DuET, or add an MH
accept/reject step.

For every configured stochastic PVB update, the frozen decoder predicts
`xhat1 = xt + (1-t) * drift`. The code differentiates the same frozen TICA
potential used by the production evaluator and shifts the proposal mean by

```text
delta = eta * sigma^2 * grad_xt(log psi(xhat1)) * dt.
```

The sampled update is corrected with the exact discrete Gaussian ratio

```text
log(p_base/q_guided)
  = -epsilon dot (delta/std) - 0.5 * ||delta/std||^2,
std = sigma * sqrt(dt).
```

These ratios remain in the inner path weight. Checkpoint potential increments
telescope along resampled ancestry; their product and the proposal corrections
form `Zhat`. The outer update remains `log Zhat - log psi(parent)`. The final
deterministic PVB update is never guided because it has no Gaussian density
ratio.

Controls are independent:

- `guidance.strength`: proposal drift coefficient `eta`; `0` is exact base-PVB
  sampling with zero proposal correction.
- `guidance.schedule`: `all_stochastic` or explicit zero-based update indices.
- `guidance.inner_resampling`: enables/disables checkpoint resampling without
  disabling guidance, weights, endpoint selection, or outer SMC.
  When disabled, candidates keep their uninterrupted original noise streams;
  checkpoint-only reseeding is not performed.
- `particles.inner_checkpoint_progresses`: selection checkpoints only; they do
  not define the guidance schedule.

The runner records guidance metadata, per-path proposal ratios, inner/outer
ESS, ancestry, potential-telescoping residual, decoder evaluations, backward
evaluations, time, and peak GPU memory. A completed run is rejected if proposal
ratios are non-finite or the potential telescope exceeds `1e-10`.

Validation completed before any production launch:

- exact Gaussian-ratio unit test;
- differentiable CA-distance and torsion TICA tests, including finite
  differences;
- guided inner weighting with checkpoint resampling both on and off;
- `eta=0` update parity (bitwise at the isolated update level, and within the
  measured CUDA base/base repeatability floor for a real full checkpoint);
- full existing `tests/duet` regression suite.

The real-checkpoint smoke was completed on A6000 GPU 1 on 2026-09-16 with
BBA, `sde_step=20`, one candidate, checkpoints 0.50/0.75, and `eta=0.01`.
It passed all checks:

- Torch/production initial-potential absolute error: `7.40e-7`;
- base/base CUDA repeat maximum coordinate difference: `2.06e-4 A`;
- eta-zero/base maximum coordinate difference: `3.13e-4 A`, below the
  measured-backend tolerance `4.12e-4 A`;
- eta-zero proposal correction: exactly `0`;
- guided proposal correction: finite (`0.0167544`);
- guidance backward evaluations: `19`;
- peak allocated GPU memory: `1,815,401,984` bytes;
- measured validation wall time: `1.76 s`.

The machine-readable result is stored on A6000 at
`/home/sejin/AI_MD_NSMC_guided_pvb_validation/validation_outputs/guided_pvb_checkpoint_gpu1.json`.
No production sweep was launched.
