# DuET-MD formulation

## State and clocks

- `H_t = (H_0, x_1, ..., x_t)` is the complete physical trajectory history.
- `x_t` is the clean molecular frame at physical step `t`.
- `m_t` is the deterministic temporal-program progress state.
- `psi_t(H_t,m_t) > 0` is the prefix potential.
- `K` is the number of outer physical-time particles.
- `M` is the number of inner reverse-diffusion particles.
- `L` is the number of reverse-diffusion decoder steps for one frame.

The frozen model defines

```math
Q_\theta(x_{1:T}\mid H_0)=\prod_{t=1}^T q_\theta(x_t\mid H_{t-1}).
```

The path target is

```math
Q_{\theta,P}(H_T,m_T)\propto Q_\theta(H_T)R_P(H_T,m_T),
\qquad R_P=\exp[-\lambda_P C_P],
```

with `psi_T = R_P`. Potentials are evaluated in log space and clipped only by
the configured positive numerical floor.

## Ideal local proposal

For a fixed outer parent,

```math
r_t^*(x_t\mid H_{t-1},m_{t-1})=
\frac{q_\theta(x_t\mid H_{t-1})\psi_t(H_t,m_t)}
{Z_t(H_{t-1},m_{t-1})},
```

where

```math
Z_t=\mathbb E_{q_\theta}[\psi_t(H_{t-1}\oplus X_t,m_t)].
```

This fully adapted construction is inherited from SMC/NSMC.

## One-checkpoint inner sampler

At the checkpoint after 75% of the `L` reverse steps by default,

```math
\phi_c^j=\exp[-\lambda_P\widehat C_{prefix}(\hat x_0^j)].
```

After resampling checkpoint ancestors `a_j`, independent SDE noise is used for
the remaining reverse steps. At the clean endpoint,

```math
\phi_0^j=\psi_t(H_{t-1}\oplus X_t^j,m_t^j),
\quad g_1^j=\phi_c^j,
\quad g_2^j=\frac{\phi_0^j}{\phi_c^{a_j}}.
```

Thus every ancestry path telescopes:

```math
g_1^{a_j}g_2^j=\phi_0^j=\psi_t.
```

The local normalizer estimate is

```math
\widehat Z_t=\left(\frac1M\sum_jg_1^j\right)
              \left(\frac1M\sum_jg_2^j\right).
```

The selected endpoint is sampled proportionally to `g_2`. The returned pair
`(X_t, Zhat_t)` is tested through the joint proper-weighting identity

```math
\mathbb E[\widehat Z_t f(X_t)\mid H_{t-1},m_{t-1}]
=\int q_\theta(x_t\mid H_{t-1})\psi_t(H_t,m_t)f(x_t)dx_t.
```

Unbiasedness of `Zhat_t` alone is not treated as sufficient.

## Outer weighting

DuET-MD uses

```math
G_t^{coupled}=\widehat Z_t/\psi_{t-1}.
```

The selected child's `psi_t` is not applied again. Systematic outer resampling
is performed every physical step and lineages are retained.

The naive-dual baseline uses the same inner sampler but applies

```math
G_t^{naive}=\psi_t/\psi_{t-1}.
```

In the ideal locally adapted limit, this mismatch produces an effective factor
proportional to `psi_t^2`. No exact finite-`M` `q psi^2` claim is made.

The complete-frame nested baseline generates `M` full next frames, selects one
proportionally to clean `psi_t`, returns their mean potential as `Zhat_t`, and
uses the same coupled outer correction. It tests whether the checkpoint branch
is better than spending the same decoder population on complete frames.

## Attribution and boundary

Nested SMC, predictive-normalizer correction, and Feynman–Kac diffusion
steering are prior methodological foundations. DuET-MD targets
surrogate-supported, user-programmed path candidates—not exact `U+b` dynamics
or kinetics.

