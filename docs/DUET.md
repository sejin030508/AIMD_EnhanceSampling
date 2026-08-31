# DuET-MD

DuET-MD tests whether a frozen, full-history protein dynamics emulator contains
dynamically coherent but low-probability transition paths that can be exposed by
an unseen inference-time temporal program.

There are two distinct particle clocks:

1. Physical-time particles (`K`) preserve complete chronological histories and
   compete across trajectory frames.
2. Reverse-diffusion particles (`M`) search within one learned next-frame
   transition and branch once, after 75% of reverse denoising steps by default.

The method returns **surrogate-consistent, user-programmed transition-path
candidates**. Throughout the code and reports, the emulator is an **MD-derived
transition prior**. No exact physical-law, `U+b`, transition-rate, MFPT,
committor, or unbiased path-probability claim is made.

## Prior foundations

The following are inherited foundations, not DuET-MD contributions:

- Full-history autoregressive trajectory generation and SE(3) diffusion decoding
  are provided by ConfRover.
- Properly weighted inner outputs and predictive-normalizer correction follow
  [Nested Sequential Monte Carlo Methods](https://proceedings.mlr.press/v37/naesseth15.html).
- Diffusion-time particle twisting follows
  [Twisted Diffusion Sampling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/63e8bc7bbf1cfea36d1d1b6538aecce5-Abstract-Conference.html)
  and [Feynman–Kac diffusion steering](https://proceedings.mlr.press/v267/singhal25a.html).
- Telescoping potential ratios, fully adapted local proposals, selected-child
  plus normalizer interfaces, and generic K/M allocation are standard SMC/NSMC
  constructions.

The tested contribution is the use of these foundations at two separated clocks
inside a frozen monomer-scale, full-history protein dynamics emulator for
chronological, user-programmed molecular events.

## Safety and claim boundary

Success means the path satisfies a configured program and remains consistent
with held-out MD intermediate features and structural validity checks. Finite
failure is reported as **unresolved under the tested particle budget**; it is not
interpreted as zero model support.

