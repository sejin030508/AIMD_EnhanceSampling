# DuET-MD integration map

## Reused assets

The existing project is treated as read-only at
`/workspace/sejin/confrover_mh_steering`. The verified ConfRover checkout commit
is `30af5c3bfaee8497f8dfdf7f1c16097e843a0246`.

| Scientific role | Existing implementation or asset | DuET integration |
|---|---|---|
| Model loading | `external/ConfRover/src/confrover/model/confrover.py::ConfRover.from_pretrained` | `src/confmh/adapters/confrover_duet.py::load_model` |
| Full-history encoding | `ConfRover._ar_sample`, structure encoder, temporal Llama | Complete stored histories are re-encoded; stale KV caches are never shared across ancestors |
| Temporal state | `outputs.past_key_values` with Offloaded/Sink cache | Correctness path recomputes the full history before optimization |
| Reverse sampler | `model/decoder/confdiff/sampler/euler.py::EulerSampler.reverse_sample` | Local checkpoint-capable mirror in `ConfRoverDuETAdapter` |
| Predicted clean frame | `model_nn` outputs `pred_rigids_0` and `pred_atom14` | Cached checkpoint output supplies `x0_hat` and is reused for the next reverse step |
| ODE/SDE control | `EulerSampler.mode`; `se3_diffuser.reverse(..., mode=...)` | Configured by `model.sampler_mode`; SDE copies receive deterministic independent seed streams |
| Particle state | Upstream rigid, score, mask, and conditioning tensors | `ConfRoverParticleState`; recursive resampling copies every batched field |
| Coordinate conversion | `atom14_to_atom37`; upstream Writer | `ConfRoverFrame` uses the same atom14-to-atom37 conversion; trajectories are saved as atom37 arrays |
| PCA/PC1 | Existing `src/confmh/pca_cv.py::PCACV` | Reused unchanged; DuET design PCA is fit on R1/R2 only |
| ATLAS data | `data/atlas/6j56_A/*R1/R2/R3*_fit.xtc` | R1/R2 task design, R3 held-out evaluation |
| Base checkpoint | `data/confrover_cache/confrover_ckpts/confrover_base_20m_v1_0.pt` | Reused, never redownloaded |
| Interpolation checkpoint | Registry entry exists; local file not yet verified | Phase E remains asset-gated |
| Official benchmark list | Not found in the current ConfRover checkout | Required manifest path is explicit; no replacement list is invented |
| NFE accounting | Existing model call structure | `NFEAccounting` counts particle decoder evaluations, predicted-clean reads, temporal encodes, rewards, and complete frames |

The ConfRover public `forward` input schema accepts one conditioning frame. It
cannot preserve branched full histories by repeatedly calling the public API.
The adapter therefore reuses the internal encoder and temporal model to encode
each complete selected history. This is slower than cache cloning but is the
correct first implementation.

No vendor file is modified. If later profiling establishes that a vendor hook
is unavoidable, it must be isolated under `patches/` and the unmodified path
must remain runnable.

