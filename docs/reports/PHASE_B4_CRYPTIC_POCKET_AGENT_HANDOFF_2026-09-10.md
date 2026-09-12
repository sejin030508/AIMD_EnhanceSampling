# Phase B4 cryptic-pocket success-search handoff

- Date: 2026-09-10 KST
- Audience: theory/coordination AI agent
- Execution role: Codex ran and audited the experiments; downstream scientific interpretation remains with the coordinating agent.
- Status: **all six prespecified B4 production runs completed**
- Main outcome: **no trajectory reached the prespecified valid holo-like endpoint or a valid anytime hit**
- Scope: DuET-only development. This phase does **not** support a method-superiority claim.

## 1. Why B4 was run

Earlier Phase B experiments did not restore a cryptic pocket to the strict
holo-like threshold. B4 therefore asked a narrower feasibility question:

> With the frozen ConfRover model and fixed DuET formulation, do stronger
> endpoint rewards produce valid candidate populations that move toward
> experimentally observed holo conformations?

B4 deliberately varied only the endpoint-reward coefficient, `a=16` versus
`a=32`, while keeping the model, seed, population size, horizon, checkpoints,
validity threshold, and target definition fixed within each protein.

The six production cells were:

```text
3 proteins x 2 reward strengths x 1 fixed seed = 6 runs
```

Frozen, Outer-only, and Complete Nested were not run. If B4 identifies a
credible steering setting, matched controls must be run later before claiming
that diffusion-time steering is superior to complete-frame or physical-time
selection.

## 2. Final protocol actually executed

| Item | Fixed setting |
|---|---|
| Proteins | PRMT6, SMARCA2, PI3Kalpha WT |
| Method | Fixed DuET only |
| Model | frozen `confrover_base_20m_v1_0` checkpoint |
| Sampler | SDE; existing 200-step reverse diffusion schedule |
| Outer/inner population | `K=4`, `M=4`; 16 next-frame candidates per transition |
| Physical horizon | `T=24` generated transitions; 25 structures including `x0` |
| Nominal lag | 2.56 ns per generated transition |
| Reward coefficients | `a=16`, `a=32` |
| Random seed | 103 for every production cell |
| Inner interventions | exactly twice, after 150 and 180 completed reverse updates: 75% and 90% |
| Outer resampling | systematic, when outer ESS is at or below `0.5K` |
| Context | full generated history; no restart, history reset, or weight reset |
| Saved analysis points | pre-resampling population and weights at T=16 and T=24 |
| Structural validity | adjacent-CA distance no greater than 5.5 Angstrom, with existing 0.001-A numerical tolerance |
| Endpoint threshold | aligned local RMSD no greater than 1.5 Angstrom |
| Resource mitigation | existing microbatch/offloading and Pairformer chunking only; no K/M/T reduction |

The 61.44 ns value from `24 x 2.56 ns` is a **nominal model horizon**. It is not
an estimate of physical opening kinetics or accelerated-MD time.

### 2.1 Endpoint potential

For generated coordinates `x`, `d(x)` is the N/CA/C RMSD of a predefined
moving region to the holo reference after alignment on a fixed protein core.
`d0` is evaluated on the starting structure using the identical function.

```text
log psi_a(x) = -a * (d(x) / d0)^2
```

The former `max(-30, ...)` potential floor was removed for B4. All inner
incremental weights, normalizers, and outer weights were evaluated in log
space with log-sum-exp operations.

The checkpoint predicted-clean score and final score use exactly the same
alignment, residue mask, atom mask, units, and reward coefficient. No contact,
ordered-event, deadline, ligand, or validity term was added to the reward.

### 2.2 Two-checkpoint Feynman--Kac weighting

The implemented inner sequence is:

1. At 75%, weight/resample with `phi75`.
2. At 90%, weight/resample with `phi90 / phi75(ancestor)`.
3. At the final clean frame, apply `psi(final) / phi90(ancestor)`.
4. Form `Zhat` from the product of mean incremental weights.
5. Update the outer history weight using `Zhat / psi(parent)`.

Every resampled branch receives independent continuation SDE noise. Thus the
second checkpoint and final correction use the potential belonging to the
actual ancestor, rather than the same array position before resampling.

### 2.3 Outcome definitions

- `valid holo-like endpoint`: the whole path is structurally valid and the
  endpoint RMSD threshold, `d <= 1.5 A`, is satisfied at both of the final two
  generated frames.
- `valid anytime hit`: the whole path is structurally valid and at least one
  generated frame satisfies `d <= 1.5 A`.
- Invalid paths remain in the denominator of all primary yield calculations.
- `weighted d/d0` is calculated from the pre-resampling outer population and
  its normalized weights. Lower is closer to the holo endpoint.
- `best valid d` is the smallest RMSD among retained paths whose full history
  is valid. It is undefined when none of the four retained paths is valid.

## 3. Protein definitions and immutable input provenance

### 3.1 PRMT6

| Field | Value |
|---|---|
| Starting construct | AlphaFold Q96LA8, residues 53--375 |
| Holo reference | PDB 6W6D |
| Moving/reward region | residues 155--165 |
| `d0` | 4.6688079085 A |
| AlphaFold source | v4; SHA256 `5b8c2685e2925101563f3d951f65e653f886fcce341044c06794a9e9fb502a9e` |
| 6W6D source SHA256 | `ff63babec1d2a699a2e89acec25ab0f026ebe864d683c142b1cedf91cc5dedea` |
| Prepared PDB SHA256 | `aaa9c1262fc28e70d46710987646cc718605f6d9eab9857f15f63a8e6eb510cb` |
| Prepared sequence SHA256 | `6690d2c4919aab251231033af64ae53c2951af9e2ca98a6d00cceb8cf747e7fc` |
| B4 manifest SHA256 | `829cc35d5c41248221c2fa294fc3e8d0e6836767578ee8a01284e5b20e40cf7a` |

The previously validated PRMT6 input, atom mapping, and alignment were reused
without changing its biological target.

### 3.2 SMARCA2

| Field | Value |
|---|---|
| Starting construct | AlphaFold P51531, residues 705--955 |
| Holo reference | PDB 6EG3 |
| Moving/reward region | residues 852--860 |
| `d0` | 9.4027980209 A |
| AlphaFold source | v6; SHA256 `7922d2e19c2ba2c23b04bf27f5af2923713193adb1ea30fbceb6d78f55ad0009` |
| 6EG3 source SHA256 | `ec57f24f4527d88ef950cfdedf9ff01ec9e67fb181482d15ab72d7109e70f139` |
| B4 manifest SHA256 | `e2dda27697b5b33f075b6a2b23e81800daf21df20935604b2738b64fd525e105` |

The 6EG3 fusion partner was excluded from reward and alignment. Sequence
mapping found zero mismatches, and all required reward backbone atoms were
present.

### 3.3 PI3Kalpha observed-mask amendment

The original full moving region, UniProt 931--957, could not be evaluated
because 8TSB lacks N/CA/C coordinates for residues 943--950. The original
`pi3ka` task remains recorded as `hold_missing_target_reward_backbone`; the
missing residues were not interpolated or silently treated as matching.

The separately named task `pi3ka_observed_mask` used the approved amendment:

| Field | Value |
|---|---|
| Starting construct | AlphaFold P42336, residues 765--1051 |
| Holo reference | PDB 8TSB |
| Original moving region | residues 931--957, 27 residues |
| Observed RMSD mask | 931--942 and 951--957, 19 residues |
| Excluded from RMSD only | 943--950, 8 residues |
| Observed coverage | 19/27 = 0.7037037 |
| Recomputed `d0` | 5.0607000520 A |
| Alignment-core atom count | 240 |
| AlphaFold source | v6; SHA256 `8042943cec4d40cf92f40dd60eec4d3e42107e70cc2da678ba3260e8ef38f244` |
| 8TSB source SHA256 | `548c788e2b7c1ad454564c468ee6d470b4ab80fd84e5343bf0afce7e062aa5a2` |
| B4 manifest SHA256 | `bc2c74538623b6846cfdf81a041684d49d176b494f1f97972b27ddf81b153a24` |

The full generated 943--950 segment remains present in path validity and
ligand-space evaluation. The alignment core excludes the full original
931--957 region and its existing guard band; excluded RMSD residues were not
returned to the core. PI3K results are therefore named
`observed_loop_rmsd`/`observed_loop_holo_like_fraction` and cannot be presented
as full-loop restoration.

## 4. Runtime audit and implementation verification

Every production run reports:

- exactly 76,416 decoder NFE;
- exactly 1,153 potential evaluations;
- checkpoint steps 150 and 180;
- both requested inner interventions observed;
- maximum telescoping residual `0.0` at stored numerical precision;
- zero potential-floor-clipped evaluations;
- a passing `checkpoint_audit.json`.

The runtime aborts if the checkpoints are absent or misordered, the ancestor
ratios fail to telescope within `1e-10`, or no-floor execution reports any
clipped evaluation. Focused DuET/B4 tests also passed before production launch.

All six runs used checkpoint SHA256:

```text
4ba22c4a83c21fc139b689cfb63ff48bbf451346ff3995d658cf158907f4c4c6
```

Relevant code-file SHA256 values at launch:

| File | SHA256 |
|---|---|
| `phase_b_runner.py` | `d4158ca210d638798111b560b64e41c19a7046dfe99160c005dbf8389ed1dfba` |
| `phase_b_pockets.py` | `420ba7bbdc4de89449d799154aa8efdf45a512bbefef5c5c8ce5244d60c0e9da` |
| `inner_fkc.py` | `4d0fcdf1f708ee5606215c9746f7afc7b296008f21d4072ec263f00612575371` |
| `outer_smc.py` | `daa7ed9556183cd4724432f779452cc4413f77a2c52e2c0448be896245149efa` |
| `confrover_duet.py` | `063876bc204db2cb362c727a03d3b55a44998a097393424269ee55f0cea91040` |
| `run_phase_b4_cell.sh` | `e6364ac67fac5ac3d7bde27e8382d50f549849d00688d9da5e95c0f0c3dace89` |

Git was not available inside the runtime image, so commit state could not be
recorded there. The immutable file/checkpoint/manifest hashes above are the
operative provenance record.

## 5. Production raw-result summary

All endpoint and anytime-hit columns below use the strict 1.5-A threshold and
whole-path validity rule. `T16/T24 valid` denotes valid retained paths out of
four. `Peak` is PyTorch maximum allocated memory, not node-wide `nvidia-smi`
usage.

| Protein | a | d0 A | T16 weighted d/d0 | T16 best valid A | T16 valid | T24 weighted d/d0 | T24 best valid A | T24 valid | Valid endpoint | Valid anytime | Decoder NFE | Wall min | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PRMT6 | 16 | 4.66881 | 0.844722 | 3.87787 | 4/4 | 0.898249 | 3.93455 | 4/4 | 0/4 | 0/4 | 76,416 | 54.24 | 26.63 |
| PRMT6 | 32 | 4.66881 | 0.766617 | -- | 0/4 | 0.723687 | -- | 0/4 | 0/4 | 0/4 | 76,416 | 73.06 | 26.63 |
| SMARCA2 | 16 | 9.40280 | 0.702426 | 5.89644 | 3/4 | 0.696743 | 5.79088 | 4/4 | 0/4 | 0/4 | 76,416 | 53.33 | 16.15 |
| SMARCA2 | 32 | 9.40280 | 0.520481 | 4.66666 | 4/4 | 0.533514 | 4.81618 | 4/4 | 0/4 | 0/4 | 76,416 | 55.69 | 16.15 |
| PI3Kalpha observed | 16 | 5.06070 | 0.916366 | 4.35043 | 3/4 | 0.866690 | 4.35104 | 4/4 | 0/4 | 0/4 | 76,416 | 64.12 | 21.06 |
| PI3Kalpha observed | 32 | 5.06070 | 0.824862 | 4.11904 | 4/4 | 0.801391 | 4.05433 | 4/4 | 0/4 | 0/4 | 76,416 | 63.75 | 21.06 |

Total production decoder work was 458,496 NFE. The sum of individual GPU wall
times was approximately 6.07 GPU-hours. The six cells were distributed across
three H100 pods, two consecutive strengths per protein.

### 5.1 Selection statistics

| Protein | a | Inner ESS 75% mean/min | Inner ESS 90% mean/min | Outer ESS mean/min/final | Outer resamples | Final log Z |
|---|---:|---:|---:|---:|---:|---:|
| PRMT6 | 16 | 2.9478 / 1.1196 | 3.1163 / 1.7464 | 2.5023 / 1.3440 / 2.7249 | 6 | -27.0243 |
| PRMT6 | 32 | 2.6293 / 1.0177 | 2.6363 / 1.0932 | 2.1531 / 1.0166 / 3.0723 | 12 | -49.8780 |
| SMARCA2 | 16 | 2.5361 / 1.0695 | 2.9632 / 1.3902 | 2.5316 / 1.1940 / 2.2757 | 6 | -22.8808 |
| SMARCA2 | 32 | 2.0294 / 1.0056 | 2.4849 / 1.0104 | 2.0962 / 1.0633 / 3.2993 | 11 | -47.8082 |
| PI3Kalpha observed | 16 | 3.0366 / 1.5427 | 2.9048 / 1.2499 | 2.4655 / 1.2631 / 1.9385 | 6 | -20.3332 |
| PI3Kalpha observed | 32 | 2.5096 / 1.1014 | 2.5018 / 1.2149 | 2.1304 / 1.0286 / 1.0286 | 9 | -45.5922 |

Strength 32 generally imposed materially stronger selection: lower inner/outer
ESS and more outer resampling. PI3Kalpha a=32 ended with outer ESS 1.0286,
which is near a single surviving outer lineage.

## 6. Raw retained populations at T=16 and T=24

These are pre-resampling populations. Distances are Angstrom; weights are
normalized outer weights in the corresponding path order.

### 6.1 PRMT6, a=16

```text
T16 d      = [4.196041, 3.877867, 3.902523, 4.039683]
T16 weight = [0.148961, 0.480711, 0.301426, 0.068902]
T16 valid  = [T, T, T, T]

T24 d      = [4.142421, 4.276812, 3.934546, 4.111558]
T24 weight = [0.229387, 0.529159, 0.069672, 0.171782]
T24 valid  = [T, T, T, T]
```

### 6.2 PRMT6, a=32

```text
T16 d      = [3.362381, 3.447114, 3.517921, 3.656840]
T16 weight = [0.151976, 0.069237, 0.132316, 0.646471]
T16 valid  = [F, F, F, F]

T24 d      = [3.494776, 3.139603, 3.573685, 3.679005]
T24 weight = [0.176810, 0.474701, 0.110356, 0.238132]
T24 valid  = [F, F, F, F]
```

### 6.3 SMARCA2, a=16

```text
T16 d      = [6.987261, 5.896439, 6.202185, 6.004645]
T16 weight = [0.600371, 0.156296, 0.137178, 0.106155]
T16 valid  = [F, T, T, T]

T24 d      = [5.790875, 7.065321, 6.718900, 6.037735]
T24 weight = [0.169573, 0.127649, 0.622892, 0.079886]
T24 valid  = [T, T, T, T]
```

### 6.4 SMARCA2, a=32

```text
T16 d      = [5.050878, 4.726511, 4.986036, 4.666662]
T16 weight = [0.144005, 0.116056, 0.516777, 0.223163]
T16 valid  = [T, T, T, T]

T24 d      = [4.816184, 4.836699, 5.076523, 5.106675]
T24 weight = [0.157073, 0.132521, 0.289942, 0.420464]
T24 valid  = [T, T, T, T]
```

### 6.5 PI3Kalpha observed mask, a=16

```text
T16 d      = [4.703797, 4.988615, 4.757238, 4.350431]
T16 weight = [0.523264, 0.157634, 0.003731, 0.315370]
T16 valid  = [T, T, F, T]

T24 d      = [4.376128, 4.544406, 4.351036, 4.758979]
T24 weight = [0.214324, 0.060007, 0.681441, 0.044228]
T24 valid  = [T, T, T, T]
```

### 6.6 PI3Kalpha observed mask, a=32

```text
T16 d      = [4.289186, 4.267559, 4.119039, 4.181883]
T16 weight = [0.037113, 0.003670, 0.187821, 0.771396]
T16 valid  = [T, T, T, T]

T24 d      = [4.244078, 4.109228, 4.054333, 4.251475]
T24 weight = [0.003351, 0.010371, 0.985954, 0.000324]
T24 valid  = [T, T, T, T]
```

## 7. Structural-validity details

| Cell | Final retained max adjacent CA | Invalid-frame observation |
|---|---:|---|
| PRMT6 a=16 | 5.36761 A | none |
| PRMT6 a=32 | 5.55908 A | all four retained paths contain the same invalid frame-4 ancestor |
| SMARCA2 a=16 | 5.45231 A | one T16 lineage had frame 11 at 5.50406 A; absent from final retained population |
| SMARCA2 a=32 | 5.45600 A | none |
| PI3Kalpha observed a=16 | 5.43521 A | one T16 lineage had frame 12 at 5.56980 A; absent from final retained population |
| PI3Kalpha observed a=32 | 5.49155 A | none |

The PRMT6 a=32 table contains four invalid frame records because one frame-4
structural defect was copied into all four final lineages by ancestry. It is
not evidence for four independent defects. Nevertheless, all four whole paths
are invalid under the prespecified rule, so their lower RMSD cannot be counted
as a valid steering success.

No production cell reported NaN coordinates or the separate major-clash flag.

## 8. Evaluation-only pocket observables

Holo-ligand coordinates were never supplied to the model and never entered the
reward. Generated protein coordinates were aligned using the same fixed core,
then protein-heavy-atom proximity to holo-ligand heavy atoms was recorded at
several cutoffs. These are continuous diagnostics, not a predefined binary
pocket-opening endpoint.

### 8.1 SMARCA2 ligand-space occupancy

| State/cell | Minimum protein-ligand distance A | Ligand atoms within 1.5 A | within 2.0 A | within 2.5 A |
|---|---:|---:|---:|---:|
| Start | 0.5407 | 11.0000 | 20.0000 | 25.0000 |
| Holo reference | 2.6765 | 0.0000 | 0.0000 | 0.0000 |
| a=16 weighted T24 | 0.4659 | 12.2611 | 21.1561 | 25.1696 |
| a=32 weighted T24 | 0.5055 | 11.0609 | 21.4531 | 23.5035 |

SMARCA2 a=32 produced the largest valid local-RMSD approach, but holo-ligand
space remained heavily occupied. The minimum distance remained approximately
0.5 A, far from the holo reference, and short-cutoff occupancy did not show
convincing clearance. Therefore local RMSD improvement is not pocket opening.

### 8.2 PI3Kalpha ligand-space occupancy

| State/cell | Minimum protein-ligand distance A | Ligand atoms within 1.5 A | within 2.0 A | within 2.5 A |
|---|---:|---:|---:|---:|
| Start | 0.4360 | 23.0000 | 27.0000 | 28.0000 |
| Holo reference | 2.6611 | 0.0000 | 0.0000 | 0.0000 |
| a=16 weighted T24 | 0.5842 | 14.8900 | 21.8299 | 27.3128 |
| a=32 weighted T24 | 0.5475 | 15.9585 | 22.0281 | 26.9726 |

The generated structures show some reduction from the starting occupancy at
1.5/2.0 A, but still strongly overlap the holo ligand space and remain far
from the holo reference. Because 8TSB also lacks coordinates for part of the
original moving loop, this result is only an observed-residue local metric,
not evidence of full-loop or pocket recovery.

### 8.3 PRMT6 hidden-contact observables

The pre-existing PRMT6 evaluation-only observables were retained:

| State/cell | H163--M373 minimum distance A | L161 centroid distance to holo A |
|---|---:|---:|
| Start | 3.8320 | 14.3949 |
| Holo | 7.3328 | approximately 0 |
| a=16 weighted T24 | 3.2898 | 11.6786 |
| a=32 weighted T24 | 5.1877 | 8.6227 |

The a=32 population moves these observables more strongly toward the holo
state, but every retained path is invalid because of the shared frame-4 defect.
It cannot be treated as a successful mechanistic trajectory.

## 9. Compact interpretation for the coordinating agent

### 9.1 Findings supported by these runs

1. Increasing the reward coefficient from 16 to 32 reduced the T24 weighted
   local `d/d0` for every protein:
   - PRMT6: 0.898 to 0.724, but the a=32 population is wholly invalid.
   - SMARCA2: 0.697 to 0.534, with 4/4 valid final paths.
   - PI3Kalpha observed mask: 0.867 to 0.801, with 4/4 valid final paths.
2. The stronger reward creates stronger selection pressure, not merely a
   rescaled reported score. This appears as lower ESS and more resampling.
3. SMARCA2 a=32 is the clearest **valid local-RMSD approach signal** in B4.
4. PI3Kalpha a=32 shows a smaller valid observed-loop approach, but the final
   population weight collapses almost completely onto one path.
5. Extending a run from T16 to T24 was not universally helpful:
   - worse: PRMT6 a=16, SMARCA2 a=32;
   - slightly better: PRMT6 a=32, SMARCA2 a=16, both PI3Kalpha cells.

### 9.2 Findings not supported

1. No cell produced a valid final-two-frame holo-like endpoint.
2. No cell produced a valid anytime 1.5-A hit.
3. Neither SMARCA2 nor PI3Kalpha cleared the experimentally observed ligand
   space sufficiently to call the pocket open.
4. PRMT6 a=32 cannot be counted as a valid approach because a structural
   defect was ancestrally propagated through the complete population.
5. One seed per cell gives no estimate of stochastic uncertainty or
   reproducibility.
6. B4 contains no baseline methods and therefore says nothing about DuET's
   advantage over Complete Nested, Outer-only, Frozen, or simple endpoint
   selection.
7. Nominal generated time must not be interpreted as kinetic acceleration.

### 9.3 Suggested categorical labels

| Cell | Conservative label | Reason |
|---|---|---|
| PRMT6 a=16 | stalled / small valid approach | valid, but remains near 3.9--4.3 A and worsens from T16 to T24 |
| PRMT6 a=32 | structural failure with confounded approach | stronger RMSD/contact movement, but 0/4 valid due to shared defect |
| SMARCA2 a=16 | partial local approach | valid final population, but no holo-like hit or ligand clearance |
| SMARCA2 a=32 | most promising local-RMSD signal; pocket not recovered | strongest valid `d/d0`, but ligand space remains occupied |
| PI3Kalpha observed a=16 | weak partial observed-loop approach | final paths valid, no endpoint or pocket clearance |
| PI3Kalpha observed a=32 | partial observed-loop approach with degeneracy | somewhat closer, but terminal outer ESS is 1.0286 and space remains occupied |

## 10. Questions for the coordinating agent

The next action should not be selected automatically from these data. The
coordinating agent should decide:

1. Whether SMARCA2 a=32 has enough valid, population-level signal to justify a
   single fresh-seed confirmation with all settings frozen.
2. Whether PI3Kalpha a=32 should be repeated despite severe population-weight
   concentration and the observed-residue-only endpoint.
3. Whether PRMT6 a=32 should first receive structural-defect diagnosis instead
   of a repeat; its current apparent improvement is invalid by protocol.
4. What continuous or preregistered structural criterion, beyond local RMSD,
   will qualify as physical pocket opening before further success-search runs.
5. If a setting is confirmed, when to freeze it and run matched DuET,
   Complete Nested, and Outer-only controls. Hyperparameters must not be chosen
   separately after observing each baseline.

No automatic a=64, alternative checkpoint, longer horizon, extra seed, or
baseline sweep was launched after the six specified B4 runs.

## 11. Artifact locations

### 11.1 Project and output roots

```text
Project:
/workspace/sejin/AI_MD_NSMC_phase_b_recovery

Production output root:
/workspace/sejin/phase_b_pockets_recovery/outputs/B4_pocket_search
```

### 11.2 Production run directories

```text
PRMT6 a=16:
/workspace/sejin/phase_b_pockets_recovery/outputs/B4_pocket_search/prmt6/b4_duet_a16_t24_nofloor_cp075_090/duet/seed_103

PRMT6 a=32:
/workspace/sejin/phase_b_pockets_recovery/outputs/B4_pocket_search/prmt6/b4_duet_a32_t24_nofloor_cp075_090/duet/seed_103

SMARCA2 a=16:
/workspace/sejin/phase_b_pockets_recovery/outputs/B4_pocket_search/smarca2/b4_duet_a16_t24_nofloor_cp075_090/duet/seed_103

SMARCA2 a=32:
/workspace/sejin/phase_b_pockets_recovery/outputs/B4_pocket_search/smarca2/b4_duet_a32_t24_nofloor_cp075_090/duet/seed_103

PI3Kalpha observed mask a=16:
/workspace/sejin/phase_b_pockets_recovery/outputs/B4_pocket_search/pi3ka_observed_mask/b4_duet_a16_t24_nofloor_cp075_090/duet/seed_103

PI3Kalpha observed mask a=32:
/workspace/sejin/phase_b_pockets_recovery/outputs/B4_pocket_search/pi3ka_observed_mask/b4_duet_a32_t24_nofloor_cp075_090/duet/seed_103
```

Each completed directory contains:

```text
assets.json
checkpoint_audit.json
command.txt
environment.json
git_state.json
intermediate_metrics.json
metrics.json
outer_ancestry.json
population_curves.json
population_curves_t16.json
population_curves_t24.json
post_resampling_population_atom37.npz
pre_final_normalized_weights.json
pre_final_population_atom37.npz
pre_resampling_normalized_weights_t16.json
pre_resampling_normalized_weights_t24.json
pre_resampling_population_t16_atom37.npz
pre_resampling_population_t24_atom37.npz
records.json
resolved_config.yaml
stderr.log
stdout.log
```

The six production result directories occupy approximately 66 MB in total.

### 11.3 Excluded attempts and preflights

Smoke/preflight results are excluded from every scientific table in this
document. They were used only to confirm mapping, memory feasibility, no-floor
execution, checkpoint indices, and the runtime audit.

Two interrupted PRMT6 attempts were preserved for provenance and excluded:

```text
/workspace/sejin/phase_b_pockets_recovery/outputs/prmt6/_aborted_pre_B4_pocket_search_namespace_a16_20260910

/workspace/sejin/phase_b_pockets_recovery/outputs/B4_pocket_search/prmt6/_aborted_before_anytime_valid_metric_20260910
```

Neither interrupted directory contains a completed production `metrics.json`
or trajectory. A transient launcher/read issue after completion of PRMT6 a=16
was resolved by restarting the independent a=32 cell; no completed production
cell was overwritten or counted twice.

## 12. Final handoff statement

B4 is computationally complete and all three H100 workers are idle. The
experiment shows that `a=32` can shift local endpoint RMSD more strongly than
`a=16`, most cleanly for SMARCA2, but **cryptic-pocket restoration was not
achieved under the prespecified outcome**. The next scientifically defensible
step is a coordination decision about a narrowly scoped fresh-seed validation
and a preregistered physical-opening criterion, followed—only if confirmed—by
matched baseline comparisons.
