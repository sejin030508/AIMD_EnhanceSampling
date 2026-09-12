# Bundle A — 기존 TPS 재평가 및 공식 ConfRover sampler 대조

작성일: 2026-09-11  
범위: 사용자가 지정한 **묶음 A만 수행**. 묶음 B 및 장거리 rollout은 실행하지 않음.

## 1. 결론

이번 이상은 **평가/파일 변환 오류도, custom adapter만의 오류도 아니다.** 가장 직접적인 원인은 현재 checkpoint가 내보내는 **raw ConfRover atom37 구조의 물리적 geometry 품질**이다.

- 공식 forward, 공식 `EulerSampler(mode=sde)`를 연결한 우리 adapter, 현재 custom unsteered SDE의 생성 frame **96/96 모두** peptide C–N 거리가 `<1.0 Å` 또는 `>1.7 Å`인 gross violation을 최소 하나 포함했다.
- Trp-cage의 공식 forward 결과 자체도 동일한 energy pipeline에서 첫/최종 생성 frame 에너지 중앙값이 `9.48e7 kJ/mol`이었다. 공식-sampler adapter는 `4.45e9`, custom SDE는 `3.65e6 kJ/mol`이었다. 세 arm 모두 정상 start/target과 수만~수백조 배 이상 다르다.
- 기존 valid-final hit 경로의 계산 가능한 maximum-energy frame 21개는 **21/21 모두 nonbonded 항이 지배**했다. 가장 큰 양의 비결합 쌍은 20/21에서 generated-heavy 대 generated-heavy였고, 거리는 중앙값 `0.235 Å`였다.
- 저장/변환은 주원인이 아니다. 선택 frame의 raw tensor→PDB roundtrip 좌표 오차는 최대 `8.49e-4 Å`; 저장된 TICA 값은 raw tensor에서 재계산한 값과 7,920 frame 전체에서 정확히 일치했다.

따라서 기존 성공 표기는 삭제하지 않되, 의미를 다음으로 제한해야 한다.

> **기존 Cα QC를 통과한 first-two-TICA hit**이지, atomistically valid transition path가 아니다.

## 2. 실행한 작업

### 2.1 기존 TPS 결과 audit

- Production 24 cells: 3 proteins × 2 methods × 2 lags × 2 seeds.
- 기존 raw `pre_final_population_atom37.npz`와 당시 TICA/validity/energy 기록을 그대로 사용.
- 모든 실제 TICA hit frame에서 backbone RMSD, common-heavy RMSD, native-contact Q, peptide geometry와 기존 whole-path validity를 다시 기록.
- 기존 valid-final hit path에서는 first generated, final, 기존 maximum-energy frame을 같은 atom-completion 및 energy pipeline에 통과시킴.
- 정상 start/target에도 같은 pipeline을 적용.
- 기존 optional ETS가 실패했던 BBA 두 경로는 max-energy frame을 추정하여 채우지 않고 `undefined`로 보존. first/final만 별도로 평가.

Energy 분해:

- bonded: `HarmonicBondForce + HarmonicAngleForce + PeriodicTorsionForce`
- nonbonded: `NonbondedForce`
- solvation: `CustomGBForce`
- 기존과 동일하게 generated heavy atom은 고정하고 template hydrogen을 완화.
- 사후 minimization 결과는 성공/validity 재분류에 사용하지 않음.

### 2.2 짧은 sampler 대조

총 `2 proteins × 3 arms × 4 trajectories × 4 transitions = 96 transitions`.

| Protein | Input / lag |
|---|---|
| Trp-cage | 기존 unfolded minimized input, stride `128 × 10 ps = 1.28 ns` |
| PRMT6 | 기존 B4 input, stride `256 × 10 ps = 2.56 ns` |

공통: base 20M checkpoint, reverse 200 steps, SDE, reward/inner/outer resampling 없음, seed 307.

| Arm | 실체 |
|---|---|
| Official forward | upstream `ConfRover._ar_sample` + upstream `EulerSampler(mode=sde)` |
| Official sampler through adapter | 우리 full-history adapter + upstream `Decoder.sample/EulerSampler(mode=sde)` |
| Current custom SDE | 현재 checkpoint-interceptable reverse loop + `diffuser.reverse(mode=sde)` |

## 3. Raw 결과

### 3.1 공식 sampler 대조 geometry

`old invalid`는 기존 Cα-adjacent 5.5 Å/CA-clash 계열 QC다. `peptide violation`은 generated frame 내 peptide C–N이 `<1.0 Å` 또는 `>1.7 Å`인 경우다.

| Protein | Arm | Generated frames | Old invalid | Peptide violation | 전체 peptide C–N 범위 (Å) |
|---|---:|---:|---:|---:|---:|
| PRMT6 | Official forward | 16 | 1 | **16** | 0.096–2.993 |
| PRMT6 | Official sampler adapter | 16 | 0 | **16** | 0.157–3.006 |
| PRMT6 | Current custom SDE | 16 | 0 | **16** | 0.094–2.882 |
| Trp-cage | Official forward | 16 | 1 | **16** | 0.073–2.007 |
| Trp-cage | Official sampler adapter | 16 | 0 | **16** | 0.180–2.133 |
| Trp-cage | Current custom SDE | 16 | 0 | **16** | 0.380–2.090 |

특이 old-QC frame:

- PRMT6 Official forward: path 0/frame 3, model residues 178–179 Cα `5.578 Å`.
- Trp-cage Official forward: path 2/frame 2, Cα clash model residues 9–17.

그러나 old QC가 94/96 generated frame을 valid로 둔 것과 달리 peptide-bond 검사는 96/96을 잡았다. 기존 Cα QC가 chain continuity를 보장하지 못한다.

### 3.2 Trp-cage 짧은 대조 energy

각 arm의 4개 path에서 first generated와 final, 총 8 frame을 같은 energy pipeline으로 평가했다.

| Arm | Energy min / median / max (kJ/mol) | 지배 항 |
|---|---:|---:|
| Official forward | `1.57e5 / 9.48e7 / 1.07e10` | nonbonded 6/8, bonded 2/8 |
| Official sampler adapter | `5.06e4 / 4.45e9 / 2.36e12` | nonbonded 7/8, bonded 1/8 |
| Current custom SDE | `2.87e5 / 3.65e6 / 6.60e17` | nonbonded 5/8, bonded 3/8 |

정상 Trp-cage start/target은 같은 pipeline에서 각각 `487.0`, `-2609.5 kJ/mol`이었다. 공식 출력도 이미 비정상 범위이므로 custom adapter가 결함의 필요조건은 아니다. 다만 custom arm의 한 frame이 가장 큰 극단값을 보였으므로 세 arm 간 상대적 품질 우열은 이 작은 표본으로 주장하지 않는다.

### 3.3 기존 TPS hit path energy 발생 지점

기존 valid-final TICA hit path:

| Protein | Path 수 | 비고 |
|---|---:|---|
| Chignolin | 0 | final valid hit 없음 |
| Trp-cage | 6 | 세 frame audit 완료 |
| BBA | 17 | 15개 max-frame audit 완료, 기존 ETS 실패 2개는 max undefined |

정상 reference:

| Protein | Start energy | Target energy | 단위 |
|---|---:|---:|---|
| Chignolin | 10.4 | -1699.0 | kJ/mol |
| Trp-cage | 487.0 | -2609.5 | kJ/mol |
| BBA | -2676.1 | -7263.7 | kJ/mol |

기존 hit paths:

| 위치 | 평가 성공 수 | Energy min / median / max (kJ/mol) | 주요 관찰 |
|---|---:|---:|---|
| First generated | 23 | `1.06e5 / 1.08e7 / 7.87e12` | nonbonded 지배 16/23, bonded 7/23 |
| Final | 22 | `3.32e4 / 1.10e9 / 3.52e12` | nonbonded 지배 20/22; BBA 1 frame OpenMM NaN |
| Max finite energy | 21 | `2.19e10 / 1.01e14 / 2.85e17` | **nonbonded 지배 21/21** |

원자 수준 원인:

- Max-energy frame의 최상위 비결합 쌍: generated-heavy/generated-heavy 20/21, generated-heavy/template-OXT 1/21.
- 해당 최상위 쌍 거리: `0.124–0.492 Å`, 중앙값 `0.235 Å`.
- First-generated frame에서도 최상위 쌍은 generated-heavy/generated-heavy 20/23이었다. H가 포함된 경우 1/23, OXT가 포함된 경우 2/23뿐이었다.
- 선택된 23개 path의 first와 final frame 모두 peptide C–N gross violation을 가졌다. First C–N 전체 범위 `0.383–2.479 Å`, final `0.356–2.473 Å`; 정상 start/target은 대략 `1.33–1.36 Å`였다.
- 즉 에너지 폭증은 추가 H/OXT만의 artifact가 아니다. generated heavy atom 간 steric overlap이 주원인이며, inter-residue peptide geometry strain도 독립적으로 존재한다.

BBA 기존 optional ETS 실패:

- `stride128/frozen/seed211/path13`
- `stride128/frozen/seed223/path11`

두 경로는 기존과 동일하게 maximum-energy frame을 정의할 수 없는 상태로 유지했다. 별도로 본 first/final은 계산됐지만, `stride16/frozen/seed211/path1`의 final 재평가에서는 OpenMM `Particle coordinate is NaN`도 재현됐다.

## 4. TICA 성공 판정 재검토

실제 TICA hit frame 수와 구조 범위:

| Protein | Actual hit frames | Whole-valid path 위 hit frames | Backbone RMSD min / median / max (Å) | Heavy native-contact Q min / median / max |
|---|---:|---:|---:|---:|
| Chignolin | 229 | 79 | 0.93 / 2.53 / 6.74 | 0.00 / 0.80 / 1.00 |
| Trp-cage | 370 | 180 | 2.06 / 4.33 / 7.96 | 0.154 / 0.538 / 0.923 |
| BBA | 593 | 467 | 2.41 / 6.12 / 11.50 | 0.267 / 0.533 / 0.933 |

해석:

- BBA의 큰 평균/중앙 RMSD만으로 TICA mapping 오류라고 단정할 수 없다. 일부 BBA hit는 RMSD `2.41 Å`, native-contact Q `0.933`까지 도달했다.
- 반대로 TICA hit 중 RMSD `11.50 Å`, Q `0.267`도 존재한다. 첫 두 TICA 좌표의 반경 `<0.75`는 low-dimensional basin 지표이지, full-structure folding 판정과 동치가 아니다.
- 따라서 TICA는 방향성/저차원 basin 관찰량으로 유지할 수 있으나, atomistic 성공은 peptide geometry, clash, RMSD/contact를 함께 통과해야 한다.

### TICA 및 파일 변환 검증

| Protein | 재계산 frame | Stored vs raw 재계산 최대 오차 | 0.001 Å 양자화 시 최대 TICA 변화 | Hit label 변화 |
|---|---:|---:|---:|---:|
| Chignolin | 2640 | 0 | 0.0275 | 1 |
| Trp-cage | 2640 | 0 | 0.0615 | 0 |
| BBA | 2640 | 0 | 0.0115 | 0 |

Chignolin의 유일한 변화는 `stride16/frozen/seed211/path7/frame4`에서 raw distance `0.750078`이 PDB 정밀도 모사 후 `0.749629`가 된 경계 사례다. final frame이 아니며 production 평가는 raw tensor로 수행했으므로 보고된 final THP에는 영향이 없다.

TICA feature path는 TPS-DPS 공식 구현과 동일하게 `mdtraj.add_backbone_torsions(cossin=True)` 후 제공된 `tica_model.pkl`의 transform을 사용한다. 저장된 값과 raw 재계산이 정확히 일치하므로 TICA 오류의 근거는 없다.

## 5. 세 원인 후보 판정

| 후보 | 판정 | 근거 |
|---|---|---|
| 평가/파일 변환 오류 | **주원인 아님** | raw→PDB 오차 `<8.5e-4 Å`; TICA raw 재계산 정확히 일치; 정상 start/target energy 정상 범위 |
| Custom adapter/SDE 문제 | **주원인 아님** | 공식 forward와 upstream sampler-adapter에서도 32/32 generated frame peptide violation; 공식 Trp-cage energy도 비정상 |
| 공식 ConfRover raw 출력의 구조 품질 | **주원인** | 세 arm 전체 96/96 peptide violation, generated-heavy steric overlap, 공식 forward 자체의 큰 energy |

보다 정확히는 checkpoint가 내놓는 **unrelaxed atom37 frame이 엄격한 peptide connectivity와 all-atom force-field plausibility를 보장하지 않는 것**이다. 이는 ConfRover가 surrogate conformation generator로 유용할 수 없다는 일반 결론이 아니라, 현재 raw 출력으로 “물리적으로 타당한 전이 경로”를 주장할 수 없다는 결론이다.

## 6. 기존 결과에 미치는 영향

- 기존 metrics, validity, THP, trajectory는 변경·삭제하지 않았다.
- Trp-cage/BBA의 TICA 접근 신호 및 DuET의 reward-direction selection 신호는 그대로 남는다.
- 그러나 이번 geometry 기준으로는 기존 hit path 중 atomistically validated path는 **0**이다.
- 기존 Cα-only validity는 논문용 physical-path validity로 사용할 수 없다.
- 묶음 B는 실행하지 않았다. 현재 판정 규칙상 공식 출력에서도 backbone/peptide 결함이 확인됐으므로, raw 구조 그대로의 8-run 우월성 검증은 보류하는 것이 맞다.

## 7. 산출물

- 통합 raw: `raw/bundle_a_merged_raw.json`
- 단백질별 TPS audit: `raw/tps_audit/{protein}/audit.json`
- 선택 frame PDB exports: `raw/tps_audit/{protein}/exports/`
- 96-transition sampler tensors/metadata: `raw/sampler_contrast/{protein}/{arm}/`
- 실행 로그: `raw/logs/`

Remote 원본 namespace:

`/workspace/sejin/AI_MD_NSMC_phase_b_recovery/outputs/bundle_a_cross_clock_audit`

