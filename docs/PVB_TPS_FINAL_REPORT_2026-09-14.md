# PVB 기반 fast-folding TPS 실험 — 최종 보고서

작성일: 2026-09-14
대상: PVB(Pretrained Variational Bridge)를 DuET-MD의 두 번째 frozen backend로 쓴
fast-folding 단백질 transition path sampling 실험
총 셀: **372** (메인 360 + K/M 보조 12), 실패 셀 0, 총 연산 **19.4 GPU-hours**

> **한 문장.** Steering을 수행하는 두 방법(DuET, Complete Nested)은 Frozen 대비
> backbone RMSD를 2.2–3.0 Å 낮추고 이는 압도적으로 유의하지만(|t| = 17–36),
> **DuET과 Complete Nested는 어떤 지표에서도 구분되지 않으며(|t| < 1)**, TICA 타깃
> 도달률에서 Frozen이 유일하게 앞서는 경우는 세 단백질 중 BBA 하나뿐이고 그마저
> 다중비교를 고려하면 유의하지 않다.

> **이 보고서는 선행 보고서 2건의 핵심 주장 2개를 정정한다.** 이전 결론이었던
> "native contact Q는 Frozen이 가장 높다"와 "DuET의 타깃 도달률이 세 단백질 모두에서
> 최하위다"는 모두 **경로 수 불일치에서 온 극값 통계 교란**이었다. §3.3과 §3.5 참조.

---

## 1. 실험의 목적

DuET-MD는 두 개의 시계(clock)를 가진 중첩 SMC다.

- **바깥 시계**: 물리적 궤적 시간 t = 0…T. 입자 K개에 대한 Feynman–Kac 필터.
- **안쪽 시계**: 각 전이 한 스텝을 만드는 diffusion/bridge 역과정. 입자 M개에 대한 FK 필터.

핵심 주장은 "동일한 연산 예산 K·M을 diffusion 중간에 재배치하면(= DuET), 완성 후에
고르는 것(= Complete Nested)보다 낫다"이다. 이 실험은 **frozen emulator를 ConfRover에서
PVB로 바꿔도 그 주장이 성립하는지**를 fast-folding TPS에서 검정한다.

PVB를 고른 이유는 DuET이 요구하는 두 조건을 만족하는 몇 안 되는 공개 모델이기 때문이다.

| 요구 조건 | 이유 | PVB | PLaTITO | ProTDyn |
| --- | --- | --- | --- | --- |
| 확률적 sampler (SDE) | 안쪽 FK가 분기할 수 있어야 함 | ✅ σ=0.2 SDE | ❌ ODE 전용 | ✅ |
| 전원자/중원자 출력 | RMSD reward·validity gate가 backbone 필요 | ✅ heavy atom | ❌ CA-only | ✅ |
| 공개 체크포인트 | frozen 사용 | ✅ | ✅ | 부분 |

PLaTITO는 **결정론적 ODE**라 안쪽 시계에서 입자가 갈라지지 않고(M개가 전부 같은 값),
**CA-only coarse-grained**라 backbone reward를 정의할 수 없다. 두 가지 모두 우회가
아니라 구조적 배제 사유여서 후보에서 제외했다.

---

## 2. 실험 설정

### 2.1 Backend — PVB

| 항목 | 값 |
| --- | --- |
| 논문 | Pretrained Variational Bridge, ICLR 2026, arXiv:2602.07588 |
| 저장소 | `yaledeus/PVB` @ `c08e5e3c` |
| 모델 클래스 | `dyVAE`, 파라미터 9,965,317개 |
| 학습 방식 | augmented bridge matching |
| 백본 | torchmdnet |
| 체크포인트 | `pvb_atlas.ckpt` (ATLAS fine-tuned), 39,966,301 B, sha256 `574de9d5…57c1b8` |
| 물리 lag | **100 ps** (ATLAS stride), 설정상 `physical_lag_in_10ps: 10` |
| sampler | SDE, σ = 0.2, **Markov** (직전 프레임만 조건) |
| 출력 | heavy atom (수소 제거) |
| **`sde_step`** | **20** (공개 기본값 10에서 상향) |

`sde_step = 20`은 편의가 아니라 **필요조건**이다. DuET의 안쪽 checkpoint는 역과정
진행률(0.25/0.50/0.75/0.90)에 대응하는 정수 스텝에 놓여야 하는데, `sde_step = 10`이면
0.90이 마지막 결정론적 스텝에 걸려 분기 폭이 정확히 0.0이 된다 — 안쪽 resampling이
아무 일도 하지 않는다. 20에서는 0.25→5, 0.50→10, 0.75→**15**, 0.90→**18**로
모두 노이즈가 남은 스텝에 떨어진다(실행 로그 `checkpoint_audit.json`의
`resolved_reverse_update_steps: [15, 18]`로 확인).

### 2.2 어댑터 구현 — `src/confmh/adapters/pvb_duet.py`

PVB는 DuET을 염두에 두고 만들어진 모델이 아니므로, 안쪽 시계를 열어 놓기 위한
구현 결정이 네 가지 있었다. 각각 근사가 아니라 **모델 자신의 정의를 역산한 것**이다.

**(a) clean endpoint 복원.** 안쪽 FK의 potential은 "이 중간 상태가 결국 어디로
갈 것인가"를 평가해야 하는데, PVB는 중간 상태 `x_t`만 내놓는다. PVB의 drift head는
`(x₁ − x_t)/(1 − t)`를 타깃으로 학습되므로 역산이 닫힌 형태로 존재한다.

```python
# Drift head target is (x1 - xt) / (1 - t); inverting it gives the
# model's own endpoint estimate.
predicted = particle_state.xt + (1.0 - t) * particle_state.cached_drift
```

이것은 별도 근사가 아니라 모델 자신의 endpoint 추정이며, 진행률에 따라 단조적으로
정확해짐을 사전 검증에서 확인했다.

**(b) 입자별 독립 노이즈.** PVB의 배치는 입자를 **원자 축에 이어 붙이는** 구조라
배치 차원으로 노이즈를 분리할 수 없다. 입자·스텝별로 결정론적 시드를 합성했다.

```python
seed = int(np.random.SeedSequence(
    [int(state.noise_seeds[index]), 1, int(state.step)]
).generate_state(1, dtype=np.uint32)[0])
generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
```

**(c) resampling 시 `x_rep` 동반 이동.** PVB의 모든 decode는 표현 텐서 `x_rep`에
조건부다. 안쪽 resampling에서 `xt`만 모으고 `x_rep`을 두고 오면 입자의 정체성이
섞인다. `resample_particle_state`는 **`xt`와 `x_rep`을 함께** gather한다.

**(d) 프레임 표현.** 출력은 `ConfRoverFrame`(atom37 레이아웃)으로 내보내
reward·observable·validity·저장 경로를 backend 교체 없이 그대로 쓴다. atom37은
중원자 전용이므로 **매핑되지 않는 원자는 0**으로 둔다.

분할된 안쪽 루프가 PVB의 공식 루프를 재현하는지는 별도로 검정했고, 차이는 PVB 자신의
CUDA 비결정성 바닥 이내였다(`scripts/duet/pvb_preflight/pvb_determinism.py`).

### 2.3 대상 단백질과 자산

출발/도착 구조와 TICA 모델은 모두 **TPS-DPS 공식 저장소**
(`kiyoung98/tps-dps` @ `61fd65ad`)에서 가져왔다. 자체 재선정은 하지 않았다.

| 단백질 | 길이 | SEQRES | d₀ (Å) | Rg 출발 / folded | 비율 |
| --- | ---: | --- | ---: | --- | ---: |
| Trp-cage | 20 | `DAYAQWLKDGGPSSGRPPPS` | 7.200 | 10.19 / 7.01 | 1.45 |
| BBA | 28 | `EQYTAKYKGRTFRNEKELRDFIEKFKGR` | 8.070 | 13.13 / 9.09 | 1.44 |
| BBL | 47 | `GSQNNDALSPAIRRLLAEWNLDASAIKGTGVGGRLTREDVEKHLAKA` | 18.074 | **23.45 / 10.35** | **2.27** |

Chignolin은 사용자 지시로 제외했고, Villin은 서열/구성 불일치로 제외했다.
출발 구조는 `unfolded_minimized_allatom.pdb`(전원자 minimization 완료)를 쓴다.

### 2.4 Reward (outer potential)

```
log ψ_t = −16 · (d_RMSD(x_t, target) / d₀)²
```

- `d_RMSD`: **공통 N/CA/C 원자**에 대한 Kabsch RMSD, 전 잔기 범위
- `reward_coefficient = 16.0`, `program.type = terminal`
- `d₀`: 단백질별 정규화 상수(위 표)

바깥 resampling은 `systematic`, ESS 임계 0.5. 안쪽 resampling도 `systematic`.

### 2.5 Validity gate

프레임이 물리적으로 성립하는지를 4중으로 본다. **전 경로의 모든 프레임이 통과해야**
그 경로가 valid로 집계된다(`whole_path_valid_fraction`).

| 검사 | 기준 |
| --- | --- |
| 좌표 유한성 | 모든 좌표 finite |
| CA 충돌 | 비인접 CA 쌍 ≥ 1.0 Å |
| CA 연결 | 인접 CA ≤ 5.501 Å (품질 임계 4.5 Å) |
| **펩타이드 결합** | **C–N ∈ [1.0, 1.7] Å** (`enforce_peptide_bond: true`) |

네 번째 항목은 이번에 새로 추가했다. 선행 ConfRover 감사(Bundle A)에서 **공식 forward
sampler를 포함해 96/96 프레임이 펩타이드 C–N 위반**을 냈는데 CA-only gate가 이를
전혀 잡지 못했던 사례가 있었기 때문이다.

### 2.6 평가 지표

| 지표 | 정의 | 비고 |
| --- | --- | --- |
| `weighted_final_backbone_rmsd_a` (wBB) | SMC 가중 평균한 최종 프레임 backbone RMSD | **reward가 직접 최적화하는 양** |
| `weighted_final_heavy_rmsd_a` | 전 중원자 Kabsch RMSD | |
| `q_final_mean` / `q_final_max` | native contact Q의 경로 평균 / 경로 최댓값 | 정의: 참조 잔기쌍(최소 중원자 거리 < 4.5 Å, \|i−j\| ≥ 3)이 참조 거리의 1.2배 이내에 형성된 비율 |
| TICA 최소거리 | 첫 두 backbone-torsion TICA 좌표에서 타깃까지의 최근접 거리 | |
| `valid_anytime_hit` | 경로 중 **한 프레임이라도** TICA 거리 < 0.75 에 들어가고 경로 전체가 valid | **벤치마크의 성공 조건** |
| `weighted_valid_thp` | 가중 transition path 확률 | hit + 가중 |

### 2.7 방법 정의

여섯 가지 baseline 중 이번 행렬에서 실행한 것은 **세 가지**다. 셋 다 decoder
population budget K·M = **64**로 고정해 연산량을 맞췄다(`decoder_nfe = 40,960` 동일).

| 방법 | K | M | 바깥 resampling | 안쪽 resampling | 성격 |
| --- | ---: | ---: | --- | --- | --- |
| **Frozen** | 64 | 1 | ❌ | ❌ | 독립 rollout 64회. steering 없음. 순수 사전확률 |
| **Complete Nested** | 8 | 8 | ✅ | ❌ (완성 후 선택) | 전이를 끝까지 만든 뒤 고름 |
| **DuET** | 8 | 8 | ✅ | ✅ (diffusion 중간) | 완성 전 중간 checkpoint에서 고름 |

**Frozen은 문자 그대로 독립 64회 실행이다** — increment = 0, resampling 없음.
따라서 checkpoint timing과 무관하며, 같은 이유로 Complete Nested도 timing에
반응하지 않는다(두 방법 모두 `denoise_to_checkpoint`를 통과하지만 거기서 아무 일도
하지 않고 동일 시드로 재시딩된다). 그래서 이 둘은 timing 하나(cp7590)에서만 돌렸다.

**DuET과 Complete Nested의 유일한 차이는 안쪽 resampling 여부**다. 나머지 —
K, M, 예산, 바깥 필터, reward, 시드 — 는 완전히 동일하다. 이것이 이 실험의
핵심 대조이며, 두 팔의 차이는 오직 "diffusion 중간에 개입하는가"로 귀속된다.

### 2.8 실험 행렬

**Horizon T = 32**, PVB lag 100 ps ⇒ 물리 시간 **3.2 ns**. stage `stride128_t32`.

checkpoint timing 4종 (역과정 진행률):

| 태그 | `inner_checkpoint_progresses` | `sde_step=20` 기준 스텝 |
| --- | --- | --- |
| `cp2550` | 0.25, 0.50 | 5, 10 |
| `cp5075` | 0.50, 0.75 | 10, 15 |
| `cp7590` | 0.75, 0.90 | 15, 18 |
| `cp255075` | 0.25, 0.50, 0.75 | 5, 10, 15 |

Seed: **20개**, `7 + 12i` (i = 0…19) ⇒ 7, 19, 31, …, 235.
등차 12로 잡은 이유는 초기 3-seed 실행에서 쓴 199 / 211 / 223을 **포함**하기 때문이다.
따라서 먼저 끝난 셀이 그대로 새 집합의 부분집합으로 유효하게 남는다.

| 블록 | 셀 수 | 구성 |
| --- | ---: | --- |
| Timing sweep | **240** | DuET × timing 4 × 단백질 3 × seed 20 |
| Controls | **120** | {Frozen, Complete Nested} × 단백질 3 × seed 20, cp7590 |
| **메인 소계** | **360** | |
| K/M gradient (보조) | 12 | DuET K2×M32, K32×M2 × {BBA, Trp-cage} × seed 3 |
| **총계** | **372** | 실패 0 |

### 2.9 실행 환경과 비용

H100 NVL × 4 pod (`sejin-h100-1-work-001/002/003/004`),
Python 3.10.20, torch 2.1.2+cu121, mdtraj 1.9.9, lightning 2.0.1, numpy 1.24.4.

셀당 실측 wall clock (단일 GPU):

| 단백질 | Frozen (K64) | Complete Nested (K8×M8) | DuET (K8×M8) |
| --- | ---: | ---: | ---: |
| Trp-cage | 374 s | 101 s | 103 s |
| BBA | 365 s | 157 s | 156 s |
| BBL | 387 s | 196 s | 200 s |

동일 NFE(40,960)인데 Frozen이 2–3.7배 느린 것은 batch 형상 차이다 — Frozen은
64입자를 원자 축에 이어 붙여 한 번에 decode하므로 메모리 대역폭에 묶인다.
**총 69,883 GPU-초 = 19.4 GPU-hours.** PVB는 DuET 셀 기준 ConfRover 대비 약 40배 싸다.

---

## 3. 결과

### 3.1 방법 비교 (cp7590, n = 20)

**weighted final backbone RMSD (Å), 평균 ± 표준편차 — 낮을수록 reward가 높음**

| 단백질 | Frozen (K64,M1) | Complete Nested (K8,M8) | DuET (K8,M8) |
| --- | ---: | ---: | ---: |
| Trp-cage | 6.10 ± 0.14 | 3.89 ± 0.29 | 3.89 ± 0.24 |
| BBA | 7.63 ± 0.11 | 5.19 ± 0.56 | 5.12 ± 0.56 |
| BBL | 18.90 ± 0.24 | 16.11 ± 0.93 | 15.88 ± 0.76 |

**전 중원자 RMSD (Å) / best-valid backbone RMSD (Å)**

| 단백질 | Frozen | Complete Nested | DuET |
| --- | --- | --- | --- |
| Trp-cage | 7.64 / 3.94 | 5.65 / 3.30 | 5.71 / 3.28 |
| BBA | 9.33 / 5.55 | 7.41 / 4.26 | 7.35 / 4.37 |
| BBL | 19.64 / 16.57 | 16.93 / 15.35 | 16.70 / 15.55 |

**whole-path validity**

| 단백질 | Frozen | Complete Nested | DuET |
| --- | ---: | ---: | ---: |
| Trp-cage | 0.93 | 1.00 | 0.99 |
| BBA | 0.99 | 0.99 | 0.99 |
| **BBL** | **0.15** | **0.22** | **0.21** |

### 3.2 Welch t 검정 (cp7590, n = 20 vs 20)

| 단백질 | 지표 | DuET − Complete Nested | DuET − Frozen |
| --- | --- | ---: | ---: |
| Trp-cage | wBB | **−0.02** (df 36) | −35.91 (df 31) |
| | Q(max) | −1.06 | −3.00 |
| | validity | −1.45 | +5.45 |
| BBA | wBB | **−0.36** (df 38) | −19.51 (df 20) |
| | Q(max) | −0.28 | −3.20 |
| | validity | +0.45 | +0.00 |
| BBL | wBB | **−0.87** (df 37) | −16.97 (df 23) |
| | Q(max) | −0.92 | −5.40 |
| | validity | −0.39 | +1.37 |

두 가지가 분명하다.

1. **Steering의 효과는 압도적이다.** DuET/CN 대 Frozen의 wBB t는 −17 ~ −36이다.
   표본 20개에서 이 크기는 사실상 확정이다.
2. **DuET과 Complete Nested는 어떤 지표에서도 구분되지 않는다.** 9개 검정 모두
   |t| < 1.5, wBB만 보면 |t| < 0.9다. 3-seed 시점에 관측됐던 0.12–0.88 Å의 DuET
   우위는 seed 표본 변동이었다.

### 3.3 [정정] native contact Q — Frozen 우위는 경로 수 교란이었다

선행 보고서는 `q_final_max`로 "Q는 세 단백질 모두 Frozen이 가장 높다 ⇒ RMSD를 낮추는
동안 native contact은 오히려 줄어든다"고 결론했다. **이 비교는 성립하지 않는다.**
`q_final_max`는 경로 집합에 대한 **최댓값**인데, Frozen은 경로가 64개이고 DuET/CN은
8개다. 최댓값은 표본 크기에 단조 증가하므로 Frozen이 유리할 수밖에 없다.

Frozen의 64경로에서 8개를 비복원 추출해 최댓값을 취하는 절차를 2,000회 반복해
경로 수를 맞췄다.

| 단백질 | 방법 | Q max (원본) | **Q max @8경로** | Q mean (경로 평균) |
| --- | --- | ---: | ---: | ---: |
| Trp-cage | Frozen | 0.333 | **0.245** | 0.123 |
| | Complete Nested | 0.304 | **0.304** | 0.182 |
| | DuET | 0.281 | **0.281** | 0.170 |
| BBA | Frozen | 0.438 | **0.388** | 0.299 |
| | Complete Nested | 0.402 | **0.402** | 0.314 |
| | DuET | 0.398 | **0.398** | 0.322 |
| BBL | Frozen | 0.313 | **0.258** | 0.187 |
| | Complete Nested | 0.275 | **0.275** | 0.203 |
| | DuET | 0.265 | **0.265** | 0.199 |

경로 수를 맞추면 **세 단백질 모두에서 순서가 뒤집힌다.** Frozen이 최하위가 되고
steering 팔이 앞선다. 경로 평균 Q는 애초에 처음부터 steering 팔이 높았다
(Trp-cage 0.123 vs 0.182/0.170 등).

> **정정된 사실: steering은 backbone RMSD를 낮추면서 native contact도 동시에 (약간)
> 올린다.** "RMSD만 최적화하고 folding 지표는 악화시킨다"는 선행 결론은 철회한다.
> 다만 개선폭은 작다(Q 0.02–0.06) — steering이 folding을 복원한다는 뜻은 아니다.

### 3.4 TICA 최소거리 — Frozen 우위는 축소되지만 남는다

같은 교란이 TICA 최소거리에도 적용된다(64경로 중 최솟값 vs 8경로 중 최솟값).
동일한 subsample 절차를 적용했다.

| 단백질 | 방법 | TICA min (원본) | **TICA min @8경로** |
| --- | --- | ---: | ---: |
| Trp-cage | Frozen | 0.898 | **1.376** |
| | Complete Nested | 1.502 | 1.502 |
| | DuET | 1.575 | 1.575 |
| BBA | Frozen | 0.382 | **0.902** |
| | Complete Nested | 1.133 | 1.133 |
| | DuET | 1.138 | 1.138 |
| BBL | Frozen | 0.460 | **0.729** |
| | Complete Nested | 0.750 | 0.750 |
| | DuET | 0.728 | 0.728 |

Q와 달리 **순서는 뒤집히지 않는다.** 그러나 격차는 크게 줄어든다 — BBA에서
0.382 vs 1.138(3.0배)이 0.902 vs 1.138(1.26배)이 되고, **BBL은 사실상 동률**
(0.729 vs 0.728/0.750)이 된다. Trp-cage만 Frozen이 유의미하게 앞선다.

### 3.5 [정정] TICA 타깃 도달률 — "DuET 최하위"는 성립하지 않는다

cp7590만 보면 DuET이 세 단백질 모두에서 최하위로 보인다. 그러나 hit은 극도로 희소해
(전체 372셀에서 147건) cp7590 한 timing의 DuET 160경로는 추정치의 분산이 너무 크다.
timing sweep이 wBB·Q·validity 어디에서도 차이를 만들지 않았으므로(§3.6), DuET의 네
timing은 **동일 과정의 4회 독립 추출**로 볼 수 있다. 이를 합쳐 640경로로 재추정했다.

| 단백질 | Frozen | Complete Nested | DuET (cp7590) | **DuET (4 timing 합산)** |
| --- | ---: | ---: | ---: | ---: |
| Trp-cage | 2/1280 = 0.0016 | 1/160 = 0.0063 | 0/160 = 0.0000 | **3/640 = 0.0047** |
| BBA | 74/1280 = 0.0578 | 8/160 = 0.0500 | 3/160 = 0.0187 | **20/640 = 0.0312** |
| BBL | 19/1280 = 0.0148 | 5/160 = 0.0312 | 1/160 = 0.0063 | **15/640 = 0.0234** |

두 비율 z 검정 (Frozen − DuET 합산):

| 단백질 | z | 방향 |
| --- | ---: | --- |
| Trp-cage | **−1.27** | DuET이 더 높음 |
| BBA | **+2.54** | Frozen이 더 높음 |
| BBL | **−1.35** | DuET이 더 높음 |

Complete Nested − DuET 합산: Trp-cage +0.25, BBA +1.15, BBL +0.57 — 전부 무의미.

> **정정된 사실: "steering 없는 Frozen만 타깃에 도달한다"는 세 단백질 중 BBA에서만
> 관측되며, Trp-cage와 BBL에서는 오히려 DuET이 더 높다.** BBA의 z = +2.54도 이
> 보고서가 수행한 9건의 비율 검정에 Bonferroni를 적용하면(α = 0.05/9, z_crit ≈ 2.77)
> 유의 기준을 넘지 못한다. 선행 보고서의 "DuET은 세 단백질 모두에서 최하위"도 철회한다.

Trp-cage는 Frozen이 1,280경로에서 hit 2건, 20 seed 중 1개 seed에서만 나왔다.
이 정도 희소성에서는 어떤 방향의 주장도 지지되지 않는다.

### 3.6 Checkpoint timing sweep — 확정적 non-result

**DuET weighted backbone RMSD (Å), n = 20**

| 단백질 | 0.25+0.50 | 0.50+0.75 | 0.75+0.90 | 0.25+0.50+0.75 |
| --- | --- | --- | --- | --- |
| Trp-cage | 3.87 ± 0.26 | 3.86 ± 0.21 | 3.89 ± 0.24 | 3.88 ± 0.27 |
| BBA | 5.14 ± 0.43 | 5.00 ± 0.36 | 5.12 ± 0.56 | 5.04 ± 0.40 |
| BBL | 15.99 ± 0.80 | 16.04 ± 0.70 | 15.88 ± 0.76 | 15.43 ± 0.84 |

Trp-cage는 네 timing이 **0.03 Å** 안에 들어온다 — 표준편차의 약 1/8이다.
BBA는 0.14 Å 범위로 역시 표준편차 이내다. BBL의 `0.25+0.50+0.75`만 0.45–0.61 Å
낮으나 표준편차 0.84를 감안하면 경계선이고, 하필 validity가 0.24로 무너진
단백질에서만 나온 신호다.

**Q (max / mean)와 validity도 같은 결론이다.**

| 단백질 | 0.25+0.50 | 0.50+0.75 | 0.75+0.90 | 0.25+0.50+0.75 |
| --- | --- | --- | --- | --- |
| Trp-cage Q | 0.277 / 0.161 | 0.298 / 0.178 | 0.281 / 0.170 | 0.285 / 0.168 |
| BBA Q | 0.404 / 0.314 | 0.402 / 0.310 | 0.398 / 0.322 | 0.414 / 0.324 |
| BBL Q | 0.265 / 0.200 | 0.252 / 0.193 | 0.265 / 0.199 | 0.268 / 0.199 |
| Trp-cage validity | 0.99 | 0.97 | 0.99 | 0.96 |
| BBA validity | 1.00 | 1.00 | 0.99 | 0.99 |
| BBL validity | 0.16 | 0.28 | 0.21 | 0.24 |

> **이것은 "timing이 무관하다"는 발견이 아니라 non-result다.** 반응하지 않는 지표
> 위에서 sweep을 하면 아무것도 측정되지 않는다. §4에서 논하듯 이 조건(3.2 ns horizon,
> RMSD reward)에서는 어떤 방법 조합도 성공 지표를 움직이지 못하므로, timing만
> 별도로 판정할 수 없다.

### 3.7 K/M gradient — 예산 배분은 설명 변수가 아니다

예산 K·M = 64 고정, seed 3개(보조 블록).

| 단백질 | 배분 | 경로 수 | wBB (Å) | Q max | validity | TICA min | hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trp-cage | K2 × M32 | 2 | 4.05 | 0.218 | 0.83 | 1.551 | 0 |
| | K8 × M8 | 8 | 3.89 | 0.281 | 0.99 | 1.575 | 0 |
| | K32 × M2 | 32 | 3.99 | 0.346 | 0.98 | 1.270 | 0 |
| | *(Frozen K64 × M1)* | *64* | *6.10* | *0.333* | *0.93* | *0.898* | *2* |
| BBA | K2 × M32 | 2 | 4.87 | 0.407 | 1.00 | 1.441 | 0 |
| | K8 × M8 | 8 | 5.12 | 0.398 | 0.99 | 1.138 | 3 |
| | K32 × M2 | 32 | 4.81 | 0.474 | 1.00 | 1.199 | 0 |
| | *(Frozen K64 × M1)* | *64* | *7.63* | *0.438* | *0.99* | *0.382* | *74* |

K를 2 → 32로 16배 올려도 wBB는 4.81–5.12에서 평평하고, hit은 0에서 움직이지 않는다.
반면 steering이 아예 없는 Frozen(K64)은 hit 74건을 낸다. **K2, K8, K32 사이보다
K32와 K64 사이의 간격이 훨씬 큰데, 그 구간에서 바뀌는 것은 경로 수가 아니라
steering의 유무다.**

다만 §3.4에서 본 대로 경로 수를 맞추면 Frozen의 TICA 우위가 크게 줄어들므로,
이 gradient가 "steering이 꼬리를 깎는다"를 확정하지는 못한다. seed 3개짜리 보조
블록이라는 점도 함께 고려해야 한다.

### 3.8 BBL의 validity 붕괴 — 출발 구조 문제

BBL은 방법과 무관하게 whole-path validity가 0.15–0.28로 무너진다.

첫 가설은 산술적인 것이었다 — whole-path validity는 **단 하나의 나쁜 결합**에도
실패하는데, BBL은 46결합 × 33프레임이고 Trp-cage는 19 × 33이니 기회가 많다는 것.
결합당 위반율이 이를 반증한다.

| 단백질 | 잔기 | 검사한 결합 수 | 위반 | **결합당 위반율** | 최악 C–N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trp-cage | 20 | 195,624 | 9 | 0.00005 | 1.91 Å |
| BBA | 28 | 277,992 | **0** | 0.00000 | — |
| BBL | 47 | 473,616 | 627 | **0.00132** | **5.76 Å** |

Trp-cage의 **26배** 비율인데 사슬 길이 비는 2.4배에 불과하고, 길이 중간인 BBA는
위반이 **0건**이다. 길이는 이 순서를 설명하지 못한다. 최악값 5.76 Å는 백본이
끊어진 상태다.

출발 구조의 팽창도가 설명한다(§2.3 표): BBL의 Rg 비는 2.27로 Trp-cage 1.45,
BBA 1.44와 질적으로 다르다. PVB-ATLAS는 평형 MD에 fine-tune되어 있어 folded 및
near-folded 상태를 담고 있는데, BBL은 사실상 완전히 펼쳐진 상태에서 출발해
학습 분포 밖이다. 단일 전이 smoke test가 통과했던 이유는 손상이 32스텝에 걸쳐
누적되기 때문이다.

---

## 4. 결론과 해석

### 4.1 확정된 것

**(1) Steering은 자기 reward 위에서 명확히 작동한다.**
DuET/Complete Nested는 Frozen 대비 backbone RMSD를 2.2–3.0 Å 낮추고, Welch t는
−17 ~ −36이다. 구현이 의도대로 동작한다는 증거로는 충분하다.

**(2) DuET은 Complete Nested보다 낫지 않다.**
이것이 이 실험의 가장 중요한 결과다. 두 방법은 K, M, 예산, reward, 시드가 전부
같고 **오직 안쪽 resampling 여부만 다른데**, wBB·Q·validity 9개 검정 전부
|t| < 1.5다. DuET의 구조적 이점인 "diffusion 중간 연산 재배치"가 이 조건에서
성능으로 전환되지 않는다.

**(3) checkpoint timing 4종은 구분되지 않는다.**
20 seed에서도 seed 산포 이내다.

**(4) BBL의 validity 붕괴는 출발 구조가 PVB 학습 분포 밖이기 때문이다.**
결합당 위반율과 Rg 비가 함께 이를 지지한다. 사슬 길이 가설은 반증됐다.

### 4.2 정정된 것

선행 보고서 2건의 주장 중 둘은 **경로 수 불일치(64 vs 8)에서 온 극값 통계 교란**이었다.

| 이전 주장 | 정정 |
| --- | --- |
| "native contact Q는 세 단백질 모두 Frozen이 가장 높다" | 경로 수를 8로 맞추면 **세 단백질 모두 순서가 뒤집힌다**. 경로 평균 Q는 처음부터 steering 팔이 높았다 |
| "DuET은 세 단백질 모두에서 TICA 도달률 최하위" | timing 4종을 합산하면 Trp-cage·BBL에서 **DuET이 Frozen보다 높다**. Frozen 우위는 BBA 단독이고 다중비교 후 유의하지 않다 |

이 정정은 "reward가 folding 방향과 어긋난다"는 해석을 **약화**시킨다. Q는 steering과
같은 방향으로 (미세하게) 움직이므로, RMSD reward가 folding 지표를 **악화**시킨다는
증거는 없다. 남는 것은 더 온건한 진술이다 — **RMSD reward는 folding 지표를 거의
움직이지 못한다.**

### 4.3 왜 성공률이 오르지 않는가

세 가지가 동시에 작용하며, 현재 데이터로는 분리되지 않는다.

1. **Horizon.** T = 32 × 100 ps = **3.2 ns**. fast folder의 folding 시간은
   마이크로초 규모다. 3.2 ns에서 hit이 희소한 것은 **예상된 결과**이며 그 자체로는
   방법에 대한 정보를 주지 않는다.
2. **Reward 정렬.** `log ψ = −16·(d_RMSD/d₀)²`는 전체 backbone RMSD를 낮추는 방향인데
   성공은 첫 두 TICA 좌표로 정의된다. RMSD가 2–3 Å 내려가는 동안 Q는 0.02–0.06,
   TICA 거리는 사실상 제자리다 — reward가 성공 지표에 거의 연결되어 있지 않다.
3. **통계적 검정력.** hit은 372셀 전체에서 147건이고, Trp-cage는 Frozen 1,280경로에서
   2건이다. 이 희소성에서는 팔 사이 비교의 검정력이 매우 낮다.

DuET이 Complete Nested와 구분되지 않는 것은 **(2)의 결과일 가능성이 높다.** 안쪽
resampling이 고를 수 있는 신호가 reward뿐인데, 그 reward가 성공 방향을 가리키지
않는다면 중간에 고르든 끝에 고르든 같은 곳에 도착한다. 이것은 **DuET 알고리즘의
실패라기보다 이 실험 조건이 DuET을 검정할 수 있는 조건이 아니었다**는 뜻이다.

### 4.4 주장 경계 — 쓰면 안 되는 표현

- ❌ "DuET이 folding pathway를 복원했다."
- ❌ "RMSD가 낮아졌으므로 folded 상태에 가깝다."
- ❌ "timing이 성능에 영향을 주지 않는다." (non-result이지 부정 증거가 아님)
- ❌ "DuET이 Complete Nested보다 낫다." (20 seed에서 반증)
- ❌ "native contact Q는 Frozen이 가장 높다." (경로 수 교란 — **이번에 철회**)
- ❌ "steering 없는 팔만 타깃에 도달한다." (BBA 단독, 다중비교 후 비유의 — **이번에 철회**)
- ❌ "PVB에서 DuET이 작동하지 않는다." (검정력 부족 구간이며, 기각이 아니라 미판정)

쓸 수 있는 표현:

- ✅ "동일 예산에서 steering은 reward를 확실히 개선한다(|t| = 17–36)."
- ✅ "안쪽 resampling의 추가 이득은 20 seed에서 관측되지 않았다(|t| < 1)."
- ✅ "3.2 ns horizon과 RMSD reward 조건에서는 성공 지표가 어떤 조작에도 반응하지 않는다."
- ✅ "PVB DuET 어댑터는 동작하며 ConfRover 대비 셀당 약 40배 싸다."

---

## 5. 실행 중 발생한 문제와 처리

| 문제 | 원인 | 처리 |
| --- | --- | --- |
| timing 12셀 전부 `evaluation_failed` | config의 `stage`(`stride128_t32_cp2550`)가 러너가 계산하는 stage(`stride128_t32`)와 불일치 | 샘플링은 12셀 모두 성공했으므로 평가만 재실행(`evaluate_timing_runs.py`). 이후 stage를 러너 계산값과 일치시키고 `SMALL_PROTEIN_OUTPUT_ROOT`로 격리 |
| BBL 전 셀 `Unsupported Phase-B protein: bbl` | `small_protein_runner.py`의 단백질 whitelist | `bbl`, `protein_b`, `homeodomain` 추가. **샘플링이 끝난 뒤에 실패**하므로 연산이 낭비됨 |
| `complete_nested` exit 64 | `run_small_protein_cell.sh` 메서드 whitelist 누락 | 추가 |
| seed 상한 199/211/223 | 셀 러너 seed whitelist | 원래 3개를 **포함하는** 등차수열 `7+12i`로 확장 ⇒ 완료된 54셀 유효 유지 |
| 서브프로세스 `ModuleNotFoundError: torch_scatter` | `small_protein_env.sh`가 `PYTHONPATH`를 덮어씀 | `PVB_SITE`를 append하도록 수정 |
| torch 2.14 + CUDA 13 (3.4 GB)가 환경 torch 2.1.2를 가림 | `pip install --target`을 `--no-deps` 없이 실행 | 전부 삭제 후 `--no-deps`로 재설치(74 MB) |
| `phase_b_runner.py` KeyError `cache_dir` | `build_confrover_adapter`를 직접 호출 | pod 사본을 `build_adapter`로 패치 (pod 사본은 repo와 **분기**되어 있어 덮어쓰기 금지) |
| 2셀 `small_protein_metrics.json` 파싱 불가 | GPU 4대로 확장하며 **동일 spec을 두 큐에 배정** — JSON이 두 번 이어 기록됨 | 샘플링 산출물은 온전 ⇒ 평가만 재실행해 복구. 러너의 충돌 방지 장치는 샘플링 전 단계만 막고 평가 단계는 막지 못함. **다중 GPU 분배 시 spec 집합을 큐 간 배타적으로 분할해야 한다** |

zsh 관련으로 두 번 실수했다. `"$m:frozen"`이 `bbaozen`이 되고(`:fr`가 history
modifier로 해석), `for sd in $SEEDS`가 단어 분할되지 않았다. 이후 spec 문자열을
Python에서 생성하고 360건 전부 정규식으로 검증했다.

---

## 6. 다음 단계

**H4 — TICA 좌표 기반 reward.** 우선순위 최상위다. reward를
`−λ·(TICA distance)²`로 바꾸고 BBA에서 DuET K8×M8을 재실행해 기존 Frozen baseline과
비교한다. config 변경이 아니라 **새 observable**이 필요하다 — TICA projection이
샘플링 루프 안에서 접근 가능해야 한다.
*반증 조건*: TICA-reward DuET도 hit 0이면 한계는 reward가 아니라 3.2 ns horizon에 있다.

**H4의 부속 설계.** DuET이 CN과 구분되지 않는 원인이 "reward가 신호를 주지 않아서"인지
"중간 개입 자체가 무력해서"인지 가르려면, H4 실행 시 checkpoint 개수를
**0(= Complete Nested) / 1 / 2**로 두고 같은 지표를 비교하면 직접 검정된다.

**H1 (보류).** BBL을 Rg ≈ 15 Å 수준의 덜 팽창된 출발 구조에서 재실행.
TPS-DPS의 `path.gro`를 복사해 두지 않았으므로 보간 + Cα 구속 minimization으로
중간 구조를 새로 만들어야 한다.

**하지 말아야 할 것.** reward를 바꾸지 않은 상태에서의 추가 timing sweep 또는
K/M sweep. 20 seed로 이미 확정적으로 무반응임이 확인됐다.

---

## 7. 산출물

**코드**
- `src/confmh/adapters/pvb_duet.py` — PVB DuET 어댑터
- `src/confmh/duet/runner.py` — `build_pvb_adapter`, `BACKENDS`, `build_adapter` 디스패치
- `scripts/duet/pvb_preflight/` — 로드/사전점검/결정성/어댑터 smoke 테스트
- `scripts/duet/fast_folders_extended/` — config 생성, 실행 체인, 평가, 분석 스크립트

**데이터**
- `scripts/duet/fast_folders_extended/results_pvb_360.json` — 360 메인 셀 통합
- `scripts/duet/fast_folders_extended/results_pvb_all_66.json` — 초기 66셀
- `outputs/pvb_full/<timing>/<protein>/stride128_t32/<method>/seed_<n>/`
  - `small_protein_metrics.json` — 셀 지표
  - `native_contact_q.json` — Q 사이드카 (경로별 시계열 포함)
  - `tica_projection.npz` — `tica_first_two`, `distance_to_folded_target`, `target_hit`, `usable`
  - `checkpoint_audit.json` — 요청 대비 실현된 checkpoint 스텝과 ESS
  - `resolved_config.yaml`, `assets.json`(sha256), `environment.json`, `git_state.json`

**분석 스크립트**
- `compute_native_contacts.py`, `diagnose_validity.py`, `analyze_gate_sensitivity.py`,
  `test_h2_tica_vs_rmsd.py`, `evaluate_timing_runs.py`, `summarize_timing_variation.py`

**선행 보고서** (이 문서가 §4.2에서 정정함)
- `docs/PVB_FAST_FOLDING_FINAL_2026-09-14.md` — 66셀
- `docs/PVB_TPS_EXPERIMENT_REPORT_2026-09-14.md` — 66셀 상세
- `docs/PVB_TPS_20SEED_UPDATE_2026-09-14.md` — 360셀 통계
- `docs/PVB_FAST_FOLDING_ROOT_CAUSE_2026-09-14.md` — 실패 원인 분석
