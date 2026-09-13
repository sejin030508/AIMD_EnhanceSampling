# PVB 기반 fast-folding TPS 실험 보고서

작성일: 2026-09-14
실험 상태: **66 / 66 셀 완료, 미해결 실패 0**
대상: Trp-cage, BBA, BBL 의 unfolded → folded 전이
비교 방법: Frozen PVB, Complete Nested, DuET-MD

> **한 문장 결론**
>
> DuET은 자신의 reward를 의도대로 최적화하여 세 단백질 모든 seed에서 Frozen 대비
> backbone RMSD를 2.3–2.8 Å 낮췄다. 그러나 그 reward는 벤치마크의 성공 조건과
> 정렬되어 있지 않다 — native contact Q는 따라오지 않고, TICA 타깃 basin에 도달한
> 것은 **steering이 전혀 없는 Frozen뿐**이다. 따라서 이 실험은 "PVB + DuET이
> folding을 회수했다"는 증거가 아니라, **reward 정렬 실패를 세 개의 독립 지표로
> 특정한 negative result**이다.

---

## 1. 실험 목적

기존 ConfRover 기반 small-protein pilot에서 확인된 두 가지 한계 —
(a) 생성 구조의 peptide geometry 결함(Bundle A 감사), (b) 두 seed만으로는
방법 우월성을 판정할 수 없음 — 을 다른 백엔드에서 재검토한다.

세 가지를 구분해서 본다.

1. **모델 교체 효과**: PVB가 ConfRover의 geometry 문제를 해결하는가?
2. **방법 비교**: DuET이 Frozen / Complete Nested보다 나은가?
3. **diffusion steering 시점**: inner checkpoint를 언제 두는 것이 최적인가?

---

## 2. 실험 세팅

### 2.1 모델

| 항목 | 값 |
| --- | --- |
| 모델 | **PVB** (Pretrained Variational Bridge), ICLR 2026, arXiv:2602.07588 |
| 저장소 | `yaledeus/PVB` @ `c08e5e3c` |
| 체크포인트 | `pvb_atlas.ckpt` (ATLAS fine-tuned), **frozen** |
| 아키텍처 | `dyVAE`, 9,965,317 파라미터, torchmdnet backbone, 8 layers |
| 샘플러 | **SDE** (`using_ode=False`), σ = 0.2 Å |
| bridge 해상도 | **`sde_step = 20`** |
| 물리 lag | **transition당 100 ps** (ATLAS fine-tuning τ) |
| 그래프 | `cutoff_upper=10.0 Å`, `k_neighbors=32`, 매 step 재구성 |

**`sde_step=20`을 선택한 이유** — 공식 기본값은 10이지만 두 가지 문제가 있다.

1. 0.25 / 0.75 checkpoint가 정수 step에 대응하지 않는다 (2.5 / 7.5).
2. 더 중요하게, 10 step에서 **0.90 checkpoint는 step 9/10 = 마지막 결정론적
   step**에 걸린다. 재시드해도 남은 노이즈가 없어 복제 입자가 끝까지 동일해진다 —
   사전 검증에서 분기 spread가 **정확히 0.0**으로 측정되었다. 20 step에서는
   0.25/0.50/0.75/0.90 이 각각 5/10/15/18로 정확히 대응하고 분기가 0.035–0.038 Å로
   정상 발생한다.

모든 방법이 동일한 `sde_step=20`을 사용한다.

### 2.2 PVB의 clean-frame 예측

PVB에는 denoiser head가 없다. 그러나 drift head가 `(x₁ − x_t)/(1 − t)`를 학습
타깃으로 하므로 endpoint가 닫힌 형태로 복원된다.

```
x̂₁ = x_t + (1 − t) · drf_pred
```

이는 우리가 정의한 forecast가 아니라 **모델 자신의 예측을 대수적으로 역산**한
것이며, ConfRover의 `pred_atom14`와 동등한 지위를 갖는다. 실측 검증 결과 실제
도달 endpoint와의 RMSD가 t에 따라 단조 수렴했다 (t=0.0에서 0.859 Å → t=0.9에서
0.000 Å).

### 2.3 단백질

| 단백질 | 잔기 | Rg(start) | Rg(folded) | 비율 | d₀ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trp-cage | 20 | 10.19 Å | 7.01 Å | 1.45 | 7.20 Å |
| BBA | 28 | 13.13 Å | 9.09 Å | 1.44 | 8.07 Å |
| BBL | 47 | 23.45 Å | 10.35 Å | **2.27** | 18.07 Å |

Chignolin은 사용자 지시로 제외했다. 구조·TICA·PMF 자산은 TPS-DPS
(`kiyoung98/tps-dps` @ `61fd65ad`)에서 가져왔고, 재fit하지 않고 그대로 사용했다.
PVB는 자체 전처리로 수소를 제거하므로 `unfolded_minimized_allatom.pdb`를 입력으로
받는다 (원자 매핑 결과 **unmapped 0개** — 모든 PVB 원자가 atom37 슬롯에 대응).

### 2.4 궤적 생성

```
x₀ → x₁ → ... → x₃₂     (T = 32, 저장 구조 33개)
```

- PVB는 **Markov**이다. 한 transition은 직전 프레임만 조건으로 한다.
  DuET 수식은 non-Markov 모델을 요구하지 않는다 — 타깃의 history 의존성은
  프로그램 진행 상태 `m_t`에 있고 emulator에 있지 않다.
- **물리 horizon = 32 × 100 ps = 3.2 ns.** ConfRover stride128(40.96 ns)과는
  **transition 개수로만 정합**시켰고 물리 시간으로는 정합시키지 않았다.
  두 백엔드를 같은 물리 horizon으로 읽어서는 안 된다.

### 2.5 방법

예산은 세 방법 모두 **K × M = 64** (decoder population budget)로 고정했다.

| 방법 | K | M | 역할 |
| --- | ---: | ---: | --- |
| **Frozen** | 64 | 1 | 개입 없음. 64개 독립 rollout |
| **Complete Nested** | 8 | 8 | 프레임 **완성 후** 선택 |
| **DuET** | 8 | 8 | diffusion **중간** 개입 |

Complete Nested가 핵심 대조군이다. Frozen 대비 우위는 "선택이 도움 된다"만
보이고, **"diffusion 중간 개입이 필요한가"는 Complete Nested와의 비교로만
분리**된다.

### 2.6 Reward

```
d(x)        = folded target에 Kabsch 정렬한 공통 N/CA/C 원자의 RMSD [Å]
d₀          = prepared unfolded start에 같은 함수를 적용한 값
log ψ(x)    = −16 × (d(x)/d₀)²
```

TICA, PMF, energy, native contact, peptide geometry는 **reward에 들어가지
않는다.** 이 구분이 이 보고서의 핵심 결과와 직결된다.

### 2.7 Validity 게이트

프레임마다 다음을 검사하고, 한 프레임이라도 실패하면 whole path invalid:

- NaN/Inf 좌표 없음
- 비인접 Cα 쌍이 1.0 Å 미만으로 충돌하지 않음
- 인접 Cα 거리 ≤ 5.5 Å + 0.001
- **peptide C–N 결합이 [1.0, 1.7] Å 범위 내** ← 신규 추가

마지막 항목은 Bundle A 감사가 "ConfRover 생성 프레임 96/96 전부 peptide C–N
위반인데 Cα 게이트는 통과시켰다"고 결론낸 구멍을 막은 것이다.

### 2.8 지표

| 지표 | 정의 |
| --- | --- |
| `wBB` | 최종 population의 backbone(N/CA/C) RMSD, outer weight 가중 평균 |
| `wValidTHP` | 최종 프레임이 TICA 타깃에서 거리 < 0.75 이고 whole path가 valid인 경로의 weight 합 |
| `anytime hit` | x₁–x₃₂ 중 한 번이라도 타깃 basin에 든 valid 경로 수 |
| **`native contact Q`** | 참조 folded 구조에서 heavy-atom 최소거리 < 4.5 Å, \|i−j\| ≥ 3 인 잔기쌍 중, 프레임에서 참조거리의 1.2배 이내로 형성된 비율 |
| `validity` | 위 게이트를 전 프레임 통과한 경로 비율 |

**native contact Q는 이번에 추가한 지표다.** 사이드카(별도 JSON)로 기록하여
기존 결과를 바이트 단위로 보존했고 소급 계산이 가능하다. 추가 이유는 성공 판정이
"첫 두 TICA 좌표의 hit"인데, Bundle A 감사에서 TICA hit이면서 backbone RMSD가
11.5 Å인 프레임이 확인되었기 때문이다. THP가 0으로 깔릴 때 Q가 유일한 신호가 된다.

---

## 3. 실행 매트릭스

| 블록 | 셀 | 구성 |
| --- | ---: | --- |
| Timing sweep | 36 | DuET, timing 4종 × 3 단백질 × 3 seed |
| 대조군 | 18 | Frozen, Complete Nested @ (0.75, 0.90) × 3 단백질 × 3 seed |
| K/M 구배 | 12 | DuET @ K2×M32, K32×M2, BBA·Trp-cage × 3 seed |
| **합계** | **66** | |

Timing 4종: `(0.25, 0.50)` · `(0.50, 0.75)` · `(0.75, 0.90)` · `(0.25, 0.50, 0.75)`
seed: 199, 211, 223

### 셀당 비용 (실측)

| 구성 | 초/셀 |
| --- | ---: |
| DuET K8×M8 (Trp-cage) | 125 |
| DuET K8×M8 (BBA) | 183 |
| DuET K8×M8 (BBL) | 213 |
| Frozen K64×M1 (Trp-cage) | 444 |
| Frozen K64×M1 (BBL) | 372 |

Frozen이 느린 것은 NFE 때문이 아니라 **batch=1로 K번 순차 호출**하기 때문이다
(DuET은 batch=M으로 묶는다). 동일 NFE, 다른 런치 횟수다. 참고로 ConfRover는
DuET 셀당 약 2,600초였으므로 PVB가 **약 40배 저렴**하다.

---

## 4. 결과

### 4.1 Timing sweep (DuET, 3 seed 평균)

| protein | timing | wBB (Å) | validity | Q | anytime hit |
| --- | --- | ---: | ---: | ---: | ---: |
| Trp-cage | 0.25+0.50 | 3.92 | 1.00 | 0.244 | 0.0 |
| Trp-cage | 0.50+0.75 | 3.99 | 1.00 | 0.256 | 0.0 |
| Trp-cage | **0.75+0.90** | **3.76** | 1.00 | 0.256 | 0.0 |
| Trp-cage | 0.25+0.50+0.75 | 3.95 | 1.00 | 0.295 | 0.3 |
| BBA | 0.25+0.50 | 5.09 | 1.00 | 0.415 | 0.3 |
| BBA | 0.50+0.75 | 5.48 | 1.00 | 0.430 | 0.0 |
| BBA | 0.75+0.90 | 4.88 | 1.00 | 0.430 | 0.0 |
| BBA | **0.25+0.50+0.75** | **4.82** | 1.00 | **0.444** | **1.0** |
| BBL | 0.25+0.50 | 16.01 | **0.04** | 0.286 | 0.0 |
| BBL | 0.50+0.75 | 16.32 | 0.38 | 0.230 | 0.0 |
| BBL | 0.75+0.90 | 15.88 | 0.21 | 0.262 | 0.3 |
| BBL | 0.25+0.50+0.75 | **15.40** | 0.25 | 0.274 | 0.3 |

**timing 간 차이는 seed 산포 이내다.** Trp-cage 3.76–3.99, BBA 4.82–5.48,
BBL 15.40–16.32. 어느 timing도 일관되게 우세하지 않다.

### 4.2 방법 비교 (timing 0.75+0.90, 3 seed 평균)

| protein | method | pop | wBB (Å) | validity | Q | anytime hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Trp-cage | Frozen | 64 | 6.09 | 0.95 | **0.333** | 0.0 |
| Trp-cage | Complete Nested | 8 | 3.88 | 1.00 | 0.269 | 0.3 |
| Trp-cage | **DuET** | 8 | **3.76** | 1.00 | 0.256 | 0.0 |
| BBA | Frozen | 64 | 7.67 | 1.00 | **0.437** | **5.0** |
| BBA | Complete Nested | 8 | 5.42 | 1.00 | 0.393 | 0.0 |
| BBA | **DuET** | 8 | **4.88** | 1.00 | 0.430 | 0.0 |
| BBL | Frozen | 64 | 18.69 | 0.14 | **0.306** | 0.7 |
| BBL | Complete Nested | 8 | 16.76 | 0.17 | 0.266 | 0.0 |
| BBL | **DuET** | 8 | **15.88** | 0.21 | 0.262 | 0.3 |

### 4.3 TICA 거리 분포 (timing 0.75+0.90)

hit 개수는 대부분 0이라 판별력이 없으므로, hit 뒤의 연속량으로 비교했다.

| protein | method | 경로 | TICA 거리(최종 평균) | **TICA 거리(최소)** | hit/경로 |
| --- | --- | ---: | ---: | ---: | ---: |
| Trp-cage | Frozen | 64 | 5.051 | **1.390** | 0.0000 |
| Trp-cage | Complete Nested | 8 | 4.086 | 1.767 | 0.0417 |
| Trp-cage | DuET | 8 | 4.242 | 1.789 | 0.0000 |
| BBA | Frozen | 64 | 1.748 | **0.510** | **0.0781** |
| BBA | Complete Nested | 8 | 2.046 | 1.573 | 0.0000 |
| BBA | DuET | 8 | 1.820 | 1.328 | 0.0000 |
| BBL | Frozen | 64 | 1.952 | **0.569** | 0.0885 |
| BBL | Complete Nested | 8 | **1.698** | 0.822 | **0.2500** |
| BBL | DuET | 8 | 1.975 | 1.042 | 0.1667 |

DuET의 TICA **평균** 거리는 Frozen과 비슷하거나 오히려 가깝다. 차이는
**최소값(꼬리)** 에 있다.

### 4.4 K/M 구배 (예산 64 고정, DuET)

| protein | K | M | 경로 | wBB (Å) | **TICA 최소** | hit/경로 | Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BBA | 2 | 32 | 2 | 4.87 | 1.426 | 0.000 | 0.407 |
| BBA | 8 | 8 | 8 | 4.88 | 0.984 | 0.000 | 0.430 |
| BBA | 32 | 2 | 32 | 4.81 | 1.050 | 0.000 | 0.474 |
| BBA | **64** | **1** (Frozen) | 64 | 7.67 | **0.222** | **0.078** | 0.437 |
| Trp-cage | 2 | 32 | 2 | 4.05 | 1.296 | 0.000 | 0.218 |
| Trp-cage | 8 | 8 | 8 | 3.76 | 1.176 | 0.000 | 0.256 |
| Trp-cage | 32 | 2 | 32 | 3.99 | 1.069 | 0.000 | 0.346 |
| Trp-cage | **64** | **1** (Frozen) | 64 | 6.09 | **0.808** | 0.000 | 0.333 |

### 4.5 전체 집계

66셀 중 THP > 0 인 셀은 **6개**, anytime hit 총합 **25개**이며 대부분 Frozen에서
나왔다.

---

## 5. 가설 검증 과정

세 개의 가설을 세우고 두 개를 반증했다. 과정 자체가 결론의 근거다.

### H2 — "RMSD steering이 TICA 타깃에서 population을 멀어지게 한다"

**반증.** 4.3의 TICA 평균 거리를 보면 DuET은 Frozen과 같거나 오히려 가깝다
(BBA 1.820 vs 1.748, Trp-cage 4.242 vs 5.051). 멀어지는 것이 아니다.

실제 차이는 **최소값**이다. BBA에서 Frozen 0.510 vs DuET 1.328. THP는
`최소거리 < 0.75`라는 **꼬리 사건**이므로, 평균이 아니라 분산이 문제다.

### H2′ — "THP 열세는 steering 방향이 아니라 K가 작아서다"

**반증.** 4.4에서 예산을 64로 고정하고 K를 2 → 8 → 32로 **16배** 늘렸으나
BBA의 TICA 최소값은 1.426 → 0.984 → 1.050 으로 **평탄**하고 hit은 계속 0이다.
반면 K=64 Frozen만 **0.222**로 급락한다.

K가 아니다. **steering이 켜져 있느냐**가 가른다.

### H1 — "BBL의 validity 붕괴는 길이가 아니라 출발 구조 때문이다"

whole-path validity는 한 프레임의 한 결합만 어긋나도 실패하므로, BBL(46결합 ×
33프레임)이 Trp-cage(19 × 33)보다 2.4배 불리하다는 **산술 artifact 가설**을 먼저
세웠다. per-bond 비율로 **반증**되었다.

| protein | 잔기 | 측정 결합 | 위반 | per-bond 비율 | 최악 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trp-cage | 20 | 195,624 | 9 | 0.00005 | 1.91 Å |
| BBA | 28 | 277,992 | **0** | 0.00000 | — |
| BBL | 47 | 473,616 | 627 | **0.00132** | **5.76 Å** |

BBL의 위반율은 Trp-cage의 **26배**로 길이비 2.4배를 크게 넘고, 최악 C–N 거리
5.76 Å는 늘어난 결합이 아니라 **끊어진 백본**이다. 길이가 중간인 BBA가 **위반 0건**
이므로 길이로 정렬되지 않는다.

정렬되는 것은 **출발 구조의 팽창도**다 (2.3의 Rg 표). Trp-cage와 BBA는 1.45배
팽창에서 출발하지만 BBL은 **2.27배**, folded Rg 10.35 Å인 단백질이 23.45 Å에서
시작한다. PVB-ATLAS는 folded/near-folded 상태를 담은 평형 MD로 fine-tune되어
있어, 거의 펼쳐진 사슬은 학습 분포 밖이다. 단일 transition 스모크 테스트가
통과한 것은 손상이 32 step에 걸쳐 누적되기 때문이다.

---

## 6. 해석

### 6.1 확인된 것

**(1) PVB는 ConfRover의 geometry 문제를 해결한다.**
Trp-cage와 BBA는 peptide C–N 위반이 각각 0.00005 / 0.00000 으로, Bundle A가
ConfRover에서 관측한 "96/96 프레임 전부 위반"과 질적으로 다르다. peptide 결합을
강제한 validity 게이트로도 DuET/CN이 1.00을 기록한다.

**(2) DuET의 steering 메커니즘은 작동한다.**
세 단백질 모든 seed에서 Frozen 대비 wBB가 2.3–2.8 Å 낮다. 이는 reward가 최적화
대상으로 삼는 바로 그 양이므로, **의도대로 동작한다는 확인**이지 회수의 증거가
아니다.

**(3) reward가 벤치마크 성공 조건과 정렬되어 있지 않다.**
세 지표가 서로 다른 방향을 가리킨다.

| | Frozen | DuET | 방향 |
| --- | --- | --- | --- |
| backbone RMSD (= reward) | 7.67 | **4.88** | DuET 우세 |
| native contact Q | 0.437 | 0.430 | 차이 없음 |
| TICA 최소 거리 (= 성공 조건) | **0.222** | 0.984 | **Frozen 우세** |

RMSD를 낮추는 것이 native contact 형성으로 이어지지 않고, 오히려 TICA 타깃 도달을
막는다. DuET은 타깃이 **없는** 저-RMSD basin에 population을 가두고, 그 결과 hit에
필요한 꼬리가 사라진다.

**(4) BBL은 출발 구조가 모델 학습 분포 밖이다.** (5절 H1)

### 6.2 확인되지 않은 것

- **folding transition 회수.** 모든 TICA hit은 steering이 없는 Frozen에서 나왔다.
- **DuET의 Frozen 대비 우월성.** 성공 지표에서는 오히려 열세다.
- **diffusion 중간 개입의 필요성.** DuET과 Complete Nested의 wBB 차이(3.76 vs
  3.88, 4.88 vs 5.42, 15.88 vs 16.76)는 일관되게 DuET 쪽이 낮지만 seed 3개로는
  유의성을 주장할 수 없다.
- **checkpoint timing의 최적값.** 4종 차이가 seed 산포 이내다.

### 6.3 timing sweep이 non-result인 이유

timing 4종이 구분되지 않은 것을 "timing이 무관하다"로 읽어서는 안 된다. 주
지표(THP)가 거의 전 조건에서 0이고 보조 지표(wBB)는 reward와 정렬된 축이므로,
**애초에 반응할 수 있는 지표로 sweep하지 않았다.** 반응하지 않는 지표를 쓸면
아무것도 측정되지 않는다.

같은 이유로 K/M sweep도 추가 정보를 주지 못했다.

### 6.4 사용해야 하는 표현

> PVB-ATLAS + DuET의 fast-folding TPS 파일럿에서, DuET은 고정 RMSD reward 방향으로
> endpoint population을 이동시키는 데 성공했고 (Frozen 대비 2.3–2.8 Å), peptide
> 결합을 강제한 구조 검증도 통과했다. 그러나 공식 first-two-TICA 타깃 basin에
> 도달한 것은 steering이 없는 Frozen뿐이었으며, native contact Q는 steering에
> 반응하지 않았다. 따라서 이 결과는 reward 정렬 실패를 특정한 것이지 folding
> 회수의 증거가 아니다.

**피해야 하는 표현**

- "DuET이 PVB에서 folding pathway를 복원했다."
- "RMSD가 낮아졌으므로 folded 상태에 가깝다."
- "timing이 성능에 영향을 주지 않는다."
- "3.2 ns 안에 folding을 가속했다."

---

## 7. 다음 단계

**H4 — reward를 TICA 좌표로 정의하면 격차가 닫히는가?**

reward를 `−λ·(TICA 거리)²`로 교체하고 BBA에서 DuET K8×M8을 3 seed 실행하여 기존
Frozen baseline과 hit/경로를 비교한다. 이는 config 변경이 아니라 **신규 observable
구현**을 요구한다 — TICA projection이 샘플링 루프 안에서 계산 가능해야 한다.

반증 조건: TICA reward로도 hit이 0이면 한계는 reward가 아니라 **3.2 ns horizon**에
있다. fast folder는 실제로 µs 규모로 접히므로 이쪽 가능성도 낮지 않다.

H4가 답해지기 전까지 추가 timing / K‑M sweep은 실행 가치가 없다.

부수적으로 **H1 검증**(BBL을 Rg ≈ 15 Å 출발 구조에서 재실행)은 독립적이고 3셀로
가능하나, 중간 구조를 보간 + Cα 구속 최소화로 직접 만들어야 한다.

---

## 8. 재현 정보

**코드**: `sejin030508/AIMD_EnhanceSampling`, 브랜치 `fast-folders-extended`

| 경로 | 내용 |
| --- | --- |
| `src/confmh/adapters/pvb_duet.py` | PVB DuET 어댑터 |
| `src/confmh/duet/runner.py` | backend 레지스트리 (`confrover` / `pvb`) |
| `scripts/duet/pvb_preflight/` | 체크포인트 로드·분해 루프·분기 검증 |
| `scripts/duet/fast_folders_extended/` | config 생성, 큐 러너, 분석 도구 |
| `scripts/duet/fast_folders_extended/results_pvb_all_66.json` | 66셀 통합 결과 |

**산출물** (`/workspace/sejin/AI_MD_NSMC_phase_b_recovery`)

```
outputs/pvb_full/<timing>/<protein>/stride128_t32/<method>/seed_<seed>/
    metrics.json
    small_protein_metrics.json
    small_protein_frame_diagnostics.json
    native_contact_q.json
    pre_final_population_atom37.npz
    tica_projection.npz
```

**환경**: `confrover-mh` (torch 2.1.2+cu121) + `/workspace/sejin/pvb_assets/site`
(`torch_scatter`, `e3nn`, `easydict`, `opt_einsum`, `emcee`; **`--no-deps` 설치 필수**
— 일반 `--target` 설치는 torch 2.14와 CUDA 13 스택을 끌어와 환경의 torch 2.1.2를
가린다). 실행 시 `PVB_SITE`를 `small_protein_env.sh` 소싱 **전에** 설정해야 한다.

**실행 중 수정한 공유 코드** (백업 보존, 재실행 시 no-op)

| 파일 | 수정 |
| --- | --- |
| `phase_b_runner.py` | `build_confrover_adapter` → `build_adapter` |
| `small_protein_env.sh` | `PYTHONPATH`에 `PVB_SITE` append |
| `small_protein_runner.py` | `_SMALL_PROTEINS`에 `bbl`, `protein_b`, `homeodomain` 추가 |
| `run_small_protein_cell.sh` | method 화이트리스트에 `complete_nested` 추가 |

마지막 두 개는 각각 BBL 전 셀 실패와 대조군 전 셀 거부를 일으키던 것으로, **샘플링이
끝난 뒤** 터져 계산을 버리는 형태였다.
