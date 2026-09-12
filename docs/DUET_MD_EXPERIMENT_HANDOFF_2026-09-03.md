# DuET-MD 실험 전체 정리 및 재개 인수인계

작성 기준 시각: **2026-09-03 16:42 KST**  
로컬 프로젝트: `/Users/sejin/Desktop/AIBL/projects/AI_MD_NSMC`  
기존 Level-1/공유 자산 프로젝트: `/Users/sejin/Desktop/AIBL/projects/confrover_mh_steering`

이 문서는 지금까지 수행한 DuET-MD 실험을 처음부터 설명하고, 다른 Codex CLI가 현재 상태를 오해하지 않고 실험을 재개할 수 있도록 만든 단일 인수인계 문서다. 단순 import 확인, 1-frame smoke run, dry-run은 과학적 결과 목록에서 제외했다. 다만 현재 실행 안전성과 직접 관련된 support-gate 및 메모리 상태는 운영 현황에 별도로 기록한다.

---

## 2026-09-05 신규 단백질 코호트 프로토콜 보충

아래 본문에 기록된 6J56, 7LP1, ABL1 결과와 당시의 R1/R2 설계·R3
held-out 정의는 **역사적 실험 정의 그대로** 유지한다. 앞으로 수행할 약 10개
단백질의 confirmatory route benchmark에는 다음 문서를 우선 적용한다.

- 주 프로토콜: [DUET_PROTEIN_BENCHMARK_PROTOCOL_V3.md](DUET_PROTEIN_BENCHMARK_PROTOCOL_V3.md)
- 사전등록 후보군: [DUET_ATLAS_CONFIRMATORY_COHORT_SCREEN_V1.md](DUET_ATLAS_CONFIRMATORY_COHORT_SCREEN_V1.md)
- 실제 동결 결과: [DUET_ATLAS_COHORT_PREPARATION_2026-09-05.md](DUET_ATLAS_COHORT_PREPARATION_2026-09-05.md)
- 설정 명세: `configs/duet/templates/prepare_atlas_route_benchmark_v3.yaml`

동결된 핵심 결정은 다음과 같다.

1. original R1/R2/R3 번호를 고정 역할로 보지 않는다. 공식 ConfRover case가
   사용한 replicate를 `D1`, 남은 것 중 번호가 가장 낮은 replicate를 `D2`,
   마지막 replicate를 `H`로 지정한다. D1/D2는 task 설계용이고 H는 task를
   정의하지 않는 선택적 2차 fidelity 자료다.
2. reference transition path는 A→B라는 구조적 route와 residue-distance
   milestone을 정하는 데 쓴다. 생성 trajectory가 reference의 좁은 시간 창을
   그대로 재현하도록 요구하지 않는다.
3. reference 첫 도달 시간 대비 생성 nominal deadline은 사전등록된 최대 10배
   압축을 허용한다. 단, 이는 frozen surrogate 안에서의 biased path discovery
   비교이지 실제 kinetics, MFPT, rate 또는 free energy의 추정이 아니다.
4. 성공은 같은 deadline 안에서 generated A, B, target이 각각 2 frame 이상
   지속되며 엄격히 A→B→target 순서로 끝나는 것이다. 안정적 target에 일찍
   도달해도 성공이며 `time_to_success`와 누적 성공곡선을 기록한다.
5. H에 같은 transition이 없으면 fidelity는 `unavailable`이다. 다른 route가
   보이면 heterogeneity로 기록하며, 어느 경우도 단독 탈락 조건이 아니다.
6. 모든 주 실험은 decoder population budget `B=64`를 사용한다. Frozen 및
   Outer-only는 `K64×M1`, Inner-only는 `K1×M64`, Complete Nested와 DuET은
   `K16×M4`로 맞춘다.

2026-09-05에 generated multi-frame persistence, strict A→B→target state
transition, first-stable target hit, deadline censoring 및 time-to-success를
runtime에 구현했고 DuET 단위 테스트 전체가 통과했다(`61 passed`). 새 코호트는
method outcome을 생성하지 않은 채 ATLAS asset, transition, nominal challenge,
A/B catalog를 동결했다. Primary 10개 중 `6in7_A`, `6lus_A`, `6ovk_R`이
pre-method gate에서 탈락했고 사전등록 reserve 1--3을 순서대로 넣어 10개 task를
확보했다. Static prelaunch audit는 10/10 통과했다. 실제 method comparison
전에는 각 단백질의 ConfRover representation과 Frozen reachability, checkpoint
predictivity, whole-path validity support gate를 GPU에서 별도로 통과해야 한다.
2026-09-05 H100 staging에서 representation/config gate와 p=.95 checkpoint
continuation-predictivity gate는 10/10 통과했다. K=1 whole-path smoke는 endpoint
10/10 valid, strict whole-path 8/10 valid였으며, K=8 Frozen pilot이 진행 중이다.
메인 5-method evaluation seed는 아직 시작하지 않았다. 자세한 실행 기록은
`docs/DUET_ATLAS_COHORT_PREPARATION_2026-09-05.md`에 있다.
2026-09-06 확인 시 K=8 pilot은 9/10 완료됐다. full success는 아직 0건,
A→B는 `6q9c_A`에서만 1/8이었으며 `6tly_A` whole-path valid는 2/8이었다.
따라서 reachability/validity gate 미해결 상태이고, 마지막 `7p46_A`만 persistent
remote job으로 재개했다.
같은 날 overnight final Frozen eligibility queue도 두 H100에 시작했다. 설정은
protein당 `K48×M1`, ordered task, seeds `20260993/20260994`이며 총 20 run이다.
GPU1은 `6tly_A`부터 즉시 시작했고 GPU2는 `7p46_A` K=8 종료 후 `6q9c_A`부터
자동 시작한다. 이 queue는 eligibility 검사이며 main 5-method 결과가 아니다.
이후 Frozen nonzero를 필수 gate로 두는 것은 rare-event benchmark를 쉬운
task로 편향시킨다는 판단에 따라 K48 queue를 첫 완료 전에 중단했다. 실제 B=64
main queue를 protein-first 방식으로 시작했으며, 두 H100은 `6jv8_A/ordered`의
서로 다른 main seed에서 다섯 방법을 모두 순차 실행한다. 모든 ordered seed를
끝낸 후 endpoint, 그 후 다음 protein으로 이동한다.
사용자 요청으로 이 five-seed queue도 첫 완료 전에 중단하고 seed `20261001`
하나만 남겼다. 현재 총계는 `5 methods x 10 proteins x 2 tasks = 100 runs`다.
GPU1은 Frozen/Outer-only/Inner-only, GPU2는 Complete Nested/DuET을 담당하고,
두 lane이 같은 protein/task를 모두 끝내야 다음 task로 이동한다.

---

## 0. 가장 먼저 읽을 요약

### 연구 질문

학습이 끝난 **frozen ConfRover**를 재학습하거나 물리적 힘을 추가하지 않고, 생성 중 후보를 선택·복제하는 것만으로 사용자가 원하는 시간 순서 조건을 더 자주 만족시킬 수 있는지 평가한다.

핵심 비교는 다음과 같다.

- **Frozen**: steering 없이 독립 trajectory를 생성한다.
- **Complete Nested**: 다음 physical frame 후보들을 reverse diffusion 끝까지 모두 생성한 뒤, 완성된 후보 중 하나를 선택한다.
- **DuET-MD**: reverse diffusion 중간에도 후보를 선택하고, 완성된 physical-frame history에도 다시 선택을 적용한다.

현재 입증하려는 조건부 주장은 다음과 같다.

> 동일한 ConfRover, 동일한 temporal task, 동일한 seed family와 decoder NFE에서, diffusion 중간 상태가 미래의 유망한 frame을 예측할 수 있는 경우 DuET-MD가 Complete Nested보다 독립적인 성공 trajectory와 성공 probability mass를 더 많이 확보할 수 있는가?

### 현재까지의 결론

1. **구현 타당성은 확보했다.** Exact toy와 ConfRover preflight가 통과했다.
2. **6J56 초기 실험은 DuET 우월성을 지지하지 않았다.** 대부분의 ordered/windowed run이 실패했고 K/M 배분만 바꿔도 해결되지 않았다.
3. **7LP1 v2 결과도 allocation에 따라 혼합됐다.** `K4×M8`, `K8×M4`, `K8×M8`에서는 DuET이 Complete Nested보다 성공 metric이 낮았다. 반면 `K16×M4` one-checkpoint DuET은 5 seeds에서 pre-resampling 독립 성공률과 success mass가 높았다.
4. **따라서 확정적 superiority는 아니다.** 유리했던 `K16×M4`에서도 bootstrap 불확실성이 남고, DuET의 held-out path distance는 약 5% 나빴다.
5. **Two-checkpoint `[0.85, 0.95]` 개선은 혼합 결과다.** Success mass는 증가했지만 독립 성공률·lineage·fidelity는 개선되지 않았다. 엄격한 사전 규칙을 적용하면 one-checkpoint `0.95`가 더 보수적인 기본 방법이다.
6. **ABL1 DFG flip case study는 production 직전까지 구현됐지만 아직 본 결과는 없다.** A6000 budget-8 support-gate 세 방법은 모두 완료됐지만 whole-path validity가 Frozen `0.875`, Complete Nested `0.0`, DuET `0.0`으로 자동 기준 `1.0`을 만족하지 못했다. Endpoint 구조는 세 방법 모두 `1.0` valid였다.
7. **H100 두 장의 기존 작업은 모두 끝났고 GPU는 유휴다.** H100에는 현재 대기 중인 queue가 없다.

### 절대 과장하면 안 되는 주장

- DuET이 실제 MD kinetics, MFPT, free energy를 복원했다고 말하면 안 된다.
- 한 성공 trajectory가 최종 resampling에서 여러 번 복제된 것을 여러 개의 독립 발견으로 세면 안 된다.
- 7LP1 endpoint 성공 증가를 reference mechanism/path fidelity 증가로 해석하면 안 된다.
- 6J56 결과를 superiority evidence로 사용하면 안 된다.
- ABL1의 두 공개 transition path를 독립 replicate처럼 사용해 route population을 추정하면 안 된다.

---

## 1. 용어와 방법

### 1.1 Physical time과 diffusion time

ConfRover는 trajectory의 다음 구조 한 장을 바로 출력하지 않는다. 각 다음 frame을 만들 때 노이즈 상태에서 시작해 약 200회의 reverse-diffusion step을 거쳐 구조를 복원한다.

- **Physical time**: trajectory의 frame 번호, 예: frame 0 → 1 → 2 → ... → 24
- **Diffusion time**: frame 하나를 만드는 내부 reverse-diffusion 진행도, 예: 0% → 85% → 95% → 완료

DuET의 “dual time”은 이 두 축을 뜻한다.

### 1.2 K와 M

- `K`: 동시에 유지하는 **outer physical-history particle 수**다. 서로 다른 trajectory 계보의 수에 해당한다.
- `M`: 각 outer particle에서 다음 frame을 만들 때 생성하는 **inner diffusion 후보 수**다.
- 한 physical step의 decoder 후보 budget은 대체로 `K × M`이다.

예를 들어 `K16 × M4`는 16개의 trajectory history 각각에서 다음 frame 후보 4개를 생성한다. Complete Nested와 DuET을 같은 `K/M`으로 비교해야 diffusion-time 선택 자체의 효과를 볼 수 있다.

### 1.3 Complete Nested와 DuET의 차이

Complete Nested:

1. 각 trajectory history에서 `M`개 reverse diffusion을 끝까지 수행한다.
2. 완성된 frame들의 temporal reward를 계산한다.
3. 완성된 후보 중 하나를 선택한다.
4. 선택된 frame을 history에 추가한다.

DuET:

1. 각 trajectory history에서 `M`개 reverse diffusion을 시작한다.
2. 미리 고정한 diffusion checkpoint에서 각 noisy state의 predicted-clean frame을 평가한다.
3. 유망한 diffusion state를 복제하고 독립 noise로 continuation한다.
4. 완성된 frame에서 endpoint correction을 적용한다.
5. physical history 수준의 outer SMC selection도 수행한다.

DuET은 force, biasing potential 또는 새로운 MD integrator가 아니다. Frozen ConfRover가 생성할 수 있는 후보들 사이에서 선택하는 inference-time steering이다.

### 1.4 PCA와 PC1

PCA는 trajectory 구조 변화에서 가장 큰 공통 변화 방향을 데이터로부터 찾는 차원 축소 방법이다. `PC1`은 그중 가장 큰 변화 방향의 좌표다. 여기서는 시작 basin, 중간 영역, 목표 basin을 간단한 숫자 구간으로 정의할 때 사용했다.

중요하게 PCA basis와 task threshold는 ATLAS의 R1/R2 trajectory로만 설계하고, R3는 held-out 평가에 사용했다.

### 1.5 Event, event A, event B

“Event”는 특정 physical-frame window 안에 미리 정의한 구조 조건을 만족하는 것을 뜻한다. Event A/B는 단백질마다 정의가 다르므로 이름만 보고 생물학적으로 같은 사건이라고 생각하면 안 된다.

#### 6J56 초기 ordered task의 event

Residue index는 생성 sequence의 **0-based index**다.

- **Event A**: residue 29와 127의 거리가 `4.0231 nm` 이하가 되어 contact가 형성되는 것. 허용 frame은 3–5다.
- **Event B**: residue 1과 126의 거리가 `1.9283 nm` 이하가 되는 것. 허용 frame은 5–7이다.
- **Terminal event**: frame 8의 PC1이 `[1.9443, 2.4443]`에 들어가는 것.
- Ordered success는 A가 먼저, B가 나중, terminal basin이 마지막에 성립해야 한다.

이 task는 R1/R2에서 transition segment가 하나뿐이어서 일반화 근거가 약하다.

#### 6J56 v3 ordered task의 event

- **Event A**: residue 2와 72의 거리가 `2.7027 nm` 이상으로 벌어지는 것. 허용 frame은 7–15다.
- **Event B**: residue 1과 126의 거리가 `1.9283 nm` 이하로 닫히는 것. 허용 frame은 15–21이다.
- **Terminal event**: frame 19–24 사이 PC1이 `[0.2390, 0.7390]`에 들어가는 것.

그러나 Event B는 시작 구조에서 이미 만족돼 있었고 task를 지지하는 R1/R2 segment가 하나뿐이었다. 따라서 이 결과는 allocation/implementation 진단으로만 해석한다.

#### 7LP1 v2 ordered task의 event

7LP1 sequence는 40 residues다. Residue index는 생성 sequence의 0-based index다.

- **Event A — N-terminal opening**: residue 0과 7 사이 거리가 초기 `0.7524 nm`에서 증가해 `1.7092 nm` 이상이 되는 것. 허용 frame은 1–5다.
- **Event B — late closing**: residue 25와 36 사이 거리가 초기 `1.9242 nm`에서 감소해 `1.7424 nm` 이하가 되는 것. 허용 frame은 19–23이다.
- **Terminal event**: frame 21–24 사이 PC1이 `[-1.5733, -0.5733]`에 들어가는 것.
- Ordered success는 opening A → closing B → terminal basin 순서를 만족해야 한다.

R1/R2의 두 transition segment 모두 A-before-B를 보였고, A/B crossing frame은 각각 `[1, 2]`, `[23, 22]`였다. 이 때문에 24-frame horizon이 실제로 긴 physical-time dependency를 포함한다.

### 1.6 성공 metric

가장 중요한 metric은 최종 resampling **이전**에 계산한다.

- **Pre-resampling unique success rate**: 마지막 frame에서 서로 다른 outer particle 중 조건을 만족한 비율
- **Pre-resampling success weight mass**: 최종 weighted population에서 성공 particle들이 가진 총 확률 질량
- **Post-resampling success rate**: 최종 출력 중 성공 비율. 생성 yield로는 유용하지만 독립 발견 수는 아니다.
- **Surviving initial ancestors**: 최종 population이 몇 개의 독립 초기 계보에서 왔는지
- **Held-out path distance**: R3 reference path와 생성 path의 거리. 낮을수록 reference path와 유사하다.
- **Structural validity**: nonfinite 좌표, CA clash, 과도한 인접 CA 거리 등이 없는지

전체 구조의 회전·병진은 분자 내부 변화가 아니므로, 이동량과 연속성은 모든 방법에 동일하게 Kabsch CA alignment 후 계산한다.

---

## 2. 데이터 분할 및 공정성 원칙

### 2.1 ATLAS 단백질

- R1/R2: PCA, contact, event window, threshold를 정하는 design split
- R3: 결과를 평가하는 held-out split
- Complete Nested 또는 DuET 결과를 본 후 threshold를 바꾸지 않는다.

### 2.2 계산량 통제

- 같은 protein-task-method 비교에서 reverse steps, horizon, seed family, base SDE sampler를 고정한다.
- Complete Nested와 DuET은 같은 `K/M`과 decoder NFE를 사용한다.
- `Best-of-budget`과 `Naive Dual`은 초기 진단에는 포함됐지만 최종 핵심 비교에서는 제외했다.

### 2.3 결과 선택 방지

- 6J56는 이미 많은 개발 판단에 사용됐으므로 최종 일반화 증거로 사용하지 않는다.
- 7LP1의 추가 K/M 및 checkpoint 탐색은 종료했다.
- ABL1 endpoint threshold와 hidden evaluator는 generated outcome을 보기 전에 고정했다.

---

## 3. 완료된 의미 있는 실험

## 3.1 Phase A — exact two-clock correctness

### 목적

작은 exact toy model에서 DuET의 Feynman–Kac weighting, inner/outer resampling, normalizer 추정이 수학적으로 목표 분포를 재현하는지 확인했다.

### 설정

- Config: `configs/duet/exact_toy.yaml`
- Output: `outputs/duet_md/phase_a_exact_toy`
- 실제 단백질이나 ConfRover를 사용하지 않는 finite-state exact benchmark
- Physical horizon: 5, outer states: 6, inner latent states: 3
- Tasks: terminal, windowed, ordered
- Methods: Frozen, Outer-only, Inner-only, Naive Dual, Complete Nested, DuET
- 기본 비교: Frozen/Outer `K16×M1`, Inner-only `K1×M16`, dual methods `K4×M4`
- Convergence scale: Complete Nested/DuET/Naive Dual에 `K=M=2,4,8,16`, 각 100 repetitions
- Proper-weighting test: `M=1,2,4,8,16`, 각 3,000 repetitions
- DuET 중간 marginal, final marginal, route probability, normalizer를 exact 값과 비교

### 결과와 해석

- Correctness gate의 6개 check가 모두 통과했다.
  - Proper-weighting L1 relative residual 최대 `0.0341`, 기준 `<0.10`
  - Local normalizer relative bias 절댓값 최대 `0.0039`
  - Inner `M=1→16`에서 normalizer estimator variance `0.0381→0.00292`로 감소
  - Scale 16 ordered task의 final TV distance: Complete Nested `0.0240`, DuET `0.0261`
  - Scale 16 ordered normalizer relative bias: Complete Nested `-0.0357`, DuET `0.0135`
  - 같은 scale의 Naive Dual bias는 `0.1570`으로 intentional mismatch가 non-degenerate함을 확인
- DuET 구현의 weighting과 resampling이 원리적으로 잘못됐다는 증거는 없다.
- 이 결과는 구현 타당성이지 단백질에서의 우월성 증거가 아니다.

---

## 3.2 Phase B — ConfRover ODE/SDE/checkpoint preflight

### 목적

ConfRover 공식 ODE 출력과 새 adapter의 SDE/checkpoint 경로가 구조를 망가뜨리지 않는지, constant-potential resampling이 기본 marginal을 과도하게 바꾸지 않는지 확인했다.

### 설정

- Protein: `6j56_A`, 129 residues
- Horizon: 8 generated frames
- Replicates: 16
- Reverse steps: ODE 200 상당, SDE adapter 실제 199 decoder steps/frame
- Config: `configs/duet/confrover_sde_preflight.yaml`
- Output: `outputs/duet_md/phase_b_confrover_preflight`
- 비교 variant:
  - 공식 ConfRover ODE
  - unconditioned SDE
  - checkpoint hook을 사용하지만 resampling하지 않는 SDE
  - constant potential에서 resampling하는 SDE

### Kabsch 수정의 배경

초기에는 raw coordinate displacement가 약 `5.7–6.0 nm`로 나타나 harsh gate를 실패했다. 이는 ConfRover가 고정된 절대 좌표계에서 단백질을 생성하지 않아 전체 회전·병진이 포함됐기 때문이다. 생성 구조 자체를 바꾸지 않고 평가할 때만 각 frame을 frame 0에 Kabsch CA alignment했다. 공식 ODE baseline에도 같은 정렬을 적용했다.

### 결과

- 모든 variant nonfinite rate: `0`
- 모든 variant clash rate: `0`
- hard geometry validity: `1.0`
- 공식 ODE mean maximum adjacent-CA distance: `4.715 Å`
- SDE variants: `4.890–4.939 Å`
- SDE/ODE ratio: `1.037–1.047`, 기준 `≤1.1`
- Unconditioned SDE 대비 PC1 Wasserstein:
  - checkpoint/no resampling: `0.203`
  - constant resampling: `0.314`
  - 기준 `≤0.5`
- Gate: **passed**

### 해석

- SDE adapter와 checkpoint hook은 공식 ConfRover 대비 구조 유효성을 훼손하지 않았다.
- Constant potential에서 checkpoint/resampling을 넣어도 PC1 marginal이 허용 범위 안에 있었다.
- 이 실험은 이후 비교를 신뢰하기 위한 fallback/adapter validation이다. DuET superiority evidence는 아니다.
- `outputs/duet_md/phase_b_confrover_preflight_pre_seedfix_20260831`은 seed 처리 수정 전의 폐기된 archive다. 최종 Phase B 집계에 섞지 않는다.

---

## 3.3 Phase C — 초기 6J56 core comparison

### 목적

초기 8-frame 조건에서 여러 steering baseline과 DuET을 같은 decoder population budget으로 비교했다.

### 설정

- Protein: `6j56_A`
- Horizon: 8
- Reverse steps: 200 configured
- Checkpoint: `0.75`
- Seeds: `20260831`, `20260901`, `20260902`
- Tasks: endpoint, windowed, ordered
- Decoder population budget per step: 16
- Methods:
  - Frozen `K16×M1`
  - Best-of-budget `K16×M1`
  - Outer-only `K16×M1`
  - Inner-only `K1×M16`
  - Naive Dual `K4×M4`
  - Complete Nested `K4×M4`
  - DuET `K4×M4`
- Config: `configs/duet/6j56_factorial.yaml`
- Output: `outputs/duet_md/phase_c_6j56_core`

### Task 정의

- Endpoint: frame 8에서 목표 PC1 basin에 진입
- Windowed: frame 3–5에서 중간 PC1 영역을 지나고 마지막에 목표 basin 진입
- Ordered: 앞서 정의한 6J56 Event A → Event B → terminal basin 순서 만족

### 결과

Post-resampling success rate의 seed별 결과다.

| Task | Method | Seed별 성공률 |
|---|---|---|
| Endpoint | DuET | `[1.0, 0.0, 0.0]` |
| Endpoint | Naive Dual | `[1.0, 0.0, 0.0]` |
| Endpoint | Outer-only | `[1.0, 0.0, 0.0]` |
| Endpoint | Frozen / Best / Inner-only / Complete Nested | 모두 `[0, 0, 0]` |
| Ordered | 모든 방법 | 모두 `[0, 0, 0]` |
| Windowed | 모든 방법 | 모두 `[0, 0, 0]` |

구조 유효성은 대부분 `1.0`이었으나 일부 old ordered/outer 결과에서 `0.75` 또는 `0.9375`가 있었다.

### 해석

- Endpoint 한 seed의 성공은 DuET에 고유하지 않았다. Naive Dual과 Outer-only도 같은 seed에서 성공했다.
- Ordered/windowed task는 모든 방법이 실패했으므로 방법 우열을 구분하지 못했다.
- 이 결과는 DuET 우월성을 지지하지 않는다. Task rarity, 짧은 horizon, weak checkpoint signal을 발견한 negative result다.

---

## 3.4 Phase D — 6J56 K/M allocation

### 목적

동일 budget 16에서 outer history 다양성 `K`와 inner diffusion discovery `M`의 배분만 바꾸면 실패가 해결되는지 확인했다.

### 설정

- Protein/task/config 기반: Phase C의 6J56 windowed 및 ordered task
- Horizon: 8
- Seeds: `20260831`, `20260901`, `20260902`
- DuET만 실행
- Allocation:
  - `K16×M1`
  - `K8×M2`
  - `K4×M4`
  - `K2×M8`
  - `K1×M16`
- Config: `configs/duet/6j56_km_grid.yaml`
- Output: `outputs/duet_md/phase_d_km_allocation`

### 결과

- 모든 allocation, 모든 seed, ordered/windowed task에서 success rate `0`
- 대부분 최종 surviving initial ancestors가 `1`
- 예외도 대체로 `2`를 넘지 않았다.

### 해석

- 단순히 `K` 또는 `M`을 늘리는 것만으로 task가 해결되지 않았다.
- 후보 자체가 희귀하고, checkpoint의 미래 예측력이 약하며, outer genealogy가 쉽게 붕괴하는 문제가 함께 있었다.
- 이 negative result가 horizon 증가, continuous distance potential, adaptive outer resampling, checkpoint diagnostics로 이어졌다.

---

## 3.5 6J56 protocol v2/v3 개발 실험

### 변경 사항

- Horizon을 16, 이후 24로 증가
- Contact를 binary indicator만이 아니라 연속 거리로 평가
- 실패 후에도 terminal 방향의 score 차이를 유지하는 failure guidance 추가
- Outer ESS가 기준 이하일 때만 resampling
- Final output 복제 전 unique success와 success mass를 계산

### Protocol v2, horizon 16

- Seed: `20260911`
- Population budget: 32
- Frozen/Outer-only `K32×M1`, Inner-only `K1×M32`, Complete Nested/DuET `K8×M4`
- Decoder NFE: 모든 방법 `101,888`
- Output: `outputs/duet_md/protocol_v2/main_ordered`

Frozen, Outer-only, Inner-only, Complete Nested, DuET 모두 pre-resampling success, success mass, post-resampling success가 `0`이었다. Structural validity는 Frozen `0.96875`, 나머지는 `1.0`이었다. Horizon 8보다 늘렸지만 이 task에서는 방법을 구분하지 못했다.

### 대표 설정

6J56 v3:

- Horizon: 24
- Population budget: 32
- Complete Nested/DuET: `K8×M4`
- Frozen/Outer-only: `K32×M1`
- Inner-only: `K1×M32`
- Checkpoint: `0.90`
- Outer resampling ESS fraction: `0.25`
- Config: `configs/duet/6j56_v3_main_ordered.yaml`
- Output: `outputs/duet_md/protocol_v3`

### 핵심 결과

- Frozen design 64 paths 중 Event A 도달: `3/64`
- Event A 다음 Event B까지 도달: `2/64`

Seed `20260913`의 full baseline 비교:

| Method | Allocation | Pre unique success | Success mass | Post success | Held-out distance | Ancestors |
|---|---|---:|---:|---:|---:|---:|
| Frozen | `K32×M1` | 0.03125 | 0.03125 | 0.03125 | **1.3741** | 32 |
| Outer-only | `K32×M1` | 0.03125 | 0.2623 | 0.2500 | 1.4273 | 23 |
| Inner-only | `K1×M32` | **1.0000** | **1.0000** | **1.0000** | 1.8737 | 1 |
| Complete Nested | `K8×M4` | 0.1250 | 0.4385 | 0.5000 | 1.4480 | 5 |
| DuET | `K8×M4` | 0.1250 | 0.4167 | 0.3750 | 1.4927 | 5 |

Complete Nested와 DuET은 final resampling 전에 각각 독립 성공 path 1개를 발견했다. Post-resampling `4/8` 대 `3/8` 차이는 주로 그 한 path의 복제 수 차이다. Inner-only의 `1.0`은 `K=1`인 단 하나의 history가 성공했다는 뜻이지 32개의 독립 성공 path가 있다는 뜻이 아니다. 그래도 이 seed에서 DuET이 최선의 방법은 아니었다.

- Seed `20260914`:
  - Complete Nested는 독립 성공 path 1개, success mass `0.1947`
  - DuET은 성공 path 없음
  - DuET inner ESS가 거의 `4/4`여서 checkpoint selection signal이 약했다.

추가 allocation pilot `K4×M8`, seeds `20260914–20260915`에서는 Complete Nested가 두 seed 모두 성공 `0`이었다. DuET은 seed `20260914`에서 pre unique `0.25`, mass `0.5251`, post `0.5`였고 seed `20260915`에서는 모두 `0`이었다. 그러나 두 방법 모두 preterminal outer resampling count가 `0`이어서 이 결과는 diffusion-time 후보 선택의 가능성만 보여주며 dual-clock superiority 증거는 아니다.

### 해석

- 6J56 task는 frozen support 밖은 아니지만 매우 희귀하다.
- DuET은 Complete Nested보다 일관되게 좋지 않았다.
- 일부 작은 pilot에서 DuET 성공이 나왔지만 seed 의존적이었고 physical-time resampling이 실제로 작동하지 않았다.
- V3 Event B가 시작 구조에서 이미 만족돼 있었고 R1/R2 transition segment도 하나뿐이므로 formal two-clock benchmark로 부적합하다.
- 6J56은 최종 논문 evidence가 아니라 개발 및 falsification case로 유지한다.

---

## 3.6 7LP1 초기 탐색 및 v2 task 선택

### 초기 탐색 결과

- 초기 endpoint task는 particle method에서 쉽게 포화됐다.
- 초기 windowed task는 어느 방법도 구분하지 못하는 unresolved task였다.
- 초기 ordered task에서 `K4×M4` DuET은 3 seeds 중 1개 성공했다.
- `K2×M8` DuET은 3 seeds 모두 성공했지만 평균적으로 surviving outer ancestor가 1개뿐이었다.

### 초기 task의 문제

- Contact pair를 median crossing time으로만 선택했다.
- 한 R1/R2 segment에서는 B가 A보다 먼저 발생했다.
- 일부 observed crossing이 event window 밖이었다.

따라서 이 결과는 탐색 기록이며 최종 evidence로 사용하지 않는다.

### v2 task 수정

- 두 R1/R2 transition segment 모두에서 A-before-B가 성립하는 contact pair만 허용
- 실제 crossing frame을 모두 포함하도록 window 고정
- R3는 계속 held-out
- Frozen design에서 nonzero이지만 nonsaturated인 reachability를 요구

최종 v2 Event A/B 정의는 1.5절에 명시했다.

---

## 3.7 7LP1 v2 frozen reachability gate

### 목적과 설정

수정한 ordered task가 frozen ConfRover의 support 안에 있으면서도 포화되지 않았는지, Complete Nested/DuET 결과를 보기 전에 확인했다.

- Method: Frozen
- Horizon: 24
- `K48×M1`
- Seeds: `20260916`, `20260917`
- Decoder NFE/seed: `229,248`
- Output: `outputs/duet_md/extra_proteins/7lp1_A/protocol_v2/frozen_gate`

### 결과

| Seed | Event A 도달 | A 다음 B 도달 | A → B → terminal 전체 성공 |
|---|---:|---:|---:|
| 20260916 | 5/48 | 5/48 | 1/48 |
| 20260917 | 5/48 | 5/48 | 4/48 |
| 합계 | 10/96 = 10.42% | 10/96 = 10.42% | 5/96 = 5.21% |

- Structural validity: 두 seed 모두 `1.0`
- Mean held-out R3 path distance: `2.4995`
- Frozen이므로 surviving initial ancestors는 seed당 48이고 outer resampling은 없다.

### 해석

Event A와 A→B는 드물지만 관찰 가능했고 전체 성공도 0이 아니며 포화되지 않았다. 즉 7LP1 v2 task는 method 비교를 수행할 최소 reachability 조건을 만족했다. 이 gate는 방법 우월성 비교가 아니라 task eligibility 검사다.

---

## 3.8 7LP1 v2 one-checkpoint population/allocation study

### 공통 설정

- Protein/task: 7LP1 v2 ordered
- Horizon: 24
- Checkpoint: `0.95`
- Seeds: `20260920–20260924`
- Complete Nested와 DuET은 각 allocation 안에서 동일한 `K/M`, seed, SDE, NFE를 사용
- Structural validity: 아래 모든 run에서 `1.0`

### Budget 32 결과

`K×M=32`, decoder NFE/cell `152,832`이다.

| Allocation | Method | Pre unique success | Success weight mass | Post success | Held-out distance | Ancestors | Preterminal resampling |
|---|---|---:|---:|---:|---:|---:|---:|
| `K4×M8` | Complete Nested | **0.3000** | **0.4601** | **0.4000** | 2.7967 | **1.4** | 1.4 |
| `K4×M8` | DuET | 0.2000 | 0.3208 | 0.3500 | **2.3082** | 1.0 | 1.8 |
| `K8×M4` | Complete Nested | **0.5250** | **0.7821** | **0.7750** | **2.5629** | **2.0** | 1.2 |
| `K8×M4` | DuET | 0.3500 | 0.6804 | 0.7000 | 2.6618 | 1.8 | 1.2 |

`K4×M8` DuET은 held-out distance만 더 낮았고 세 성공 metric과 lineage가 낮았다. `K8×M4` DuET도 success, fidelity, lineage에서 모두 Complete Nested보다 낮았다.

### Budget 64 결과

`K×M=64`, decoder NFE/cell `305,664`이다.

| Allocation | Method | Pre unique success | Success weight mass | Post success | Held-out distance | Ancestors | Preterminal resampling |
|---|---|---:|---:|---:|---:|---:|---:|
| `K8×M8` | Complete Nested | **0.5000** | **0.8378** | **0.8000** | **2.6115** | **2.2** | 1.6 |
| `K8×M8` | DuET | 0.3000 | 0.6137 | 0.6250 | 2.7883 | 1.6 | 2.2 |
| `K4×M16` | Complete Nested | 0.5000 | **0.8660** | 0.8500 | 2.6325 | 1.4 | 0.8 |
| `K4×M16` | DuET | **0.6500** | 0.8447 | 0.8500 | **2.6008** | **1.8** | 1.0 |
| `K16×M4` | Complete Nested | 0.4625 | 0.8536 | 0.8625 | **2.5523** | 3.2 | 1.6 |
| `K16×M4` | DuET | **0.6000** | **0.8908** | **0.9000** | 2.6818 | 3.2 | 1.8 |

### 전체 allocation 해석

- 단순히 총 budget을 64로 늘린 `K8×M8`에서는 DuET이 Complete Nested보다 모든 핵심 성공 metric, held-out fidelity, lineage가 낮았다.
- Inner 후보에 치우친 `K4×M16`은 DuET unique success와 fidelity가 좋아졌지만 success mass는 낮고 post success는 같았다. Outer lineage가 평균 2개 미만이어서 dual-clock 주장의 주 설정으로는 약하다.
- Outer history를 충분히 유지한 `K16×M4`가 DuET에 가장 유리한 allocation이었다. Unique success와 success mass가 함께 높고 ancestor 수가 유지됐다.
- 하지만 여러 allocation을 이미 비교한 development task에서 `K16×M4`를 골랐다는 점을 숨기면 안 된다. 이 5-seed 결과만으로 독립 confirmatory superiority를 주장할 수 없다.

---

## 3.9 7LP1 v2 selected `K16×M4` one-checkpoint 결과

### 목적

수정된 ordered task에서 Complete Nested와 DuET을 같은 `K/M/NFE`로 비교하고, outer/inner population을 충분히 키웠을 때 두-clock advantage가 나타나는지 확인했다.

### 선택된 개발 설정

- Protein: `7lp1_A`, 40 residues
- Start structure: ATLAS R2 frame 5000
- Horizon: 24
- Physical lag: 256 × 10 ps nominal stride
- Reverse steps: 200 configured, 실제 decoder NFE는 frame당 candidate당 199
- Complete Nested: `K16×M4`, checkpoint `0.95`
- DuET: `K16×M4`, checkpoint `0.95`
- Outer ESS fraction: `0.5`
- Seeds: `20260920–20260924`
- Decoder NFE per cell: `305,664`
- Output: `outputs/duet_md/extra_proteins/7lp1_A/protocol_v2/factorial/K16_M4_p95`

### 결과

| Metric, 5-seed mean | Complete Nested | DuET | DuET − baseline 방향 |
|---|---:|---:|---|
| Pre-resampling unique success rate | 0.4625 | **0.6000** | +0.1375 |
| Pre-resampling success weight mass | 0.8536 | **0.8908** | +0.0372 |
| Post-resampling success rate | 0.8625 | **0.9000** | +0.0375 |
| Structural validity | 1.0000 | 1.0000 | 동일 |
| Surviving initial ancestors | 3.2 | 3.2 | 동일 |
| Held-out R3 path distance | **2.5523** | 2.6818 | DuET 약 5.1% 악화 |
| Preterminal outer resampling count | 1.6 | 1.8 | DuET에서 약간 많음 |

Primary success metric의 seed별 원자료:

| Seed | Complete pre unique | DuET pre unique | Complete success mass | DuET success mass | Complete post | DuET post |
|---|---:|---:|---:|---:|---:|---:|
| 20260920 | 0.6250 | 0.3750 | 0.9972 | 0.5653 | 1.0000 | 0.5625 |
| 20260921 | 0.6875 | 0.6875 | 0.9294 | 0.9530 | 0.9375 | 0.9375 |
| 20260922 | 0.0625 | 0.5000 | 0.6038 | 0.9672 | 0.6250 | 1.0000 |
| 20260923 | 0.3125 | 0.6250 | 0.7557 | 0.9958 | 0.7500 | 1.0000 |
| 20260924 | 0.6250 | 0.8125 | 0.9821 | 0.9729 | 1.0000 | 1.0000 |

### 해석

- 현재까지 DuET을 가장 강하게 지지하는 결과다.
- Final output 복제 이전에도 독립 성공 path와 success mass가 높으므로 단순 복제 artifact만은 아니다.
- 구조 유효성과 평균 surviving lineage는 유지됐다.
- 그러나 paired bootstrap interval에 0이 포함됐다. 5 seeds만으로 확정적 통계 우월성은 아니다.
- Held-out path distance가 악화됐으므로 DuET이 reference pathway를 더 잘 재현했다고 말할 수 없다.
- Complete Nested wall time은 seed당 약 `29.8–37.3분`, DuET은 약 `39.6–40.5분`이었다.

---

## 3.10 7LP1 multi-checkpoint refinement

### 목적

기존 `0.95` checkpoint는 reverse diffusion 종료까지 약 10 steps만 남아 복제된 후보가 독립 continuation으로 갈라질 시간이 짧았다. Frozen 성공 history의 checkpoint diagnostics를 이용해 더 이른 `0.85`를 추가했다.

### 사전 진단

- Candidate progresses: `0.75`, `0.85`, `0.90`, `0.95`
- `0.75`: event-relevant rank correlation이 `0.26–0.31`까지 낮음
- `0.85`: 모든 진단 step에서 positive selection gain, 대체로 rank correlation `≥0.67`
- 고정 schedule: `[0.85, 0.95]`
- 각 checkpoint는 incremental potential ratio와 independent continuation noise를 사용
- Endpoint correction이 potential telescope를 닫음
- Decoder NFE는 one-checkpoint와 동일

### 동일 development seed 비교

Seeds `20260920–20260923`, `K16×M4`에서 DuET one-checkpoint와 two-checkpoint를 비교했다.

| Metric | One checkpoint `0.95` | Two checkpoints `[0.85,0.95]` |
|---|---:|---:|
| Pre-resampling unique success rate | **0.5469** | 0.5313 |
| Pre-resampling success weight mass | 0.8703 | **0.8978** |
| Post-resampling success rate | 0.8750 | **0.8906** |
| Held-out path distance | **2.6373** | 2.6776 |
| Surviving initial ancestors | **3.25** | 3.00 |

### Fresh confirmation

Seeds `20260925`, `20260926`에서 Complete Nested `0.95`와 two-checkpoint DuET을 비교했다.

| Metric, 2-seed mean | Complete Nested | Two-checkpoint DuET |
|---|---:|---:|
| Pre-resampling unique success rate | 0.6250 | **0.7188** |
| Pre-resampling success weight mass | 0.9066 | **0.9598** |
| Post-resampling success rate | 0.8750 | **1.0000** |
| Structural validity | 1.0000 | 1.0000 |
| Held-out path distance | **2.5812** | 2.8004 |
| Surviving initial ancestors | **4.0** | 3.5 |
| Preterminal outer resampling count | 1.5 | 1.5 |
| Decoder NFE | 305,664 | 305,664 |

Seed별 post-resampling success:

- Complete Nested: `0.8125`, `0.9375`
- DuET: `1.0`, `1.0`

Cell wall clock:

- Complete Nested: 약 `36.6–37.2분`
- Two-checkpoint DuET: 약 `39.1–40.6분`

### 해석 및 최종 판정

- Fresh seeds에서는 DuET이 성공 metric 모두에서 높았다.
- 그러나 development matched seeds에서는 unique success가 소폭 감소했고 lineage와 fidelity도 개선되지 않았다.
- Two-checkpoint가 one-checkpoint보다 명확히 우월하다는 결론은 성립하지 않는다.
- 사전 결정 규칙을 엄격히 적용하면 one-checkpoint `0.95`를 기본 DuET으로 유지하고 two-checkpoint를 ablation/variant로 보고하는 것이 보수적이다.
- 현재 ABL1 config는 generated ABL1 outcome을 보기 전에 two-checkpoint를 고정했지만, 논문에서는 이것을 “7LP1에서 확정된 개선안”이라고 표현하면 안 된다.

---

## 4. 완료 결과의 총체적 해석

### 4.1 지지되는 것

- DuET의 수학적 weighting과 state resampling 구현은 exact benchmark에서 타당하다.
- ConfRover checkpoint/SDE adapter는 구조 유효성을 유지한다.
- 적절히 선택된 7LP1 ordered task의 `K16×M4` setting에서 DuET은 Complete Nested보다 높은 독립 성공률과 success mass를 보였다.
- Fresh seed 2개에서도 같은 방향이 유지됐다.

### 4.2 아직 지지되지 않는 것

- 여러 단백질에서 반복되는 일반적 superiority
- 7LP1의 allocation 전반에서 일관된 superiority
- Reference path 또는 transition mechanism fidelity의 향상
- Two-checkpoint가 one-checkpoint보다 우월하다는 주장
- Kinetics, equilibrium population, MFPT, free energy 관련 주장

### 4.3 가장 큰 현재 병목

1. 긍정 결과가 사실상 7LP1 한 task에 집중돼 있다.
2. 7LP1 success는 높지만 held-out path distance가 악화됐다.
3. 6J56은 benchmark로 부적합했다.
4. ABL1 같은 독립 biological case의 본 결과가 아직 없다.
5. A6000 48GB에서 ABL1 full-history 메모리 peak가 매우 높다.

따라서 다음 핵심은 추가 tuning이 아니라, 사전에 고정된 ABL1 평가를 안전하게 완료해 독립 case에서 결과 방향을 확인하는 것이다.

---

## 5. ABL1 DFG-flip case study

## 5.1 연구 목적

ABL1 kinase의 DFG-in → DFG-out 전이는 알려진 concerted/staggered 경로를 가진 생물학적으로 의미 있는 전이다. Reward에는 최종 DFG-out endpoint만 사용하고, 전이 순서와 route는 hidden evaluation으로만 측정한다.

핵심 질문:

> Mechanism 정보를 steering reward에 넣지 않고도 DuET이 Frozen/Complete Nested보다 DFG-out endpoint와 물리적으로 설명 가능한 전이 path를 더 자주 회수하는가?

## 5.2 데이터

- PDB: `6XR6`, `6XR7`
- Public WT WESTPA archive: Zenodo `11194787`
- Selected WT paths:
  - `wt_concerted_flip.trr`
  - `wt_staggered_flip.trr`
- WE topology sequence: canonical ABL1 residues 227–513, 총 287 residues
- Generated residue index:
  - Tyr253: 26
  - Lys271: 44
  - Glu286: 59
  - Val299: 72
  - Thr315: 88
  - Ala380: 153
  - Asp381: 154
  - Phe382: 155

ConfRover parser는 PDB residue 번호를 사실상 배열 index로 사용하므로 prepared start PDB는 residue `1–287`로 재번호화했다. 로컬과 A6000의 corrected 파일을 사용해야 한다.

## 5.3 Reward와 hidden evaluation firewall

### Reward에 사용하는 정보

최종 Asp381/Phe382 pseudo-dihedral 두 개의 composite normalized distance만 사용한다.

- Asp center: `265.029°`, scale `31.039°`
- Phe center: `33.492°`, scale `18.434°`
- Endpoint success: composite distance `≤0.85387555`
- 정의 데이터: 두 selected WT path의 마지막 10 frames
- Generated ABL1 결과를 보기 전에 threshold 고정

### Reward에 사용하지 않는 hidden evaluation

- DFG-inter intermediate contact
- Asp/Phe completion order
- Concerted/staggered route family
- Lys271–Glu286 distance
- Path continuity와 genealogy

Reference event frames `(Asp complete, Phe complete, DFG-inter contact)`:

- Concerted: `[394, 670, 558]`
- Staggered: `[289, 614, 341]`

### 해석 한계

공개 archive에는 독립적인 WT replicate 여러 개가 아니라 WT WESTPA HDF5 하나와 selected path 두 개가 있다. 따라서 route 비교는 descriptive/unweighted다. Route population, rate, MFPT, free energy를 주장할 수 없다.

## 5.4 고정 production 설정

- Config: `configs/duet/case_studies/abl1_dfg_flip_production.yaml`
- Horizon: 24
- Reverse steps: 200 configured
- Sampler: SDE
- Frozen: `K64×M1`
- Complete Nested: `K16×M4`, checkpoint `0.95`
- DuET: `K16×M4`, checkpoints `[0.85, 0.95]`
- Seeds: `20261001–20261005`
- Decoder population budget: 64
- Expected decoder NFE/cell: `24 × 199 × 64 = 305,664`
- A6000 decoder microbatch size: 2
- Output root: `outputs/duet_md/case_studies/abl1_dfg_flip`

중요: Two-checkpoint 사용은 ABL1 outcome 이전에 고정됐지만, 7LP1에서 one-checkpoint보다 확정적으로 우수했던 것은 아니다.

---

## 6. 현재 A6000 상태

Snapshot: **2026-09-03 16:42 KST / 07:42 UTC**

### 서버

- SSH alias: `sejinA6000`
- 접속 정보는 로컬 `~/.ssh/config`의 `sejinA6000` 항목을 사용한다. 공개 저장소에는 실제 IP와 key 경로를 기록하지 않는다.
- Hostname: `a6000-4-003`
- Project: `/home/sejin/AI_MD_NSMC`
- Shared assets: `/home/sejin/confrover_mh_steering` → `/mnt/ssd0/sejin/confrover_mh_steering`
- Python: `/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python`
- Python version: 3.10.21

### 프로세스

| GPU | 작업 | 설정 | 마지막 확인 상태 |
|---|---|---|---|
| 0 | Frozen support-gate | seed `20260930`, `K8×M1`, T24 | 완료, 현재 유휴 |
| 1 | 없음 | — | 유휴 |
| 2 | DuET support-gate | seed `20260930`, `K4×M2`, T24 | 완료, 현재 유휴 |
| 3 | Complete Nested support-gate | seed `20260930`, `K4×M2`, T24 | 완료, 현재 유휴 |

마지막 확인에서 네 GPU 모두 약 `4–5 MiB`, utilization `0%`였다. Support-gate child process는 모두 종료됐다. 다음 parent wrapper만 production 자동 진입을 막기 위해 여전히 `SIGSTOP` 상태다.

- GPU2 parent wrapper: `1663861`, state `T` (의도적으로 SIGSTOP)
- GPU3 parent wrapper: `1663984`, state `T` (의도적으로 SIGSTOP)

PID는 시간이 지나면 바뀌거나 사라질 수 있으므로 재개 Codex는 반드시 다시 조회해야 한다.

### 세 support-gate 결과

| Method | Output paths | Endpoint validity | Whole-path validity | Invalid frame 수 | Max Kabsch step (nm) | Wall time (min) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen `K8×M1` | 8 | 1.0 | **0.875** | 1 | 2.5421 | 95.69 |
| Complete Nested `K4×M2` | 4 | 1.0 | **0.0** | 4 | 1.8623 | 89.54 |
| DuET `K4×M2` | 4 | 1.0 | **0.0** | 4 | 2.4729 | 89.36 |

공통 결과:

- Joint endpoint success와 valid endpoint success: 모두 `0.0`
- Decoder NFE: 모두 `38,208`
- Peak GPU memory metric: 모두 `42,555,556,864 bytes ≈ 39.6 GiB`
- Nonfinite coordinate와 CA clash: invalid frame에서도 모두 `0`

`metrics.json`의 `path_validity`를 확인한 정확한 invalid 위치는 다음과 같다. Frame index는 저장된 generated-path 배열의 0-based index다.

- Frozen: path 1, frame index 1, maximum adjacent-CA distance `5.557086 Å`
- Complete Nested: path 0–3 모두 frame index 1, 각각 `5.546187 Å`
- DuET: path 0–3 모두 frame index 6, 각각 `5.503004 Å`

Hard cutoff는 tolerance를 포함해 `5.501 Å`다. DuET은 cutoff를 `0.002 Å` 넘은 매우 경계적인 실패지만, Complete Nested와 Frozen도 cutoff를 실제로 넘었다. 세 방법 모두 마지막 endpoint만 보면 valid였으나, trajectory 중간에 invalid frame이 있어 자동 whole-path gate는 실패했다. 이 support-gate는 과학적 비교 결과가 아니라 production 안전성 진단이다.

### 왜 parent wrapper를 정지했는가

Budget 8 gate가 A6000에서 최대 약 46.8GB까지 사용했다. Production의 inner `M=4`를 한 batch로 실행하면 48GB에서 OOM 위험이 높다. 따라서 다음을 수행했다.

- Scientific `K16×M4`, NFE, seed는 유지
- Decoder forward만 최대 2개 후보씩 micro-batch하도록 구현
- Config에 `decoder_microbatch_size: 2` 추가
- DuET test 48개 통과
- M2 support-gate는 old process로 완료
- Production 부모는 SIGSTOP 상태로 유지해 무검증 production 진입을 차단

### 현 상태에서 하면 안 되는 것

- `kill -CONT`로 parent wrapper를 바로 재개하지 말 것
- 세 방법의 whole-path validity 실패 처리 원칙을 결정하지 않고 production을 시작하지 말 것
- GPU2/3에 계산 child는 현재 없고 parent wrapper만 정지돼 있다. PID를 재확인하지 않고 signal을 보내지 말 것.
- A6000 project root를 로컬 dirty tree로 무차별 overwrite하지 말 것

---

## 7. H100 상태와 결과 위치

### Kubernetes

- Namespace: `dept-dh`
- Pod 1: `sejin-h100-1-work-001-zhr2l`
- Pod 2: `sejin-h100-1-work-004-96566`
- Project: `/workspace/sejin/AI_MD_NSMC`
- Shared Level-1 assets: `/workspace/sejin/confrover_mh_steering`

2026-09-03 확인 결과:

- 두 H100 모두 memory `0 MiB`, utilization `0%`
- DuET/7LP1 관련 실행 프로세스 없음
- 두 H100의 기존 queue 완료
- 7LP1 metric files 61개 확인

주요 결과 위치:

```text
/workspace/sejin/AI_MD_NSMC/outputs/duet_md/phase_b_confrover_preflight
/workspace/sejin/AI_MD_NSMC/outputs/duet_md/phase_c_6j56_core
/workspace/sejin/AI_MD_NSMC/outputs/duet_md/phase_d_km_allocation
/workspace/sejin/AI_MD_NSMC/outputs/duet_md/protocol_v2
/workspace/sejin/AI_MD_NSMC/outputs/duet_md/protocol_v3
/workspace/sejin/AI_MD_NSMC/outputs/duet_md/extra_proteins/7lp1_A/protocol_v2
```

현재 H100에 pending queue는 없다. ABL1 assets와 representation이 H100에 준비됐는지는 검증하지 않았으므로 A6000 명령을 그대로 H100에서 실행하면 안 된다.

---

## 8. 아직 실행되지 않은 기존 예정 실험

이 절은 새 계획 제안이 아니라, 이전에 실제로 존재했던 계획을 기록한다.

### 8.1 ABL1 production

- 상태: **미실행**
- 예정: Frozen/Complete Nested/DuET × 5 seeds = 15 cells
- 현재 blocker:
  - 세 support-gate whole-path validity가 각각 `0.875/0.0/0.0`
  - M4 microbatch 실제 GPU smoke 미완료
- A6000 두 장만 사용하면 예상 60–100시간
- H100까지 준비해 병렬화하면 단축 가능하지만 H100 ABL1 환경은 아직 검증되지 않음

### 8.2 SMARCA2 cryptic-pocket case study

- 상태: 구조 조사까지만 완료, production config 없음
- 확보: PDB `5DKC`
- 후보 영역: alphaZ/ZA loop/alphaA 주변 cryptic pocket opening
- 미확보: 정확한 open/closed reference trajectory와 source notebook
- Zenodo archive가 `notebooks.tar.gz` 약 2.1GB, `simulations.tar.gz` 약 26GB의 all-system 묶음
- 사용자가 허용한 “해당 단백질 자산만 받고 전체 benchmark는 받지 않음” 범위와 충돌할 수 있어 다운로드 중단

### 8.3 `6rrv_A`

- 원래 `7lp1_A`와 함께 추가 ATLAS 단백질로 계획됨
- 7LP1 이후 ABL1/SMARCA2 case study로 우선순위가 이동
- 데이터 준비와 실험 모두 미실행

### 8.4 48-frame half-lag ablation

- 24-frame 결과의 temporal resolution 의존성을 확인하기 위한 조건부 계획
- R1/R2에서 event window를 다시 설계하고 R3 held-out을 유지해야 함
- Primary protocol 고정 후에만 실행하도록 문서화됨
- 실제 queue에는 올라가지 않았음

### 8.5 Interpolation/ProAR 초기 roadmap

- Config와 script는 존재한다.
- 필요한 interpolator/ProAR checkpoint 또는 manifest가 없어 실행 가능한 상태가 아니었다.
- 현재 핵심 DuET superiority 흐름에서는 우선순위가 내려갔다.

### 8.6 명시적으로 중단한 탐색

- Best-of-budget 추가 반복
- Naive Dual 추가 반복
- 7LP1 추가 K/M search
- 7LP1 추가 checkpoint search

이유: 중심 주장과 직접 관련이 낮거나, 이미 결과를 본 뒤 계속 조정하면 post-hoc tuning 문제가 생긴다.

---

## 9. 로컬 프로젝트와 파일시스템

## 9.1 저장소 역할 분리

### 새 구현 저장소

```text
/Users/sejin/Desktop/AIBL/projects/AI_MD_NSMC
```

DuET 코드, 신규 config, 실험 스크립트, ABL1/SMARCA2 case-study metadata를 둔다. 모든 신규 구현은 이 저장소에 유지한다.

### 기존 자산 저장소

```text
/Users/sejin/Desktop/AIBL/projects/confrover_mh_steering
```

기존 ConfRover checkout, checkpoint, ATLAS trajectory, reference asset을 읽기 전용으로 재사용한다. Level-1 결과나 기존 코드를 수정하지 않는다.

## 9.2 AI_MD_NSMC 주요 구조

```text
AI_MD_NSMC/
├── benchmarks/                 # exact/ATLAS/interpolation benchmark 명세
├── configs/duet/               # 모든 DuET 실행 YAML
│   └── case_studies/           # ABL1 production config
├── data/duet/                  # task catalog, PCA, selection report, case metadata
│   ├── 6j56_A/
│   ├── 6j56_A_v3/
│   ├── 7lp1_A_v2/
│   └── case_studies/
│       ├── abl1_dfg_flip/
│       └── smarca2_cryptic/
├── docs/                       # formulation, protocol, evaluation 및 본 문서
├── outputs/duet_md/            # 로컬 출력; 원격 결과 전체가 복사돼 있지는 않음
├── scripts/duet/               # 실행·요약·서버 setup script
│   └── case_studies/           # ABL1 preprocess/launch/resume
├── src/confmh/adapters/        # ConfRover/ProAR adapter
├── src/confmh/duet/            # DuET core, metrics, programs, evaluator
└── tests/duet/                 # DuET unit/integration tests
```

## 9.3 핵심 파일

### 방법 구현

- `src/confmh/duet/outer_smc.py`: physical-history particle와 outer resampling
- `src/confmh/duet/inner_fkc.py`: one/multi-checkpoint Feynman–Kac inner sampler
- `src/confmh/duet/baselines.py`: Complete Nested 등 baseline
- `src/confmh/adapters/confrover_duet.py`: ConfRover reverse SDE checkpoint adapter와 decoder microbatch
- `src/confmh/duet/potentials.py`: temporal prefix potential
- `src/confmh/duet/programs.py`: terminal/windowed/ordered event semantics
- `src/confmh/duet/evaluation_metrics.py`: pre-resampling stable metric
- `src/confmh/duet/case_study_eval.py`: reward/hidden firewall 및 ABL1 hidden evaluator
- `src/confmh/duet/runner.py`: 실행, validity, Kabsch continuity, 결과 저장

### 평가 설계

- `docs/DUET_EVALUATION_V2.md`: 현재 evaluation policy와 decision rule
- `data/duet/7lp1_A_v2/duet_programs.yaml`: 7LP1 v2 event 정의
- `data/duet/7lp1_A_v2/selection_report.yaml`: R1/R2 task selection 근거
- `data/duet/case_studies/abl1_dfg_flip/benchmark_spec.yaml`: ABL1 reward/hidden split
- `data/duet/case_studies/abl1_dfg_flip/reference_mapping_report.md`: residue mapping
- `configs/duet/case_studies/abl1_dfg_flip_production.yaml`: ABL1 본 실험 설정

### 실행 및 요약

- `scripts/duet/run_protocol_v2.py`: method/seed/K/M/checkpoint/horizon override 실행기
- `scripts/duet/summarize_stable_evaluation.py`: final resampling 전 metric 재계산
- `scripts/duet/run_7lp1_multicheckpoint_final.sh`: 완료된 H100 final queue
- `scripts/duet/case_studies/resume_abl1_a6000.sh`: support-gate 후 production wrapper
- `scripts/duet/case_studies/launch_phase_e_a6000.sh`: 5-seed production shard 실행
- `scripts/duet/case_studies/preprocess_abl1.py`: ABL1 reference preprocessing

## 9.4 Git 상태

- Branch: `main`
- Local HEAD: `565d0e56a86c5b22e7d69d47e73d1fc81707c3e9`
- A6000 HEAD: 같은 hash
- Worktree: **매우 dirty**
- 다수의 tracked 수정과 untracked config/script/data가 존재
- 최신 실험 코드는 commit에 포함돼 있지 않다.

따라서 새 Codex는 다음을 하면 안 된다.

- `git reset --hard`
- `git checkout -- .`
- clean checkout을 원격 project 위에 덮어쓰기
- untracked 파일 삭제

재현성을 위해 현재 상태를 검토한 뒤 별도 snapshot commit 또는 archive를 만드는 것이 필요하지만, 기존 변경을 잃지 않도록 먼저 diff와 asset 제외 범위를 확인해야 한다.

## 9.5 결과 디렉터리 사용 주의

다음 결과는 존재하더라도 최종 evidence 집계에서 제외한다.

- `phase_b_confrover_preflight_pre_seedfix_20260831`: seed 수정 전 결과
- `protocol_v2/pilot_*`, `protocol_v3/pilot_*`: task/horizon/threshold 개발 중 소규모 pilot
- `extra_proteins/7lp1_A/protocol_v2/smoke`: multi-checkpoint 구현 smoke
- 7LP1 초기 task 결과: event pair/window가 v2에서 교체됐으므로 탐색 기록으로만 유지
- ABL1 `support_gate_budget8`: production 운영 검증이며 biological comparison 결과가 아님

최종 분석 스크립트를 output root 전체에 무차별 적용하지 말고, 본 문서에 적은 확정 디렉터리만 명시적으로 입력해야 한다.

---

## 10. 다른 Codex CLI를 위한 재개 절차

## 10.1 첫 단계: 상태만 확인

로컬:

```bash
cd /Users/sejin/Desktop/AIBL/projects/AI_MD_NSMC
git status --short
git diff --check
```

A6000:

```bash
ssh sejinA6000
date
nvidia-smi
pgrep -af 'resume_abl1_a6000|run_protocol_v2'
find /home/sejin/AI_MD_NSMC/outputs/duet_md/case_studies/abl1_dfg_flip \
  -name metrics.json -print | sort
```

H100:

```bash
kubectl get pods -n dept-dh -o wide
kubectl exec -n dept-dh sejin-h100-1-work-001-zhr2l -- nvidia-smi
kubectl exec -n dept-dh sejin-h100-1-work-004-96566 -- nvidia-smi
```

## 10.2 완료된 A6000 support-gate 재확인

기대 경로:

```text
/home/sejin/AI_MD_NSMC/outputs/duet_md/case_studies/abl1_dfg_flip/
  support_gate_budget8/endpoint/frozen/seed_20260930/metrics.json
  support_gate_budget8/endpoint/complete_nested/seed_20260930/metrics.json
  support_gate_budget8/endpoint/duet/seed_20260930/metrics.json
```

세 파일은 모두 생성됐다. 다시 확인할 때 다음을 비교한다.

- `decoder_nfe == 38208`
- `structural_validity_rate`
- `path_structural_validity_rate`
- `valid_endpoint_success_rate`
- `maximum_kabsch_aligned_frame_step_nm`
- `peak_gpu_memory_bytes`
- `wall_clock_s`

현재 결과는 Frozen `0.875`, Complete Nested `0.0`, DuET `0.0`이다. 세 run 모두 endpoint validity는 `1.0`이고 invalidity는 중간 frame의 adjacent-CA hard cutoff 초과에서 발생했다.

## 10.3 Invalid path 진단

각 run의 다음 파일을 읽는다.

```text
metrics.json
records.json
trajectories_atom37_a.npz
resolved_config.yaml
```

Per-path/per-frame validity는 별도 파일이 아니라 `metrics.json`의 `path_validity` 배열에 들어 있다.
Support-gate 실행 로그는 run directory가 아니라 다음 launcher log에 있다.

```text
outputs/duet_md/case_studies/abl1_dfg_flip/_launcher/resume/gpu2_duet_then_shard0.log
outputs/duet_md/case_studies/abl1_dfg_flip/_launcher/resume/gpu3_complete_then_shard1.log
```

확인할 내용:

- 어느 physical frame에서 `ca_adjacent_max_a > 5.501 Å`인지
- nonfinite 또는 CA clash가 있는지
- 시작 PDB numbering이 1–287인지
- invalidity가 특정 method/seed에만 있는지
- endpoint는 valid인데 중간 frame만 invalid한지

Generated outcome을 보고 hard threshold를 임의로 넓히면 안 된다. 수치 허용오차는 이미 `0.001 Å`만 허용한다.

## 10.4 M4 microbatch 실GPU 확인

GPU가 유휴인 상태에서 GPU 하나만 사용해 짧게 확인한다. `run_protocol_v2.py`는 `--horizon` override를 지원한다.

예시:

```bash
cd /home/sejin/AI_MD_NSMC
export DUET_PROJECT_ROOT=/home/sejin/AI_MD_NSMC
export DUET_ASSET_ROOT=/home/sejin/confrover_mh_steering
export PYTHONPATH=/home/sejin/AI_MD_NSMC/src
export XDG_CACHE_HOME=/mnt/ssd0/sejin/cache
export CUDA_VISIBLE_DEVICES=2

/mnt/ssd0/sejin/conda_envs/confrover-mh/bin/python \
  scripts/duet/run_protocol_v2.py \
  --config configs/duet/case_studies/abl1_dfg_flip_production.yaml \
  --methods duet \
  --seed 20260998 \
  --outer-k 1 \
  --inner-m 4 \
  --horizon 1 \
  --output-directory outputs/duet_md/case_studies/_m4_memory_gate \
  --resume
```

확인 조건:

- OOM 없음
- decoder NFE 예상값과 일치
- `decoder_microbatch_size: 2`가 resolved config에 기록
- structural validity와 peak memory 기록

## 10.5 Production 재개 조건

다음이 모두 성립하기 전에는 production을 시작하지 않는다.

1. Frozen/Complete Nested/DuET 세 `metrics.json` 생성 완료
2. M4 microbatch 실GPU 확인 통과
3. Invalid path의 원인과 처리 원칙 결정
4. ABL1 endpoint/hidden threshold를 변경하지 않음
5. Output directory와 seeds 중복 여부 확인

Parent wrapper는 SIGSTOP 상태일 수 있다. PID를 새로 확인한 뒤에만 재개한다. PID가 이미 종료됐으면 script를 새로 실행한다.

Production script:

```bash
bash scripts/duet/case_studies/launch_phase_e_a6000.sh 2 0 abl1
bash scripts/duet/case_studies/launch_phase_e_a6000.sh 3 1 abl1
```

실제로는 `nohup`과 별도 log를 사용하고, 두 shard가 같은 seed-method cell을 중복 실행하지 않는지 먼저 dry inspection해야 한다.

## 10.6 결과 요약

Stable metric 생성:

```bash
python scripts/duet/summarize_stable_evaluation.py \
  outputs/duet_md/case_studies/abl1_dfg_flip \
  --output outputs/duet_md/case_studies/abl1_dfg_flip/stable_summary.json
```

최종 비교에서 반드시 보고할 것:

- Seed별 raw value
- Pre-resampling unique success rate
- Pre-resampling success weight mass
- Post-resampling success yield
- Valid endpoint success
- Whole-path validity
- Genealogy와 surviving ancestors
- Hidden DFG-inter recovery/order/route family
- Decoder NFE와 wall clock
- Paired seed difference

---

## 11. 예상 실행시간

A6000 ABL1 budget-8 Frozen gate 실측:

- `38,208 NFE`
- `95.69분`

Production cell은 `305,664 NFE`로 정확히 약 8배다. 단순 선형 추정은 cell당 약 12.8시간이지만 method와 microbatch 효율에 따라 약 `8–13시간/cell` 범위를 예상한다.

15 cells를 A6000 두 장에 8/7개 순차 분할하면 약 `60–100시간`, 즉 약 `2.5–4일` 범위다.

H100의 7LP1 `305,664 NFE` cell은 약 37–41분이었지만 ABL1은 287 residues로 7LP1 40 residues보다 훨씬 크므로 이 속도를 ABL1에 직접 적용하면 안 된다.

---

## 12. 재개 Codex의 권장 판단 순서

1. 현재 A6000 GPU와 stopped parent PID를 다시 확인한다.
2. 세 support-gate의 `metrics.json`을 보존하고 validity/memory/NFE를 재확인한다.
3. Whole-path validity 실패를 production blocker로 유지할지, 별도 QC metric으로 보고 진행할지 사전에 결정한다. Generated outcome을 보고 cutoff를 바꾸면 안 된다.
4. M4 microbatch one-frame gate를 수행한다.
5. 현재 dirty worktree의 정확한 diff를 보존한다.
6. Production 시작 여부를 결정한다.
7. Production 결과를 본 뒤 ABL1 threshold나 hidden definition을 바꾸지 않는다.
8. ABL1 완료 전에는 SMARCA2/6RRV로 범위를 불필요하게 넓히지 않는다.

---

## 13. 최종 한 문장 상태

> DuET-MD는 구현 타당성과 7LP1에서의 유망한 task-success 우세를 확보했지만 일반적 superiority와 mechanism fidelity는 아직 입증되지 않았으며, ABL1 DFG-flip support-gate 세 개는 완료됐으나 whole-path validity 실패와 M4 memory 검증이 남아 production은 아직 시작하지 않은 상태다.
