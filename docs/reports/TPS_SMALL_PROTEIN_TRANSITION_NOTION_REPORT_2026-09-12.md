# TPS small-protein transition recovery 실험 정리

작성일: 2026-09-12  
실험 상태: **생산 실험 24/24 완료, 후속 geometry·energy 감사 완료**  
대상: Chignolin, Trp-cage, BBA의 unfolded → folded 전이  
비교 방법: Frozen ConfRover, DuET-MD

> **한 문장 결론**  
> DuET은 세 단백질 모두에서 folded target 방향의 RMSD를 낮추는 신호를 보였고, Trp-cage와 BBA 일부 조건에서는 coarse TICA target hit의 가중 확률도 높였다. 그러나 생성 구조의 peptide geometry와 원자 충돌을 사후 조사한 결과, 현재 경로 중 **원자 수준에서 물리적으로 검증된 전이 경로는 0개**다. 따라서 이 실험은 “DuET이 reward 방향으로 ConfRover 출력을 선택할 수 있다”는 탐색적 증거이지, “물리적으로 타당한 folding transition을 복원했다”는 증거가 아니다.

---

## 1. 이 실험이 무엇인지

이 실험의 질문은 다음과 같다.

> Unfolded 단백질 구조에서 시작해 ConfRover가 여러 단계의 구조를 순차적으로 생성할 때, diffusion 생성 과정 안에서 후보를 선택하는 DuET이 아무 개입 없이 생성하는 Frozen ConfRover보다 folded target에 도달하는 경로를 더 잘 찾는가?

여기서 `TPS`라는 명칭은 공식 `tps-dps` 저장소가 제공하는 small-protein 시작 구조, 목표 구조, TICA 모델 및 평가 자산을 사용했다는 뜻이다. **TPS-DPS 논문의 sampling algorithm이나 published score를 재현한 실험은 아니다.** 실제 trajectory 생성기는 frozen ConfRover이고, 그 위에서 Frozen과 DuET을 비교했다.

이 실험은 다음 세 가지를 구분해서 본다.

1. **목표 접근:** 생성 구조가 folded target과 RMSD상 가까워지는가?
2. **저차원 basin 도달:** TPS-DPS의 공식 TICA 좌표에서 folded basin에 들어가는가?
3. **물리적 구조 타당성:** peptide bond, 원자 충돌, force-field energy가 정상적인가?

초기 생산 평가에서는 1과 2를 중심으로 봤고, 이후 별도의 감사에서 3을 상세히 조사했다. 세 조건은 서로 동치가 아니다.

---

## 2. 데이터와 단백질

공식 데이터 출처:

- 저장소: `https://github.com/kiyoung98/tps-dps.git`
- 고정 commit: `61fd65ad2e2f110d65c176a8c8f5c2fe8bdab034`
- 사용 파일: 각 단백질의 `unfolded.pdb`, `folded.pdb`, `tica_model.pkl`, `pmf.npy`, `xs.npy`, `ys.npy`
- 참고 구현: 공식 metric, TICA, dynamics, force-field 코드

| 단백질 | PDB에서 확인한 서열 | 길이 | 준비된 start→target backbone RMSD `d0` | 공식 TICA start→target 거리 |
|---|---|---:|---:|---:|
| Chignolin | `GYDPETGTWG` | 10 aa | 6.085456 Å | 4.797010 |
| Trp-cage | `DAYAQWLKDGGPSSGRPPPS` | 20 aa | 7.199678 Å | 11.113707 |
| BBA | `EQYTAKYKGRTFRNEKELRDFIEKFKGR` | 28 aa | 8.069854 Å | 2.959433 |

각 단백질에는 두 고정 구조가 있다.

- **Start:** 공식 `unfolded.pdb`. ConfRover에 실제 입력되는 초기 구조다.
- **Target:** 공식 `folded.pdb`. reward 및 평가에만 사용하며 ConfRover의 history 입력에는 넣지 않는다.

Start와 target은 공식 TPS-DPS `BaseDynamics` 설정을 재현해 각각 한 번 energy minimization한 뒤 고정했다. 생성된 frame에는 sampling 전후로 minimization이나 구조 repair를 적용하지 않았다. 즉 생산 결과는 raw ConfRover 생성 구조다.

모든 target은 공식 TICA에서 자기 자신과의 거리가 0이었고, 모든 start는 고정 target basin cutoff 0.75 밖에 있었다. 따라서 시작부터 성공 조건을 만족하는 trivial case는 없었다.

`path.gro`와 Trp-cage H5 파일도 보존했지만 unbiased reference trajectory라는 provenance가 확인되지 않아, reference-path coverage는 의도적으로 `NA`로 두었다.

---

## 3. trajectory가 생성되는 방식

한 run은 start 구조 `x0`에서 시작하여 32개의 다음 구조를 autoregressive하게 생성한다.

```text
x0 → x1 → x2 → ... → x32
```

- 총 physical transition: `T=32`
- start를 포함한 저장 구조: 33개
- 각 `x_t`는 이전 history 전체 `x_0:t-1`를 조건으로 생성한다.
- 중간에 best path로 재시작하거나 history를 초기화하지 않았다.
- ConfRover model parameter는 학습하거나 업데이트하지 않았다.

### Lag의 의미

두 가지 ConfRover stride를 사용했다.

| 설정 | 한 generated transition의 nominal lag | T=32 nominal horizon |
|---|---:|---:|
| stride 16 | 160 ps = 0.16 ns | 5.12 ns |
| stride 128 | 1.28 ns | 40.96 ns |

이 값은 ConfRover conditioning에 사용되는 nominal MD lag다. 생성 경로가 실제 kinetics를 정확히 보존한다는 의미는 아니며, 5.12 ns와 40.96 ns 결과를 실제 folding time으로 해석하면 안 된다. 두 lag는 모델이 서로 다른 시간 간격의 전이를 얼마나 생성하는지 보는 조건이다.

---

## 4. 비교 방법

### 4.1 Frozen ConfRover

- `K=16`, `M=1`
- 16개 trajectory를 서로 독립적으로 생성한다.
- diffusion 중 후보 선택 없음
- physical-time outer resampling 없음
- 최종 weight는 균등하다.

Frozen은 “아무 steering 없이 같은 frozen model을 실행했을 때”의 기준선이다.

### 4.2 DuET-MD

- outer trajectory population `K=4`
- 각 parent에서 동시에 유지하는 diffusion 후보 `M=4`
- 한 physical transition마다 총 `K×M=16`개 후보 계산
- reverse diffusion 완료 비율 75%와 90%에서 총 두 번 inner resampling
- 200-step convention에서 실제 checkpoint는 reverse update step 150과 180
- checkpoint 이후 복제된 후보는 서로 독립적인 SDE noise로 나머지 diffusion을 진행
- physical-time outer population은 기존 systematic resampling을 사용하며, outer ESS가 `0.5K` 이하일 때 resampling
- full history와 lineage를 유지하며 중간 restart나 weight reset 없음

DuET의 inner 과정은 “다음 frame 하나를 완성하기 전에” 중간 predicted-clean 구조를 평가해 유망한 후보에 남은 계산을 재배분한다.

두 checkpoint의 Feynman–Kac incremental weight는 개념적으로 다음처럼 연결된다.

```text
75% checkpoint:  phi75
90% checkpoint:  phi90 / phi75(해당 ancestor)
final:           psi(final) / phi90(해당 ancestor)
```

각 단계의 평균 incremental weight를 곱해 parent history의 normalizer 추정치 `Zhat_t`를 만들고, outer update에는 `Zhat_t / psi(parent)`를 사용한다. 따라서 inner에서 선택된 최종 frame의 reward를 outer에서 단순히 한 번 더 곱하는 double counting을 피한다.

실행 로그에서 다음을 확인했다.

- 75%, 90% checkpoint가 모두 실제 호출됨
- checkpoint step: 150, 180/200
- ancestor indexing 검사 통과
- 최대 telescoping log 오차: `2.22×10^-16`
- reward floor/clipping 호출: 0회

### 4.3 계산량 matching

Frozen은 `16×1`, DuET은 `4×4`이므로 한 physical step에서 평가하는 decoder population은 모두 16개다.

- 실제 candidate당 decoder NFE: 199
- run당 decoder NFE: `32 × 16 × 199 = 101,888`
- 24개 run 총 decoder NFE: `2,445,312`

75%와 90% checkpoint는 진행 중 decoder가 이미 제공하는 predicted-clean 구조를 사용하므로 별도의 완전한 decoder rollout을 추가하지 않는다. 따라서 비교는 nominal decoder NFE 기준으로 matched되어 있다.

다만 반환되는 최종 population은 Frozen 16개, DuET 4개다. 따라서 단순 hit 개수만 직접 비교하면 안 되며, DuET은 최종 outer weight를 포함한 weighted metric을 함께 봐야 한다.

---

## 5. Reward

Reward에는 folded target과의 **전체 단백질 backbone RMSD 하나만** 사용했다.

```text
d(x)  = target에 Kabsch 정렬한 뒤 공통 N/CA/C 원자의 RMSD [Å]
d0    = prepared unfolded start에 같은 함수를 적용한 값
log psi(x) = -16 × (d(x)/d0)^2
```

해석은 간단하다.

- `d/d0 = 1`: 시작 구조와 같은 수준으로 target에서 멀다.
- `d/d0 < 1`: 시작점보다 target에 가까워졌다.
- `d/d0 → 0`: folded target에 접근했다.
- 계수 16은 거리가 작은 후보에 더 큰 상대 weight를 주는 고정 steering strength다.

다음 항목은 reward에 들어가지 않았다.

- TICA 또는 THP
- PMF
- energy
- native contact
- peptide geometry
- 별도의 ordered-event 또는 pocket reward

Reward는 log-space/logsumexp로 계산했고, `max(-30, ...)` 같은 floor나 clipping은 사용하지 않았다. Checkpoint predicted-clean과 final frame은 같은 atom mask, Kabsch alignment 및 Å 단위를 사용했다.

이 구분이 중요하다. DuET이 reward를 따라 RMSD를 낮출 수는 있지만, TICA basin이나 물리적으로 올바른 geometry는 reward가 직접 보장하지 않는다.

---

## 6. 고정 실험 matrix

| 축 | 값 |
|---|---|
| Proteins | Chignolin, Trp-cage, BBA |
| Methods | Frozen, DuET |
| Lags | stride 16, stride 128 |
| Seeds | 211, 223 |
| Horizon | T=32 |
| 총 run | `3 × 2 × 2 × 2 = 24` |
| Model | frozen `confrover_base_20m_v1_0` |
| Reverse sampler | SDE, 200-step schedule |
| Context | full history |
| Analysis snapshots | T=8, 16, 32; 평가용이며 추가 resampling 없음 |

OOM 대응은 microbatch 1, KV offloading, Pairformer chunk 32로 제한했다. K, M, T, lag 또는 history를 결과에 따라 줄이지 않았다. 실행 중 사용자의 요청으로 남은 DuET 셀을 Frozen보다 먼저 처리하도록 GPU scheduling 순서만 바뀌었으며, 설정·seed·reward·평가 규칙은 변하지 않았다.

---

## 7. 평가 지표

### 7.1 Final backbone RMSD (`wBB`)

각 run의 최종 pre-resampling population에서 folded target까지의 N/CA/C Kabsch RMSD를 계산하고 최종 outer weight로 평균했다. 낮을수록 target에 가깝다.

### 7.2 Final heavy-atom RMSD (`wHeavy`)

Target과 generated frame에 공통으로 존재하는 heavy atom을 Kabsch 정렬해 RMSD를 계산했다. ConfRover가 terminal OXT를 출력하지 않으므로 본 결과는 공식 full-heavy metric이 아니라 명시적으로 **common-heavy RMSD**다.

### 7.3 공식 TICA target hit와 THP

공식 folded-PDB topology에서 backbone torsion feature를 `cossin=True`로 계산하고, 배포된 `tica_model.pkl`을 다시 fitting하지 않고 그대로 적용했다.

```text
target hit = 첫 두 TICA 좌표에서 folded target까지 Euclidean distance < 0.75
```

- `wTHP`: 최종 target hit에 놓인 weight의 합
- `wValidTHP`: 최종 target hit이면서 trajectory 전체가 기존 validity를 통과한 path에 놓인 weight의 합
- `valid final hits`: 실제 반환 population에서 위 두 조건을 만족한 path 수
- `valid anytime hits`: 최종 frame이 아니더라도 x1–x32 중 한 번 이상 target basin에 들어간 whole-path-valid trajectory 수

Primary metric은 `wValidTHP`다.

```text
wValidTHP = Σ_i w_i × I(final TICA hit_i) × I(whole path valid_i)
```

Invalid path는 분모에서 제거하거나 나머지 weight를 재정규화하지 않았다. 따라서 invalid target hit은 `wTHP`에는 잡힐 수 있지만 `wValidTHP`에는 기여하지 않는다.

### 7.4 생산 당시 whole-path validity

각 frame에서 다음을 검사했다.

- NaN/Inf 좌표가 없어야 함
- sequence상 이웃하지 않은 Cα pair가 1.0 Å 미만으로 충돌하지 않아야 함
- 모든 인접 residue Cα 거리가 `5.5 Å + 0.001 Å` 이하여야 함

한 frame이라도 실패하면 whole path invalid다. 인접 Cα 4.5 Å 이상은 별도 quality warning으로 기록했지만 hard fail cutoff는 5.5 Å였다.

이 validity는 생산 실험 당시 고정 규칙이다. 이후 감사에서 peptide C–N과 heavy-atom geometry를 충분히 잡지 못하는 것으로 밝혀졌으므로, 최종 물리적 타당성 판정으로는 사용할 수 없다.

### 7.5 Path diversity

서로 다른 valid final target-reaching path가 최소 2개 있을 때, first-two-TICA trajectory 사이의 normalized DTW distance를 계산했다. 완전히 같은 ancestry와 좌표의 복제품은 제거했지만, 공통 조상 이후 실제로 분기한 path는 유지했다. 성공 path가 2개 미만이면 `NA`다.

### 7.6 Sampled-frame ETS

Valid final target path에 한해 start를 포함한 33개 저장 frame의 potential energy를 평가하고, path별 maximum을 계산했다.

- Force field: `protein.ff14SBonlysc + implicit/gbn2`
- NoCutoff
- bias/restraint energy 제외
- generated heavy atom은 고정
- template hydrogen만 완화
- 생성 mask에 없는 terminal OXT는 정렬된 target template 위치에 고정

이는 sparse saved frame에서 얻은 **조건부 energy diagnostic**일 뿐 실제 transition-state energy가 아니다. 성공 경로가 없으면 0이 아니라 `NA`다.

---

## 8. 24개 production run raw 결과

표 읽는 법:

- `Lag`: generated transition 하나의 nominal 간격
- `wBB`, `best-valid BB`, `wHeavy`: Å, 낮을수록 좋음
- `wTHP`: validity를 무시한 최종 TICA-hit weight
- `wValid`: whole-path validity까지 통과한 최종 TICA-hit weight
- `hit`: valid final hit 수 / 반환 population
- `any`: valid anytime-hit path 수
- `valid`: whole-path-valid population 비율
- `DTW`: 서로 다른 valid final hit path의 TICA diversity 평균; 2개 미만이면 `—`
- `ETS`: qualifying path들의 sampled-frame maximum potential 평균, kJ/mol
- `min`: ConfRover sampling wall time; 후속 ETS 시간 제외

### 8.1 Chignolin

| Lag | Method | Seed | wBB | best-valid BB | wHeavy | wTHP | wValid | hit | any | valid | DTW | ETS | min |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.16 ns | Frozen | 211 | 3.756 | 3.425 | 5.495 | 0.063 | **0.000** | 0/16 | 6 | 0.563 | — | — | 44.2 |
| 0.16 ns | Frozen | 223 | 3.566 | 2.718 | 5.262 | 0.063 | **0.000** | 0/16 | 6 | 0.438 | — | — | 43.8 |
| 0.16 ns | DuET | 211 | 2.106 | — | 3.281 | 0.401 | **0.000** | 0/4 | 0 | 0.000 | — | — | 38.6 |
| 0.16 ns | DuET | 223 | 2.389 | 2.260 | 4.672 | 0.000 | **0.000** | 0/4 | 3 | 0.750 | — | — | 42.8 |
| 1.28 ns | Frozen | 211 | 4.007 | 2.883 | 5.850 | 0.000 | **0.000** | 0/16 | 7 | 0.625 | — | — | 39.9 |
| 1.28 ns | Frozen | 223 | 3.858 | 2.256 | 5.286 | 0.000 | **0.000** | 0/16 | 13 | 0.875 | — | — | 44.0 |
| 1.28 ns | DuET | 211 | 2.798 | — | 4.504 | 0.000 | **0.000** | 0/4 | 0 | 0.000 | — | — | 41.5 |
| 1.28 ns | DuET | 223 | 1.762 | — | 2.888 | 0.787 | **0.000** | 0/4 | 0 | 0.000 | — | — | 37.8 |

핵심 관찰:

- 네 대응 조건 모두에서 DuET의 weighted final backbone RMSD가 Frozen보다 낮았다.
- 평균 wBB는 Frozen `3.797 Å`, DuET `2.263 Å`였다.
- 하지만 valid final TICA hit은 모든 조건에서 0이었다.
- 특히 DuET stride128/seed223은 `wTHP=0.787`이지만 반환 path 4개가 모두 whole-path invalid여서 `wValidTHP=0`이다.
- 결론: target 방향 접근 신호는 있으나 유효한 folding endpoint 회수는 확인되지 않았다.

### 8.2 Trp-cage

| Lag | Method | Seed | wBB | best-valid BB | wHeavy | wTHP | wValid | hit | any | valid | DTW | ETS | min |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.16 ns | Frozen | 211 | 4.899 | 5.234 | 6.468 | 0.250 | **0.000** | 0/16 | 0 | 0.063 | — | — | 46.6 |
| 0.16 ns | Frozen | 223 | 5.108 | 5.363 | 6.516 | 0.188 | **0.063** | 1/16 | 2 | 0.250 | — | `1.04×10^15` | 41.7 |
| 0.16 ns | DuET | 211 | 4.329 | 3.855 | 6.285 | 0.000 | **0.000** | 0/4 | 0 | 1.000 | — | — | 40.0 |
| 0.16 ns | DuET | 223 | 4.062 | 3.419 | 5.795 | 0.000 | **0.000** | 0/4 | 0 | 1.000 | — | — | 40.3 |
| 1.28 ns | Frozen | 211 | 4.837 | 2.501 | 6.420 | 0.125 | **0.063** | 1/16 | 6 | 0.438 | — | `7.35×10^12` | 46.3 |
| 1.28 ns | Frozen | 223 | 4.959 | 2.949 | 6.460 | 0.313 | **0.125** | 2/16 | 3 | 0.188 | 0.703 | `1.34×10^13` | 40.9 |
| 1.28 ns | DuET | 211 | 2.719 | 2.354 | 4.294 | 0.757 | **0.333** | 1/4 | 2 | 0.500 | — | `1.48×10^11` | 40.5 |
| 1.28 ns | DuET | 223 | 3.317 | 2.581 | 4.832 | 0.120 | **0.120** | 1/4 | 4 | 1.000 | — | `3.46×10^16` | 40.2 |

핵심 관찰:

- 평균 wBB는 Frozen `4.951 Å`, DuET `3.607 Å`였다.
- 0.16 ns 조건에서는 DuET valid final hit이 없고 Frozen seed223에서만 1개가 있었다.
- 1.28 ns/seed211에서는 DuET wValidTHP `0.333`, Frozen `0.063`으로 DuET 신호가 컸다.
- 1.28 ns/seed223에서는 DuET `0.120`, Frozen `0.125`로 사실상 비슷하며 DuET이 수치상 약간 낮다.
- 1.28 ns DuET 두 seed 모두 Frozen보다 RMSD는 크게 낮았다.
- 따라서 Trp-cage 1.28 ns는 가장 일관된 DuET target-approach 신호지만, 두 seed만으로 우월성을 주장할 수는 없다.

### 8.3 BBA

| Lag | Method | Seed | wBB | best-valid BB | wHeavy | wTHP | wValid | hit | any | valid | DTW | ETS | min |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.16 ns | Frozen | 211 | 7.334 | 5.143 | 8.999 | 0.250 | **0.250** | 4/16 | 8 | 0.813 | 0.498 | `7.13×10^16` | 46.9 |
| 0.16 ns | Frozen | 223 | 6.671 | 5.097 | 8.316 | 0.375 | **0.250** | 4/16 | 7 | 0.500 | 0.420 | `8.96×10^15` | 45.2 |
| 0.16 ns | DuET | 211 | 3.240 | 3.791 | 5.513 | 0.000 | **0.000** | 0/4 | 0 | 0.250 | — | — | 43.4 |
| 0.16 ns | DuET | 223 | 3.262 | 2.971 | 5.200 | 0.599 | **0.599** | 1/4 | 4 | 1.000 | — | `1.90×10^16` | 43.5 |
| 1.28 ns | Frozen | 211 | 6.629 | 4.637 | 8.484 | 0.375 | **0.250** | 4/16 | 5 | 0.313 | 0.422 | `3.73×10^14`* | 45.9 |
| 1.28 ns | Frozen | 223 | 6.105 | 3.539 | 7.706 | 0.250 | **0.125** | 2/16 | 3 | 0.375 | 0.394 | `2.85×10^17`* | 46.8 |
| 1.28 ns | DuET | 211 | 3.236 | 2.410 | 5.267 | 0.656 | **0.656** | 2/4 | 4 | 1.000 | 0.114 | `1.20×10^14` | 44.2 |
| 1.28 ns | DuET | 223 | 3.597 | 3.263 | 6.035 | 0.000 | **0.000** | 0/4 | 0 | 0.750 | — | — | 43.6 |

`*` BBA stride128 Frozen의 두 셀은 qualifying path 중 각각 1개에서 기존 ETS 계산이 실패했다. 표의 값은 성공적으로 평가된 3개 및 1개 path의 평균이며 실패 path를 추정해 채우지 않았다.

핵심 관찰:

- 평균 wBB는 Frozen `6.685 Å`, DuET `3.334 Å`로 DuET이 훨씬 낮았다.
- Frozen은 네 셀 모두 valid final hit을 생성했다. wValidTHP는 `0.125–0.250`이었다.
- DuET은 두 셀에서는 0이지만 나머지 두 셀에서는 `0.599`, `0.656`의 높은 weight를 성공 path에 집중했다.
- 즉 DuET은 성공하면 작은 수의 경로에 높은 확률을 집중하지만 seed/lag에 따른 변동이 크다.
- BBA의 높은 coarse hit 수는 이후 geometry 감사에서 물리적 folding 성공으로 인정되지 않았다.

---

## 9. 집계 결과

### 9.1 전체 method 집계

| Method | 셀 수 | 평균 wValidTHP | valid final hit이 1개 이상인 셀 | 평균 sampling 시간 |
|---|---:|---:|---:|---:|
| Frozen | 12 | 0.0938 | 7/12 | 44.4분 |
| DuET | 12 | 0.1423 | 4/12 | 41.4분 |

### 9.2 단백질별 집계

| Protein | Method | 평균 wBB | 평균 wValidTHP | hit cell/4 | 평균 기존 whole-path validity |
|---|---|---:|---:|---:|---:|
| Chignolin | Frozen | 3.797 | 0.000 | 0/4 | 0.625 |
| Chignolin | DuET | 2.263 | 0.000 | 0/4 | 0.188 |
| Trp-cage | Frozen | 4.951 | 0.063 | 3/4 | 0.234 |
| Trp-cage | DuET | 3.607 | 0.113 | 2/4 | 0.875 |
| BBA | Frozen | 6.685 | 0.219 | 4/4 | 0.500 |
| BBA | DuET | 3.334 | 0.314 | 2/4 | 0.750 |

이 집계는 significance test가 아니다. 단백질과 lag가 섞여 있고, seed가 2개뿐이며, DuET particle은 resampling 때문에 독립 표본이 아니다. Frozen과 DuET의 반환 population 크기도 각각 16과 4로 다르다. 평균은 완료된 matrix를 압축해 보여 주는 기술통계일 뿐이다.

---

## 10. 생산 결과만 보았을 때의 해석

### 10.1 DuET steering은 작동했는가?

**Reward 방향으로는 작동했다.** 세 단백질 모두에서 DuET의 평균 final backbone RMSD가 Frozen보다 낮았고, 대응하는 12개 protein×lag×seed 조건 모두에서도 DuET wBB가 Frozen보다 낮았다. 이는 diffusion-time checkpoint selection과 outer weighting이 target RMSD가 작은 후보를 실제로 선호했다는 증거다.

### 10.2 Folded target에 도달했는가?

Coarse TICA 기준으로는 일부 도달했다.

- Chignolin: valid final hit 없음
- Trp-cage: DuET 2개 셀, Frozen 3개 셀에서 valid final hit
- BBA: DuET 2개 셀, Frozen 4개 셀에서 valid final hit

하지만 TICA는 backbone torsion에서 얻은 첫 두 저차원 좌표만 사용한다. TICA basin hit은 전체 원자 구조가 정상이라는 뜻이 아니다.

### 10.3 DuET이 Frozen보다 우월한가?

이 pilot만으로는 결론낼 수 없다.

- DuET은 RMSD에서는 일관된 개선을 보였다.
- Weighted valid THP 전체 평균도 DuET이 높았다.
- 하지만 valid-hit cell 수는 Frozen이 7/12, DuET이 4/12로 더 많았다.
- BBA DuET은 두 조건에서 강한 성공 weight를 보였지만 두 조건에서는 완전히 0이었다.
- seed가 2개뿐이라 분산과 재현성을 추정하기 어렵다.
- Complete Nested나 Outer-only가 포함되지 않아 improvement가 diffusion 중간 개입 자체에서 왔는지 분리할 수 없다.

따라서 적절한 표현은 “일부 task에서 DuET의 reward-directed concentration 신호를 관찰했다”이다.

---

## 11. 후속 geometry·energy 감사

생산 실험의 coarse validity를 통과한 TICA hit 경로에서 ETS가 `10^11–10^17 kJ/mol` 수준으로 매우 높게 나타났다. 이에 기존 결과를 변경하지 않고 raw tensor, PDB conversion, TICA mapping, peptide geometry 및 force-field energy를 별도로 감사했다.

### 11.1 TICA hit frame 자체의 구조 범위

최종 frame뿐 아니라 전체 trajectory에서 발생한 모든 TICA hit frame을 다시 조사했다.

| Protein | 전체 actual TICA-hit frames | whole-valid path 위 hit frames | Backbone RMSD min / median / max | Heavy native-contact Q min / median / max |
|---|---:|---:|---:|---:|
| Chignolin | 229 | 79 | 0.93 / 2.53 / 6.74 Å | 0.000 / 0.800 / 1.000 |
| Trp-cage | 370 | 180 | 2.06 / 4.33 / 7.96 Å | 0.154 / 0.538 / 0.923 |
| BBA | 593 | 467 | 2.41 / 6.12 / 11.50 Å | 0.267 / 0.533 / 0.933 |

여기서 frame 수는 독립적인 성공 trajectory 수가 아니다. 모든 run·particle·timepoint에서 TICA cutoff를 통과한 frame appearance 수다.

일부 hit은 target과 실제로 가까웠지만, BBA의 경우 TICA hit이면서 backbone RMSD가 11.50 Å이고 native-contact Q가 0.267인 frame도 있었다. 따라서 first-two-TICA cutoff만으로 folded structure를 정의하기에는 정보가 부족하다.

### 11.2 파일 변환이나 TICA 구현 오류였는가?

주원인이 아니었다.

- 선택 frame raw tensor → PDB roundtrip 좌표 오차: 최대 `8.49×10^-4 Å`
- 7,920개 frame에서 저장 TICA와 raw tensor 재계산 TICA: 정확히 일치
- 0.001 Å PDB 양자화를 모사했을 때 Trp-cage와 BBA hit label 변화: 0개
- Chignolin 변화: cutoff 0.75 바로 근처의 non-final frame 1개뿐이며 production final THP에는 영향 없음
- 정상 start/target은 동일 energy pipeline에서 정상 범위를 보임

즉 production TICA 계산이나 PDB 저장이 coarse hit 신호를 인위적으로 만든 것은 아니다.

### 11.3 energy가 비정상적으로 큰 원인

기존 valid-final TICA-hit 경로는 Trp-cage 6개, BBA 17개였다. 이 23개 경로의 first generated frame과 final frame, 그리고 계산 가능한 maximum-energy frame을 같은 pipeline으로 조사했다.

정상 reference energy:

| Protein | Start | Folded target |
|---|---:|---:|
| Chignolin | 10.4 | -1,699.0 kJ/mol |
| Trp-cage | 487.0 | -2,609.5 kJ/mol |
| BBA | -2,676.1 | -7,263.7 kJ/mol |

기존 hit path frame:

| 위치 | 평가 성공 | Energy min / median / max | 관찰 |
|---|---:|---:|---|
| First generated | 23 | `1.06×10^5 / 1.08×10^7 / 7.87×10^12` kJ/mol | nonbonded 지배 16/23 |
| Final | 22 | `3.32×10^4 / 1.10×10^9 / 3.52×10^12` kJ/mol | nonbonded 지배 20/22; 1 frame NaN |
| Maximum finite-energy frame | 21 | `2.19×10^10 / 1.01×10^14 / 2.85×10^17` kJ/mol | nonbonded 지배 21/21 |

원자 수준 원인:

- maximum-energy frame의 최상위 nonbonded pair는 20/21에서 generated-heavy/generated-heavy였다.
- 최상위 충돌 pair 거리는 `0.124–0.492 Å`, 중앙값 `0.235 Å`였다.
- first frame에서도 top pair 20/23이 generated-heavy/generated-heavy였다.
- 선택된 23개 hit path의 first와 final frame 모두 peptide C–N gross violation을 포함했다.
- 정상 peptide C–N은 대략 1.33–1.36 Å인 반면, first frame 범위는 0.383–2.479 Å, final은 0.356–2.473 Å였다.

즉 추가 hydrogen이나 terminal OXT 처리만의 artifact가 아니라, **ConfRover가 생성한 heavy atom 사이의 steric overlap과 inter-residue peptide geometry strain**이 주원인이었다.

### 11.4 custom DuET adapter만의 문제였는가?

그것만의 문제는 아니었다. 별도의 짧은 sampler 대조에서 공식 ConfRover forward, 공식 EulerSampler를 연결한 adapter, 현재 custom unsteered SDE를 비교했다. TPS에 직접 해당하는 Trp-cage에서도 세 arm 모두 generated frame 16/16에 peptide C–N violation이 있었다.

Trp-cage first/final frame energy 비교:

| Sampler arm | Energy min / median / max | 지배 항 |
|---|---:|---|
| Official ConfRover forward | `1.57×10^5 / 9.48×10^7 / 1.07×10^10` kJ/mol | nonbonded 6/8, bonded 2/8 |
| Official sampler through adapter | `5.06×10^4 / 4.45×10^9 / 2.36×10^12` | nonbonded 7/8, bonded 1/8 |
| Current custom unsteered SDE | `2.87×10^5 / 3.65×10^6 / 6.60×10^17` | nonbonded 5/8, bonded 3/8 |

공식 forward에서도 같은 종류의 geometry 문제가 확인됐으므로 custom adapter는 결함의 필요조건이 아니다. 다만 표본이 작고 custom arm에 가장 큰 extreme frame이 있었으므로, 세 sampler의 상대적 구조 품질 우열까지 주장하지 않는다.

---

## 12. 좌표 정렬과 PyMOL에서 보이는 큰 회전

Raw atom37 trajectory를 그대로 PDB로 내보내면 단백질 중심은 비슷해도 frame마다 전체 단백질이 크게 회전해 보일 수 있다. 이는 내부 구조 변화와 별개인 global rigid-body coordinate freedom이다.

공식 ConfRover trajectory writer는 최종 XTC를 저장할 때 다음을 적용한다.

1. 각 frame center_coordinates
2. 모든 frame을 첫 frame에 superpose

우리의 초기 raw PDB export는 공식 writer를 우회했기 때문에 이 표시용 정렬이 빠져 있었다. 따라서 PyMOL에서는 first-frame rigid alignment를 적용한 경로를 보는 것이 공식 ConfRover 출력 의도와 맞다.

중요한 구분:

- Rigid alignment는 frame 전체를 회전·이동할 뿐 내부 거리, 결합, RMSD after Kabsch, torsion-TICA 또는 energy를 바꾸지 않는다.
- 그러므로 앞서 본 큰 global rotation은 정렬로 제거할 수 있다.
- 정렬 후에도 남는 peptide distortion과 atom clash는 실제 내부 geometry 문제이며 정렬로 해결되지 않는다.

생산 reward의 Kabsch RMSD와 torsion 기반 TICA는 원래부터 global translation/rotation에 불변이므로, raw export에서 정렬이 빠졌다고 생산 수치가 바뀌지는 않는다.

---

## 13. 사후 restrained minimization 실험

Raw generated frame을 간단한 OpenMM minimization으로 고칠 수 있는지 진단했다. 이는 production sampling에 포함되지 않았고, 기존 성공 여부도 변경하지 않았다.

### 13.1 고정 절차

- Force field: `protein.ff14SBonlysc + implicit/gbn2`
- 모든 present atom은 이동 가능
- topology를 구성하기 위한 표준 hydrogen만 추가
- 모든 heavy atom을 **그 frame 자체의 generated 위치**에 `10 kcal mol^-1 Å^-2` harmonic restraint
- target/folded 구조는 restraint에 사용하지 않음
- OpenMM L-BFGS, tolerance `10 kJ mol^-1 nm^-1`, 최대 2,000 iterations
- 분석 대상: Trp-cage와 BBA에서 method별 raw-hit path 및 raw-non-hit path를 사전 고정하여 총 8개 path, 264 frame appearances

초기 감사에서 atom37의 CB와 O index가 뒤바뀐 mapping 오류가 한 번 있었으며 해당 결과는 무효로 보존했다. 아래 수치는 공식 atom37 순서 `N, CA, C, CB, O, ...`로 수정한 최종 감사 결과만 사용한다.

### 13.2 결과

| System | Unique frames | Median energy raw → relaxed | Gross peptide C–N defects | Heavy clashes <1 Å | Median / max displacement | force tolerance 통과 |
|---|---:|---:|---:|---:|---:|---:|
| Trp-cage | 129 | `7.52×10^10 → -1.75×10^3` kJ/mol | 802 → 1 | 223 → 5 | 2.78 / 8.79 Å | 0/129 |
| BBA | 129 | `4.50×10^11 → -5.73×10^3` kJ/mol | 1,309 → 14 | 375 → 0 | 3.45 / 11.61 Å | 0/129 |

Minimization은 enormous energy와 대부분의 clash를 줄였지만 작은 보정이 아니었다. 모든 frame이 2,000-iteration cap에서 force tolerance를 통과하지 못했고, 원자가 최대 8.79–11.61 Å 이동했다.

TICA 상태도 완전히 보존되지 않았다.

| System | Raw TICA-hit frames | Relaxed hit frames | 유지 | 소실 | 새로 발생 |
|---|---:|---:|---:|---:|---:|
| Trp-cage | 42 | 38 | 32 | 10 | 6 |
| BBA | 33 | 25 | 18 | 15 | 7 |

더 중요한 문제는 minimization 후 template-sign 검사상 chirality inversion이 Trp-cage 124/129, BBA 128/129 frame에서 나타났다는 것이다. 따라서 이 minimization은 현재 형태로는 production repair, proposal kernel 또는 성공 검증에 사용할 수 없다.

정상 folded-target control은 relaxation 전후 TICA state를 유지했다. 따라서 topology/atom37 변환 자체가 정상 target을 basin 밖으로 보내는 문제는 아니었다.

---

## 14. 최종 과학적 판정

### 확인된 것

1. **DuET의 cross-clock 계산은 의도대로 실행됐다.** 두 checkpoint, ancestor ratio, telescoping, `Zhat/psi(parent)` outer interface가 검증됐다.
2. **동일 decoder NFE에서 DuET은 target backbone RMSD를 일관되게 낮췄다.** 세 단백질의 모든 대응 조건에서 이 경향이 관찰됐다.
3. **Trp-cage 1.28 ns와 BBA 일부 조건에서 TICA target에 weight를 집중하는 신호가 있었다.**
4. **TICA 및 raw tensor→PDB 변환 오류는 주요 원인이 아니었다.**

### 확인되지 않은 것

1. 물리적으로 타당한 unfolded→folded transition recovery
2. 실제 folding kinetics 또는 transition time
3. DuET이 Frozen보다 일반적으로 우월하다는 통계적 결론
4. diffusion-time steering이 complete-frame selection보다 낫다는 결론
5. post-hoc minimization으로 raw path 성공을 검증하거나 복원할 수 있다는 결론

### 현재 사용해야 하는 표현

> Frozen ConfRover + DuET small-protein pilot에서 DuET은 고정 RMSD reward 방향으로 endpoint population을 이동시키는 신호를 보였다. Trp-cage와 BBA 일부 조건에서는 공식 first-two-TICA target basin에도 도달했다. 그러나 peptide connectivity와 all-atom geometry 감사 결과 해당 hit들은 atomistically valid transition으로 인정할 수 없었다.

피해야 하는 표현:

- “DuET이 folding pathway를 복원했다.”
- “TICA hit이므로 물리적으로 folded state다.”
- “40.96 ns 안에 folding을 가속했다.”
- “Minimization 후 energy가 낮아졌으므로 원래 경로도 valid하다.”

---

## 15. 다음 단계의 우선순위

1. Production validity에 peptide C–N, backbone bond/angle 및 heavy-atom clash 검사를 추가한다.
2. ConfRover raw output을 물리적 path로 사용할 수 있는 구조 보정 방법을 별도 검증한다. 보정은 target-independent이고 chirality/lineage/path를 보존해야 한다.
3. 보정된 구조를 다시 history에 넣을지, 평가에만 사용할지 구분해 formulation을 고정한다. 이는 단순 후처리가 아니라 proposal dynamics를 바꿀 수 있는 결정이다.
4. raw trajectory 시각화에는 공식 writer와 같은 first-frame rigid alignment를 적용한다.
5. atomistic validity가 확보된 뒤 fresh seeds로 Trp-cage stride128 후보 설정을 재검증한다.
6. 최종 방법론 비교에는 최소한 Frozen, Outer-only, Complete Nested, DuET의 matched-budget 비교가 필요하다.

현재 단계에서 reward, TICA cutoff 또는 성공 기준을 기존 결과에 맞춰 사후 변경해서는 안 된다. 기존 raw 결과는 그대로 보존하고 새로운 validity를 병렬 지표로 추가해야 한다.

---

## 16. 실행 시간과 계산 자원

- 24개 production run 모두 완료
- run당 ConfRover sampling: 약 37.8–46.9분
- Frozen 평균: 44.4분
- DuET 평균: 41.4분
- TICA/RMSD/파일 저장 평가는 대체로 수십 초 이내
- ETS는 valid final hit path가 많을 때 추가 병목
- BBA Frozen stride16/seed211: sampling 46.88분 + ETS 포함 후처리 23.98분 = 총 70.86분
- 세 H100에 단백질별 chain을 배정해 실행

Sampling wall time 차이를 알고리즘 속도 우월성으로 해석하면 안 된다. GPU scheduling, sequence length, conditional ETS 실행 수가 섞여 있으며 모든 run의 주 비교 기준은 wall time이 아니라 matched decoder NFE다.

---

## 17. 원본 산출물과 재현 경로

Remote production namespace:

```text
/workspace/sejin/AI_MD_NSMC_phase_b_recovery/outputs/small_protein_transition_pilot
```

주요 파일:

```text
summary_all.csv
summary_all.json
report.md
<protein>/stride{16|128}_t32/<method>/seed_<seed>/metrics.json
<protein>/stride{16|128}_t32/<method>/seed_<seed>/small_protein_metrics.json
<protein>/stride{16|128}_t32/<method>/seed_<seed>/small_protein_frame_diagnostics.json
<protein>/stride{16|128}_t32/<method>/seed_<seed>/tica_projection.npz
<protein>/stride{16|128}_t32/<method>/seed_<seed>/tica_pmf_overlay.png
```

Local source reports:

- `reports/SMALL_PROTEIN_TRANSITION_PILOT_PREFLIGHT_2026-09-10.md`
- `reports/SMALL_PROTEIN_TRANSITION_PILOT_FINAL_AGENT_REPORT_2026-09-11.md`
- `reports/BUNDLE_A_CROSS_CLOCK_AUDIT_2026-09-11/BUNDLE_A_FINAL_REPORT_2026-09-11.md`
- `reports/restrained_relaxation_audit_2026-09-11_atom37_corrected/RESTRAINED_RELAXATION_AUDIT_REPORT.md`

사람이 확인할 수 있는 raw/relaxed path와 PyMOL script:

```text
visual_examples/restrained_relaxation_paths_2026-09-12/
```

`view_*_aligned.pml`은 frame 전체의 global rotation/translation을 제거해 내부 구조 변화를 보기 위한 파일이고, raw/relaxed PDB 좌표 자체를 수정하지 않는다.

---

## 18. 최종 요약

이 TPS small-protein pilot은 계산적으로는 계획된 24-run matrix를 완전하게 수행했고, DuET의 두 diffusion checkpoint 및 cross-clock weighting도 검증했다. DuET은 folded target RMSD를 낮추는 데는 분명한 신호를 보였다. 하지만 coarse TICA hit과 기존 Cα validity만으로는 물리적 folding transition을 판정할 수 없었고, 실제 all-atom 감사에서는 severe peptide defect와 heavy-atom clash가 확인됐다.

따라서 현재 결과의 논문적 위치는 **method feasibility 및 failure-mode discovery**다. DuET의 steering mechanism은 작동했지만, ConfRover raw proposal을 물리적으로 타당한 trajectory로 연결하는 구조 품질 계층이 아직 해결되지 않았다.
