# TICA reward 진단 및 실험 — 결과 보고서 (agent 전달용)

작성일: 2026-09-15
구성: Phase R0 오프라인 진단 (195셀 재분석, GPU 0) + H4 TICA reward 실험 (신규 20셀)

> **한 문장.** RMSD reward는 입자를 충분히 구분하지만 그 구분이 성공 조건과 거의 직교했고
> (R1 순위상관 0.05–0.24, R3 선택 정확성 ≈ 0), reward를 TICA 좌표로 교체하자
> **THP가 0.009–0.036 → 0.56–0.67로 올랐다.** 단 이 비교에는 순환성과 교락이 있으며
> DuET과 Complete Nested는 여전히 구분되지 않는다 (THP Welch t = −0.63).

---

## 0. 이 문서를 읽는 agent를 위한 안내

- §3·§5에 **셀별 raw 표** 전체가 있다. 집계는 모두 거기서 재계산 가능하다.
- §6의 해석은 **관측 / 가설 / 미측정**을 분리했다. 인과 주장을 그대로 인용하지 말 것.
- **반드시 함께 읽을 주의사항이 §6.2에 있다** (순환성, 교락, 중간 집계 오류).
- 선행 문서:
  - `PVB_TPS_FINAL_REPORT_2026-09-14.md` — RMSD reward 360셀 기준선
  - `PVB_DUET_K_REDUCTION_2026-09-14.md` — K 축소 실험 420셀
  - `PVB_PHASE0_DIAGNOSTICS_2026-09-14.md` — 안쪽/바깥 필터 진단
    **단 그 문서의 "중복(redundancy)" 인과 설명은 미검증 가설이며 K 축소 실험이 그 예측을 지지하지 않았다.**

---

## 1. 배경 — 왜 reward를 의심했는가

선행 실험 두 건이 모두 "DuET의 이득 검출 실패"로 끝났다.

| 실험 | 규모 | 결과 |
| --- | ---: | --- |
| RMSD reward 기준선 | 360셀 | DuET ≡ CN (18개 검정 전부 \|t\| < 1.5) |
| K 축소 (예산 고정) | 420셀 | K=1까지 줄여도 DuET 이득 없음 (통합 g +0.256) |

동시에 별도로 관측된 것:

- **T=32에서** RMSD가 2–3 Å 내려가는 동안 TICA 거리는 제자리 → "reward 오정렬" 의심
- **T=96 BBL에서** 같은 RMSD reward로 120경로 중 79개(65.8%)가 TICA 타깃 진입
  → **"reward가 틀린 방향"에 대한 반례**

이 둘이 모순되지 않으려면 질문을 바꿔야 한다.

> ~~"RMSD reward가 TICA와 다른 방향인가?"~~
> **"RMSD reward가 유용한 *선택 신호*를 주는가?"**

SMC의 resampling은 reward의 절댓값이 아니라 **같은 시점 입자들 사이의 순위**만 쓴다.
따라서 궤적 수준의 정렬과 선택 수준의 정렬은 별개의 문제다.

---

## 2. Phase R0 — 설정

전부 이미 디스크에 있는 로그로 계산했다. 신규 샘플링 없음, GPU 0.

| # | 계산 내용 | 데이터 출처 | 답하는 질문 |
| --- | --- | --- | --- |
| **R1** | 전이별 입자 간 backbone RMSD ↔ TICA 거리 Spearman | `small_protein_frame_diagnostics.json`의 `whole_backbone_rmsd_a`, `tica_distance_to_folded_target` | 두 지표가 같은 것을 재는가 |
| **R2** | 전이별 입자 간 `log ψ` 표준편차, 유도 ESS | `records.json`의 `log_psi` | reward가 입자를 구분하는가 |
| **R3** | 전이 t의 reward 순위 ↔ 그 후손의 타깃 도달 여부 | `pre_final_lineages` + `target_hit` | 선택이 결과를 맞히는가 |
| **R4** | 같은 입자에 두 potential 적용, 순위·argmax·유도 ESS 비교 | R1과 동일 | reward를 바꾸면 다른 결정을 내리는가 |

대상: `cp7590` 180셀 (T=32, 3단백질 × 3방법 × 20seed) + `t96` 15셀 (T=96 BBL) = **195셀**

potential 정의 (둘 다 계수 16):
```
RMSD : log ψ = −16 · (d_RMSD / d₀)²          d₀ = 7.200 / 8.070 / 18.074 Å
TICA : log ψ = −16 · (d_TICA / 0.75)²        0.75 = 성공 반경
```

**R4의 한계 (중요):** 이것은 **한 스텝짜리 반사실**이다. 실제로 reward를 바꾸면 첫 resampling부터
다른 입자가 살아남아 궤적이 갈라지므로, 후반 비교는 "TICA reward였다면 겪었을 상황"이 아니다.
판정은 비대칭이다 — **순위가 어디서나 일치하면 H4 무의미(강한 배제), 다르면 실행 근거(약한 지지).**

---

## 3. Phase R0 — 결과

#### R1 — 정렬 (전이별 입자 간 RMSD↔TICA 순위상관)

| 조건 | n | 전체 | 초반 1/3 | 후반 1/3 |
| --- | ---: | ---: | ---: | ---: |
| T=32 Trp-cage | 60 | 0.108 | 0.239 | -0.015 |
| T=32 BBA | 60 | 0.174 | 0.258 | 0.094 |
| T=32 BBL | 60 | 0.048 | 0.090 | 0.043 |
| T=96 BBL | 15 | 0.240 | 0.417 | 0.133 |

#### R2 — 구분력 (log ψ 입자 간 분산)

| 조건 | n | log ψ 표준편차 | 유도 ESS/K |
| --- | ---: | ---: | ---: |
| T=32 Trp-cage | 60 | 2.1236 | 0.422 |
| T=32 BBA | 60 | 1.9410 | 0.332 |
| T=32 BBL | 60 | 1.4867 | 0.422 |
| T=96 BBL | 15 | 1.6150 | 0.465 |

#### R3 — 선택 정확성 (도달 경로 조상 − 미도달 경로 조상의 reward 순위차)

| 조건 | n | 순위차 | 도달 비율 |
| --- | ---: | ---: | ---: |
| T=32 Trp-cage | 60 | **+0.0970** | 0.003 |
| T=32 BBA | 60 | **-0.0431** | 0.042 |
| T=32 BBL | 60 | **-0.0274** | 0.131 |
| T=96 BBL | 15 | **+0.0124** | 0.658 |

#### R4 — 반사실 (두 potential의 가중치 비교)

| 조건 | n | 순위상관 | 초반 | 후반 | **argmax 불일치율** | 유도 ESS/K (RMSD) | 유도 ESS/K (TICA) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T=32 Trp-cage | 60 | 0.084 | 0.200 | -0.019 | **0.789** | 0.473 | 0.170 |
| T=32 BBA | 60 | 0.174 | 0.258 | 0.094 | **0.767** | 0.396 | 0.223 |
| T=32 BBL | 60 | 0.048 | 0.090 | 0.043 | **0.870** | 0.461 | 0.150 |
| T=96 BBL | 15 | 0.240 | 0.417 | 0.133 | **0.672** | 0.630 | 0.361 |

---

## 4. H4 TICA reward 실험 — 설정

### 4.1 구현

신규 파일 `src/confmh/duet/tica_potential.py`:

- `TicaEndpointMetric` — 프레임을 atom37 → 공식 `folded.pdb` 토폴로지로 매핑,
  pyemma featurizer(`add_backbone_torsions(cossin=True)`, 108차원) → 공식 `tica_model.pkl` 투영 → 첫 두 좌표
- `TicaEndpointPotential` — `log ψ = −16·(d_TICA/d₀)²`, RMSD potential과 동일 인터페이스
  (`values`, `log_psi`, `candidate_log_psi`, `evaluations`, `clipped_evaluations`)

**평가기와 수치적으로 동일한 양을 쓴다** — 동일 featurizer, 동일 모델, 타깃은 manifest의
`tica.target_first_two`. 검증: `minimized_target_allatom` 투영 시 타깃까지 거리 0.0000.

러너 분기: `program.potential: tica` (**새 키**). `program.type`은 `config.py`가
`{terminal, windowed, ordered}`로 검증하므로 건드리지 않았다.

**d₀ 처리**: TICA potential의 d₀는 출발 구조 자신의 TICA 거리 = **2.9594** (BBA).
manifest의 `d0_a` 8.070 Å은 RMSD용이며 그대로 남아 있다.

**RMSD metric은 유지**했다 — 모든 observable·validity·진단이 기존과 동일하게 기록된다.
따라서 `metrics.json`의 `per_path_final_d_a`와 `d0_a`는 여전히 **RMSD(Å)**이고,
`records.json`의 `values.endpoint_distance_a`는 **TICA 거리**다. 집계 시 혼동 주의.

투영 비용: 단일 프레임 0.02 ms. 셀 wall clock에 유의한 영향 없음.

### 4.2 실험 행렬

| 항목 | 값 |
| --- | --- |
| 단백질 | BBA (L=28, d₀_RMSD 8.070 Å, d₀_TICA 2.9594) |
| **Horizon** | **T = 48** (4.8 ns @ 100 ps lag) |
| **Reward** | **TICA 좌표** (`program.potential: tica`), 계수 16 |
| 방법 | DuET, Complete Nested |
| 배분 | K=8 × M=8 (예산 64) |
| checkpoint | 0.75, 0.90 (sde_step 20 → step 15, 18) |
| Seed | 10개 (7, 19, 31, 43, 55, 67, 79, 91, 103, 115) |
| 셀 | **20**, 실패 0 |
| NFE | 61,440/셀 |
| 출력 | `outputs/pvb_full/tica48/` |

기준선(`cp7590` BBA)과 달라진 것은 **reward와 horizon 두 가지**다. 나머지(모델, sde_step,
K, M, checkpoint, validity gate, 출발 구조, 시드)는 전부 동일하다.

### 4.3 실행 중 발견된 평가기 동작

`SampledFrameEnergy`가 **valid 타깃 도달 경로의 모든 프레임마다 OpenMM `minimizeEnergy()`**를
실행한다. 기존 360셀은 `sampled_frame_ets_evaluated_path_count = 0`이어서 이 분기가 한 번도
실행된 적이 없었다. TICA reward가 도달 경로를 만들면서 처음 작동했고, 셀당 30분 이상을 소모했다.

`sampled_frame_ets_*` 필드만 채우는 진단 전용 지표이므로 `SMALL_PROTEIN_SKIP_ETS=1`로
건너뛰도록 했다(`patch_eval_skip_ets.py`, 백업 보존). **샘플링에는 개입하지 않으며,
RMSD·Q·TICA·validity·THP는 모두 정상 계산된다.**

---

## 5. H4 실험 — 결과

### 5.1 셀별 raw (20행)

컬럼: `valid경로` = validity gate 통과 경로 수(최대 8), `TICAmin` = usable 프레임 전체의
타깃까지 최소 TICA 거리, `도달경로` = 한 프레임이라도 TICA < 0.75에 들어간 경로 수(gate 무관),
`validhit` = 도달 + 전 경로 validity 통과, `ESS1/ESS2` = 안쪽 체크포인트 0.75/0.90의 셀 평균 ESS/M,
`outerESS` = 바깥 ESS/K 평균, `outer_rs` = 48전이 중 바깥 resampling 발동 횟수.

| method | seed | T | K | M | wBB | heavy | valid | valid경로 | Qmax | Qmean | TICAmin | 도달경로 | validhit | hit_fin | THP | ESS1 | ESS2 | outerESS | outer_rs | NFE | wall_s |
| --- | --- |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |  ---: |
| duet | 7 | 48 | 8 | 8 | 7.638 | 8.961 | 1.000 | 8 | 0.4889 | 0.3861 | 0.446 | 7 | 7 | 6 | 0.7594 | 0.818 | 0.972 | 0.670 | 9 | 61440 | 224 |
| duet | 19 | 48 | 8 | 8 | 7.405 | 9.257 | 1.000 | 8 | 0.4222 | 0.3472 | 0.181 | 8 | 8 | 7 | 0.9448 | 0.818 | 0.977 | 0.653 | 9 | 61440 | 227 |
| duet | 31 | 48 | 8 | 8 | 7.593 | 9.394 | 1.000 | 8 | 0.4000 | 0.3361 | 0.697 | 1 | 1 | 1 | 0.1801 | 0.749 | 0.965 | 0.696 | 9 | 61440 | 224 |
| duet | 43 | 48 | 8 | 8 | 6.829 | 8.255 | 0.000 | 0 | 0.4222 | 0.3111 | 0.387 | 6 | 0 | 0 | 0.0000 | 0.761 | 0.970 | 0.648 | 11 | 61440 | 224 |
| duet | 55 | 48 | 8 | 8 | 7.447 | 9.147 | 1.000 | 8 | 0.3111 | 0.2722 | 0.761 | 0 | 0 | 0 | 0.0000 | 0.751 | 0.965 | 0.661 | 10 | 61440 | 226 |
| duet | 67 | 48 | 8 | 8 | 6.726 | 8.500 | 1.000 | 8 | 0.4000 | 0.3444 | 0.331 | 3 | 3 | 2 | 0.7138 | 0.781 | 0.971 | 0.677 | 7 | 61440 | 231 |
| duet | 79 | 48 | 8 | 8 | 7.874 | 9.643 | 1.000 | 8 | 0.4000 | 0.3722 | 0.076 | 8 | 8 | 7 | 0.9864 | 0.818 | 0.982 | 0.706 | 7 | 61440 | 220 |
| duet | 91 | 48 | 8 | 8 | 7.341 | 9.170 | 1.000 | 8 | 0.4222 | 0.3528 | 0.412 | 4 | 4 | 4 | 0.4114 | 0.784 | 0.961 | 0.687 | 9 | 61440 | 227 |
| duet | 103 | 48 | 8 | 8 | 6.768 | 7.971 | 1.000 | 8 | 0.4222 | 0.3306 | 0.345 | 7 | 7 | 4 | 0.7279 | 0.698 | 0.951 | 0.615 | 14 | 61440 | 223 |
| duet | 115 | 48 | 8 | 8 | 7.642 | 9.223 | 1.000 | 8 | 0.3778 | 0.3250 | 0.281 | 6 | 6 | 6 | 0.8446 | 0.815 | 0.975 | 0.705 | 6 | 61440 | 228 |
| complete_nested | 7 | 48 | 8 | 8 | 7.156 | 8.742 | 1.000 | 8 | 0.4000 | 0.3667 | 0.208 | 8 | 8 | 8 | 1.0000 | — | — | 0.615 | 11 | 61440 | 220 |
| complete_nested | 19 | 48 | 8 | 8 | 7.955 | 9.499 | 1.000 | 8 | 0.3778 | 0.3333 | 0.044 | 4 | 4 | 4 | 0.8137 | — | — | 0.646 | 11 | 61440 | 221 |
| complete_nested | 31 | 48 | 8 | 8 | 6.076 | 7.755 | 1.000 | 8 | 0.4444 | 0.4056 | 0.318 | 7 | 7 | 7 | 0.8217 | — | — | 0.624 | 6 | 61440 | 221 |
| complete_nested | 43 | 48 | 8 | 8 | 7.474 | 9.147 | 1.000 | 8 | 0.4667 | 0.3417 | 0.253 | 7 | 7 | 2 | 0.2755 | — | — | 0.668 | 10 | 61440 | 229 |
| complete_nested | 55 | 48 | 8 | 8 | 7.745 | 9.578 | 1.000 | 8 | 0.4222 | 0.3778 | 0.168 | 8 | 8 | 8 | 1.0000 | — | — | 0.674 | 8 | 61440 | 227 |
| complete_nested | 67 | 48 | 8 | 8 | 6.510 | 8.592 | 1.000 | 8 | 0.4222 | 0.3500 | 0.200 | 5 | 5 | 5 | 0.8641 | — | — | 0.658 | 11 | 61440 | 224 |
| complete_nested | 79 | 48 | 8 | 8 | 8.154 | 9.649 | 1.000 | 8 | 0.3778 | 0.3556 | 0.258 | 6 | 6 | 5 | 0.9225 | — | — | 0.664 | 10 | 61440 | 224 |
| complete_nested | 91 | 48 | 8 | 8 | 7.088 | 8.932 | 1.000 | 8 | 0.4222 | 0.3694 | 0.113 | 7 | 7 | 7 | 0.9357 | — | — | 0.694 | 7 | 61440 | 221 |
| complete_nested | 103 | 48 | 8 | 8 | 6.121 | 8.036 | 1.000 | 8 | 0.5111 | 0.3889 | 0.275 | 2 | 2 | 1 | 0.0343 | — | — | 0.648 | 10 | 61440 | 224 |
| complete_nested | 115 | 48 | 8 | 8 | 7.900 | 9.288 | 1.000 | 8 | 0.3333 | 0.2750 | 0.933 | 0 | 0 | 0 | 0.0000 | — | — | 0.634 | 8 | 61440 | 229 |
### 5.2 집계 및 DuET vs Complete Nested 검정 (각 n=10)

| 지표 | DuET | Complete Nested | 차이 | Welch t | df |
| --- | ---: | ---: | ---: | ---: | ---: |
| THP | 0.557 | 0.667 | -0.110 | **-0.63** | 18 |
| 도달 경로 / 8 | 5.00 | 5.40 | -0.40 | **-0.32** | 18 |
| validated hit / 8 | 4.40 | 5.40 | -1.00 | **-0.75** | 17 |
| 최종프레임 hit / 8 | 3.70 | 4.70 | -1.00 | **-0.78** | 18 |
| validity | 0.900 | 1.000 | -0.100 | **-1.00** | 9 |
| valid 경로 수 / 8 | 7.20 | 8.00 | -0.80 | **-1.00** | 9 |
| Qmax | 0.4067 | 0.4178 | -0.0111 | **-0.52** | 18 |
| Qmean | 0.3378 | 0.3564 | -0.0186 | **-1.23** | 18 |
| wBB (Å) | 7.326 | 7.218 | +0.108 | **+0.40** | 14 |
| TICAmin | 0.392 | 0.277 | +0.115 | **+1.13** | 18 |
| 바깥 ESS/K | 0.672 | 0.652 | +0.019 | **+1.63** | 17 |
| 바깥 resampling/48 | 9.10 | 9.20 | -0.10 | **-0.11** | 17 |

총 도달 경로: DuET 50/80, CN 54/80

### 5.3 RMSD reward 기준선과의 대조 (참고)

기준선은 T=32·RMSD reward이므로 **reward와 horizon이 동시에 다르다.** 단독 귀속 불가.

| 조건 | n | THP | Qmean | validity | wBB (Å) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **TICA reward T=48 DuET** | 10 | **0.557** | 0.338 | 0.90 | 7.33 |
| **TICA reward T=48 CN** | 10 | **0.667** | 0.356 | 1.00 | 7.22 |
| RMSD reward T=32 DuET | 20 | 0.009 | 0.322 | 0.99 | 5.12 |
| RMSD reward T=32 CN | 20 | 0.024 | 0.314 | 0.99 | 5.19 |
| RMSD reward T=32 Frozen | 20 | 0.036 | 0.299 | 0.99 | 7.63 |

### 5.4 안쪽 체크포인트 ESS (DuET만 해당)

| 조건 | n | ckpt 0.75 ESS/M | ckpt 0.90 ESS/M |
| --- | ---: | ---: | ---: |
| **TICA reward T=48 (BBA)** | 10 | **0.779** | 0.969 |
| RMSD reward T=32 (BBA) | 20 | 0.679 | 0.995 |
| CN (참고, 안쪽 resampling 없음) | 10 | — | — |

---

## 6. 해석

### 6.1 관측된 사실

**(1) RMSD reward는 입자를 구분하지만, 그 구분이 성공 조건과 거의 무관하다.**

- R2: `log ψ` 입자 간 표준편차 1.49–2.12, 유도 ESS/K 0.33–0.47 → **구분력 자체는 정상**
- R1: 같은 전이의 입자들 사이에서 RMSD 순위 ↔ TICA 순위 상관 **0.048–0.240**
  (후반 1/3에서는 Trp-cage −0.015) → **어느 입자가 타깃에 가까운지 RMSD가 알려주지 못함**
- R3: 타깃 도달 경로의 조상과 미도달 경로의 조상 사이 reward 순위차 **−0.043 ~ +0.097**
  → **선택이 도달률을 높이지 못함**. BBA·BBL(T=32)에서는 부호가 음수

> 궤적 수준(시간에 따른 진행)과 선택 수준(같은 시점 입자 간 비교)은 다른 문제다.
> T=96 BBL에서 RMSD reward만으로 65.8%가 타깃에 도달한 것은 전자이고,
> R1·R3이 재는 것은 후자다. 둘은 모순되지 않는다.

**(2) 두 reward는 서로 다른 입자를 고른다.**

R4: argmax 불일치율 **0.672–0.870**. 전이의 2/3 이상에서 1순위가 다르다.
반사실 기준 유도 ESS도 TICA 쪽이 훨씬 낮다 (0.150–0.361 vs 0.396–0.630).

**(3) reward를 TICA로 바꾸자 THP가 두 자릿수 배로 올랐다.**

| | THP | 도달 경로 |
| --- | ---: | ---: |
| TICA reward T=48 (DuET/CN) | **0.557 / 0.667** | 104/160 = 65% |
| RMSD reward T=32 (DuET/CN/Frozen) | 0.009 / 0.024 / 0.036 | 약 2–6% |

**(4) 독립 지표 Qmean도 같은 방향으로 움직였다.** 0.314–0.322 → 0.338–0.356.
다만 폭이 작다(+0.02–0.04).

**(5) DuET과 Complete Nested는 이 조건에서도 구분되지 않는다.**
14개 검정 전부 \|t\| ≤ 1.63, THP는 t = −0.63.

**(6) R4의 예측이 실제 샘플링에서 빗나갔다.**
R4는 TICA potential이 훨씬 날카로울 것(유도 ESS 0.223)으로 예측했으나,
실제 TICA reward 실행의 안쪽 ESS/M은 0.779로 RMSD 조건(0.679)보다 **오히려 높다**.
반사실 계산은 실제로 방문한 입자 구성에 대한 것이고, reward를 바꾸면 입자 구성 자체가
달라지기 때문이다 — R4 한계(§2)의 직접적 사례다.

**(7) DuET에서만 validity 붕괴가 발생했다.**
20셀 중 DuET seed 43 한 셀에서 8경로 전부가 gate 실패(도달은 6/8 했으나 THP 0).
CN은 10셀 80경로 전부 valid. 1건이라 통계적으로 약하지만(t = −1.00),
방향은 T=96 BBL 실험과 같다 — **안쪽 resampling이 있는 팔에서만 깨진 경로가 나온다.**

### 6.2 반드시 함께 읽어야 할 주의사항

| # | 주의 | 내용 |
| --- | --- | --- |
| **A** | **순환성** | reward를 TICA 거리로 정의하고 성공을 "TICA 거리 < 0.75"로 재므로, THP 상승 자체는 부분적으로 정의상 귀결이다. **독립 지표는 Q와 RMSD뿐이며, Q는 +0.02–0.04로 작다.** |
| **B** | **교락** | reward(RMSD→TICA)와 horizon(32→48)을 동시에 바꿨다. **T=48 + RMSD reward 대조군이 없어 단독 귀속이 불가능하다.** |
| **C** | **중간 집계 오류** | n=9 시점에 "CN이 DuET보다 낫다(THP 0.741 vs 0.525)"고 보고했으나, 10셀을 채우자 0.667 vs 0.557로 좁혀졌고 t = −0.63으로 구분되지 않는다. 이 문서의 수치가 최종본이다. |
| **D** | **라벨 주의** | `metrics.json`의 `per_path_final_d_a`·`d0_a`는 **RMSD(Å)**, `records.json`의 `values.endpoint_distance_a`는 **TICA 거리**다. 중간 보고에서 전자를 TICA로 오기한 적이 있다. |
| **E** | **검정력** | n=10. THP 표준편차가 크다(DuET 0.00–0.99, CN 0.00–1.00). \|t\| < 1은 **동등성이 아니라 검출 실패**다. |
| **F** | **단일 단백질** | BBA만. Trp-cage·BBL에서 재현되는지 미확인. |

### 6.3 미측정 — 다음에 해야 할 것

**D2 (최우선).** 체크포인트에서의 clean-endpoint 추정 TICA 거리 ↔ 완성 후 실제 TICA 거리 상관.
TICA는 backbone torsion의 비선형 함수라 작은 좌표 오차에 민감한 반면 RMSD는 전역 평균이라
둔감하다. **TICA reward에서는 중간 추정의 품질이 RMSD 때보다 나쁠 수 있고, 그렇다면 DuET은
노이즈로 고르고 CN은 완성된 실제 구조로 고르는 셈이 된다** — 이것이 (5)와 (7)을 동시에
설명할 수 있는 가설이다. 현재 로그에 추정값이 저장되지 않아 **로깅 추가 후 소규모 재실행 필요
(~1 GPU-h).**

**horizon 대조군.** T=48 + RMSD reward, BBA, 10 seed → §6.2-B 교락 해소 (~1 GPU-h).

**seed 확대.** 현재 10 → 20이면 검정력 약 2배 (~1.5 GPU-h).

**다른 단백질.** Trp-cage(저비용) 우선, BBL은 validity가 별도 병목이므로 후순위.

### 6.4 주장 경계

쓸 수 없는 표현:

- ❌ "TICA reward가 folding을 복원했다." — Q 증가폭 +0.02–0.04, RMSD는 오히려 악화
- ❌ "TICA reward가 성공률을 30배 올렸다." — 순환성(A)과 교락(B)을 명시하지 않은 인용
- ❌ "CN이 DuET보다 낫다." — t = −0.63, 검출 실패
- ❌ "DuET과 CN은 동등하다." — 비유의는 동등성이 아님
- ❌ "안쪽 필터가 과잉 수렴한다." — 실제 안쪽 ESS는 오히려 상승 (관측 6)

쓸 수 있는 표현:

- ✅ "RMSD reward는 입자 구분력은 정상이나, 그 구분이 성공 조건과 거의 직교한다 (R1 0.05–0.24, R3 ≈ 0)."
- ✅ "두 potential은 전이의 67–87%에서 다른 입자를 1순위로 고른다 (R4)."
- ✅ "TICA reward + T=48 조건에서 BBA의 THP는 0.56–0.67, 타깃 도달 경로는 65%다."
- ✅ "DuET의 CN 대비 이득은 이 조건에서도 검출되지 않았다 (\|t\| ≤ 1.63)."

---

## 7. 산출물

루트: `/workspace/sejin/AI_MD_NSMC_phase_b_recovery`

| 경로 | 내용 |
| --- | --- |
| `outputs/pvb_full/tica48/bba/stride128_t32/{duet,complete_nested}/seed_*/` | TICA reward 20셀 |
| `scripts/fast_folders_extended/results_phase_r0.json` | R0 195셀 셀별 결과 |
| `scripts/fast_folders_extended/phase_r0_reward_diagnostics.py` | R1–R4 계산 |
| `src/confmh/duet/tica_potential.py` | TICA potential 구현 |
| `scripts/fast_folders_extended/patch_runner_tica_program.py` | 러너 분기 패치 (`program.potential`) |
| `scripts/fast_folders_extended/patch_eval_skip_ets.py` | ETS 진단 우회 (`SMALL_PROTEIN_SKIP_ETS=1`) |
| `small_protein_pilot/configs_pvb_tica48/bba_stride128_t32.yaml` | 실험 설정 |
| `outputs/pvb_full/_queues/tica48_q{1,2,3}.log` | 큐 실행 로그 |

셀 디렉토리 내 주요 파일 (기존과 동일):
`small_protein_metrics.json`, `metrics.json`(`per_path_valid`, `per_path_final_d_a`, `outer_ess`),
`records.json`(`log_psi`, `log_z_hat`, `inner_checkpoint_ess`, `inner_selected_index`),
`checkpoint_audit.json`, `outer_ancestry.json`, `population_curves_t{16,32,48}.json`,
`native_contact_q.json`, `tica_projection.npz`, `pre_final_population_atom37.npz`,
`small_protein_frame_diagnostics.json`
