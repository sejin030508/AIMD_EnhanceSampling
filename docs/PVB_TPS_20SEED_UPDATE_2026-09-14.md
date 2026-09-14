# PVB fast-folding TPS — 20-seed 재검정 (360셀)

작성일: 2026-09-14
seed: 3 → **20** (7, 19, 31, … 235; 등차 12, 기존 3개 포함)
셀: 54 → **360** (timing 4종 × 3 단백질 × 20 seed + cp7590 대조군 2종 × 3 × 20)

3 seed 시점의 결론 중 **하나는 유지되고, 하나는 뒤집혔으며, 하나는 약화**되었다.

---

## 1. 뒤집힌 결론 — DuET은 Complete Nested보다 낫지 않다

3 seed에서는 DuET이 세 단백질 모두 CN보다 wBB가 낮아 "중간 개입이 근소하게 유리"로 읽혔다.
20 seed에서 **이 차이는 사라진다.**

| 단백질 | DuET | Complete Nested | 차이 | Welch t |
| --- | --- | --- | ---: | ---: |
| Trp-cage | 3.89 ± 0.24 | 3.89 ± 0.29 | −0.00 | **−0.02** |
| BBA | 5.12 ± 0.56 | 5.19 ± 0.56 | −0.06 | **−0.36** |
| BBL | 15.88 ± 0.76 | 16.13 ± 0.95 | −0.25 | **−0.92** |

세 경우 모두 |t| < 1 로, **구분되지 않는다.** 3 seed에서 보였던 0.12–0.88 Å 우위는
seed 표본 변동이었다.

> **diffusion 중간 개입(DuET)은 완성 후 선택(Complete Nested) 대비 측정 가능한 이득이 없다.**
> 이것이 20 seed 확장으로 얻은 가장 중요한 결과다. DuET의 유일한 구조적 이점인
> "연산 재배치"가 이 조건에서는 성능으로 전환되지 않는다.

---

## 2. 유지된 결론 — steering은 RMSD만 최적화한다

| 단백질 | Frozen (K=64) | Complete Nested | DuET |
| --- | ---: | ---: | ---: |
| Trp-cage wBB | 6.10 ± 0.14 | 3.89 ± 0.29 | 3.89 ± 0.24 |
| BBA wBB | 7.63 ± 0.11 | 5.19 ± 0.56 | 5.12 ± 0.56 |
| BBL wBB | 18.90 ± 0.24 | 16.13 ± 0.95 | 15.88 ± 0.76 |

선택이 있는 두 방법이 Frozen보다 2.2–3.0 Å 낮다. 표준편차가 작아 이 차이는 확실하다.

그러나 native contact Q는 반대 방향이다.

| 단백질 | Frozen | Complete Nested | DuET |
| --- | ---: | ---: | ---: |
| Trp-cage | **0.333** | 0.304 | 0.281 |
| BBA | **0.438** | 0.402 | 0.398 |
| BBL | **0.313** | 0.273 | 0.265 |

**세 단백질 모두 Frozen이 가장 높다.** RMSD를 2–3 Å 낮추는 동안 native contact은
오히려 줄어든다. reward가 향하는 방향과 folding 진행 방향이 어긋나 있다는 3 seed
결론이 20 seed에서 더 선명해졌다.

---

## 3. 약화된 결론 — "TICA 타깃 도달은 Frozen만"은 부분적으로만 맞다

3 seed에서는 hit이 거의 Frozen에서만 나왔다. 20 seed에서는 양상이 갈린다.

| 단백질 | Frozen | Complete Nested | DuET |
| --- | ---: | ---: | ---: |
| Trp-cage hit/path | 0.0016 | **0.0063** | 0.0000 |
| BBA hit/path | **0.0578** | 0.0500 | 0.0187 |
| BBL hit/path | 0.0148 | **0.0263** | 0.0063 |

- **Frozen 독점은 아니다.** Trp-cage와 BBL에서는 Complete Nested가 가장 높다.
- **DuET은 세 단백질 모두에서 최하위다.** 이것은 3 seed 때보다 오히려 강해진 패턴이다.

즉 정확한 진술은 "Frozen만 도달한다"가 아니라 다음과 같다.

> **DuET은 세 단백질 모두에서 TICA 타깃 도달률이 가장 낮다.**
> Complete Nested는 Frozen과 비슷하거나(BBA) 더 높다(Trp-cage, BBL).

DuET과 CN은 wBB·Q·validity가 사실상 동일한데 hit/path만 DuET이 1/3 수준이다.
두 방법의 유일한 차이가 중간 개입이므로, **중간 resampling이 타깃 도달에 필요한
꼬리를 깎는다**는 해석이 가능하다. 다만 hit 자체가 희소해(전체 146건) 이 차이의
통계적 강도는 wBB 비교만큼 견고하지 않다.

---

## 4. timing sweep — 여전히 non-result, 이제는 확정적으로

DuET wBB, 20 seed 평균 ± 표준편차:

| 단백질 | 0.25+0.50 | 0.50+0.75 | 0.75+0.90 | 0.25+0.50+0.75 |
| --- | --- | --- | --- | --- |
| Trp-cage | 3.87 ± 0.26 | 3.86 ± 0.21 | 3.89 ± 0.24 | 3.88 ± 0.27 |
| BBA | 5.14 ± 0.43 | 5.00 ± 0.36 | 5.12 ± 0.56 | 5.04 ± 0.40 |
| BBL | 15.99 ± 0.80 | 16.04 ± 0.70 | 15.88 ± 0.76 | **15.43 ± 0.84** |

Trp-cage는 네 timing이 0.03 Å 안에 들어온다 — 표준편차의 1/8 수준이다.
BBA도 0.14 Å 범위로 표준편차 안이다.

BBL의 `0.25+0.50+0.75`만 0.45–0.61 Å 낮은데, 표준편차 0.84를 감안하면
경계선이다. 세 단백질 중 하나에서만, 그것도 validity가 0.2 수준으로 무너진
단백질에서 나온 신호이므로 **timing 효과의 증거로 삼기 어렵다.**

> 3 seed 때 "seed 산포 이내"라고 한 판단이 20 seed에서 확정되었다.
> 이는 timing이 무관하다는 뜻이 아니라, **3.2 ns horizon·RMSD reward 조건에서는
> 반응할 수 있는 지표가 없다**는 뜻이다.

---

## 5. BBL validity — 3 seed 관측 유지

| 단백질 | Frozen | Complete Nested | DuET |
| --- | ---: | ---: | ---: |
| Trp-cage | 0.93 | 1.00 | 0.99 |
| BBA | 0.99 | 0.99 | 0.99 |
| **BBL** | **0.15** | **0.22** | **0.21** |

BBL만 방법과 무관하게 붕괴한다. 출발 구조가 folded 대비 2.27배 팽창(Rg 23.45 vs
10.35 Å)해 PVB-ATLAS 학습 분포 밖이라는 3 seed 해석이 20 seed에서도 그대로 유지된다.

---

## 6. 실행 중 발생한 문제

**중복 배정 경합.** GPU 4대로 확장하면서 동일 spec을 두 큐에 배정한 결과,
두 셀(`cp7590/bba/frozen/seed_103`, `cp7590/bbl/complete_nested/seed_103`)의
`small_protein_metrics.json`에 JSON이 두 번 이어 기록되어 파싱 불가 상태가 되었다.
샘플링 결과(`metrics.json`, population npz)는 온전했으므로 **평가만 재실행하여
복구**했다. 재샘플링은 불필요했다.

러너의 안전장치(`metrics.json` 없이 디렉토리만 있으면 exit 2로 거부)가 대부분의
충돌은 막았으나, 평가 단계까지 도달한 동시 실행은 막지 못했다. 향후 다중 GPU
배분 시 spec 집합을 **큐 간 배타적으로** 분할해야 한다.

---

## 7. 갱신된 결론

> PVB-ATLAS 기반 fast-folding TPS에서, 선택을 수행하는 두 방법(DuET, Complete
> Nested)은 Frozen 대비 backbone RMSD를 2.2–3.0 Å 낮춘다. 그러나 native contact Q는
> 세 단백질 모두에서 Frozen이 가장 높고, TICA 타깃 도달률은 DuET이 가장 낮다.
> **DuET과 Complete Nested는 RMSD·Q·validity 어디에서도 구분되지 않으며(|t| < 1),
> diffusion 중간 개입의 이득은 관측되지 않았다.** checkpoint timing 4종도 20 seed
> 기준 구분되지 않는다.
>
> 이는 DuET 알고리즘의 실패가 아니라 **reward 정렬 실패**다.
> `log ψ = −16·(d_RMSD/d₀)²` 가 최적화하는 방향이 벤치마크의 성공 조건(첫 두 TICA
> 좌표)과 어긋나 있어, 어떤 선택 전략을 써도 성공률로 전환되지 않는다.

**피해야 하는 표현** (3 seed 보고서에서 이어짐, 1건 추가)

- "DuET이 folding pathway를 복원했다."
- "RMSD가 낮아졌으므로 folded 상태에 가깝다."
- "timing이 성능에 영향을 주지 않는다."
- **"DuET이 Complete Nested보다 낫다."** ← 20 seed에서 반증됨

---

## 8. 다음 단계

H4(TICA 좌표 기반 reward)의 우선순위가 더 높아졌다. 20 seed로도 RMSD reward
조건에서는 어떤 방법·timing 조합도 성공 지표를 움직이지 못한다는 것이 확정되었으므로,
reward를 바꾸지 않는 한 추가 sweep은 정보를 주지 않는다.

부수적으로, DuET이 CN 대비 hit/path에서만 일관되게 낮은 점(세 단백질 모두 1/3 수준)은
**중간 resampling이 꼬리를 깎는다**는 가설로 이어진다. H4 실행 시 checkpoint 개수를
0(=CN) / 1 / 2 로 두고 hit/path를 비교하면 직접 검정할 수 있다.

---

## 산출물

- `scripts/duet/fast_folders_extended/results_pvb_360.json` — 360셀 통합
- `outputs/pvb_full/<timing>/…/native_contact_q.json` — 셀별 Q 사이드카
