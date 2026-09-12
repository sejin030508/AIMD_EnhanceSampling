# Phase A Diagnosis v2 — frozen-evidence handoff

## 실행 상태와 보존 규칙

- 기존 Phase A: `/Users/sejin/Desktop/AIBL/projects/AI_MD_NSMC/outputs/duet_md/phase_a_cross_clock` (163/163 cells, cell당 5,000 repetitions)
- 새 진단: `/Users/sejin/Desktop/AIBL/projects/AI_MD_NSMC/outputs/duet_md/phase_a_diagnosis_v2`
- frozen evidence read-only 검증: `true`
- 기존 `metrics.json` SHA-256: `d063e5d4ac9dff838fe17e0f3c352dba3e960fd813f2d19580fe76ddcca37de2`
- smoke run은 포함하지 않았다.
- 새 셀 primary는 final outer resampling 직전 weighted population이다. 기존 frozen Phase A primary는 resampling 후 unweighted population이며, frozen raw만으로 pre 값을 재구성할 수 없다.

## 고정 실험 사양

- canonical informative settings 4개, horizon 5, epsilon 0.05, float64, master seed family 20260908.
- 모든 새 셀은 repetition당 reverse-transition 480회로 동일하다.
- adaptive action set: `(K,M) = (12,2), (6,4), (3,8)`; 첫 step은 `K32,M1`.
- Stage 1 pilot: 1,000 repetitions; 필요한 27개 비교만 5,000으로 확정.
- Stage 2: `K=16`, `sum_t M_t=10`, Y2 유지. Static `[2,2,2,2,2]`; concentrated는 한 step만 6, 나머지는 1.
- 관측 wall clock: reanalysis 1.8s, pilot 287.3s, confirmation 899.3s.
- H100 실행 환경에서 Phase A 원본+diagnosis unit tests 17/17 통과.

## Stage 0 — 기존 raw 재분석

| t | rho²=0 | InnerNeed=0 | OuterNeed=0 | balanced from tie | K12M2 | K6M4 | K3M8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1.0000 | 1.0000 | 0.8209 | 0.8209 | 0.1791 | 0.8209 | 0.0000 |
| 3 | 0.6485 | 0.6485 | 0.8906 | 0.7076 | 0.0282 | 0.7140 | 0.2578 |
| 4 | 0.7712 | 0.7712 | 0.4918 | 0.4135 | 0.4042 | 0.5001 | 0.0957 |
| 5 | 0.7768 | 0.7768 | 0.4916 | 0.4153 | 0.4074 | 0.4987 | 0.0939 |

Frozen raw에는 centered phi/psi variance가 없어 `rho²=0` 중 zero empirical variance가 원인인 비율은 replay 없이 식별 불가능하다.

Adaptive/Global Static의 variance ratio:

| quantity | mean ratio | median ratio |
| --- | --- | --- |
| terminal_success | 2.0204 | 1.9801 |
| route_l | 2.0101 | 1.9486 |
| route_r | 2.0101 | 1.9486 |

Joint residual은 applicable 453개 중 431개(95.14%)의 95% CI가 0을 포함했고, 최대 절대 z-score는 3.0715였다.
Normalizer는 proper 151개 셀 중 143개의 95% CI가 0 relative bias를 포함했다. 평균/최대 absolute relative bias는 각각 0.005361/0.022938였다.
Proper cell의 marginal bias와 MSE 중 variance 비율:

| quantity | mean abs bias | median abs bias | max abs bias | mean variance/MSE | median variance/MSE |
| --- | --- | --- | --- | --- | --- |
| terminal_success | 0.019859 | 0.019568 | 0.041623 | 0.9908 | 0.9942 |
| route_l | 0.012077 | 0.007307 | 0.045672 | 0.9961 | 0.9993 |
| route_r | 0.012077 | 0.007307 | 0.045672 | 0.9961 | 0.9993 |

Self-normalized finite-particle marginal estimate는 정확히 unbiased일 필요가 없으므로 이 bias 자체를 correctness 실패로 판정하지 않는다. 반면 unnormalized normalizer와 joint residual이 직접적인 proper-weighting audit이다.

## Stage 1 — 원인 분리 확정 결과 (primary: pre-resampling)

| setting | method | success RMSE | route RMSE | combined RMSE | TV | Z rel. bias |
| --- | --- | --- | --- | --- | --- | --- |
| common_low_informative | exact_stat_allocator | 0.295435 | 0.341078 | 0.318256 | 0.033019 | -0.000133 |
| common_low_informative | current_probe_allocator | 0.265148 | 0.323108 | 0.294128 | 0.029990 | -0.004807 |
| common_low_informative | matched_production_k12_m2 | 0.183274 | 0.233662 | 0.208468 | 0.026255 | 0.005762 |
| common_low_informative | global_static_k16_m2 | 0.157524 | 0.203733 | 0.180629 | 0.020875 | -0.004542 |
| rare_low_informative | exact_stat_allocator | 0.287261 | 0.331939 | 0.309600 | 0.149827 | -0.007193 |
| rare_low_informative | current_probe_allocator | 0.252312 | 0.279266 | 0.265789 | 0.133202 | 0.001089 |
| rare_low_informative | matched_production_k12_m2 | 0.223440 | 0.224640 | 0.224040 | 0.111184 | 0.002338 |
| rare_low_informative | global_static_k16_m2 | 0.204105 | 0.204283 | 0.204194 | 0.093144 | 0.000817 |
| common_high_informative | exact_stat_allocator | 0.288552 | 0.339748 | 0.314150 | 0.033943 | -0.009023 |
| common_high_informative | current_probe_allocator | 0.270013 | 0.317086 | 0.293549 | 0.030722 | -0.001362 |
| common_high_informative | matched_production_k12_m2 | 0.183675 | 0.234270 | 0.208973 | 0.027426 | 0.002057 |
| common_high_informative | global_static_k16_m2 | 0.154783 | 0.204089 | 0.179436 | 0.020398 | 0.001637 |
| rare_high_informative | exact_stat_allocator | 0.289962 | 0.333933 | 0.311947 | 0.147633 | -0.020219 |
| rare_high_informative | current_probe_allocator | 0.251213 | 0.285314 | 0.268264 | 0.131932 | -0.003709 |
| rare_high_informative | matched_production_k12_m2 | 0.225852 | 0.234093 | 0.229972 | 0.108164 | 0.006750 |
| rare_high_informative | global_static_k16_m2 | 0.202450 | 0.213634 | 0.208042 | 0.091540 | 0.001309 |

첫 step의 고정 `K32,M1`을 제외한 t=2–5 allocation 선택 비율:

| setting | allocator | K12M2 | K6M4 | K3M8 |
| --- | --- | --- | --- | --- |
| common_low_informative | exact_stat_allocator | 0.2135 | 0.2672 | 0.5193 |
| common_low_informative | current_probe_allocator | 0.2492 | 0.5138 | 0.2371 |
| common_low_informative | matched_production_k12_m2 | 1.0000 | 0.0000 | 0.0000 |
| rare_low_informative | exact_stat_allocator | 0.1530 | 0.5834 | 0.2636 |
| rare_low_informative | current_probe_allocator | 0.1805 | 0.7797 | 0.0398 |
| rare_low_informative | matched_production_k12_m2 | 1.0000 | 0.0000 | 0.0000 |
| common_high_informative | exact_stat_allocator | 0.4663 | 0.0154 | 0.5182 |
| common_high_informative | current_probe_allocator | 0.2759 | 0.4855 | 0.2386 |
| common_high_informative | matched_production_k12_m2 | 1.0000 | 0.0000 | 0.0000 |
| rare_high_informative | exact_stat_allocator | 0.4062 | 0.3297 | 0.2641 |
| rare_high_informative | current_probe_allocator | 0.2045 | 0.7599 | 0.0356 |
| rare_high_informative | matched_production_k12_m2 | 1.0000 | 0.0000 | 0.0000 |

평균 combined RMSE에서 matched production은 current adaptive보다 +21.98% 낮았고, global K16M2는 matched보다 추가로 +11.47% 낮았다. Exact-stat allocator는 current probe allocator 대비 +12.00% 변했다(양수는 악화).

## Checkpoint information test

Complete와 Y2는 기존 frozen post-resampling 결과를 재사용했다. 새 Y1의 primary는 pre이지만, 아래 직접 비교 표에는 기존 결과와 population을 맞추기 위해 Y1의 post 값을 쓴다.

| setting | checkpoint | success RMSE | route RMSE | combined RMSE | TV | Z rel. bias |
| --- | --- | --- | --- | --- | --- | --- |
| common_low_informative | complete | 0.211784 | 0.252351 | 0.232068 | 0.025431 | -0.003510 |
| common_low_informative | Y2 | 0.211196 | 0.254865 | 0.233031 | 0.025760 | -0.002832 |
| common_low_informative | Y1 | 0.206977 | 0.249925 | 0.228451 | 0.018140 | -0.002874 |
| rare_low_informative | complete | 0.231937 | 0.247013 | 0.239475 | 0.119941 | 0.007419 |
| rare_low_informative | Y2 | 0.226566 | 0.242957 | 0.234762 | 0.112383 | 0.000161 |
| rare_low_informative | Y1 | 0.213007 | 0.243715 | 0.228361 | 0.106335 | -0.004412 |
| common_high_informative | complete | 0.208685 | 0.256140 | 0.232413 | 0.023955 | -0.001277 |
| common_high_informative | Y2 | 0.208032 | 0.253764 | 0.230898 | 0.030742 | -0.003525 |
| common_high_informative | Y1 | 0.206671 | 0.256981 | 0.231826 | 0.027285 | 0.000727 |
| rare_high_informative | complete | 0.233678 | 0.256677 | 0.245177 | 0.119652 | -0.006325 |
| rare_high_informative | Y2 | 0.224018 | 0.257737 | 0.240877 | 0.108050 | 0.009779 |
| rare_high_informative | Y1 | 0.209583 | 0.254767 | 0.232175 | 0.097552 | 0.001878 |

No-information control (frozen post-resampling):

| setting | method | success RMSE | route RMSE | combined RMSE | TV | Z rel. bias |
| --- | --- | --- | --- | --- | --- | --- |
| rare_low_no_information | complete | 0.233984 | 0.243696 | 0.238840 | 0.132258 | -0.022840 |
| rare_low_no_information | Y2 | 0.236650 | 0.245223 | 0.240936 | 0.116666 | 0.005401 |
| rare_high_no_information | complete | 0.236146 | 0.256887 | 0.246517 | 0.116460 | 0.004482 |
| rare_high_no_information | Y2 | 0.232137 | 0.253487 | 0.242812 | 0.125001 | -0.007351 |

Y1은 checkpoint information이 큰 diagnostic control일 뿐이며 main checkpoint를 사후 변경한 결과가 아니다. 기존 no-information controls에서는 Y2 DuET의 이득이 일관되지 않았으므로 ‘DuET always better’로 해석하지 않는다.

## Stage 2 — temporal budget schedule 확정 결과

| setting | schedule | success RMSE | route RMSE | combined RMSE | TV | combined reduction vs static |
| --- | --- | --- | --- | --- | --- | --- |
| common_low_informative | static_m2 | 0.155998 | 0.206324 | 0.181161 | 0.017816 | — |
| common_low_informative | concentrated_t3 | 0.151042 | 0.183306 | 0.167174 | 0.009972 | +7.72% |
| rare_low_informative | static_m2 | 0.201864 | 0.202173 | 0.202018 | 0.093266 | — |
| rare_low_informative | concentrated_t3 | 0.154698 | 0.193540 | 0.174119 | 0.056965 | +13.81% |
| common_high_informative | static_m2 | 0.157664 | 0.202599 | 0.180131 | 0.020209 | — |
| common_high_informative | concentrated_t3 | 0.150472 | 0.181123 | 0.165798 | 0.007139 | +7.96% |
| rare_high_informative | static_m2 | 0.202520 | 0.214602 | 0.208561 | 0.090203 | — |
| rare_high_informative | concentrated_t3 | 0.152576 | 0.206861 | 0.179718 | 0.062613 | +13.83% |

5,000회 확장 대상을 고른 1,000회 pilot의 여섯 schedule 전체 raw summary:

| setting | schedule | success RMSE | route RMSE | combined RMSE | TV | Z rel. bias |
| --- | --- | --- | --- | --- | --- | --- |
| common_low_informative | static_m2 | 0.152547 | 0.208922 | 0.180734 | 0.021923 | -0.010656 |
| common_low_informative | concentrated_t1 | 0.196537 | 0.247429 | 0.221983 | 0.048017 | -0.006885 |
| common_low_informative | concentrated_t2 | 0.195272 | 0.246151 | 0.220711 | 0.047004 | 0.007028 |
| common_low_informative | concentrated_t3 | 0.154987 | 0.181112 | 0.168049 | 0.010764 | -0.006887 |
| common_low_informative | concentrated_t4 | 0.194079 | 0.235626 | 0.214853 | 0.048335 | -0.002990 |
| common_low_informative | concentrated_t5 | 0.200276 | 0.240497 | 0.220386 | 0.050642 | -0.010870 |
| rare_low_informative | static_m2 | 0.202111 | 0.199906 | 0.201008 | 0.089572 | -0.003841 |
| rare_low_informative | concentrated_t1 | 0.241949 | 0.214358 | 0.228153 | 0.122867 | 0.007276 |
| rare_low_informative | concentrated_t2 | 0.242591 | 0.207923 | 0.225257 | 0.141380 | -0.004285 |
| rare_low_informative | concentrated_t3 | 0.152697 | 0.192985 | 0.172841 | 0.071882 | -0.002431 |
| rare_low_informative | concentrated_t4 | 0.242265 | 0.215472 | 0.228869 | 0.132736 | -0.012643 |
| rare_low_informative | concentrated_t5 | 0.240663 | 0.211761 | 0.226212 | 0.128371 | -0.001324 |
| common_high_informative | static_m2 | 0.152524 | 0.201488 | 0.177006 | 0.021496 | -0.007060 |
| common_high_informative | concentrated_t1 | 0.188690 | 0.245277 | 0.216983 | 0.046077 | 0.001878 |
| common_high_informative | concentrated_t2 | 0.205549 | 0.239268 | 0.222408 | 0.046837 | 0.022611 |
| common_high_informative | concentrated_t3 | 0.144236 | 0.178266 | 0.161251 | 0.010180 | 0.014776 |
| common_high_informative | concentrated_t4 | 0.203464 | 0.243497 | 0.223480 | 0.052922 | -0.004639 |
| common_high_informative | concentrated_t5 | 0.195097 | 0.241648 | 0.218373 | 0.042676 | 0.015325 |
| rare_high_informative | static_m2 | 0.203749 | 0.218260 | 0.211005 | 0.079177 | 0.020071 |
| rare_high_informative | concentrated_t1 | 0.246424 | 0.227220 | 0.236822 | 0.121029 | 0.011630 |
| rare_high_informative | concentrated_t2 | 0.240415 | 0.218682 | 0.229548 | 0.138983 | -0.018945 |
| rare_high_informative | concentrated_t3 | 0.150061 | 0.204963 | 0.177512 | 0.054325 | 0.013110 |
| rare_high_informative | concentrated_t4 | 0.243697 | 0.225881 | 0.234789 | 0.111097 | 0.026151 |
| rare_high_informative | concentrated_t5 | 0.242907 | 0.218494 | 0.230700 | 0.128074 | -0.006504 |

4개 setting 모두 concentrated_t3가 static보다 우수하므로 판정은 Case B다. 즉 allocation opportunity는 존재하지만, 현재 online diagnostic/risk rule이 그 위치를 찾지 못한다.

## Q1–Q4 결론

### Q1. Sampling formulation 자체에 correctness 문제 증거가 있는가?

현재 결과에서는 sampler correctness issue의 증거가 없다. Unnormalized normalizer와 joint proper-weighting residual의 coverage가 nominal 수준이고, marginal RMSE는 평균적으로 99% 이상 variance가 설명한다. Self-normalized finite-particle marginal bias는 원리상 0일 필요가 없다. 새 셀의 pre/post 동시 기록으로 기존 final resampling의 추가 Monte Carlo noise도 분리했다.

### Q2. Current adaptive failure의 주 원인은 무엇인가?

상대 중요도는 C(risk formula/action selection) > B(probe overhead/reduced K) >> A(probe estimator noise)이며, D(no useful allocation opportunity)는 Stage 2가 반박한다. Exact statistics로도 회복되지 않아 A가 주원인이라는 가설은 지지되지 않는다.

### Q3. Fixed DuET에 실제 signal이 있는가?

Y2의 signal은 작고 setting-dependent하다. Frozen held-out 20개 평균 combined gain은 약 +1.55%, 16/20 positive였지만 canonical 및 no-information controls까지 포함하면 모든 지표·setting에서 일관된 우위는 아니다. 정보가 더 큰 Y1은 terminal RMSE 개선 경향을 강화하므로 pre-completion information의 역할을 지지하지만, main checkpoint 변경 근거로 사용하지 않는다.

### Q4. Allocation research를 계속할 근거가 있는가?

Conditional continue. Case B이므로 physical step별 compute 재배치 가능성은 확인됐다. 다만 현재 allocator를 확장·튜닝할 근거는 없고, 후속 연구가 있다면 유리한 step을 online으로 식별하는 진단 문제로 한정해야 한다.

## Raw evidence index

- `reanalysis/stepwise_adaptive_diagnostics.{json,csv}`
- `reanalysis/bias_variance_decomposition.{json,csv}`
- `reanalysis/adaptive_vs_static_bias_variance.{json,csv}`
- `reanalysis/joint_proper_weighting.{json,csv}`
- `reanalysis/pre_post_resampling_audit.json`
- `reanalysis/frozen_evidence_manifest.json`
- 각 cell의 `summary.json`과 `raw_repetitions.npz`
- `report/pilot_metrics.json`, `pilot_results_{pre,post}_long.csv`
- `report/confirmation_plan.json`, `confirmation_metrics.json`, `confirmation_results_{pre,post}_long.csv`
- `report/metrics.json`, `compute_accounting.json`

이 보고서는 요청된 Q1–Q4까지만 판정하며, 결과 개선을 위한 추가 sweep·allocator·reward·topology 탐색은 수행하지 않았다.
