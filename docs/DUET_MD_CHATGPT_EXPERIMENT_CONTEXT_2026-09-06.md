# DuET-MD 전체 실험 컨텍스트 — ChatGPT 이론/논문 고도화용

스냅샷: **2026-09-06 15:54 KST**  
로컬 코드: `/Users/sejin/Desktop/AIBL/projects/AI_MD_NSMC`  
Frozen-model/ATLAS asset: `/Users/sejin/Desktop/AIBL/projects/confrover_mh_steering`  
Git: `main @ 33556dc88db77a4735b180bbdeeaa41c8339de3c` + 미커밋 변경 다수

이 문서는 ChatGPT가 다른 대화의 맥락 없이 DuET-MD의 이론과 논문 설계를 검토할 수 있도록 만든 **설정·결과 동결 스냅샷**이다. 단순 import, dry-run, 1-frame/K=1 smoke 및 명시적으로 `smoke` 디렉터리에 저장된 결과는 제외했다. 대신 실제 비교, negative result, pilot, preflight, support gate, 중단된 본 실험의 완료 cell은 숨기지 않았다.

수집된 비-smoke `metrics.json`은 총 **294개**다: seed/method별 run **290개**, 묶음형 exact/preflight 결과 **4개**. 현재 10-protein 본 실험 완료분은 **15/100 cells**다. 이 문서의 표는 평균만이 아니라 seed별 raw scalar를 모두 포함한다.

## 1. 연구 질문과 주장 경계

Frozen ConfRover를 재학습하거나 물리 force를 추가하지 않고, 생성 중 후보를 선택·복제하여 사용자가 지정한 temporal program을 더 자주 만족시키는지를 묻는다. 핵심 비교는 **DuET-MD 대 Complete Nested**다. 두 방법은 같은 `K`, `M`, frozen model, SDE, seed, physical horizon과 decoder NFE를 사용하고, reverse diffusion이 끝나기 전에 계산을 재배치하는지 여부만 다르게 만든다.

정확한 주장 범위는 **frozen surrogate prior 안에서의 bias-conditioned transition-path discovery**다. 실제 `U+b` dynamics, equilibrium distribution, free energy, MFPT, rate 또는 물리 kinetics를 복원했다는 주장은 현재 결과로 할 수 없다. ATLAS reference 시간 대비 nominal compression은 난이도 설정일 뿐 kinetic acceleration 추정치가 아니다.

## 2. 두 clock과 방법의 코드상 실체

- Physical clock: trajectory frame `t=1...T`.
- Diffusion clock: 다음 frame 하나를 만드는 약 200 reverse steps. 실제 SDE decoder evaluation은 후보·frame당 199회다.
- `K`: 동시에 유지하는 outer full-history particle 수.
- `M`: 각 outer history에서 다음 frame을 만들 때 쓰는 inner reverse-diffusion 후보 수.
- 한 physical step의 nominal decoder population budget은 `B=K*M`.

Frozen model prior는 `Q_theta(x_1:T|H_0)=prod_t q_theta(x_t|H_t-1)`이고, path target은 `Q_theta,P proportional to Q_theta * R_P`, `R_P=exp(-lambda*C_P)`다. Prefix potential은 `psi_t=exp(-lambda*C_t)`이며 `potential_floor` 아래로만 numerical clipping한다.

| Method | 실제 동작 | Outer weight | 핵심 의미 |
|---|---|---|---|
| Frozen | `M=1`; 각 history를 독립적으로 끝까지 rollout; resampling 없음 | 없음 | 기본 frozen surrogate sampling |
| Outer-only | `M=1`; 완성된 다음 frame을 받은 뒤 physical histories를 ESS 기준 선택/복제 | `psi_t/psi_(t-1)` | 이미 생성된 경로 중 좋은 경로 선택 |
| Inner-only | `K=1`; diffusion checkpoint predicted-clean 점수로 M개 noisy state를 선택/복제 | 없음 | 한 history 안에서 다음 frame 발견에 계산 집중 |
| Complete Nested | 부모마다 M개 frame을 모두 완성; `psi_t` 비례로 하나 선택 | `Zhat_t/psi_(t-1)`, `Zhat=mean(psi_t^j)` | 같은 budget을 완성 후보에 쓰는 핵심 대조군 |
| DuET-MD | checkpoint에서 predicted-clean potential로 resample, 독립 noise로 continuation, clean endpoint correction | `Zhat_t/psi_(t-1)` | diffusion-time 계산 재배치 + physical-time SMC |
| Naive Dual | DuET inner를 쓰고 outer에서 선택된 child의 `psi_t/psi_(t-1)`를 다시 적용 | 잘못된 중복 보상 | 이상적 limit에서 `psi_t^2` 방향의 target distortion 진단 |
| Best-of-budget | 전체 path 후보를 끝까지 만든 뒤 최종 score 최고 path 선택 | 해당 없음 | 초기 진단용, 최종 baseline 아님 |

DuET one-checkpoint의 incremental inner weights는 checkpoint `g1=phi_c`, endpoint `g2=phi_0/phi_c(ancestor)`라서 ancestry별로 `g1*g2=phi_0=psi_t`로 telescope한다. Local normalizer는 `Zhat=(mean g1)(mean g2)`이고, selected child의 `psi_t`를 outer에서 다시 곱하지 않는다.

현재 runtime의 outer resampling은 weighted method에서 **`ESS <= fraction*K` 또는 final step**일 때 systematic resampling한다. 초기 문서 중 “매 step resampling” 문구보다 실제 resolved config와 runtime 조건을 우선한다. Inner resampling도 systematic이다.

## 3. Reward/potential의 코드상 정의

다음 event의 observable 값 `v`와 허용 interval `I` 사이 거리를 `d(v,I)`라 하면, 성공 전 prefix cost는 다음 구현이다.

```text
C_prefix = distance_weight*(d/distance_scale)^2
           *(1 + deadline_weight/(remaining_frames+1))
           + remaining_event_weight*(미완료 event 수)
log psi_t = max(-lambda_program*C_prefix, log(potential_floor))
```

이미 program failure 상태라면 `failure_penalty + failure_guidance_weight*(terminal distance/scale)^2`; final deadline에 아직 성공하지 못했으면 `terminal_failure_penalty + distance_weight*(next-event distance/scale)^2`; 성공이면 final cost 0이다. Binary success와 continuous steering reward를 구분해야 한다. Reward는 “A/B를 만족하면 1점”만 주는 것이 아니라, **현재 다음 stage까지의 연속 거리 + deadline + 미완료 stage penalty**를 prefix potential로 바꾼 것이다.

## 4. 데이터 분할과 route-v3.1 benchmark 구성

새 10-protein protocol에서는 original R1/R2/R3 번호를 고정 역할로 쓰지 않는다. 공식 ConfRover 사례가 사용한 replicate를 `D1`, 남은 것 중 낮은 번호를 `D2`, 마지막을 `H`로 둔다. D1/D2만 PCA·basin·A/B 설계에 사용하고 H는 comparable transition이 있을 때만 optional fidelity 평가에 쓴다. H에 같은 route가 없더라도 탈락시키지 않는다.

Reference path는 D1 initial structure에서 start basin에 머문 뒤의 **stable last exit**부터 target basin의 **stable first entry**까지 기존 MD frame을 추출한다. 25개 analysis state를 고를 때 좌표 선형보간은 하지 않고 원래 MD frame index를 균등 subsampling한다. Reference time은 A/B route를 설계하고 nominal challenge를 정할 뿐 generated trajectory에 좁은 reference event window를 강제하지 않는다.

Generated ordered success는 공통 deadline frame 24 안에서 A, B, target이 각각 2 consecutive frames 지속되고, 한 frame에 두 stage를 동시에 완료하지 않으며, 엄격히 `A -> B -> target` 순서로 완료되는 것이다. 안정적 target에 일찍 도달해도 성공이고 time-to-success를 기록한다.

### 4.1 동결된 10개 단백질과 시간 설정

| Protein | Residues | D1/D2/H | Lag ps | D1 first passage ns | Last-exit path ns | Model T=24 ns | Compression | Route support | H fidelity |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 6jv8_A | 76 | R3/R1/R2 | 420 | 20.4 | 20.1 | 10.08 | 2.024 | D1+D2 | unavailable |
| 7bwf_B | 92 | R3/R1/R2 | 200 | 9.5 | 7.2 | 4.80 | 1.979 | D1 only | unavailable |
| 6tly_A | 99 | R2/R1/R3 | 370 | 17.8 | 17.7 | 8.88 | 2.005 | D1+D2 | available |
| 7s86_A | 99 | R3/R1/R2 | 240 | 11.7 | 6.9 | 5.76 | 2.031 | D1 only | available |
| 6rrv_A | 127 | R1/R2/R3 | 420 | 20.3 | 5.7 | 10.08 | 2.014 | D1 only | unavailable |
| 6gus_A | 155 | R2/R1/R3 | 310 | 15.1 | 12.8 | 7.44 | 2.030 | D1+D2 | available |
| 6q9c_A | 155 | R3/R1/R2 | 200 | 8.9 | 6.7 | 4.80 | 1.854 | D1 only | unavailable |
| 7rm7_A | 228 | R3/R1/R2 | 200 | 5.7 | 4.1 | 4.80 | 1.188 | D1 only | unavailable |
| 7aex_A | 275 | R3/R1/R2 | 200 | 6.4 | 3.5 | 4.80 | 1.333 | D1 only | unavailable |
| 7p46_A | 282 | R3/R1/R2 | 320 | 15.8 | 12.4 | 7.68 | 2.057 | D1 only | unavailable |

Primary 후보 `6in7_A`, `6lus_A`, `6ovk_R`은 method outcome을 보기 전 pre-method gate에서 탈락했고 reserve 1–3인 `7rm7_A`, `7aex_A`, `7p46_A`가 이 순서로 대체됐다.

### 4.2 단백질별 정확한 A/B/terminal binary success 정의

Residue index는 추출 chain의 0-based index다. `<=`는 closing/contact, `>=`는 opening이다. 모든 조건은 2-frame persistence를 요구한다.

| Protein | A | B | Terminal PC1 interval |
|---|---|---|---|
| 6jv8_A | d(0,67) <= 1.496516943 nm | d(25,74) >= 3.519251466 nm | [0.531737044, 1.531737044] |
| 7bwf_B | d(41,80) <= 3.793600798 nm | d(41,82) <= 3.809341073 nm | [1.557720264, 2.557720264] |
| 6tly_A | d(1,97) >= 4.020670176 nm | d(0,78) >= 3.120464325 nm | [-1.724851325, -0.724851325] |
| 7s86_A | d(5,52) >= 3.224613190 nm | d(0,52) >= 4.000426173 nm | [1.104406423, 2.104406423] |
| 6rrv_A | d(0,48) >= 3.810573578 nm | d(1,41) >= 4.273632050 nm | [1.774485271, 2.774485271] |
| 6gus_A | d(0,64) <= 2.020712614 nm | d(0,63) <= 1.916823149 nm | [0.971507389, 1.971507389] |
| 6q9c_A | d(17,125) >= 4.286140561 nm | d(18,84) >= 4.667618752 nm | [1.179918608, 2.179918608] |
| 7rm7_A | d(4,116) <= 4.287963748 nm | d(0,146) <= 7.607475519 nm | [-0.111642421, 0.888357579] |
| 7aex_A | d(108,157) >= 1.170583844 nm | d(74,157) >= 3.609910011 nm | [-0.201393033, 0.798606967] |
| 7p46_A | d(2,102) >= 2.340170145 nm | d(0,102) >= 2.720981061 nm | [-2.077466483, -1.077466483] |

### 4.3 단백질별 generated x0와 reference endpoint

`x0`는 AI rollout이 시작하는 실제 구조다. 아래 start/end frame은 D1 reference에서 benchmark task를 정의할 때 사용한 source frame이며, generated path가 이 MD 좌표열을 복사하거나 같은 시간에 사건을 재현한다는 뜻이 아니다.

| Protein | D1 | Generated x0/source start | Reference endpoint | Start PC1 | Start basin PC1 |
|---|---|---|---|---:|---|
| 6jv8_A | R3 | `6jv8_A_R3F1000_start.pdb` / F1000 | `6jv8_A_R3F3048_end.pdb` / F3048 | -1.887977 | [-2.137977,-1.637977] |
| 7bwf_B | R3 | `7bwf_B_R3F3000_start.pdb` / F3000 | `7bwf_B_R3F4024_end.pdb` / F4024 | -2.066311 | [-2.316311,-1.816311] |
| 6tly_A | R2 | `6tly_A_R2F1000_start.pdb` / F1000 | `6tly_A_R2F3048_end.pdb` / F3048 | 1.904487 | [1.654487,2.154487] |
| 7s86_A | R3 | `7s86_A_R3F5000_start.pdb` / F5000 | `7s86_A_R3F7048_end.pdb` / F7048 | 0.246775 | [-0.003225,0.496775] |
| 6rrv_A | R1 | `6rrv_A_R1F1000_start.pdb` / F1000 | `6rrv_A_R1F3048_end.pdb` / F3048 | -1.136965 | [-1.386965,-0.886965] |
| 6gus_A | R2 | `6gus_A_R2F7000_start.pdb` / F7000 | `6gus_A_R2F9048_end.pdb` / F9048 | -1.652867 | [-1.902867,-1.402867] |
| 6q9c_A | R3 | `6q9c_A_R3F3000_start.pdb` / F3000 | `6q9c_A_R3F5048_end.pdb` / F5048 | -0.531497 | [-0.781497,-0.281497] |
| 7rm7_A | R3 | `7rm7_A_R3F3000_start.pdb` / F3000 | `7rm7_A_R3F4024_end.pdb` / F4024 | -0.925509 | [-1.175509,-0.675509] |
| 7aex_A | R3 | `7aex_A_R3F5000_start.pdb` / F5000 | `7aex_A_R3F7048_end.pdb` / F7048 | 0.804314 | [0.554314,1.054314] |
| 7p46_A | R3 | `7p46_A_R3F5000_start.pdb` / F5000 | `7p46_A_R3F7048_end.pdb` / F7048 | 0.592636 | [0.342636,0.842636] |

### 4.4 역사적 6J56/7LP1 task 정의

| Case/version | Horizon | Ordered A | Ordered B | Terminal | Windowed intermediate |
|---|---:|---|---|---|---|
| 6J56 initial | 8 | d(29,127)<=4.023118 nm, f3–5 | d(1,126)<=1.928276 nm, f5–7 | PC1 [1.944262,2.444262], f8 | PC1 [0.515456,1.015456], f3–5 |
| 6J56 v2 | 16 | d(2,72)>=2.702696 nm, f5–9 | d(1,126)<=1.928276 nm, f10–14 | PC1 [1.694262,2.694262], f14–16 | PC1 [0.515456,1.015456], f6–10 |
| 6J56 v3 | 24 | d(2,72)>=2.702696 nm, f7–15 | d(1,126)<=1.928276 nm, f15–21 | PC1 [0.239032,0.739032], f19–24 | PC1 [0.515456,1.015456], f9–15 |
| 7LP1 initial | 8 | contact(0,24)=0, f1–3 | contact(2,36)=1, f3–5 | PC1 [-1.323268,-0.823268], f8 | PC1 [0.178593,0.678593], f3–5 |
| 7LP1 v2 | 24 | d(0,7)>=1.709183 nm, f1–5 | d(25,36)<=1.742410 nm, f19–23 | PC1 [-1.573268,-0.573268], f21–24 | PC1 [0.178593,0.678593], f9–15 |

Historical 6J56 x0는 `${DUET_ASSET_ROOT}/external/ConfRover/examples/6j56_A_start.pdb`, 7LP1 x0는 `7lp1_A_R2F5000_start.pdb`다. 두 case 모두 model lag config는 `256*10 ps=2.56 ns/generated frame`이었다. 이 historical timing을 현재 route-v3.1 protocol과 혼합하면 안 된다.

## 5. 현재 confirmatory main matrix

공통: frozen ConfRover base-20m-v1.0 checkpoint, SDE, reverse steps 200 configured/199 decoder calls, horizon 24, one checkpoint `p=0.95`, systematic resampling, outer ESS fraction 0.5, float32, offloaded KV cache, seed `20261001` 한 개. 원래 config에는 seeds `20261001...20261005`가 남아 있지만 **현재 실행 queue의 effective seed는 20261001 하나뿐**이다.

| Method | K | M | B | Outer | Inner | Correct outer increment |
|---|---:|---:|---:|---|---|---|
| Frozen | 64 | 1 | 64 | 없음 | 없음 | 없음 |
| Outer-only | 64 | 1 | 64 | ESS SMC | 없음 | `psi_t/psi_(t-1)` |
| Inner-only | 1 | 64 | 64 | 없음 | p=.95 | 없음 |
| Complete Nested | 16 | 4 | 64 | ESS SMC | 완성 후 M개 중 선택 | `Zhat_t/psi_(t-1)` |
| DuET-MD | 16 | 4 | 64 | ESS SMC | p=.95 resample + continuation + endpoint correction | `Zhat_t/psi_(t-1)` |

각 full cell의 expected decoder NFE는 `24*199*64=305,664`다. 단백질별 ordered와 endpoint 두 task를 모두 실행하므로 전체는 `10 proteins*2 tasks*5 methods=100 cells`, seed 추가 전 총 NFE는 30,566,400이다.

### 5.1 현재 queue 상태

- Scheduler: 두 H100이 method 단위 atomic lock을 공유하는 no-barrier queue.
- 우선순위: protein 순서 `6jv8_A, 7bwf_B, 6tly_A, 7s86_A, 6rrv_A, 6gus_A, 6q9c_A, 7rm7_A, 7aex_A, 7p46_A`; protein 안에서 ordered 후 endpoint; method 우선순위 Frozen, Complete, DuET, Outer, Inner.
- 15:54 KST snapshot active: GPU1=`7bwf_B/endpoint/frozen`; GPU2=`7bwf_B/endpoint/complete_nested`.
- 그 전에 시도한 K48 Frozen mandatory gate는 첫 완료 전에 취소했다. finite Frozen sample의 0 success가 zero support를 뜻하지 않고 rare-event benchmark를 쉬운 task로 편향시킬 수 있기 때문이다.
- five-seed main queue도 첫 완료 전에 취소하고 seed 1개 coverage-first 전략으로 전환했다. 취소된 run에는 scientific raw result가 없다.

## 6. 실험별 결과 해석 요약

### 6.1 Exact toy와 adapter preflight

Exact toy gate 6개가 모두 통과했다. Proper-weighting L1 residual 최대 0.0341, local normalizer relative bias 절댓값 최대 0.0039, M=1→16에서 Zhat variance 0.03813→0.002925였다. Ordered scale 16 final TV는 Complete 0.0240, DuET 0.0261, normalizer bias는 -0.0357와 +0.0135였고 Naive Dual bias는 +0.1570이었다. 이는 weighting 구현 타당성이지 protein superiority가 아니다.

Kabsch-aligned final Phase-B preflight는 ODE/SDE/checkpoint/constant-resampling 모두 nonfinite=0, clash=0, hard geometry validity=1.0이었다. SDE adjacent-CA quality는 ODE 대비 약 1.037–1.047배, PC1 Wasserstein은 unconditioned 대비 checkpoint/no-resampling 0.203, constant-resampling 0.314로 gate를 통과했다. `pre_seedfix` 결과는 Kabsch/seed 처리 전 폐기 archive이며 evidence에서 제외하되 raw table에는 invalidated history로 보존한다.

### 6.2 6J56 개발/negative result

Horizon-8 Phase C에서 endpoint는 seed 20260831의 DuET/Naive/Outer만 post success 1.0이었고 나머지 seed는 0; ordered/windowed는 모든 method/seed가 0이었다. Budget-16 K/M grid도 모든 allocation의 ordered/windowed가 0이었다. Horizon-16 protocol-v2 역시 다섯 방법 모두 0이었다.

Horizon-24 protocol-v3 seed 20260913에서는 Frozen 0.03125, Outer 0.03125, Inner-only 1.0, Complete 0.125, DuET 0.125의 pre-unique success였다. Complete mass 0.4385/post 0.5, DuET mass 0.4167/post 0.375였다. Seed 20260914에서는 Complete가 1개 성공, DuET은 0이었다. K4×M8 pilot은 DuET이 한 seed에서 성공했지만 preterminal outer resampling이 0이었다. Event B가 x0에서 이미 만족되고 design transition도 하나뿐이어서 6J56는 formal confirmatory evidence가 아니라 개발/falsification 사례다.

### 6.3 7LP1 initial task와 v2

초기 horizon-8 task는 A=(0,24) contact loss in frames 1–3, B=(2,36) contact formation in frames 3–5, terminal PC1 [-1.323268,-0.823268] at frame 8이었다. 한 reference segment에서 B가 A보다 먼저이고 일부 crossing이 window 밖이어서 탐색 기록으로만 유지한다.

V2는 A=d(0,7)>=1.709183 nm at frames 1–5, B=d(25,36)<=1.742410 nm at frames 19–23, terminal PC1 [-1.573268,-0.573268] at frames 21–24로 고쳤다. R1/R2 두 transition 모두 A-before-B였다. Frozen K48×M1 seeds 20260916/17에서 full success는 1/48와 4/48, 합계 5/96였다.

One-checkpoint p=.95 allocation 결과는 혼합됐다. K4×M8, K8×M4, K8×M8에서는 Complete가 성공 metric에서 우세했고, K4×M16은 DuET pre-unique/fidelity가 나았지만 lineage가 약했다. 선택된 K16×M4 five-seed development에서는 Complete 대 DuET 평균 pre-unique 0.4625 대 0.6000, success mass 0.8536 대 0.8908, post 0.8625 대 0.9000이었다. 하지만 이 allocation은 여러 조건을 본 뒤 선택됐고 paired bootstrap CI가 0을 포함하며 held-out path distance는 DuET이 2.5523→2.6818로 약 5.1% 나빴다.

Two-checkpoint [0.85,0.95]는 matched development seeds에서 unique success/lineage/fidelity를 개선하지 못했다. Fresh seeds 20260925/26에서는 Complete 대 DuET pre-unique 0.6250 대 0.7188, mass 0.9066 대 0.9598, post 0.875 대 1.0이었지만 held-out path distance와 lineage는 DuET이 나빴다. 따라서 p=.95 one-checkpoint가 보수적 기본이고 multi-checkpoint는 ablation이다.

### 6.4 ABL1 DFG-flip case study

Reward는 최종 Asp381/Phe382 pseudo-dihedral composite distance만 사용한다: Asp center 265.029°, scale 31.039°; Phe center 33.492°, scale 18.434°; success threshold <=0.85387555. DFG-inter contact, Asp/Phe order, concerted/staggered route, Lys271-Glu286, continuity/genealogy는 hidden evaluation이며 steering reward에 넣지 않았다.

Production 고정안은 T24, B64, Frozen K64M1, Complete/DuET K16M4, seeds 20261001–05, Complete p=.95, DuET [0.85,.95], A6000 decoder microbatch 2다. 그러나 production은 아직 실행하지 않았다. Budget-8 support gate seed 20260930에서 Frozen/Complete/DuET endpoint success는 모두 0, endpoint validity는 모두 1.0, whole-path validity는 0.875/0/0이었다. Invalidity는 adjacent-CA 5.501 Å hard cutoff를 일부 중간 frame이 근소하게 넘은 문제였다. 이것은 biological comparison이 아니라 안전성 진단이다.

### 6.5 ATLAS 10-protein route-v3.1 현재 초기 신호

K=8 Frozen diagnostic pilot은 9개 완료: A→B는 6q9c_A에서만 1/8, full A→B→target은 전부 0; 6tly_A whole-path valid가 2/8이었다. 이는 main result가 아니라 난이도/geometry 진단이다.

현재 completed main 결과에서 6jv8_A ordered는 다섯 방법 모두 full success 0이다. Complete는 pre-final population에서 stage-2(A→B) 1개가 있었고 DuET은 stage-1(A) 3개까지였다. 6jv8_A endpoint는 DuET과 Outer-only가 success 1.0, Frozen/Inner/Complete가 0이었다. 7bwf_B ordered도 다섯 방법 모두 full success 0; Complete stage-1 4개, DuET stage-0 16개였다. Seed 하나·단백질 두 개뿐이므로 결론은 아니지만, ordered의 초기 상대 신호는 DuET에 불리하고 endpoint에는 긍정 신호가 섞여 있다.

### 6.6 K=8 diagnostic raw counts

| Protein | A reached | A→B | Full | Whole-path valid |
|---|---:|---:|---:|---:|
| 6jv8_A | 3/8 | 0/8 | 0/8 | 7/8 |
| 6gus_A | 3/8 | 0/8 | 0/8 | 8/8 |
| 6q9c_A | 1/8 | 1/8 | 0/8 | 7/8 |
| 6rrv_A | 2/8 | 0/8 | 0/8 | 7/8 |
| 6tly_A | 1/8 | 0/8 | 0/8 | 2/8 |
| 7aex_A | 0/8 | 0/8 | 0/8 | 7/8 |
| 7bwf_B | 0/8 | 0/8 | 0/8 | 7/8 |
| 7rm7_A | 0/8 | 0/8 | 0/8 | 8/8 |
| 7s86_A | 0/8 | 0/8 | 0/8 | 7/8 |

`7p46_A` K=8은 queue 전환 과정에서 완료 전 중단되어 raw result가 없다.

## 7. 실행 설정 registry

아래 값은 각 run의 `resolved_config.yaml`과 `metrics.json`에서 다시 읽은 effective setting이다. `p`는 inner checkpoint reverse progress이고, ESS는 outer resampling threshold fraction이다.

| Experiment group | Case | H | Lag (×10 ps) | Reverse | p | ESS | Tasks | Methods/allocations | Seeds | Cells |
|---|---|---:|---:|---:|---|---:|---|---|---|---:|
| 6J56 Phase C: horizon-8 factorial | 6j56_A | 8 | 256 | 200 | 0.75 | 1.0 | endpoint,ordered,windowed | best_of_budget K16M1, complete_nested K4M4, duet K4M4, frozen K16M1, inner_only K1M16, naive_dual K4M4, outer_only K16M1 | 20260831,20260901,20260902 | 63 |
| 6J56 Phase D: budget-16 K/M grid | 6j56_A | 8 | 256 | 200 | 0.75 | 1.0 | ordered,windowed | duet K16M1, duet K1M16, duet K2M8, duet K4M4, duet K8M2 | 20260831,20260901,20260902 | 30 |
| 6J56 protocol-v2: main_ordered | 6j56_A | 16 | 256 | 200 | 0.75 | 0.5 | ordered | complete_nested K8M4, duet K8M4, frozen K32M1, inner_only K1M32, outer_only K32M1 | 20260911 | 5 |
| 6J56 protocol-v2: pilot_ordered | 6j56_A | 16 | 128 | 200 | 0.75 | 0.5 | ordered | duet K4M2 | 20260901 | 1 |
| 6J56 protocol-v2: pilot_ordered_s256 | 6j56_A | 16 | 256 | 200 | 0.75 | 0.5 | ordered | duet K4M2 | 20260902 | 1 |
| 6J56 protocol-v2: pilot_windowed | 6j56_A | 16 | 128 | 200 | 0.75 | 0.5 | windowed | duet K4M2 | 20260901 | 1 |
| 6J56 protocol-v2: pilot_windowed_s256 | 6j56_A | 16 | 256 | 200 | 0.75 | 0.5 | windowed | duet K4M2 | 20260902 | 1 |
| 6J56 protocol-v3: allocation_pilot | 6j56_A | 24 | 256 | 200 | 0.9 | 0.25 | ordered | complete_nested K4M8, duet K4M8 | 20260914,20260915 | 4 |
| 6J56 protocol-v3: main_ordered | 6j56_A | 24 | 256 | 200 | 0.9 | 0.25 | ordered | complete_nested K8M4, duet K8M4, frozen K32M1, inner_only K1M32, outer_only K32M1 | 20260913,20260914 | 7 |
| 6J56 protocol-v3: pilot_ordered | 6j56_A | 24 | 256 | 200 | 0.9 | 0.25 | ordered | duet K4M2 | 20260911,20260912 | 2 |
| 6J56 protocol-v3: pilot_ordered_fixed | 6j56_A | 24 | 256 | 200 | 0.9 | 0.25 | ordered | duet K4M2 | 20260911,20260912 | 2 |
| 6J56 protocol-v3: pilot_ordered_q85 | 6j56_A | 24 | 256 | 200 | 0.9 | 0.25 | ordered | duet K4M2 | 20260911,20260912 | 2 |
| 7LP1 initial budget-16 K/M grid | 7lp1_A | 8 | 256 | 200 | 0.75 | 1.0 | ordered,windowed | duet K16M1, duet K1M16, duet K2M8, duet K4M4, duet K8M2 | 20260831,20260901,20260902 | 30 |
| 7LP1 initial horizon-8 factorial | 7lp1_A | 8 | 256 | 200 | 0.75 | 1.0 | endpoint,ordered,windowed | best_of_budget K16M1, complete_nested K4M4, duet K4M4, frozen K16M1, inner_only K1M16, naive_dual K4M4, outer_only K16M1 | 20260831,20260901,20260902 | 63 |
| 7LP1-v2 Frozen reachability gate | 7lp1_A | 24 | 256 | 200 | 0.9 | 0.5 | ordered | frozen K48M1 | 20260916,20260917 | 2 |
| 7LP1-v2 allocation: K16_M4_p95 | 7lp1_A | 24 | 256 | 200 | 0.95 | 0.5 | ordered | complete_nested K16M4, duet K16M4 | 20260920,20260921,20260922,20260923,20260924 | 10 |
| 7LP1-v2 allocation: K4_M16_p95 | 7lp1_A | 24 | 256 | 200 | 0.95 | 0.5 | ordered | complete_nested K4M16, duet K4M16 | 20260920,20260921,20260922,20260923,20260924 | 10 |
| 7LP1-v2 allocation: K4_M8_p95 | 7lp1_A | 24 | 256 | 200 | 0.95 | 0.5 | ordered | complete_nested K4M8, duet K4M8 | 20260920,20260921,20260922,20260923,20260924 | 10 |
| 7LP1-v2 allocation: K8_M4_p95 | 7lp1_A | 24 | 256 | 200 | 0.95 | 0.5 | ordered | complete_nested K8M4, duet K8M4 | 20260920,20260921,20260922,20260923,20260924 | 10 |
| 7LP1-v2 allocation: K8_M8_p95 | 7lp1_A | 24 | 256 | 200 | 0.95 | 0.5 | ordered | complete_nested K8M8, duet K8M8 | 20260920,20260921,20260922,20260923,20260924 | 10 |
| 7LP1-v2 multi-checkpoint development | 7lp1_A | 24 | 256 | 200 | [0.85,0.95] | 0.5 | ordered | duet K16M4 | 20260920,20260921,20260922,20260923 | 4 |
| 7LP1-v2 multi-checkpoint fresh confirmation | 7lp1_A | 24 | 256 | 200 | 0.95,[0.85,0.95] | 0.5 | ordered | complete_nested K16M4, duet K16M4 | 20260925,20260926 | 4 |
| ABL1 DFG-flip budget-8 support gate | abl1_dfg_flip | 24 | 256 | 200 | [0.85,0.95] | 0.5 | endpoint | complete_nested K4M2, duet K4M2, frozen K8M1 | 20260930 | 3 |
| ATLAS 10-protein route-v3.1 main seed-1 | 6jv8_A,7bwf_B | 24 | 20,42 | 200 | 0.95 | 0.5 | endpoint,ordered | complete_nested K16M4, duet K16M4, frozen K64M1, inner_only K1M64, outer_only K64M1 | 20261001 | 15 |

Potential coefficient signature는 `(lambda_program, distance_weight, remaining_event_weight, deadline_weight, failure_penalty, terminal_failure_penalty, failure_guidance_weight, distance_scale, potential_floor)` 순서다.

| Experiment group | Potential coefficient signatures | Decoder microbatch | Catalogs |
|---|---|---|---|
| 6J56 Phase C: horizon-8 factorial | `(1.0, 1.0, 0.1, 1.0, 20.0, 20.0, 0.0, 0.25, 1e-30)` | — | `data/duet/6j56_A/duet_programs.yaml` |
| 6J56 Phase D: budget-16 K/M grid | `(1.0, 1.0, 0.1, 1.0, 20.0, 20.0, 0.0, 0.25, 1e-30)` | — | `data/duet/6j56_A/duet_programs.yaml` |
| 6J56 protocol-v2: main_ordered | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v2/duet_programs.yaml` |
| 6J56 protocol-v2: pilot_ordered | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v2/duet_programs.yaml` |
| 6J56 protocol-v2: pilot_ordered_s256 | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v2/duet_programs.yaml` |
| 6J56 protocol-v2: pilot_windowed | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v2/duet_programs.yaml` |
| 6J56 protocol-v2: pilot_windowed_s256 | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v2/duet_programs.yaml` |
| 6J56 protocol-v3: allocation_pilot | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v3/duet_programs_fixed.yaml` |
| 6J56 protocol-v3: main_ordered | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v3/duet_programs_fixed.yaml` |
| 6J56 protocol-v3: pilot_ordered | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v3/duet_programs.yaml` |
| 6J56 protocol-v3: pilot_ordered_fixed | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v3/duet_programs_fixed.yaml` |
| 6J56 protocol-v3: pilot_ordered_q85 | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/6j56_A_v3/duet_programs_fixed.yaml` |
| 7LP1 initial budget-16 K/M grid | `(1.0, 1.0, 0.1, 1.0, 20.0, 20.0, 0.0, 0.25, 1e-30)` | — | `data/duet/7lp1_A/duet_programs.yaml` |
| 7LP1 initial horizon-8 factorial | `(1.0, 1.0, 0.1, 1.0, 20.0, 20.0, 0.0, 0.25, 1e-30)` | — | `data/duet/7lp1_A/duet_programs.yaml` |
| 7LP1-v2 Frozen reachability gate | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/7lp1_A_v2/duet_programs.yaml` |
| 7LP1-v2 allocation: K16_M4_p95 | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/7lp1_A_v2/duet_programs.yaml` |
| 7LP1-v2 allocation: K4_M16_p95 | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/7lp1_A_v2/duet_programs.yaml` |
| 7LP1-v2 allocation: K4_M8_p95 | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/7lp1_A_v2/duet_programs.yaml` |
| 7LP1-v2 allocation: K8_M4_p95 | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/7lp1_A_v2/duet_programs.yaml` |
| 7LP1-v2 allocation: K8_M8_p95 | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/7lp1_A_v2/duet_programs.yaml` |
| 7LP1-v2 multi-checkpoint development | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/7lp1_A_v2/duet_programs.yaml` |
| 7LP1-v2 multi-checkpoint fresh confirmation | `(0.5, 1.0, 0.05, 0.5, 4.0, 4.0, 1.0, 0.5, 1e-30)` | — | `data/duet/7lp1_A_v2/duet_programs.yaml` |
| ABL1 DFG-flip budget-8 support gate | `(0.5, 1.0, 0.0, 0.0, 4.0, 4.0, 1.0, 1.0, 1e-30)` | — | `data/duet/case_studies/abl1_dfg_flip/prepared/endpoint_program.yaml` |
| ATLAS 10-protein route-v3.1 main seed-1 | `(1.0, 1.0, 0.0, 1.0, 25.0, 25.0, 1.0, 1.0, 1e-30)` | — | `/workspace/sejin/AI_MD_NSMC_v31_stage/data/duet/protein_benchmark/route_v3_1/6jv8_A/duet_programs.yaml, /workspace/sejin/AI_MD_NSMC_v31_stage/data/duet/protein_benchmark/route_v3_1/7bwf_B/duet_programs.yaml` |

## 8. 묶음형 raw 결과

### 8.1 Phase A exact two-clock benchmark

상태 6개, inner latent 3개, horizon 5; 전체 wall time `141.2821 s`; gate pass=`True`.

| Task | Method | K×M | Repetitions | Samples | Final TV | Route probability | Program success | Normalizer rel. bias | Normalizer variance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| terminal | frozen | 16×1 | 200 | 3,200 | 0.3457 | 0.1625 | 0.0756 | — | — |
| terminal | outer_only | 16×1 | 200 | 3,200 | 0.0658 | 0.2244 | 0.3191 | 0.0221 | 0.0063 |
| terminal | inner_only | 1×16 | 200 | 200 | 0.1317 | 0.1800 | 0.2800 | — | — |
| terminal | naive_dual | 4×4 | 200 | 800 | 0.0902 | 0.1725 | 0.4750 | 0.6579 | 0.0398 |
| terminal | complete_nested | 4×4 | 200 | 800 | 0.0686 | 0.2037 | 0.3162 | -0.0027 | 0.0216 |
| terminal | duet | 4×4 | 200 | 800 | 0.0679 | 0.1938 | 0.3212 | -0.0497 | 0.0260 |
| windowed | frozen | 16×1 | 200 | 3,200 | 0.1004 | 0.1837 | 0.0194 | — | — |
| windowed | outer_only | 16×1 | 200 | 3,200 | 0.0225 | 0.1928 | 0.0853 | -0.0001 | 0.0018 |
| windowed | inner_only | 1×16 | 200 | 200 | 0.0331 | 0.2600 | 0.1000 | — | — |
| windowed | naive_dual | 4×4 | 200 | 800 | 0.0611 | 0.1950 | 0.1300 | 0.1804 | 0.0100 |
| windowed | complete_nested | 4×4 | 200 | 800 | 0.0467 | 0.2387 | 0.0712 | -0.0047 | 0.0049 |
| windowed | duet | 4×4 | 200 | 800 | 0.0496 | 0.1913 | 0.0975 | 0.0003 | 0.0057 |
| ordered | frozen | 16×1 | 200 | 3,200 | 0.0439 | 0.1781 | 0.0122 | — | — |
| ordered | outer_only | 16×1 | 200 | 3,200 | 0.0374 | 0.1609 | 0.0553 | 0.0209 | 0.0015 |
| ordered | inner_only | 1×16 | 200 | 200 | 0.0539 | 0.1750 | 0.0500 | — | — |
| ordered | naive_dual | 4×4 | 200 | 800 | 0.0845 | 0.2025 | 0.1087 | 0.1468 | 0.0161 |
| ordered | complete_nested | 4×4 | 200 | 800 | 0.0276 | 0.1613 | 0.0413 | -0.0127 | 0.0123 |
| ordered | duet | 4×4 | 200 | 800 | 0.0728 | 0.1512 | 0.0488 | -0.0397 | 0.0122 |

Exact-toy ordered convergence raw values:

| Method | Scale K=M | Repetitions | Final TV | Route probability | Program success | Normalizer rel. bias | Variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| complete_nested | 2 | 100 | 0.0624 | 0.2000 | 0.0350 | -0.0241 | 0.0115 |
| duet | 2 | 100 | 0.0389 | 0.1550 | 0.0400 | -0.0138 | 0.0178 |
| naive_dual | 2 | 100 | 0.0711 | 0.1700 | 0.0650 | 0.1101 | 0.0125 |
| complete_nested | 4 | 100 | 0.0779 | 0.1825 | 0.0425 | 0.0423 | 0.0155 |
| duet | 4 | 100 | 0.0729 | 0.2350 | 0.0650 | 0.0721 | 0.0176 |
| naive_dual | 4 | 100 | 0.0661 | 0.2500 | 0.0875 | 0.1199 | 0.0084 |
| complete_nested | 8 | 100 | 0.0461 | 0.1675 | 0.0537 | 0.0071 | 0.0057 |
| duet | 8 | 100 | 0.0272 | 0.1588 | 0.0512 | 0.0111 | 0.0063 |
| naive_dual | 8 | 100 | 0.1068 | 0.2387 | 0.1562 | 0.1431 | 0.0051 |
| complete_nested | 16 | 100 | 0.0240 | 0.1613 | 0.0537 | -0.0357 | 0.0022 |
| duet | 16 | 100 | 0.0261 | 0.1512 | 0.0537 | 0.0135 | 0.0033 |
| naive_dual | 16 | 100 | 0.1004 | 0.2325 | 0.1613 | 0.1570 | 0.0021 |

DuET proper-weighting raw values:

| M | Repetitions | Exact local Z | Mean Zhat | Rel. bias | Variance | L1 proper-weighting residual |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,000 | 0.7299 | 0.7327 | 0.0039 | 0.0381 | 0.0324 |
| 2 | 3,000 | 0.7299 | 0.7297 | -0.0002 | 0.0205 | 0.0179 |
| 4 | 3,000 | 0.7299 | 0.7287 | -0.0016 | 0.0108 | 0.0244 |
| 8 | 3,000 | 0.7299 | 0.7291 | -0.0011 | 0.0053 | 0.0171 |
| 16 | 3,000 | 0.7299 | 0.7294 | -0.0007 | 0.0029 | 0.0341 |

### 8.2 `extra_proteins/7lp1_A/preflight`

공통 preflight setting: horizon 8, physical lag 256×10 ps, reverse steps 200, checkpoint p=.75, 16 replicates. Phase-B 파일은 6j56_A, `extra_proteins/7lp1_A/preflight`는 7lp1_A다.

| Variant | n | Mode | Checkpoint hook | M | Nonfinite | Geometry valid | Clash | Mean max adjacent CA (Å) | PC1 W1 vs uncond. | Decoder NFE | Wall min |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_ode | 16 | ode | 0 | 1 | 0 | 1.0000 | 0 | 4.6479 | 0.2570 | 25,600 | 18.1383 |
| unconditioned_sde | 16 | sde | 0 | 1 | 0 | 1.0000 | 0 | 4.8488 | 0 | 25,472 | 13.6973 |
| sde_checkpoint_no_resampling | 16 | sde | 1 | 1 | 0 | 1.0000 | 0 | 4.7848 | 0.2322 | 25,472 | 13.9205 |
| sde_constant_resampling | 16 | sde | 1 | 16 | 0 | 1.0000 | 0 | 4.6658 | 0.2566 | 407,552 | 31.3743 |

### 8.3 `phase_b_confrover_preflight`

공통 preflight setting: horizon 8, physical lag 256×10 ps, reverse steps 200, checkpoint p=.75, 16 replicates. Phase-B 파일은 6j56_A, `extra_proteins/7lp1_A/preflight`는 7lp1_A다.

| Variant | n | Mode | Checkpoint hook | M | Nonfinite | Geometry valid | Clash | Mean max adjacent CA (Å) | PC1 W1 vs uncond. | Decoder NFE | Wall min |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_ode | 16 | ode | 0 | 1 | 0 | 1.0000 | 0 | 4.7152 | 0.1311 | 25,600 | 10.8003 |
| unconditioned_sde | 16 | sde | 0 | 1 | 0 | 1.0000 | 0 | 4.9043 | 0 | 25,472 | 12.3013 |
| sde_checkpoint_no_resampling | 16 | sde | 1 | 1 | 0 | 1.0000 | 0 | 4.9385 | 0.2027 | 25,472 | 12.5374 |
| sde_constant_resampling | 16 | sde | 1 | 16 | 0 | 1.0000 | 0 | 4.8906 | 0.3139 | 407,552 | 56.6453 |

### 8.4 `phase_b_confrover_preflight_pre_seedfix_20260831`

공통 preflight setting: horizon 8, physical lag 256×10 ps, reverse steps 200, checkpoint p=.75, 16 replicates. Phase-B 파일은 6j56_A, `extra_proteins/7lp1_A/preflight`는 7lp1_A다.

| Variant | n | Mode | Checkpoint hook | M | Nonfinite | Geometry valid | Clash | Mean max adjacent CA (Å) | PC1 W1 vs uncond. | Decoder NFE | Wall min |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_ode | 16 | ode | 0 | 1 | 0 | 0.0625 | 0 | — | 0.2153 | 25,600 | 10.7715 |
| unconditioned_sde | 16 | sde | 0 | 1 | 0 | 0 | 0 | 4.9760 | 0 | 25,472 | 11.8017 |
| sde_checkpoint_no_resampling | 16 | sde | 1 | 1 | 0 | 0 | 0 | 4.9875 | 0.1961 | 25,472 | 12.2357 |
| sde_constant_resampling | 16 | sde | 1 | 16 | 0 | 0 | 0 | 4.9360 | 0.2152 | 407,552 | 55.9583 |

## 9. Seed/method별 raw 결과

`Pre-U`는 final outer resampling 직전의 unique success rate, `Pre-W`는 같은 시점의 success probability mass, `Post`는 저장된 final output의 성공률이다. `S0/1/2/3`은 final pre-resampling population의 progress-stage count다. `EndV`는 endpoint-frame structural validity, `PathV`는 whole-path validity다. 과거 schema에 `PathV`가 없으면 `—`다. `R`은 final step 이전 outer resampling 횟수다.

### 6J56 Phase C: horizon-8 factorial

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | endpoint | best_of_budget | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 2.0527 | 1 | 7 | 25,472 | 10.8090 |
| 6j56_A | endpoint | best_of_budget | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.5758 | 1 | 7 | 25,472 | 10.8438 |
| 6j56_A | endpoint | best_of_budget | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.4694 | 1 | 7 | 25,472 | 10.8583 |
| 6j56_A | endpoint | complete_nested | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.2932 | 1 | 7 | 25,472 | 4.7972 |
| 6j56_A | endpoint | complete_nested | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3856 | 1 | 7 | 25,472 | 4.8014 |
| 6j56_A | endpoint | complete_nested | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.4605 | 1 | 7 | 25,472 | 4.7772 |
| 6j56_A | endpoint | duet | 20260831 | 4×4 | 0.5000 | 1.0000 | 1.0000 | 2/2/0/0 | 1.0000 | — | — | 1.1526 | 1 | 7 | 25,472 | 4.8123 |
| 6j56_A | endpoint | duet | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.2521 | 1 | 7 | 25,472 | 4.8074 |
| 6j56_A | endpoint | duet | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.4503 | 1 | 7 | 25,472 | 4.8415 |
| 6j56_A | endpoint | frozen | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.8654 | 16 | 0 | 25,472 | 10.8612 |
| 6j56_A | endpoint | frozen | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.7577 | 16 | 0 | 25,472 | 10.8809 |
| 6j56_A | endpoint | frozen | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.7757 | 16 | 0 | 25,472 | 10.9707 |
| 6j56_A | endpoint | inner_only | 20260831 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.4176 | 1 | 0 | 25,472 | 3.0789 |
| 6j56_A | endpoint | inner_only | 20260901 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.2895 | 1 | 0 | 25,472 | 3.0700 |
| 6j56_A | endpoint | inner_only | 20260902 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.3597 | 1 | 0 | 25,472 | 3.0713 |
| 6j56_A | endpoint | naive_dual | 20260831 | 4×4 | 0.5000 | 1.0000 | 1.0000 | 2/2/0/0 | 1.0000 | — | — | 1.1526 | 1 | 7 | 25,472 | 4.8304 |
| 6j56_A | endpoint | naive_dual | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.2572 | 1 | 7 | 25,472 | 4.8234 |
| 6j56_A | endpoint | naive_dual | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.4720 | 1 | 7 | 25,472 | 4.7863 |
| 6j56_A | endpoint | outer_only | 20260831 | 16×1 | 0.0625 | 1.0000 | 1.0000 | 15/1/0/0 | 1.0000 | — | — | 1.6397 | 1 | 7 | 25,472 | 10.9370 |
| 6j56_A | endpoint | outer_only | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 0.9375 | — | — | 1.8511 | 12 | 7 | 25,472 | 10.9877 |
| 6j56_A | endpoint | outer_only | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.2725 | 1 | 7 | 25,472 | 10.8671 |
| 6j56_A | ordered | best_of_budget | 20260831 | 16×1 | 0 | 0 | 0 | 13/0/3/0 | 1.0000 | — | — | 2.0527 | 1 | 7 | 25,472 | 10.8991 |
| 6j56_A | ordered | best_of_budget | 20260901 | 16×1 | 0 | 0 | 0 | 12/0/4/0 | 1.0000 | — | — | 1.7858 | 1 | 7 | 25,472 | 10.8876 |
| 6j56_A | ordered | best_of_budget | 20260902 | 16×1 | 0 | 0 | 0 | 13/0/3/0 | 1.0000 | — | — | 1.4694 | 1 | 7 | 25,472 | 10.8798 |
| 6j56_A | ordered | complete_nested | 20260831 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | 1.4711 | 1 | 7 | 25,472 | 4.8330 |
| 6j56_A | ordered | complete_nested | 20260901 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | 1.5161 | 1 | 7 | 25,472 | 4.9019 |
| 6j56_A | ordered | complete_nested | 20260902 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | 1.6203 | 1 | 7 | 25,472 | 4.8507 |
| 6j56_A | ordered | duet | 20260831 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | 1.6722 | 1 | 7 | 25,472 | 4.8759 |
| 6j56_A | ordered | duet | 20260901 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | 1.4747 | 2 | 7 | 25,472 | 4.7653 |
| 6j56_A | ordered | duet | 20260902 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 0.7500 | — | — | 2.4849 | 1 | 7 | 25,472 | 4.8295 |
| 6j56_A | ordered | frozen | 20260831 | 16×1 | 0 | 0 | 0 | 13/0/3/0 | 1.0000 | — | — | 1.8654 | 16 | 0 | 25,472 | 11.0100 |
| 6j56_A | ordered | frozen | 20260901 | 16×1 | 0 | 0 | 0 | 12/0/4/0 | 1.0000 | — | — | 1.7577 | 16 | 0 | 25,472 | 10.9493 |
| 6j56_A | ordered | frozen | 20260902 | 16×1 | 0 | 0 | 0 | 13/0/3/0 | 1.0000 | — | — | 1.7757 | 16 | 0 | 25,472 | 10.9450 |
| 6j56_A | ordered | inner_only | 20260831 | 1×16 | 0 | 0 | 0 | 0/1/0/0 | 1.0000 | — | — | 1.9653 | 1 | 0 | 25,472 | 3.0975 |
| 6j56_A | ordered | inner_only | 20260901 | 1×16 | 0 | 0 | 0 | 0/1/0/0 | 1.0000 | — | — | 1.5991 | 1 | 0 | 25,472 | 3.1530 |
| 6j56_A | ordered | inner_only | 20260902 | 1×16 | 0 | 0 | 0 | 0/0/1/0 | 1.0000 | — | — | 1.2732 | 1 | 0 | 25,472 | 3.1059 |
| 6j56_A | ordered | naive_dual | 20260831 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | 1.5737 | 2 | 7 | 25,472 | 4.8751 |
| 6j56_A | ordered | naive_dual | 20260901 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | 1.4241 | 3 | 7 | 25,472 | 4.9129 |
| 6j56_A | ordered | naive_dual | 20260902 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 0.7500 | — | — | 2.6256 | 1 | 7 | 25,472 | 4.9512 |
| 6j56_A | ordered | outer_only | 20260831 | 16×1 | 0 | 0 | 0 | 0/0/16/0 | 1.0000 | — | — | 1.9165 | 1 | 7 | 25,472 | 10.9163 |
| 6j56_A | ordered | outer_only | 20260901 | 16×1 | 0 | 0 | 0 | 0/16/0/0 | 1.0000 | — | — | 1.9172 | 1 | 7 | 25,472 | 10.8899 |
| 6j56_A | ordered | outer_only | 20260902 | 16×1 | 0 | 0 | 0 | 0/16/0/0 | 1.0000 | — | — | 1.5219 | 1 | 7 | 25,472 | 11.1598 |
| 6j56_A | windowed | best_of_budget | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 2.0527 | 1 | 7 | 25,472 | 10.8890 |
| 6j56_A | windowed | best_of_budget | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.7858 | 1 | 7 | 25,472 | 10.9808 |
| 6j56_A | windowed | best_of_budget | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.4694 | 1 | 7 | 25,472 | 10.8681 |
| 6j56_A | windowed | complete_nested | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.4632 | 2 | 7 | 25,472 | 4.7978 |
| 6j56_A | windowed | complete_nested | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.2736 | 1 | 7 | 25,472 | 4.8262 |
| 6j56_A | windowed | complete_nested | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.0296 | 1 | 7 | 25,472 | 4.7976 |
| 6j56_A | windowed | duet | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3618 | 1 | 7 | 25,472 | 4.7979 |
| 6j56_A | windowed | duet | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.2198 | 1 | 7 | 25,472 | 4.9102 |
| 6j56_A | windowed | duet | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.0925 | 1 | 7 | 25,472 | 4.9317 |
| 6j56_A | windowed | frozen | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.8654 | 16 | 0 | 25,472 | 10.8886 |
| 6j56_A | windowed | frozen | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.7577 | 16 | 0 | 25,472 | 10.8537 |
| 6j56_A | windowed | frozen | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.7757 | 16 | 0 | 25,472 | 10.9559 |
| 6j56_A | windowed | inner_only | 20260831 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.0625 | 1 | 0 | 25,472 | 3.0674 |
| 6j56_A | windowed | inner_only | 20260901 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.0986 | 1 | 0 | 25,472 | 3.0661 |
| 6j56_A | windowed | inner_only | 20260902 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.2007 | 1 | 0 | 25,472 | 3.0555 |
| 6j56_A | windowed | naive_dual | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3148 | 2 | 7 | 25,472 | 4.8326 |
| 6j56_A | windowed | naive_dual | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3953 | 1 | 7 | 25,472 | 4.8053 |
| 6j56_A | windowed | naive_dual | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.1313 | 1 | 7 | 25,472 | 4.8008 |
| 6j56_A | windowed | outer_only | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.1281 | 1 | 7 | 25,472 | 10.8579 |
| 6j56_A | windowed | outer_only | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 0.9375 | — | — | 1.3915 | 1 | 7 | 25,472 | 10.8732 |
| 6j56_A | windowed | outer_only | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.2280 | 1 | 7 | 25,472 | 10.9660 |

### 6J56 Phase D: budget-16 K/M grid

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | ordered | duet | 20260831 | 16×1 | 0 | 0 | 0 | 0/16/0/0 | 0.9375 | — | — | — | 1 | 7 | 25,472 | 11.8784 |
| 6j56_A | ordered | duet | 20260831 | 1×16 | 0 | 0 | 0 | 0/1/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 3.5017 |
| 6j56_A | ordered | duet | 20260831 | 2×8 | 0 | 0 | 0 | 0/2/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 4.1584 |
| 6j56_A | ordered | duet | 20260831 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 5.1866 |
| 6j56_A | ordered | duet | 20260831 | 8×2 | 0 | 0 | 0 | 0/8/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 7.1406 |
| 6j56_A | ordered | duet | 20260901 | 16×1 | 0 | 0 | 0 | 0/16/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 11.8436 |
| 6j56_A | ordered | duet | 20260901 | 1×16 | 0 | 0 | 0 | 0/1/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 3.5062 |
| 6j56_A | ordered | duet | 20260901 | 2×8 | 0 | 0 | 0 | 0/2/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 4.0848 |
| 6j56_A | ordered | duet | 20260901 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | — | 2 | 7 | 25,472 | 5.1802 |
| 6j56_A | ordered | duet | 20260901 | 8×2 | 0 | 0 | 0 | 0/8/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 7.2090 |
| 6j56_A | ordered | duet | 20260902 | 16×1 | 0 | 0 | 0 | 0/16/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 11.8202 |
| 6j56_A | ordered | duet | 20260902 | 1×16 | 0 | 0 | 0 | 0/0/1/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 3.4965 |
| 6j56_A | ordered | duet | 20260902 | 2×8 | 0 | 0 | 0 | 0/2/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 4.0902 |
| 6j56_A | ordered | duet | 20260902 | 4×4 | 0 | 0 | 0 | 0/4/0/0 | 0.7500 | — | — | — | 1 | 7 | 25,472 | 5.2181 |
| 6j56_A | ordered | duet | 20260902 | 8×2 | 0 | 0 | 0 | 0/8/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 7.2276 |
| 6j56_A | windowed | duet | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 11.9529 |
| 6j56_A | windowed | duet | 20260831 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 3.5290 |
| 6j56_A | windowed | duet | 20260831 | 2×8 | 0 | 0 | 0 | 2/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 4.1042 |
| 6j56_A | windowed | duet | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 5.2717 |
| 6j56_A | windowed | duet | 20260831 | 8×2 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 7.0883 |
| 6j56_A | windowed | duet | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 11.7999 |
| 6j56_A | windowed | duet | 20260901 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 3.5000 |
| 6j56_A | windowed | duet | 20260901 | 2×8 | 0 | 0 | 0 | 2/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 4.0805 |
| 6j56_A | windowed | duet | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 5.1754 |
| 6j56_A | windowed | duet | 20260901 | 8×2 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | — | 2 | 7 | 25,472 | 7.0932 |
| 6j56_A | windowed | duet | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | — | 2 | 7 | 25,472 | 12.1682 |
| 6j56_A | windowed | duet | 20260902 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 3.5658 |
| 6j56_A | windowed | duet | 20260902 | 2×8 | 0 | 0 | 0 | 2/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 4.0742 |
| 6j56_A | windowed | duet | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 5.2067 |
| 6j56_A | windowed | duet | 20260902 | 8×2 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | — | 1 | 7 | 25,472 | 7.1184 |

### 6J56 protocol-v2: main_ordered

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | ordered | complete_nested | 20260911 | 8×4 | 0 | 0 | 0 | 0/0/8/0 | 1.0000 | — | — | 1.1760 | 2 | 3 | 101,888 | 17.5011 |
| 6j56_A | ordered | duet | 20260911 | 8×4 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | 1.2741 | 1 | 3 | 101,888 | 17.5626 |
| 6j56_A | ordered | frozen | 20260911 | 32×1 | 0 | 0 | 0 | 30/0/2/0 | 0.9688 | — | — | 1.7520 | 32 | 0 | 101,888 | 37.4929 |
| 6j56_A | ordered | inner_only | 20260911 | 1×32 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.0903 | 1 | 0 | 101,888 | 12.4838 |
| 6j56_A | ordered | outer_only | 20260911 | 32×1 | 0 | 0 | 0 | 25/0/7/0 | 1.0000 | — | — | 1.7654 | 4 | 6 | 101,888 | 40.7904 |

### 6J56 protocol-v2: pilot_ordered

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | ordered | duet | 20260901 | 4×2 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.9649 | 1 | 2 | 25,472 | 7.2052 |

### 6J56 protocol-v2: pilot_ordered_s256

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | ordered | duet | 20260902 | 4×2 | 0 | 0 | 0 | 0/0/4/0 | 1.0000 | — | — | 1.2931 | 1 | 3 | 25,472 | 6.0239 |

### 6J56 protocol-v2: pilot_windowed

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | windowed | duet | 20260901 | 4×2 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.7382 | 1 | 3 | 25,472 | 6.8559 |

### 6J56 protocol-v2: pilot_windowed_s256

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | windowed | duet | 20260902 | 4×2 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.4249 | 1 | 5 | 25,472 | 5.2825 |

### 6J56 protocol-v3: allocation_pilot

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | ordered | complete_nested | 20260914 | 4×8 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.4521 | 3 | 0 | 152,832 | 23.3746 |
| 6j56_A | ordered | complete_nested | 20260915 | 4×8 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.8054 | 2 | 0 | 152,832 | 23.5189 |
| 6j56_A | ordered | duet | 20260914 | 4×8 | 0.2500 | 0.5251 | 0.5000 | 3/0/0/1 | 1.0000 | — | — | 1.3196 | 3 | 0 | 152,832 | 25.0711 |
| 6j56_A | ordered | duet | 20260915 | 4×8 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.1470 | 3 | 0 | 152,832 | 24.9043 |

### 6J56 protocol-v3: main_ordered

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | ordered | complete_nested | 20260913 | 8×4 | 0.1250 | 0.4385 | 0.5000 | 6/1/0/1 | 1.0000 | — | — | 1.4480 | 5 | 0 | 152,832 | 26.6671 |
| 6j56_A | ordered | complete_nested | 20260914 | 8×4 | 0.1250 | 0.1947 | 0.1250 | 7/0/0/1 | 1.0000 | — | — | 1.3299 | 6 | 0 | 152,832 | 29.4547 |
| 6j56_A | ordered | duet | 20260913 | 8×4 | 0.1250 | 0.4167 | 0.3750 | 5/0/2/1 | 1.0000 | — | — | 1.4927 | 5 | 0 | 152,832 | 29.5186 |
| 6j56_A | ordered | duet | 20260914 | 8×4 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | 1.2982 | 5 | 0 | 152,832 | 26.7474 |
| 6j56_A | ordered | frozen | 20260913 | 32×1 | 0.0312 | 0.0312 | 0.0312 | 29/1/1/1 | 1.0000 | — | — | 1.3741 | 32 | 0 | 152,832 | 67.8535 |
| 6j56_A | ordered | inner_only | 20260913 | 1×32 | 1.0000 | 1.0000 | 1.0000 | 0/0/0/1 | 1.0000 | — | — | 1.8737 | 1 | 0 | 152,832 | 16.5278 |
| 6j56_A | ordered | outer_only | 20260913 | 32×1 | 0.0312 | 0.2623 | 0.2500 | 29/1/1/1 | 1.0000 | — | — | 1.4273 | 23 | 0 | 152,832 | 57.2383 |

### 6J56 protocol-v3: pilot_ordered

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | ordered | duet | 20260911 | 4×2 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.1204 | 2 | 0 | 38,208 | 8.2303 |
| 6j56_A | ordered | duet | 20260912 | 4×2 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.2712 | 2 | 0 | 38,208 | 11.1724 |

### 6J56 protocol-v3: pilot_ordered_fixed

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | ordered | duet | 20260911 | 4×2 | 0 | 0 | 0 | 3/0/1/0 | 1.0000 | — | — | 1.0989 | 1 | 0 | 38,208 | 8.0194 |
| 6j56_A | ordered | duet | 20260912 | 4×2 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.2422 | 1 | 0 | 38,208 | 10.9496 |

### 6J56 protocol-v3: pilot_ordered_q85

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6j56_A | ordered | duet | 20260911 | 4×2 | 0.2500 | 0.8540 | 1.0000 | 3/0/0/1 | 1.0000 | — | — | 1.3828 | 1 | 0 | 38,208 | 8.1760 |
| 6j56_A | ordered | duet | 20260912 | 4×2 | 0.2500 | 0.7052 | 0.7500 | 3/0/0/1 | 1.0000 | — | — | 1.0984 | 2 | 0 | 38,208 | 11.1927 |

### 7LP1 initial budget-16 K/M grid

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | ordered | duet | 20260831 | 16×1 | 0 | 0 | 1.0000 | 0/0/16/0 | 1.0000 | — | — | 1.1656 | 1 | 7 | 25,472 | 13.4805 |
| 7lp1_A | ordered | duet | 20260831 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.2300 | 1 | 7 | 25,472 | 1.9116 |
| 7lp1_A | ordered | duet | 20260831 | 2×8 | 0 | 0 | 1.0000 | 0/0/2/0 | 1.0000 | — | — | 1.2875 | 1 | 7 | 25,472 | 2.4873 |
| 7lp1_A | ordered | duet | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.5169 | 2 | 7 | 25,472 | 4.0015 |
| 7lp1_A | ordered | duet | 20260831 | 8×2 | 0 | 0 | 1.0000 | 0/0/8/0 | 1.0000 | — | — | 1.2661 | 1 | 7 | 25,472 | 7.2083 |
| 7lp1_A | ordered | duet | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.6324 | 7 | 7 | 25,472 | 13.4938 |
| 7lp1_A | ordered | duet | 20260901 | 1×16 | 0 | 0 | 1.0000 | 0/0/1/0 | 1.0000 | — | — | 1.3671 | 1 | 7 | 25,472 | 1.9567 |
| 7lp1_A | ordered | duet | 20260901 | 2×8 | 0 | 0 | 1.0000 | 0/0/2/0 | 1.0000 | — | — | 1.3617 | 1 | 7 | 25,472 | 2.5235 |
| 7lp1_A | ordered | duet | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.6253 | 3 | 7 | 25,472 | 4.0206 |
| 7lp1_A | ordered | duet | 20260901 | 8×2 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | 1.7217 | 4 | 7 | 25,472 | 7.1841 |
| 7lp1_A | ordered | duet | 20260902 | 16×1 | 0 | 0 | 1.0000 | 0/0/16/0 | 1.0000 | — | — | 1.7696 | 1 | 7 | 25,472 | 13.4749 |
| 7lp1_A | ordered | duet | 20260902 | 1×16 | 0 | 0 | 1.0000 | 0/0/1/0 | 1.0000 | — | — | 1.2971 | 1 | 7 | 25,472 | 1.9691 |
| 7lp1_A | ordered | duet | 20260902 | 2×8 | 0 | 0 | 1.0000 | 0/0/2/0 | 1.0000 | — | — | 1.3726 | 1 | 7 | 25,472 | 2.4604 |
| 7lp1_A | ordered | duet | 20260902 | 4×4 | 0 | 0 | 1.0000 | 0/0/4/0 | 1.0000 | — | — | 1.3238 | 1 | 7 | 25,472 | 4.0377 |
| 7lp1_A | ordered | duet | 20260902 | 8×2 | 0 | 0 | 1.0000 | 0/0/8/0 | 1.0000 | — | — | 1.1693 | 1 | 7 | 25,472 | 7.2779 |
| 7lp1_A | windowed | duet | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.4240 | 1 | 7 | 25,472 | 13.7175 |
| 7lp1_A | windowed | duet | 20260831 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.3704 | 1 | 7 | 25,472 | 1.9491 |
| 7lp1_A | windowed | duet | 20260831 | 2×8 | 0 | 0 | 0 | 2/0/0/0 | 1.0000 | — | — | 1.6920 | 1 | 7 | 25,472 | 2.4572 |
| 7lp1_A | windowed | duet | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3096 | 1 | 7 | 25,472 | 4.0511 |
| 7lp1_A | windowed | duet | 20260831 | 8×2 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | 1.6455 | 1 | 7 | 25,472 | 7.3200 |
| 7lp1_A | windowed | duet | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.4813 | 1 | 7 | 25,472 | 13.6882 |
| 7lp1_A | windowed | duet | 20260901 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.1189 | 1 | 7 | 25,472 | 1.9333 |
| 7lp1_A | windowed | duet | 20260901 | 2×8 | 0 | 0 | 0 | 2/0/0/0 | 1.0000 | — | — | 1.5035 | 1 | 7 | 25,472 | 2.4615 |
| 7lp1_A | windowed | duet | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3773 | 1 | 7 | 25,472 | 4.0489 |
| 7lp1_A | windowed | duet | 20260901 | 8×2 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | 1.4683 | 2 | 7 | 25,472 | 7.1641 |
| 7lp1_A | windowed | duet | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.3856 | 1 | 7 | 25,472 | 13.5703 |
| 7lp1_A | windowed | duet | 20260902 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.2564 | 1 | 7 | 25,472 | 1.9107 |
| 7lp1_A | windowed | duet | 20260902 | 2×8 | 0 | 0 | 0 | 2/0/0/0 | 1.0000 | — | — | 1.6034 | 1 | 7 | 25,472 | 2.4963 |
| 7lp1_A | windowed | duet | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.6129 | 1 | 7 | 25,472 | 4.0663 |
| 7lp1_A | windowed | duet | 20260902 | 8×2 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | 1.5087 | 1 | 7 | 25,472 | 7.2583 |

### 7LP1 initial horizon-8 factorial

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | endpoint | best_of_budget | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.5165 | 1 | 7 | 25,472 | 13.4134 |
| 7lp1_A | endpoint | best_of_budget | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.3750 | 1 | 7 | 25,472 | 13.4845 |
| 7lp1_A | endpoint | best_of_budget | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.4325 | 1 | 7 | 25,472 | 13.2137 |
| 7lp1_A | endpoint | complete_nested | 20260831 | 4×4 | 0.7500 | 1.0000 | 1.0000 | 1/3/0/0 | 1.0000 | — | — | 1.3003 | 1 | 7 | 25,472 | 3.9205 |
| 7lp1_A | endpoint | complete_nested | 20260901 | 4×4 | 0.7500 | 1.0000 | 1.0000 | 1/3/0/0 | 1.0000 | — | — | 1.4876 | 1 | 7 | 25,472 | 3.8829 |
| 7lp1_A | endpoint | complete_nested | 20260902 | 4×4 | 0.7500 | 1.0000 | 1.0000 | 1/3/0/0 | 1.0000 | — | — | 1.3686 | 1 | 7 | 25,472 | 4.0079 |
| 7lp1_A | endpoint | duet | 20260831 | 4×4 | 0.7500 | 1.0000 | 1.0000 | 1/3/0/0 | 1.0000 | — | — | 1.1981 | 1 | 7 | 25,472 | 3.8734 |
| 7lp1_A | endpoint | duet | 20260901 | 4×4 | 1.0000 | 1.0000 | 1.0000 | 0/4/0/0 | 1.0000 | — | — | 1.1354 | 1 | 7 | 25,472 | 3.8585 |
| 7lp1_A | endpoint | duet | 20260902 | 4×4 | 1.0000 | 1.0000 | 1.0000 | 0/4/0/0 | 1.0000 | — | — | 1.4814 | 1 | 7 | 25,472 | 4.0244 |
| 7lp1_A | endpoint | frozen | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.6772 | 16 | 0 | 25,472 | 13.3522 |
| 7lp1_A | endpoint | frozen | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.6890 | 16 | 0 | 25,472 | 13.3631 |
| 7lp1_A | endpoint | frozen | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.6512 | 16 | 0 | 25,472 | 13.1790 |
| 7lp1_A | endpoint | inner_only | 20260831 | 1×16 | 1.0000 | 1.0000 | 1.0000 | 0/1/0/0 | 1.0000 | — | — | 1.5876 | 1 | 0 | 25,472 | 1.9095 |
| 7lp1_A | endpoint | inner_only | 20260901 | 1×16 | 1.0000 | 1.0000 | 1.0000 | 0/1/0/0 | 1.0000 | — | — | 1.6636 | 1 | 0 | 25,472 | 1.9176 |
| 7lp1_A | endpoint | inner_only | 20260902 | 1×16 | 1.0000 | 1.0000 | 1.0000 | 0/1/0/0 | 1.0000 | — | — | 1.4443 | 1 | 0 | 25,472 | 1.9160 |
| 7lp1_A | endpoint | naive_dual | 20260831 | 4×4 | 0.7500 | 1.0000 | 1.0000 | 1/3/0/0 | 1.0000 | — | — | 1.6301 | 1 | 7 | 25,472 | 3.9430 |
| 7lp1_A | endpoint | naive_dual | 20260901 | 4×4 | 1.0000 | 1.0000 | 1.0000 | 0/4/0/0 | 1.0000 | — | — | 1.5425 | 1 | 7 | 25,472 | 3.9766 |
| 7lp1_A | endpoint | naive_dual | 20260902 | 4×4 | 1.0000 | 1.0000 | 1.0000 | 0/4/0/0 | 1.0000 | — | — | 1.3499 | 1 | 7 | 25,472 | 3.9450 |
| 7lp1_A | endpoint | outer_only | 20260831 | 16×1 | 0.5625 | 1.0000 | 1.0000 | 7/9/0/0 | 1.0000 | — | — | 1.7255 | 1 | 7 | 25,472 | 13.3744 |
| 7lp1_A | endpoint | outer_only | 20260901 | 16×1 | 0.3750 | 1.0000 | 1.0000 | 10/6/0/0 | 1.0000 | — | — | 1.3060 | 1 | 7 | 25,472 | 13.3460 |
| 7lp1_A | endpoint | outer_only | 20260902 | 16×1 | 0.2500 | 1.0000 | 1.0000 | 12/4/0/0 | 1.0000 | — | — | 1.4113 | 1 | 7 | 25,472 | 13.4098 |
| 7lp1_A | ordered | best_of_budget | 20260831 | 16×1 | 0 | 0 | 0 | 13/0/3/0 | 1.0000 | — | — | 1.6291 | 1 | 7 | 25,472 | 13.3616 |
| 7lp1_A | ordered | best_of_budget | 20260901 | 16×1 | 0 | 0 | 0 | 14/0/2/0 | 1.0000 | — | — | 1.7082 | 1 | 7 | 25,472 | 13.5157 |
| 7lp1_A | ordered | best_of_budget | 20260902 | 16×1 | 0 | 0 | 0 | 13/2/1/0 | 1.0000 | — | — | 1.5881 | 1 | 7 | 25,472 | 13.6219 |
| 7lp1_A | ordered | complete_nested | 20260831 | 4×4 | 0 | 0 | 1.0000 | 0/0/4/0 | 1.0000 | — | — | 1.4206 | 1 | 7 | 25,472 | 4.0060 |
| 7lp1_A | ordered | complete_nested | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.6054 | 3 | 7 | 25,472 | 3.9674 |
| 7lp1_A | ordered | complete_nested | 20260902 | 4×4 | 0 | 0 | 1.0000 | 0/0/4/0 | 1.0000 | — | — | 1.5912 | 1 | 7 | 25,472 | 4.0106 |
| 7lp1_A | ordered | duet | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.5169 | 2 | 7 | 25,472 | 3.9590 |
| 7lp1_A | ordered | duet | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.6253 | 3 | 7 | 25,472 | 3.9479 |
| 7lp1_A | ordered | duet | 20260902 | 4×4 | 0 | 0 | 1.0000 | 0/0/4/0 | 1.0000 | — | — | 1.3238 | 1 | 7 | 25,472 | 3.9931 |
| 7lp1_A | ordered | frozen | 20260831 | 16×1 | 0 | 0 | 0 | 13/0/3/0 | 1.0000 | — | — | 1.6772 | 16 | 0 | 25,472 | 13.4094 |
| 7lp1_A | ordered | frozen | 20260901 | 16×1 | 0 | 0 | 0 | 14/0/2/0 | 1.0000 | — | — | 1.6890 | 16 | 0 | 25,472 | 13.3384 |
| 7lp1_A | ordered | frozen | 20260902 | 16×1 | 0 | 0 | 0 | 13/2/1/0 | 1.0000 | — | — | 1.6512 | 16 | 0 | 25,472 | 13.4170 |
| 7lp1_A | ordered | inner_only | 20260831 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.2300 | 1 | 0 | 25,472 | 1.9582 |
| 7lp1_A | ordered | inner_only | 20260901 | 1×16 | 0 | 0 | 1.0000 | 0/0/1/0 | 1.0000 | — | — | 1.3671 | 1 | 0 | 25,472 | 1.9271 |
| 7lp1_A | ordered | inner_only | 20260902 | 1×16 | 0 | 0 | 1.0000 | 0/0/1/0 | 1.0000 | — | — | 1.2971 | 1 | 0 | 25,472 | 1.9478 |
| 7lp1_A | ordered | naive_dual | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.5917 | 2 | 7 | 25,472 | 3.9843 |
| 7lp1_A | ordered | naive_dual | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.6605 | 3 | 7 | 25,472 | 3.9625 |
| 7lp1_A | ordered | naive_dual | 20260902 | 4×4 | 0 | 0 | 1.0000 | 0/0/4/0 | 1.0000 | — | — | 1.3817 | 1 | 7 | 25,472 | 4.0029 |
| 7lp1_A | ordered | outer_only | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.6135 | 1 | 7 | 25,472 | 13.3850 |
| 7lp1_A | ordered | outer_only | 20260901 | 16×1 | 0 | 0 | 1.0000 | 0/0/16/0 | 1.0000 | — | — | 1.2277 | 1 | 7 | 25,472 | 13.4256 |
| 7lp1_A | ordered | outer_only | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.5760 | 2 | 7 | 25,472 | 13.2743 |
| 7lp1_A | windowed | best_of_budget | 20260831 | 16×1 | 0 | 0 | 0 | 14/2/0/0 | 1.0000 | — | — | 1.6291 | 1 | 7 | 25,472 | 13.2723 |
| 7lp1_A | windowed | best_of_budget | 20260901 | 16×1 | 0 | 0 | 0 | 11/5/0/0 | 1.0000 | — | — | 1.7082 | 1 | 7 | 25,472 | 13.3371 |
| 7lp1_A | windowed | best_of_budget | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.5881 | 1 | 7 | 25,472 | 13.5061 |
| 7lp1_A | windowed | complete_nested | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3524 | 1 | 7 | 25,472 | 3.9736 |
| 7lp1_A | windowed | complete_nested | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3875 | 1 | 7 | 25,472 | 3.9610 |
| 7lp1_A | windowed | complete_nested | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.8649 | 1 | 7 | 25,472 | 3.9105 |
| 7lp1_A | windowed | duet | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3096 | 1 | 7 | 25,472 | 3.9283 |
| 7lp1_A | windowed | duet | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3773 | 1 | 7 | 25,472 | 4.0869 |
| 7lp1_A | windowed | duet | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.6129 | 1 | 7 | 25,472 | 4.0246 |
| 7lp1_A | windowed | frozen | 20260831 | 16×1 | 0 | 0 | 0 | 14/2/0/0 | 1.0000 | — | — | 1.6772 | 16 | 0 | 25,472 | 13.3447 |
| 7lp1_A | windowed | frozen | 20260901 | 16×1 | 0 | 0 | 0 | 11/5/0/0 | 1.0000 | — | — | 1.6890 | 16 | 0 | 25,472 | 13.3918 |
| 7lp1_A | windowed | frozen | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.6512 | 16 | 0 | 25,472 | 13.2650 |
| 7lp1_A | windowed | inner_only | 20260831 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.3704 | 1 | 0 | 25,472 | 1.9376 |
| 7lp1_A | windowed | inner_only | 20260901 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.1189 | 1 | 0 | 25,472 | 1.9153 |
| 7lp1_A | windowed | inner_only | 20260902 | 1×16 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | — | — | 1.2564 | 1 | 0 | 25,472 | 1.9232 |
| 7lp1_A | windowed | naive_dual | 20260831 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.2668 | 1 | 7 | 25,472 | 3.9875 |
| 7lp1_A | windowed | naive_dual | 20260901 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.3579 | 1 | 7 | 25,472 | 4.0144 |
| 7lp1_A | windowed | naive_dual | 20260902 | 4×4 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 1.4670 | 1 | 7 | 25,472 | 4.0018 |
| 7lp1_A | windowed | outer_only | 20260831 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.5807 | 1 | 7 | 25,472 | 13.4281 |
| 7lp1_A | windowed | outer_only | 20260901 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.5587 | 2 | 7 | 25,472 | 13.2760 |
| 7lp1_A | windowed | outer_only | 20260902 | 16×1 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | — | — | 1.6701 | 1 | 7 | 25,472 | 13.3325 |

### 7LP1-v2 Frozen reachability gate

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | ordered | frozen | 20260916 | 48×1 | 0.0208 | 0.0208 | 0.0208 | 43/0/4/1 | 1.0000 | — | — | 2.4699 | 48 | 0 | 229,248 | 79.8757 |
| 7lp1_A | ordered | frozen | 20260917 | 48×1 | 0.0833 | 0.0833 | 0.0833 | 43/0/1/4 | 1.0000 | — | — | 2.5292 | 48 | 0 | 229,248 | 102.9261 |

### 7LP1-v2 allocation: K16_M4_p95

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | ordered | complete_nested | 20260920 | 16×4 | 0.6250 | 0.9972 | 1.0000 | 0/4/2/10 | 1.0000 | — | — | 2.5221 | 3 | 2 | 305,664 | 29.8410 |
| 7lp1_A | ordered | complete_nested | 20260921 | 16×4 | 0.6875 | 0.9294 | 0.9375 | 0/3/2/11 | 1.0000 | — | — | 2.5998 | 4 | 1 | 305,664 | 37.2952 |
| 7lp1_A | ordered | complete_nested | 20260922 | 16×4 | 0.0625 | 0.6038 | 0.6250 | 0/15/0/1 | 1.0000 | — | — | 2.3192 | 1 | 3 | 305,664 | 30.7574 |
| 7lp1_A | ordered | complete_nested | 20260923 | 16×4 | 0.3125 | 0.7557 | 0.7500 | 0/6/5/5 | 1.0000 | — | — | 2.7027 | 5 | 1 | 305,664 | 30.7480 |
| 7lp1_A | ordered | complete_nested | 20260924 | 16×4 | 0.6250 | 0.9821 | 1.0000 | 0/4/2/10 | 1.0000 | — | — | 2.6179 | 3 | 1 | 305,664 | 29.8251 |
| 7lp1_A | ordered | duet | 20260920 | 16×4 | 0.3750 | 0.5653 | 0.5625 | 5/2/3/6 | 1.0000 | — | — | 2.7824 | 4 | 3 | 305,664 | 40.2679 |
| 7lp1_A | ordered | duet | 20260921 | 16×4 | 0.6875 | 0.9530 | 0.9375 | 2/2/1/11 | 1.0000 | — | — | 2.6408 | 4 | 1 | 305,664 | 39.5540 |
| 7lp1_A | ordered | duet | 20260922 | 16×4 | 0.5000 | 0.9672 | 1.0000 | 0/5/3/8 | 1.0000 | — | — | 2.5623 | 3 | 2 | 305,664 | 40.1687 |
| 7lp1_A | ordered | duet | 20260923 | 16×4 | 0.6250 | 0.9958 | 1.0000 | 0/6/0/10 | 1.0000 | — | — | 2.5636 | 2 | 2 | 305,664 | 40.4825 |
| 7lp1_A | ordered | duet | 20260924 | 16×4 | 0.8125 | 0.9729 | 1.0000 | 0/2/1/13 | 1.0000 | — | — | 2.8598 | 3 | 1 | 305,664 | 39.5645 |

### 7LP1-v2 allocation: K4_M16_p95

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | ordered | complete_nested | 20260920 | 4×16 | 0.2500 | 0.6917 | 0.5000 | 1/2/0/1 | 1.0000 | — | — | 2.3444 | 2 | 1 | 305,664 | 12.1012 |
| 7lp1_A | ordered | complete_nested | 20260921 | 4×16 | 0.7500 | 0.9804 | 1.0000 | 0/1/0/3 | 1.0000 | — | — | 2.3183 | 1 | 1 | 305,664 | 14.6788 |
| 7lp1_A | ordered | complete_nested | 20260922 | 4×16 | 0.5000 | 0.9829 | 1.0000 | 1/1/0/2 | 1.0000 | — | — | 2.8760 | 2 | 0 | 305,664 | 14.6967 |
| 7lp1_A | ordered | complete_nested | 20260923 | 4×16 | 0.2500 | 0.6813 | 0.7500 | 1/2/0/1 | 1.0000 | — | — | 2.6626 | 1 | 1 | 305,664 | 14.5371 |
| 7lp1_A | ordered | complete_nested | 20260924 | 4×16 | 0.7500 | 0.9939 | 1.0000 | 0/1/0/3 | 1.0000 | — | — | 2.9613 | 1 | 1 | 305,664 | 14.6780 |
| 7lp1_A | ordered | duet | 20260920 | 4×16 | 1.0000 | 1.0000 | 1.0000 | 0/0/0/4 | 1.0000 | — | — | 2.6641 | 2 | 1 | 305,664 | 13.4505 |
| 7lp1_A | ordered | duet | 20260921 | 4×16 | 0.2500 | 0.7180 | 0.7500 | 0/1/2/1 | 1.0000 | — | — | 3.1624 | 1 | 1 | 305,664 | 16.0338 |
| 7lp1_A | ordered | duet | 20260922 | 4×16 | 0.5000 | 0.5152 | 0.5000 | 0/2/0/2 | 1.0000 | — | — | 2.3751 | 2 | 1 | 305,664 | 16.1866 |
| 7lp1_A | ordered | duet | 20260923 | 4×16 | 0.7500 | 1.0000 | 1.0000 | 0/1/0/3 | 1.0000 | — | — | 2.8303 | 2 | 1 | 305,664 | 15.9342 |
| 7lp1_A | ordered | duet | 20260924 | 4×16 | 0.7500 | 0.9904 | 1.0000 | 0/1/0/3 | 1.0000 | — | — | 1.9720 | 2 | 1 | 305,664 | 15.9759 |

### 7LP1-v2 allocation: K4_M8_p95

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | ordered | complete_nested | 20260920 | 4×8 | 0.5000 | 0.9822 | 0.7500 | 0/1/1/2 | 1.0000 | — | — | 2.4197 | 1 | 1 | 152,832 | 10.7178 |
| 7lp1_A | ordered | complete_nested | 20260921 | 4×8 | 0.5000 | 0.9741 | 1.0000 | 1/1/0/2 | 1.0000 | — | — | 2.3438 | 2 | 0 | 152,832 | 10.6887 |
| 7lp1_A | ordered | complete_nested | 20260922 | 4×8 | 0 | 0 | 0 | 3/1/0/0 | 1.0000 | — | — | 2.8663 | 1 | 2 | 152,832 | 8.8020 |
| 7lp1_A | ordered | complete_nested | 20260923 | 4×8 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 3.2341 | 1 | 3 | 152,832 | 10.7631 |
| 7lp1_A | ordered | complete_nested | 20260924 | 4×8 | 0.5000 | 0.3442 | 0.2500 | 1/0/1/2 | 1.0000 | — | — | 3.1194 | 2 | 1 | 152,832 | 10.5906 |
| 7lp1_A | ordered | duet | 20260920 | 4×8 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 2.7370 | 1 | 3 | 152,832 | 11.6581 |
| 7lp1_A | ordered | duet | 20260921 | 4×8 | 0 | 0 | 0 | 0/4/0/0 | 1.0000 | — | — | 1.6428 | 1 | 1 | 152,832 | 11.5954 |
| 7lp1_A | ordered | duet | 20260922 | 4×8 | 0.2500 | 0.9384 | 1.0000 | 0/3/0/1 | 1.0000 | — | — | 2.2589 | 1 | 1 | 152,832 | 11.6559 |
| 7lp1_A | ordered | duet | 20260923 | 4×8 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | — | — | 2.6615 | 1 | 2 | 152,832 | 11.5820 |
| 7lp1_A | ordered | duet | 20260924 | 4×8 | 0.7500 | 0.6656 | 0.7500 | 0/0/1/3 | 1.0000 | — | — | 2.2406 | 1 | 2 | 152,832 | 12.0251 |

### 7LP1-v2 allocation: K8_M4_p95

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | ordered | complete_nested | 20260920 | 8×4 | 0.5000 | 0.9210 | 0.8750 | 0/1/3/4 | 1.0000 | — | — | 2.8868 | 1 | 1 | 152,832 | 18.1811 |
| 7lp1_A | ordered | complete_nested | 20260921 | 8×4 | 0.6250 | 0.9933 | 1.0000 | 0/2/1/5 | 1.0000 | — | — | 2.1166 | 2 | 1 | 152,832 | 15.0535 |
| 7lp1_A | ordered | complete_nested | 20260922 | 8×4 | 0.5000 | 0.9638 | 1.0000 | 1/3/0/4 | 1.0000 | — | — | 2.4851 | 2 | 1 | 152,832 | 15.3635 |
| 7lp1_A | ordered | complete_nested | 20260923 | 8×4 | 0.1250 | 0.2729 | 0.2500 | 0/5/2/1 | 1.0000 | — | — | 2.4347 | 2 | 1 | 152,832 | 15.2395 |
| 7lp1_A | ordered | complete_nested | 20260924 | 8×4 | 0.8750 | 0.7595 | 0.7500 | 1/0/0/7 | 1.0000 | — | — | 2.8915 | 3 | 2 | 152,832 | 15.3006 |
| 7lp1_A | ordered | duet | 20260920 | 8×4 | 0.2500 | 0.8628 | 0.8750 | 0/6/0/2 | 1.0000 | — | — | 2.3629 | 2 | 2 | 152,832 | 14.9564 |
| 7lp1_A | ordered | duet | 20260921 | 8×4 | 0.2500 | 0.3976 | 0.3750 | 3/2/1/2 | 1.0000 | — | — | 2.5002 | 2 | 1 | 152,832 | 19.7431 |
| 7lp1_A | ordered | duet | 20260922 | 8×4 | 0.3750 | 0.9989 | 1.0000 | 0/5/0/3 | 1.0000 | — | — | 2.8168 | 1 | 1 | 152,832 | 19.7906 |
| 7lp1_A | ordered | duet | 20260923 | 8×4 | 0.3750 | 0.2822 | 0.3750 | 1/4/0/3 | 1.0000 | — | — | 3.0284 | 2 | 1 | 152,832 | 20.4509 |
| 7lp1_A | ordered | duet | 20260924 | 8×4 | 0.5000 | 0.8607 | 0.8750 | 1/3/0/4 | 1.0000 | — | — | 2.6007 | 2 | 1 | 152,832 | 19.8405 |

### 7LP1-v2 allocation: K8_M8_p95

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | ordered | complete_nested | 20260920 | 8×8 | 0.1250 | 0.6907 | 0.6250 | 0/5/2/1 | 1.0000 | — | — | 2.0618 | 3 | 2 | 305,664 | 21.2627 |
| 7lp1_A | ordered | complete_nested | 20260921 | 8×8 | 1.0000 | 1.0000 | 1.0000 | 0/0/0/8 | 1.0000 | — | — | 4.1013 | 1 | 2 | 305,664 | 17.7556 |
| 7lp1_A | ordered | complete_nested | 20260922 | 8×8 | 0.3750 | 0.9416 | 0.8750 | 0/4/1/3 | 1.0000 | — | — | 2.0424 | 2 | 1 | 305,664 | 21.6858 |
| 7lp1_A | ordered | complete_nested | 20260923 | 8×8 | 0.2500 | 0.7010 | 0.6250 | 0/5/1/2 | 1.0000 | — | — | 2.3945 | 3 | 1 | 305,664 | 18.0575 |
| 7lp1_A | ordered | complete_nested | 20260924 | 8×8 | 0.7500 | 0.8555 | 0.8750 | 1/1/0/6 | 1.0000 | — | — | 2.4575 | 2 | 2 | 305,664 | 21.9366 |
| 7lp1_A | ordered | duet | 20260920 | 8×8 | 0.5000 | 0.9925 | 1.0000 | 0/4/0/4 | 1.0000 | — | — | 3.0451 | 1 | 2 | 305,664 | 23.2318 |
| 7lp1_A | ordered | duet | 20260921 | 8×8 | 0.3750 | 0.8941 | 0.8750 | 1/3/1/3 | 1.0000 | — | — | 2.8681 | 2 | 1 | 305,664 | 23.1822 |
| 7lp1_A | ordered | duet | 20260922 | 8×8 | 0.1250 | 0.2271 | 0.2500 | 5/1/1/1 | 1.0000 | — | — | 2.4236 | 2 | 3 | 305,664 | 23.9136 |
| 7lp1_A | ordered | duet | 20260923 | 8×8 | 0.5000 | 0.9547 | 1.0000 | 1/1/2/4 | 1.0000 | — | — | 2.7839 | 1 | 1 | 305,664 | 23.4860 |
| 7lp1_A | ordered | duet | 20260924 | 8×8 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | — | — | 2.8210 | 2 | 4 | 305,664 | 23.8857 |

### 7LP1-v2 multi-checkpoint development

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | ordered | duet | 20260920 | 16×4 | 0.5000 | 0.9708 | 1.0000 | 0/5/3/8 | 1.0000 | — | — | 2.5353 | 4 | 1 | 305,664 | 36.8005 |
| 7lp1_A | ordered | duet | 20260921 | 16×4 | 0.5625 | 0.7730 | 0.7500 | 1/4/2/9 | 1.0000 | — | — | 2.7003 | 4 | 2 | 305,664 | 38.2901 |
| 7lp1_A | ordered | duet | 20260922 | 16×4 | 0.5000 | 0.9100 | 0.9375 | 0/5/3/8 | 1.0000 | — | — | 2.8396 | 2 | 3 | 305,664 | 30.0302 |
| 7lp1_A | ordered | duet | 20260923 | 16×4 | 0.5625 | 0.9374 | 0.8750 | 0/4/3/9 | 1.0000 | — | — | 2.6352 | 2 | 2 | 305,664 | 37.3728 |

### 7LP1-v2 multi-checkpoint fresh confirmation

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7lp1_A | ordered | complete_nested | 20260925 | 16×4 | 0.6875 | 0.8910 | 0.8125 | 1/3/1/11 | 1.0000 | — | — | 2.4184 | 4 | 2 | 305,664 | 37.1940 |
| 7lp1_A | ordered | complete_nested | 20260926 | 16×4 | 0.5625 | 0.9221 | 0.9375 | 1/4/2/9 | 1.0000 | — | — | 2.7441 | 4 | 1 | 305,664 | 36.6041 |
| 7lp1_A | ordered | duet | 20260925 | 16×4 | 0.6875 | 0.9374 | 1.0000 | 1/3/1/11 | 1.0000 | — | — | 2.8281 | 4 | 1 | 305,664 | 40.6253 |
| 7lp1_A | ordered | duet | 20260926 | 16×4 | 0.7500 | 0.9823 | 1.0000 | 0/4/0/12 | 1.0000 | — | — | 2.7727 | 3 | 2 | 305,664 | 39.1390 |

### ABL1 DFG-flip budget-8 support gate

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abl1_dfg_flip | endpoint | complete_nested | 20260930 | 4×2 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | 0 | 0 | — | 1 | 10 | 38,208 | 89.5378 |
| abl1_dfg_flip | endpoint | duet | 20260930 | 4×2 | 0 | 0 | 0 | 4/0/0/0 | 1.0000 | 0 | 0 | — | 1 | 13 | 38,208 | 89.3595 |
| abl1_dfg_flip | endpoint | frozen | 20260930 | 8×1 | 0 | 0 | 0 | 8/0/0/0 | 1.0000 | 0.8750 | 0 | — | 8 | 0 | 38,208 | 95.6911 |

### ATLAS 10-protein route-v3.1 main seed-1

| Case | Task | Method | Seed | K×M | Pre-U | Pre-W | Post | S0/1/2/3 | EndV | PathV | Valid-end success | Held-out path | Anc. | R | NFE | Wall min |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6jv8_A | endpoint | complete_nested | 20261001 | 16×4 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | 0.8750 | 0 | — | 2 | 7 | 305,664 | 34.4042 |
| 6jv8_A | endpoint | duet | 20261001 | 16×4 | 0.3750 | 1.0000 | 1.0000 | 10/6/0/0 | 1.0000 | 1.0000 | 1.0000 | — | 3 | 8 | 305,664 | 34.6676 |
| 6jv8_A | endpoint | frozen | 20261001 | 64×1 | 0 | 0 | 0 | 64/0/0/0 | 1.0000 | 0.8438 | 0 | — | 64 | 0 | 305,664 | 129.4082 |
| 6jv8_A | endpoint | inner_only | 20261001 | 1×64 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | 1.0000 | 0 | — | 1 | 0 | 305,664 | 17.1255 |
| 6jv8_A | endpoint | outer_only | 20261001 | 64×1 | 0.0156 | 1.0000 | 1.0000 | 63/1/0/0 | 1.0000 | 1.0000 | 1.0000 | — | 1 | 22 | 305,664 | 129.9611 |
| 6jv8_A | ordered | complete_nested | 20261001 | 16×4 | 0 | 0 | 0 | 12/3/1/0 | 1.0000 | 1.0000 | 0 | — | 8 | 0 | 305,664 | 34.6452 |
| 6jv8_A | ordered | duet | 20261001 | 16×4 | 0 | 0 | 0 | 13/3/0/0 | 1.0000 | 0.8750 | 0 | — | 8 | 0 | 305,664 | 40.0369 |
| 6jv8_A | ordered | frozen | 20261001 | 64×1 | 0 | 0 | 0 | 39/25/0/0 | 1.0000 | 0.8438 | 0 | — | 64 | 0 | 305,664 | 130.6803 |
| 6jv8_A | ordered | inner_only | 20261001 | 1×64 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | 1.0000 | 0 | — | 1 | 0 | 305,664 | 16.9927 |
| 6jv8_A | ordered | outer_only | 20261001 | 64×1 | 0 | 0 | 0 | 39/25/0/0 | 1.0000 | 0.9062 | 0 | — | 20 | 0 | 305,664 | 130.5747 |
| 7bwf_B | ordered | complete_nested | 20261001 | 16×4 | 0 | 0 | 0 | 12/4/0/0 | 1.0000 | 1.0000 | 0 | — | 1 | 4 | 305,664 | 45.2981 |
| 7bwf_B | ordered | duet | 20261001 | 16×4 | 0 | 0 | 0 | 16/0/0/0 | 1.0000 | 0.8125 | 0 | — | 1 | 4 | 305,664 | 45.3849 |
| 7bwf_B | ordered | frozen | 20261001 | 64×1 | 0 | 0 | 0 | 64/0/0/0 | 1.0000 | 0.9062 | 0 | — | 64 | 0 | 305,664 | 135.5393 |
| 7bwf_B | ordered | inner_only | 20261001 | 1×64 | 0 | 0 | 0 | 1/0/0/0 | 1.0000 | 1.0000 | 0 | — | 1 | 0 | 305,664 | 23.5018 |
| 7bwf_B | ordered | outer_only | 20261001 | 64×1 | 0 | 0 | 0 | 64/0/0/0 | 1.0000 | 0.7344 | 0 | — | 16 | 3 | 305,664 | 133.7939 |

## 10. Raw-data provenance와 해석 규칙

- Historical H100 root: `/workspace/sejin/AI_MD_NSMC/outputs/duet_md`.
- Current route-v3.1 root: `/workspace/sejin/AI_MD_NSMC_v31_stage/outputs/duet_md/protein_benchmark/route_v3_1`.
- A6000 root: `/home/sejin/AI_MD_NSMC/outputs/duet_md`.
- 각 run의 source of truth: `metrics.json`, `records.json`, `resolved_config.yaml`; 구조가 필요하면 `trajectories_atom37_a.npz`와 `final_paths.pdb`류를 추가로 본다.
- 이 MD의 `Pre-U`, `Pre-W`, `R`, stage count는 raw `records.json`과 resolved ESS threshold로 deterministic reconstruction했다. `Post`, validity, distance, ancestry, NFE, wall time은 raw `metrics.json` scalar다.
- 한 successful pre-resampling path가 final resampling에서 여러 번 복제될 수 있으므로 Post output 개수를 독립 발견 수로 해석하지 않는다.
- Cross-protein 통계 단위는 trajectory가 아니라 protein이다. Protein별 DuET−Complete paired difference를 만든 뒤 macro mean/median과 protein bootstrap CI를 보고해야 한다.
- Invalid generated paths는 compute에서 제거하지 않는다. Unconditional success, valid-endpoint success, endpoint validity, whole-path validity를 분리 보고한다.
- H held-out path가 `unavailable`이면 fidelity missing으로 기록할 뿐 task 실패나 protein 탈락으로 바꾸지 않는다.
- 6J56와 7LP1는 development/falsification 자료다. Route-v3.1 10-protein cohort가 일반화 주장용 confirmatory unit이다.
- 현재 route-v3.1은 seed 하나이므로 coverage/디버깅 단계다. 모든 protein-method-task가 끝난 뒤에만 seed 20261002–05 확장 여부를 결정한다.

## 11. ChatGPT가 이론 고도화 전에 반드시 인지할 쟁점

1. DuET의 주 비교는 Frozen이 아니라 **matched Complete Nested**다. Frozen zero success 자체는 DuET의 실패 증거가 아니다.
2. 현재 p=.95 one-checkpoint는 마지막 약 5% reverse steps에만 계산을 재배치한다. Complete와의 algorithmic 차이가 작을 수 있으며, 이것이 실제 negative result의 가능한 원인이다. 다만 현재 main outcome을 본 뒤 protein별 p를 조정하면 post-hoc tuning이 된다.
3. 7LP1 K16×M4의 긍정 결과는 allocation search 뒤 선택된 development evidence다. Confirmatory superiority로 서술하면 안 된다.
4. Binary program success와 continuous prefix potential을 혼동하면 안 된다. A/B/target은 평가 state machine이고, steering은 다음 미완료 event까지의 연속 거리 cost를 사용한다.
5. Generated nominal time은 frozen surrogate의 frame lag다. Reference MD보다 짧은 deadline에서 성공하더라도 physical kinetic acceleration이나 enhanced-sampling rate를 직접 의미하지 않는다.
6. 현재 가장 중요한 falsification은 DuET이 Complete보다 ordered pre-unique success, success mass, lineage를 함께 개선하는지이며, fidelity/validity를 악화시키지 않는지도 동시 보고해야 한다.
7. Route-v3.1에서 모든 방법이 계속 0이면 DuET만의 문제로 결론 내리지 말고 stage curves(A, A→B, terminal), checkpoint rank predictivity, proposal reachability, whole-path validity를 분해해야 한다.
