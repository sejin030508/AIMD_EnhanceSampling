# Phase B4 — Cryptic-pocket DuET development experiment

**Status:** 6/6 production runs complete. No valid holo-like endpoint was obtained.

## Goal

Frozen ConfRover surrogate에서 DuET diffusion-time steering이 cryptic-pocket
holo 구조 방향으로 **유효한 구조적 접근**을 만들 수 있는지 확인하는 development
experiment. Method superiority 비교가 아니라 steering feasibility 탐색이다.

## Shared setting

| Item | Setting |
|---|---|
| Model | frozen `confrover_base_20m_v1_0`, SDE, 200 reverse steps/frame |
| DuET population | outer `K=4`, inner `M=4` (16 candidates/transition) |
| Physical trajectory | `T=24` transitions, start 포함 25 structures; lag 2.56 ns/frame (nominal horizon 61.44 ns, kinetics claim 아님) |
| Seed | 103 |
| Reward strengths | `a=16`, `a=32` |
| Reward | `log ψ(x) = -a [d(x)/d0]^2`; fixed-core aligned moving-region N/CA/C RMSD to holo |
| Inner steering | reverse updates 150/200 (75%)와 180/200 (90%)에서 정확히 두 번 resampling |
| Proper weighting | `φ75`; `φ90/φ75(ancestor)`; `ψfinal/φ90(ancestor)`; outer update `Zhat/ψ(parent)` |
| Validity | adjacent-CA ≤5.5 Å; invalid path도 outcome 분모에 유지 |
| Endpoint success | whole-path valid + 마지막 두 frame 모두 RMSD ≤1.5 Å |
| Runtime checks | potential floor 제거, log-space weights, telescoping residual 0, checkpoints 150/180 모두 audit pass |

Reward에는 ligand/contact/deadline/validity 항을 넣지 않았다. Holo ligand 공간은
evaluation-only 관찰량이다.

## Protein-specific structures and target definitions

| Protein | Start (apo-like) | Holo target | Construct | Reward moving region | `d0` |
|---|---|---|---|---|---:|
| PRMT6 | AlphaFold `AF-Q96LA8-F1-model_v4` | PDB `6W6D` | 53–375 | 155–165 | 4.6688 Å |
| SMARCA2 | AlphaFold `AF-P51531-F1-model_v6` | PDB `6EG3` | 705–955 | 852–860 | 9.4028 Å |
| PI3Kα WT | AlphaFold `AF-P42336-F1-model_v6` | PDB `8TSB` | 765–1051 | original 931–957; observed mask 931–942 + 951–957 | 5.0607 Å |

PI3Kα holo `8TSB`에는 943–950 backbone이 없어 해당 8 residues는 RMSD 계산에서만
제외했다. 생성 구조·validity·ligand-space 분석에는 full 931–957 segment를 유지했다.
따라서 PI3Kα 결과는 full-pocket restoration이 아니라 `observed_loop_rmsd` 결과다.

## Results (production raw summary)

`weighted d/d0`는 final pre-resampling outer population의 weighted mean이며 낮을수록
holo loop에 가깝다. Best valid는 whole-path-valid trajectory 중 최저 final RMSD다.

| Protein | a | T=16 weighted d/d0 | T=24 weighted d/d0 | T=24 valid paths | Best valid final RMSD | Valid endpoint / anytime hit | Outer ESS final | Outer resamples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PRMT6 | 16 | 0.8447 | 0.8982 | 4/4 | 3.9345 Å | 0/4 / 0/4 | 2.7249 | 6 |
| PRMT6 | 32 | 0.7666 | 0.7237 | 0/4 | — | 0/4 / 0/4 | 3.0723 | 12 |
| SMARCA2 | 16 | 0.7024 | 0.6967 | 4/4 | 5.7909 Å | 0/4 / 0/4 | 2.2757 | 6 |
| SMARCA2 | 32 | 0.5205 | 0.5335 | 4/4 | 4.8162 Å | 0/4 / 0/4 | 3.2993 | 11 |
| PI3Kα observed | 16 | 0.9164 | 0.8667 | 4/4 | 4.3510 Å | 0/4 / 0/4 | 1.9385 | 6 |
| PI3Kα observed | 32 | 0.8249 | 0.8014 | 4/4 | 4.0543 Å | 0/4 / 0/4 | 1.0286 | 9 |

All cells: 76,416 decoder NFE/run, 1,153 potential evaluations/run, no NaN,
no potential clipping. Total: 458,496 NFE and approximately 6.07 GPU-hours.

## Structural/pocket interpretation

- **PRMT6 a=32:** lower RMSD and hidden-contact movement toward holo, but all
  retained paths inherited the same invalid frame-4 structure (max adjacent CA
  5.559 Å). This cannot count as a valid success.
- **SMARCA2 a=32:** strongest valid local-RMSD approach (`d/d0=0.5335`, 4/4
  valid), but holo J7G ligand space remained substantially protein-occupied;
  no convincing physical pocket clearance.
- **PI3Kα a=32:** modest valid observed-loop approach, but final outer
  ESS=1.0286 means nearly one surviving lineage; UIW ligand space remained
  occupied. Not pocket restoration.
- Increasing `a` from 16 to 32 improved final local RMSD for all three
  proteins, while generally increasing selection pressure/resampling. T=16 to
  T=24 improvement was not monotonic across proteins.

## Conclusion / scope boundary

B4 shows a single-seed **local endpoint steering signal**, clearest for valid
SMARCA2 a=32, but **does not demonstrate cryptic-pocket recovery**: no valid
1.5-Å endpoint, no valid anytime hit, and no convincing ligand-space clearance.
This DuET-only search also makes no claim versus Frozen, Outer-only, or Complete
Nested. A follow-up requires a pre-fixed physical-opening criterion and,
if a setting is fresh-seed confirmed, matched baseline comparisons.

## Artifacts

- Full agent handoff: `reports/PHASE_B4_CRYPTIC_POCKET_AGENT_HANDOFF_2026-09-10.md`
- Local PyMOL package: `outputs/phase_b_b4_pymol_2026-09-10/`
- Server output root: `/workspace/sejin/phase_b_pockets_recovery/outputs/B4_pocket_search`
