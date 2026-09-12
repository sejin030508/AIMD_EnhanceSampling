# Chirality 최소 재검증 보고서

## 결론

독립적인 이름 기반 검사와 rigid/mirror 대조군은 정상 작동했다. 좌표가 보존된 부분집합에서는 relaxed 구조에서 raw 대비 Cα signed-volume 부호가 실제로 다수 바뀌었고, 따라서 **단순한 배열 index/mapping 오류만으로 설명되지는 않는다. 실제 입체배치 반전이 확인된다.**

다만 corrected audit의 290개 unique frame 전체에 대한 좌표가 저장되어 있지 않다. 이번 재검산은 보존된 complete-path 좌표 213 frame appearances에 한정되므로, **전체 290개에 대한 정확한 발생률은 미확정**이다. 기존 124/129, 128/129, 31/32 결과를 덮어쓰지 않는다.

## 범위와 방법

- 대상: 기존 corrected relaxation audit의 저장된 raw/relaxed PDB. 새 rollout, minimization, parameter tuning 없음.
- 좌표 범위: Trp-cage 99, BBA 99, PRMT6 15 frame appearances (총 213). 누락된 corrected frames는 재구성하지 않음.
- 각 path의 저장된 시작/후반 frame은 포함했지만, 별도 파일로 저장된 정상 start/target control 좌표는 확인되지 않았다. 따라서 정상성 대조는 보존된 reference identity와 rigid transform control로 수행했고, 이 한계는 전체 판정률과 분리해 기록한다.
- residue 대응: 배열 index가 아니라 `chain index + residue number + residue name + atom name`으로 N, CA, C, CB를 찾음.
- signed volume: `dot(N-CA, cross(C-CA, CB-CA))` (Å³).
- 판정 tolerance: `|v| < 0.5 Å³`는 near-planar/판정불가. Glycine, 원자 누락도 판정불가.
- reference 부호와 비교하여 `raw/relaxed 자체의 부호`, `raw→relaxed 변화`, `reference 대비 inversion`을 별도 기록.
- RDKit 교차검사: 좌표에서 결합을 추론하지 않고, OpenMM PDB topology의 기존 bond graph를 사용. 기존 stereo tag 제거 후 3D에서 재할당. L-amino acid를 일괄 S로 가정하지 않음.

## 독립 signed-volume 결과

| system | side | inversion frame / evaluable frame | inverted Cα / evaluable Cα | non-evaluable Cα |
|---|---:|---:|---:|---:|
| Trp-cage | raw | 0 / 99 | 0 / 1,683 | 297 |
| Trp-cage | relaxed | 89 / 99 | 328 / 1,678 | 302 |
| BBA | raw | 0 / 99 | 0 / 2,574 | 198 |
| BBA | relaxed | 97 / 99 | 592 / 2,551 | 221 |
| PRMT6 | raw | 0 / 15 | 0 / 4,425 | 420 |
| PRMT6 | relaxed | 11 / 15 | 202 / 4,424 | 421 |

이 저장 subset에서는 raw가 모두 reference 부호와 일치하고, relaxed에서만 inversion이 발생했다. 따라서 관측된 relaxed inversion은 이 subset 기준으로 raw 입력에 이미 있던 이상을 단순히 재표시한 것이 아니다.

## 검사 정상 작동 대조군

| system | reference identity | rigid rotation + translation | mirror-x |
|---|---:|---:|---:|
| Trp-cage | 0 / 17 inverted | 0 / 17 | 17 / 17 |
| BBA | 0 / 26 | 0 / 26 | 26 / 26 |
| PRMT6 | 0 / 295 | 0 / 295 | 295 / 295 |

형식은 `inverted / evaluable Cα`이다. 회전·평행이동은 부호를 보존했고, 거울상은 모두 반전으로 검출됐다.

## RDKit 대표 교차검사

아래는 시스템별 최대 3개 대표 사례다. `CW→CCW`는 RDKit의 Cα 3D tag 변화이며, signed-volume 결과와 일치하는 사례와 불일치하는 사례를 함께 보존했다.

| system | path/frame | residue | signed volume raw → relaxed (Å³) | RDKit raw → relaxed | 판정 |
|---|---|---|---:|---|---|
| Trp-cage | duet seed211 p3 / 0 | GLN 5 | +2.547 → −1.544 | CW → CCW | 일치 |
| Trp-cage | duet seed211 p3 / 0 | TRP 6 | +2.537 → −1.661 | CW → CCW | 일치 |
| Trp-cage | duet seed211 p3 / 0 | ASP 9 | +2.567 → −1.855 | CCW → CCW | 불일치 |
| BBA | duet seed223 p1 / 0 | GLU 1 | +2.518 → −1.830 | CW → CCW | 일치 |
| BBA | duet seed223 p1 / 0 | TYR 3 | +2.564 → −1.684 | CCW → CCW | 불일치 |
| BBA | duet seed223 p1 / 0 | LYS 6 | +2.336 → −2.029 | CW → CCW | 일치 |
| PRMT6 | official forward p3 / 1 | ASP 1 | +2.511 → −2.309 | CW → CCW | 일치 |
| PRMT6 | official forward p3 / 1 | SER 3 | +2.508 → −1.783 | CCW → CCW | 불일치 |
| PRMT6 | official forward p3 / 1 | GLU 6 | +2.506 → −1.931 | CCW → CCW | 불일치 |

RDKit은 전체 대표 27개 중 Trp-cage 6개, BBA 5개, PRMT6 3개에서 raw→relaxed tag 변화를 보였다(같은 path에서 같은 residue를 3개 path에 걸쳐 포함). 불일치는 두 검사가 서로 다른 원자/결합 주변 3D chirality를 반영할 수 있음을 보여주므로, signed-volume 수치를 RDKit 결과로 무조건 대체하지 않았다.

## 기존 결과와의 비교

기존 corrected audit 기록은 Trp-cage 124/129, BBA 128/129, PRMT6 31/32 frame inversion이었다. 이번 수치는 저장된 좌표 subset과 독립적인 residue-name mapping/tolerance를 사용했으므로 분모와 평가 가능 residue가 다르다. 따라서 **정확한 수치 일치가 아니라, 세 시스템 모두 relaxed에서 높은 빈도의 inversion이 관측된다는 질적 결론만 일치**한다.

## 산출물

- [요약 JSON](./chirality_recheck_summary.json)
- [frame-level CSV](./chirality_frame_rows.csv)
- [residue-level CSV](./chirality_residue_rows.csv)
- [대표 사례 원자 좌표 JSON](./representative_coordinates.json)
- 재현 스크립트: `relaxation_audit_payload/independent_chirality_recheck.py`

기존 성공률·validity와 기존 corrected audit 원본은 변경하지 않았다.
