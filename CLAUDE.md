# AI_MD_NSMC — 작업 규칙과 자산 배치

## 원칙

앞으로의 모든 실험은 **`AI_MD_NSMC` 한 곳**을 root로 삼는다. 새 실험 트리를
별도 이름(`*_recovery`, `*_work`, 날짜 접미사 등)으로 파생시키지 않는다.

## 현재 배치 (2026-09-15 실측)

| 호스트 | 경로 | 상태 |
|---|---|---|
| Mac | `~/Desktop/AIBL/projects/AI_MD_NSMC` | git source of truth (`sejin030508/AIMD_EnhanceSampling`) |
| H100 | `/workspace/sejin/AI_MD_NSMC_phase_b_recovery` | **live 실행 트리** (42 GB, 이름과 달리 동결본이 아님) |
| H100 | `/workspace/sejin/AI_MD_NSMC` | 2026-09-03 정지된 오래된 사본. `STALE.md` 참조 |
| A6000 | `~/AI_MD_NSMC` | 코드 + 소규모 결과 (271 MB) |

`$HOME`은 A6000에서 99% 차 있다(여유 15 GB). A6000의 대용량 산출물은 반드시
`/mnt/ssd0/sejin` 아래에 두고 symlink로 연결한다.

## 공용 asset

ConfRover runtime cache(checkpoint / MSA / folding_repr, 7.1 GB)는 복사하지 않고
각 root의 `assets/confrover`에서 symlink로 참조한다.

- H100: `assets/confrover` -> `/workspace/sejin/confrover_mh_steering/data/confrover_cache`
- A6000: `assets/confrover` -> `/mnt/ssd0/sejin/confrover_mh_steering/data/confrover_cache`

`assets/`는 git에 올리지 않는다.

## 미완료 통합 작업

H100의 live 트리를 `AI_MD_NSMC`로 승격하는 일은 **아직 하지 않았다**. 실행 중인
작업이 `/workspace/sejin/AI_MD_NSMC_phase_b_recovery`를 절대경로로 물고 있어서,
작업이 모두 끝난 뒤에 다음 순서로 처리한다.

1. `AI_MD_NSMC` (stale) -> `AI_MD_NSMC_stale_20260903`으로 이동
2. `AI_MD_NSMC_phase_b_recovery` -> `AI_MD_NSMC`로 rename (같은 파일시스템, 복사 없음)
3. 옛 절대경로가 깨지지 않도록 `AI_MD_NSMC_phase_b_recovery -> AI_MD_NSMC` 호환 symlink 유지
