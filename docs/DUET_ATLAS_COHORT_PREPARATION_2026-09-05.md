# DuET-MD ATLAS confirmatory cohort preparation

Prepared on 2026-09-05 under route protocol v3.1, before inspecting any
Frozen, Outer-only, Inner-only, Complete Nested, or DuET outcome.

## Outcome

- Target: 10 frozen protein tasks.
- Frozen: 10 (`7` primary candidates and `3` predeclared reserves).
- Pre-method attrition: 3 primary candidates.
- Static prelaunch audit: pass for all 10 frozen tasks.
- Main B=64 method runs: started on 2026-09-06 with a protein-first queue.
- ConfRover/OpenFold representation and config-resolution gate: pass for all 10.
- Checkpoint continuation-predictivity gate at reverse progress 0.95: pass for
  all 10 at the D1 A-event step (8 checkpoints x 4 continuations; diagnostic
  seed 20260991).
- K=1 generated whole-path smoke gate: all 10 jobs completed; endpoint geometry
  was valid for 10/10 paths and strict whole-path geometry for 8/10 paths.
- Frozen K=8 reachability/validity pilot: 9/10 complete; the remaining
  `7p46_A` was resumed as a persistent remote job on 2026-09-06.

The three unresolved primary candidates remain in the machine-readable
attrition report:

- `6in7_A`: minimum trained lag (`20 x 10 ps`) makes the 24-frame nominal
  horizon 4.8 ns, slightly longer than its 4.6 ns D1 first passage
  (`compression=0.958 < 1`). It therefore fails the predeclared accelerated
  challenge.
- `6lus_A`: the predeclared D1 contains no stable initial-state last-exit path
  to the official endpoint basin.
- `6ovk_R`: the same D1 transition-path gate fails.

These failures were determined without running or inspecting any method.
Reserves were admitted strictly in the frozen order: `7rm7_A`, `7aex_A`, then
`7p46_A`. Later reserves were not downloaded or evaluated.

## Frozen ten-protein cohort

`Lag` is one generated-frame interval. `T model` is the 24-frame nominal
deadline. `Compression = D1 first passage / T model`; it is a challenge ratio,
not a kinetic-rate estimate.

| Protein | Source | Residues | D1/D2/H | Lag (ps) | D1 first passage (ns) | Last-exit path (ns) | T model (ns) | Compression | Route support | H path |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `6jv8_A` | primary | 76 | R3/R1/R2 | 420 | 20.4 | 20.1 | 10.08 | 2.024 | D1+D2 | unavailable |
| `7bwf_B` | primary | 92 | R3/R1/R2 | 200 | 9.5 | 7.2 | 4.80 | 1.979 | D1 only | unavailable |
| `6tly_A` | primary | 99 | R2/R1/R3 | 370 | 17.8 | 17.7 | 8.88 | 2.005 | D1+D2 | available |
| `7s86_A` | primary | 99 | R3/R1/R2 | 240 | 11.7 | 6.9 | 5.76 | 2.031 | D1 only | available |
| `6rrv_A` | primary | 127 | R1/R2/R3 | 420 | 20.3 | 5.7 | 10.08 | 2.014 | D1 only | unavailable |
| `6gus_A` | primary | 155 | R2/R1/R3 | 310 | 15.1 | 12.8 | 7.44 | 2.030 | D1+D2 | available |
| `6q9c_A` | primary | 155 | R3/R1/R2 | 200 | 8.9 | 6.7 | 4.80 | 1.854 | D1 only | unavailable |
| `7rm7_A` | reserve 1 | 228 | R3/R1/R2 | 200 | 5.7 | 4.1 | 4.80 | 1.188 | D1 only | unavailable |
| `7aex_A` | reserve 2 | 275 | R3/R1/R2 | 200 | 6.4 | 3.5 | 4.80 | 1.333 | D1 only | unavailable |
| `7p46_A` | reserve 3 | 282 | R3/R1/R2 | 320 | 15.8 | 12.4 | 7.68 | 2.057 | D1 only | unavailable |

H availability is optional secondary evidence and was not used to define or
reject a task. Three tasks have independently supported D1+D2 routes; seven
are explicitly labelled single-reference-route benchmarks.

## Frozen A/B distance milestones

Residue indices are zero-based indices in the extracted protein chain.
`close` means distance at or below the threshold; `open` means distance at or
above it. All A, B, and terminal target conditions must persist for two
generated frames. They are available over frames 1--24, but the state machine
requires strict A then B then target completion and forbids completing two
successive stages on the same frame.

| Protein | A milestone | B milestone |
|---|---|---|
| `6jv8_A` | 0--67 close, <=1.4965 nm | 25--74 open, >=3.5193 nm |
| `7bwf_B` | 41--80 close, <=3.7936 nm | 41--82 close, <=3.8093 nm |
| `6tly_A` | 1--97 open, >=4.0207 nm | 0--78 open, >=3.1205 nm |
| `7s86_A` | 5--52 open, >=3.2246 nm | 0--52 open, >=4.0004 nm |
| `6rrv_A` | 0--48 open, >=3.8106 nm | 1--41 open, >=4.2736 nm |
| `6gus_A` | 0--64 close, <=2.0207 nm | 0--63 close, <=1.9168 nm |
| `6q9c_A` | 17--125 open, >=4.2861 nm | 18--84 open, >=4.6676 nm |
| `7rm7_A` | 4--116 close, <=4.2880 nm | 0--146 close, <=7.6075 nm |
| `7aex_A` | 108--157 open, >=1.1706 nm | 74--157 open, >=3.6099 nm |
| `7p46_A` | 2--102 open, >=2.3402 nm | 0--102 open, >=2.7210 nm |

Exact target PC1 intervals, source frames, sampled transition-path indices,
support crossings, and H diagnostics are retained in each protein's
`selection_report.yaml` and `duet_programs.yaml`.

## Baseline launch matrix

Every method uses the same frozen ConfRover checkpoint, protein-specific lag,
24-frame horizon, endpoint and ordered tasks, five seed IDs, and nominal
decoder-population budget `B=64`.

| Method | K | M | Outer SMC | Inner action | Outer incremental weight |
|---|---:|---:|---|---|---|
| Frozen | 64 | 1 | no | none | none |
| Outer-only | 64 | 1 | yes | none | `psi_t / psi_(t-1)` |
| Inner-only | 1 | 64 | no | checkpoint 0.95 steering | none |
| Complete Nested | 16 | 4 | yes | choose after all M frames finish | proper `Zhat_t / psi_(t-1)` |
| DuET-MD | 16 | 4 | yes | resample at reverse-diffusion progress 0.95, then endpoint correction | proper `Zhat_t / psi_(t-1)` |

Outer resampling is systematic at `ESS <= 0.5 K`. Inner resampling is
systematic. Complete Nested and DuET use exactly the same `K=16, M=4`; their
primary difference is whether the same candidate computation is reallocated
inside diffusion time. `naive_dual` remains a diagnostic double-counting
ablation and is not part of this five-method confirmatory suite.

### Relation to the historical 7LP1 experiments

The new five-method budget-64 matrix is **not** a literal rerun of one earlier
7LP1 table. The 7LP1 work proceeded in stages:

1. The original horizon-8 exploratory factorial used budget 16: Frozen and
   Outer-only `K16 x M1`, Inner-only `K1 x M16`, and Complete Nested/DuET
   `K4 x M4` (with Best-of-budget and Naive Dual diagnostics).
2. After the v2 ordered task was rebuilt at horizon 24, its outcome-blind
   Frozen reachability gate used `K48 x M1` on seeds 20260916--20260917.
3. The subsequent allocation study compared **only Complete Nested and DuET**:
   budget-32 `K4 x M8` and `K8 x M4`; budget-64 `K8 x M8`, `K4 x M16`, and
   `K16 x M4`.
4. The selected five-seed development comparison used Complete Nested and
   DuET at `K16 x M4`, checkpoint 0.95, seeds 20260920--20260924. Final-v2
   Outer-only and Inner-only cells matching that selected comparison were not
   run.

Some historical 7LP1 `resolved_config.yaml` files retain the bootstrap field
`decoder_population_budget: 16` because the old command-line K/M override did
not update that metadata field. The executed metrics and NFE accounting are
unambiguous: both selected methods used `K=16`, `M=4`, and 305,664 reverse
decoder evaluations per ordered-task seed, i.e. actual population budget 64.
The override helper now updates this metadata for future runs.

## Artifacts and remaining boundary

- Cohort/attrition report:
  `data/duet/protein_benchmark/route_v3_1/cohort_preparation_report.yaml`
- Static audit:
  `data/duet/protein_benchmark/route_v3_1/cohort_prelaunch_audit.yaml`
- Per-protein configs:
  `configs/duet/protein_benchmark/route_v3_1/*_main.yaml`
- Frozen catalogs and selection reports:
  `data/duet/protein_benchmark/route_v3_1/<protein>/`
- Public ATLAS archives, extracted trajectories, endpoint PDBs, and checksums:
  `${DUET_ASSET_ROOT}/data/atlas*`

Static construction is complete. Before a main five-method comparison can be
launched on the experiment host, each protein must still pass the predeclared
Frozen reachability and generated whole-path validity decision. The model
representations and checkpoint-predictivity diagnostic have passed. The K=1
smoke run is not a reachability estimate: all ten Frozen paths missed the full
ordered task, and two paths (`7bwf_B`, `6rrv_A`) exceeded the 5.501 Angstrom
hard adjacent-CA threshold in one or more intermediate frames. These isolated
paths are retained, not silently discarded, and their recurrence is being
measured in the K=8 pilot and the larger K=48 eligibility run.
Failure at a support gate remains pre-method attrition and must not be repaired
after inspecting Complete Nested or DuET.

### GPU support-gate record (2026-09-05)

The remote staging root is
`/workspace/sejin/AI_MD_NSMC_v31_stage`; public ATLAS assets and cached
representations are under `/workspace/sejin/confrover_mh_steering`. No main
five-method evaluation seed has been launched.

For checkpoint diagnostics, each protein used its saved K=1 Frozen history at
the physical step immediately preceding the D1 median A crossing. At reverse
progress 0.95, eight paired checkpoint candidates were each continued four
times with independent noise. All ten had positive checkpoint-selection gain
and Spearman rank correlation above 0.5 (observed range 0.833--1.000). This is
a one-step ranking diagnostic, not evidence that DuET beats Complete Nested.

The completed K=8 Frozen pilot cases available on 2026-09-06 were:

| Protein | A reached | A then B | Full A-B-target | Whole-path valid |
|---|---:|---:|---:|---:|
| `6jv8_A` | 3/8 | 0/8 | 0/8 | 7/8 |
| `6gus_A` | 3/8 | 0/8 | 0/8 | 8/8 |
| `6q9c_A` | 1/8 | 1/8 | 0/8 | 7/8 |
| `6rrv_A` | 2/8 | 0/8 | 0/8 | 7/8 |
| `6tly_A` | 1/8 | 0/8 | 0/8 | 2/8 |
| `7aex_A` | 0/8 | 0/8 | 0/8 | 7/8 |
| `7bwf_B` | 0/8 | 0/8 | 0/8 | 7/8 |
| `7rm7_A` | 0/8 | 0/8 | 0/8 | 8/8 |
| `7s86_A` | 0/8 | 0/8 | 0/8 | 7/8 |

The K=8 pilot uses diagnostic seed 20260992. It is an outcome-blind escalation
check and does not replace the final Frozen reachability sample-size rule. At
this stage only `6q9c_A` has an observed A-then-B path and no protein has a
full A-B-target success. `6tly_A` also has a material whole-path geometry issue
(2/8 strict-valid paths). Therefore the main comparison must not be launched
from these pilot outcomes without resolving the reachability and validity
gates. The final `7p46_A` pilot was resumed in the background after the client
connection interrupted the original queue.

### Overnight Frozen eligibility launch (2026-09-06)

The final Frozen support queue was launched persistently on two H100 pods. It
uses only the ordered task, `K=48, M=1`, and two diagnostic seeds
`20260993--20260994` per protein. Each protein-seed run has 229,248 reverse
decoder evaluations; the full 20-run queue has 4,584,960. Existing nonempty
`metrics.json` files are skipped, so the queue is restart-safe.

- GPU1 order: `6tly_A`, `6jv8_A`, `7bwf_B`, `7s86_A`, `6rrv_A`.
- GPU2 order: `6q9c_A`, `6gus_A`, `7rm7_A`, `7aex_A`, `7p46_A`.
- GPU1 started immediately with `6tly_A`, seed `20260993`.
- GPU2 waits for the final `7p46_A` K=8 pilot process and then starts
  automatically with `6q9c_A`, seed `20260993`.

This is an eligibility run, not a five-method result. Its outputs determine
whether each task has nonzero but nonsaturated Frozen reachability and whether
whole-path geometry is acceptable before any main evaluation seed is used.

### Main B=64 launch and revised Frozen rule (2026-09-06)

The K=48 eligibility queue was cancelled before completing its first run. A
finite Frozen sample with zero full successes does not establish zero proposal
support; requiring a nonzero Frozen success would also bias a rare-event
benchmark toward easy tasks. Frozen zero-success outcomes are therefore kept
as valid baseline results with uncertainty, rather than used as a mandatory
pre-method exclusion gate. Task definitions remain frozen and are not retuned
after method outcomes.

The main queue now processes one protein at a time. Two H100 workers atomically
claim different seed blocks for the same protein. A block completes all five
methods for one protein/task/seed in this order: Frozen, Complete Nested,
DuET-MD, Outer-only, Inner-only. All five ordered-task seeds are completed
before endpoint runs begin, and both tasks finish before the queue advances to
the next protein. The first active blocks are `6jv8_A/ordered/20261001` and
`6jv8_A/ordered/20261002`; both began with resolved Frozen `K=64, M=1` and
budget 64.

The initial five-seed queue was then stopped before completing a run, at the
user's request to finish coverage across proteins before adding replicate
seeds. The active queue uses only main seed `20261001`, for a total of
`5 methods x 10 proteins x 2 tasks = 100` runs. GPU1 evaluates Frozen,
Outer-only, and Inner-only; GPU2 evaluates Complete Nested and DuET. Both lanes
must finish the same protein/task before either advances. The active initial
runs are `6jv8_A/ordered/frozen/20261001` and
`6jv8_A/ordered/complete_nested/20261001`.

After 14/100 runs completed, the fixed two-lane barrier was removed because
the Frozen/Outer lane was materially slower than the Complete/DuET lane. Both
H100 workers now share an atomic method-level work queue. They prioritize the
earliest protein and task, but if that unit is already locked by the other GPU
they immediately claim the next incomplete method instead of waiting. At the
transition, GPU1 started `7bwf_B/endpoint/Frozen` while GPU2 restarted the
unfinished `7bwf_B/ordered/Inner-only` run.
