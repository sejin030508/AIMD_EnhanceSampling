# Small-protein transition recovery pilot — launch-ready report

Date: 2026-09-10 (Asia/Seoul)  
Status: **READY FOR THE FIXED 24-RUN MATRIX; production runs not started**

## 1. Scope and isolation

This setup implements the supplied Chignolin/Trp-cage/BBA unfolded-to-folded
pilot without changing the existing Phase A or cryptic-pocket source, configs,
or results.

- Remote project: `/workspace/sejin/AI_MD_NSMC_phase_b_recovery`
- Isolated code: `/workspace/sejin/AI_MD_NSMC_phase_b_recovery/small_protein_pilot`
- Prepared data: `/workspace/sejin/AI_MD_NSMC_phase_b_recovery/data/small_protein_transition_pilot`
- Output namespace: `/workspace/sejin/AI_MD_NSMC_phase_b_recovery/outputs/small_protein_transition_pilot`
- Production `metrics.json` count at handoff: **0**
- H100 assignment: Chignolin=`work-001`, Trp-cage=`work-002`, BBA=`work-004`
- All three H100s were idle after the preflight finished.

The small-protein entry point monkey-patches only the pocket-specific hidden
observable inside its own process. No shared `src/` or `scripts/` file was
modified; remote `git diff --name-only -- src scripts` was empty at the final
audit.

## 2. Fixed data provenance

Official source: `https://github.com/kiyoung98/tps-dps.git`  
Pinned commit: `61fd65ad2e2f110d65c176a8c8f5c2fe8bdab034`

For every protein, the following official files were copied and SHA256-fixed in
its manifest: `unfolded.pdb`, `folded.pdb`, `tica_model.pkl`, `pmf.npy`,
`xs.npy`, and `ys.npy`. The official metric/dynamics code and
`data/protein.ff14SBonlysc.xml` are also hashed. `path.gro` and the Trp-cage H5
file are preserved but not treated as unbiased reference trajectories; reference
path coverage therefore remains `NA`.

| Protein | Sequence extracted from PDB | Length | Prepared start-to-target BB RMSD d0 (Å) | Official TICA start-to-target distance |
|---|---|---:|---:|---:|
| Chignolin | `GYDPETGTWG` | 10 | 6.085456 | 4.797010 |
| Trp-cage | `DAYAQWLKDGGPSSGRPPPS` | 20 | 7.199678 | 11.113707 |
| BBA | `EQYTAKYKGRTFRNEKELRDFIEKFKGR` | 28 | 8.069854 | 2.959433 |

All folded targets project to themselves with TICA distance 0, and every start
is outside the fixed 0.75 target basin.

Start and target were each minimized once by reproducing TPS-DPS BaseDynamics:
`protein.ff14SBonlysc.xml + implicit/gbn2.xml`, NoCutoff, HBonds constraints,
zero external bias force, VVVR integrator settings, and
`Simulation.minimizeEnergy()`. Original and minimized all-atom structures are
both retained. Generated frames are never minimized or repaired for sampling,
reward, RMSD, TICA, or validity.

Topology validation requires identical residue order, heavy-atom names, atom
count, element order, and per-residue hydrogen count. The official Chignolin
files use the chemically equivalent first-terminal hydrogen aliases `H` and
`H1`; this naming-only difference is explicitly tolerated and no heavy-atom
mapping is relaxed.

## 3. Frozen experimental matrix

| Dimension | Fixed value |
|---|---|
| Proteins | Chignolin, Trp-cage, BBA |
| Lag | stride 16 = 160 ps; stride 128 = 1.28 ns |
| Horizon | T=32 transitions; 33 structures including x0 |
| Frozen | K=16, M=1; no inner or outer resampling |
| DuET | K=4, M=4; inner resampling at 75% and 90% |
| Seeds | 211, 223 |
| Total | 3 × 2 × 2 × 2 = **24 runs** |
| Model | frozen `confrover_base_20m_v1_0.pt`, SDE, 200 reverse steps |
| Context | full history/context, no restart or reset |
| OOM policy | microbatch=1, offloaded KV, Pairformer chunk=32; never shrink K/M/T/lag/history |

Reward for both final and predicted-clean checkpoint evaluation:

`log psi(x) = -16 * (d(x) / d0)^2`, where `d` is whole-protein common N/CA/C
Kabsch RMSD to the prepared folded target in Å. There is no log-reward floor or
potential clipping. TICA, PMF, energy, and any pocket/event term are excluded
from the reward. DuET uses the existing two-checkpoint FKC and
`Zhat / psi(parent)` outer update; outer systematic resampling retains the
existing `ESS <= 0.5K` policy.

Analysis-only populations and weights are configured at T=8, 16, and 32. They
do not trigger extra resampling.

## 4. Evaluation adapter

The adapter writes final pre-resampling weighted and best-valid whole-backbone
RMSD, explicitly named common-heavy RMSD when the full official atom mask is not
available, fixed official TICA/THP, validity without removing invalid paths from
the denominator, TICA-PMF overlays, lineage-aware TICA-DTW, CA step displacement,
first-hit frame, and conditional sampled-frame ETS.

- THP: Euclidean distance to folded target in the first two official TICA
  coordinates `< 0.75`.
- TICA feature: official folded-PDB topology backbone torsions with
  `cossin=True`; the distributed TICA model is never refit.
- Evaluation environment is separate from ConfRover and pins `deeptime=0.4.4`,
  matching the serialized official model. Installed versions include Python
  3.10, NumPy 1.26.4, OpenMM 8.6, MDTraj 1.10.3, and PyEMMA 2.5.12.
- A TICA/PMF or energy failure is caught as a metric-specific `NA`; completed
  generation and other metrics remain available.
- ETS uses the official protein force field and implicit solvent with no bias or
  restraint energy. Its evaluation-only system is unconstrained so generated
  heavy atoms can be exactly massless/fixed while only hydrogens minimize.
  ConfRover does not emit terminal OXT; the aligned minimized-target OXT is kept
  fixed and is never inserted into RMSD/TICA generated coordinates. This is
  recorded in every ETS result.
- A direct folded-target energy audit was finite for all proteins, with maximum
  generated-heavy displacement below `4e-15 Å`.

## 5. H100 short execution validation (excluded from scientific results)

Validation used T=1, stride 16, seed 199. Both methods had the same nominal
decoder population budget and exactly `16 × 199 = 3,184` decoder NFE.

| Protein | Method | NFE | Wall (s) | Valid path fraction | Weighted final BB RMSD (Å) | Best valid BB RMSD (Å) | Weighted valid THP |
|---|---|---:|---:|---:|---:|---:|---:|
| Chignolin | Frozen | 3,184 | 84.78 | 1.00 | 4.4336 | 2.7851 | 0.125 |
| Chignolin | DuET | 3,184 | 80.05 | 1.00 | 2.6713 | 2.1147 | 0.000 |
| Trp-cage | Frozen | 3,184 | 78.27 | 1.00 | 6.7767 | 5.2604 | 0.000 |
| Trp-cage | DuET | 3,184 | 84.24 | 1.00 | 5.4757 | 5.2233 | 0.000 |
| BBA | Frozen | 3,184 | 84.59 | 1.00 | 7.4853 | 4.5242 | 0.000 |
| BBA | DuET | 3,184 | 92.77 | 1.00 | 5.3325 | 5.2496 | 0.000 |

These T=1 numbers only validate execution and output semantics. In particular,
the two Chignolin Frozen TICA hits are not pilot evidence and must not be pooled
with the 24 production runs.

DuET audit across all three proteins:

- configured completed-reverse fractions: 0.75 and 0.90
- actual reverse update steps: 150 and 180 of 200
- four parent-level checkpoint records per run
- maximum telescoping absolute log error: `2.220446049250313e-16`
- reward/potential clipping evaluations: 0
- checkpoint audit: PASS for every DuET preflight
- all TICA projections and PMF plots completed without error
- all qualifying Chignolin ETS paths completed after the documented H/OXT handling

## 6. Launch procedure

The production matrix is intentionally not running at this handoff. Launch one
protein chain on each H100. Existing preflights are detected and skipped. A
failed case writes its barrier and does not prevent the other proteins from
continuing. All proteins complete seed 211 before seed 223 begins; the seed-211
summary and final 24-cell summary are generated automatically.

```bash
kubectl exec -n dept-dh sejin-h100-1-work-001-zhr2l -- sh -lc \
  '/workspace/sejin/AI_MD_NSMC_phase_b_recovery/small_protein_pilot/run_small_protein_protein_chain.sh chignolin'

kubectl exec -n dept-dh sejin-h100-1-work-002-84brf -- sh -lc \
  '/workspace/sejin/AI_MD_NSMC_phase_b_recovery/small_protein_pilot/run_small_protein_protein_chain.sh trpcage'

kubectl exec -n dept-dh sejin-h100-1-work-004-96566 -- sh -lc \
  '/workspace/sejin/AI_MD_NSMC_phase_b_recovery/small_protein_pilot/run_small_protein_protein_chain.sh bba'
```

The commands should be started concurrently by the controller. No additional
reward sweep, baseline, allocator, horizon extension, or adaptive setting change
is included.

## 7. Readiness decision

**All three proteins are ready; no user intervention is required.** The fixed
data, model representation, sampling path, two-checkpoint arithmetic, evaluator,
per-case failure isolation, barriers, and summaries have all been exercised or
read-only verified. The remaining action is only to start the three production
chains above.

## 8. Production launch update

Launched at **2026-09-10 18:20 KST** without changing the frozen matrix:

| Protein | H100 pod | Chain PID | First active cell |
|---|---|---:|---|
| Chignolin | `sejin-h100-1-work-001-zhr2l` | 3817277 | stride 16, Frozen, seed 211 |
| Trp-cage | `sejin-h100-1-work-002-84brf` | 7085 | stride 16, Frozen, seed 211 |
| BBA | `sejin-h100-1-work-004-96566` | 1510674 | stride 16, Frozen, seed 211 |

All three chain processes were re-parented to PID 1 and confirmed alive; each
had its ConfRover sampler attached to GPU 0. The chains continue through the
fixed seed-211 cells, synchronize, and then continue through seed 223.

## 9. Scheduling amendment: DuET-first execution

At the user's request on 2026-09-10, the already-running cells were preserved
and the remaining execution order was changed **without changing any run
configuration, seed, reward, checkpoint, or evaluation rule**.

- The incumbent Chignolin/Trp-cage `stride16 / DuET / seed211` cells and BBA
  `stride16 / Frozen / seed211` cell continue unchanged.
- After each incumbent cell finishes, all remaining DuET cells for that protein
  run first: seed 211 then 223, stride 16 then 128.
- Only after all three DuET queues complete do the remaining Frozen cells run.

This is a resource-priority amendment, not an adaptive scientific decision:
the seed-223 DuET settings are fixed in advance and are not selected from
seed-211 outcomes. It is recorded because the original first-seed barrier is
no longer the execution order.

## 10. Observed evaluation-time accounting

Production `stride16 / Frozen / seed211` timing separates the sampler wall time
from the interval between `metrics.json` (sampling complete) and
`small_protein_metrics.json` (evaluation complete):

| Protein | Sampling | Post-sampling evaluation | Total | Valid final THP paths | ETS paths |
|---|---:|---:|---:|---:|---:|
| Chignolin | 44.25 min | 0.14 min | 44.38 min | 0 | 0 |
| Trp-cage | 46.56 min | 0.25 min | 46.81 min | 0 | 0 |
| BBA | 46.88 min | 23.98 min | 70.86 min | 4 | 4 |

The Chignolin/Trp-cage post-sampling interval includes TICA projection, PMF
plotting, RMSD/path diagnostics, and file output, so it establishes that those
operations are negligible here. BBA's additional roughly 23.7 minutes is
conditional ETS: four valid final target-reaching paths × 33 saved structures
including x0 = 132 H-only energy minimizations. Thus ETS is about 34% of that
BBA cell's end-to-end time (or 51% extra relative to sampling alone), but is not
a universal bottleneck when there are no qualifying paths.
