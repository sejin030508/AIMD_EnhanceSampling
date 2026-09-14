#!/usr/bin/env python
"""Phase R0 - is the RMSD reward a useful selection signal?

R1 alignment      : per-transition Spearman across particles, backbone RMSD vs TICA distance
R2 discrimination : spread of log psi across particles at each transition
R3 accuracy       : particle weight at t vs whether its descendants reach the target
R4 counterfactual : ranking under the RMSD potential vs under a TICA potential

All four read logs already on disk. No sampling.
"""
from __future__ import annotations
import json, glob, os, sys
import numpy as np

ROOT = '/workspace/sejin/AI_MD_NSMC_phase_b_recovery'
D0 = {'trpcage': 7.200, 'bba': 8.070, 'bbl': 18.074}
TICA_SCALE = 0.75   # success radius, used as the TICA potential's d0 analogue
COEF = 16.0


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3: return np.nan
    a, b = a[ok], b[ok]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/den) if den > 0 else np.nan


def ess(w):
    w = np.asarray(w, float)
    s = w.sum()
    if not np.isfinite(s) or s <= 0: return np.nan
    w = w/s
    return float(1.0/np.sum(w*w))


def cells(pattern):
    for f in sorted(glob.glob(pattern)):
        yield os.path.dirname(f)


def analyse(cell):
    d = cell
    fd = json.load(open(os.path.join(d, 'small_protein_frame_diagnostics.json')))
    m  = json.load(open(os.path.join(d, 'small_protein_metrics.json')))
    b  = json.load(open(os.path.join(d, 'metrics.json')))
    prot = m['protein']; d0 = D0[prot]
    rmsd = np.asarray(fd['whole_backbone_rmsd_a'], float)            # (K, T+1)
    tica = np.asarray(fd['tica_distance_to_folded_target'], float)   # (K, T+1)
    hit  = np.asarray(fd['target_hit'], bool)                        # (K, T+1)
    K, T1 = rmsd.shape; T = T1 - 1
    out = dict(cell=d, protein=prot, method=m['method'], seed=m['seed'],
               horizon=T, K=K, tag=d.split('pvb_full/')[1].split('/')[0])

    # ---- R1 / R4 : per-transition, across particles -------------------
    r1, r4 = [], []
    for t in range(1, T1):
        r, k = rmsd[:, t], tica[:, t]
        r1.append(spearman(-r, -k))                     # closer is better on both
        wr = np.exp(-COEF*(r/d0)**2)
        wk = np.exp(-COEF*(k/TICA_SCALE)**2)
        r4.append(spearman(wr, wk))
    out['R1_spearman_mean'] = float(np.nanmean(r1))
    out['R1_spearman_early'] = float(np.nanmean(r1[:T//3]))
    out['R1_spearman_late']  = float(np.nanmean(r1[-(T//3):]))
    out['R4_rank_corr_mean'] = float(np.nanmean(r4))
    out['R4_rank_corr_early'] = float(np.nanmean(r4[:T//3]))
    out['R4_rank_corr_late']  = float(np.nanmean(r4[-(T//3):]))
    # fraction of transitions where the two potentials pick a different argmax
    diff = []
    for t in range(1, T1):
        wr = np.exp(-COEF*(rmsd[:, t]/d0)**2)
        wk = np.exp(-COEF*(tica[:, t]/TICA_SCALE)**2)
        if np.all(np.isfinite(wr)) and np.all(np.isfinite(wk)):
            diff.append(int(np.argmax(wr) != np.argmax(wk)))
    out['R4_argmax_differs'] = float(np.mean(diff)) if diff else np.nan
    # ESS each potential would induce
    er, ek = [], []
    for t in range(1, T1):
        er.append(ess(np.exp(-COEF*(rmsd[:, t]/d0)**2)))
        ek.append(ess(np.exp(-COEF*(tica[:, t]/TICA_SCALE)**2)))
    out['R4_ess_rmsd'] = float(np.nanmean(er))/K
    out['R4_ess_tica'] = float(np.nanmean(ek))/K

    # ---- R2 : spread of log psi across particles ----------------------
    rec_path = os.path.join(d, 'records.json')
    if os.path.exists(rec_path):
        rec = json.load(open(rec_path))
        by_t = {}
        for r in rec:
            by_t.setdefault(r['t'], []).append(r.get('log_psi'))
        sd, es = [], []
        for t, vals in by_t.items():
            v = np.asarray([x for x in vals if x is not None], float)
            if v.size >= 2:
                sd.append(float(np.std(v)))
                w = np.exp(v - v.max())
                es.append(ess(w)/len(v))
        out['R2_logpsi_sd_mean'] = float(np.mean(sd)) if sd else np.nan
        out['R2_induced_ess_frac'] = float(np.mean(es)) if es else np.nan

    # ---- R3 : weight at t vs descendant target-reaching ---------------
    lin = fd.get('pre_final_lineages')
    if lin:
        lin = np.asarray(lin, int)                # (K, T) ancestor index at each transition
        reached = hit[:, 1:].any(axis=1)          # (K,) final path reached target
        r3 = []
        for t in range(lin.shape[1]):
            anc = lin[:, t]
            # mean reward-rank of ancestors, grouped by whether descendant reached
            w = np.exp(-COEF*(rmsd[:, t+1]/d0)**2)
            rank = np.argsort(np.argsort(w)).astype(float)/max(K-1, 1)
            a = rank[anc][reached]; bq = rank[anc][~reached]
            if a.size and bq.size:
                r3.append(float(a.mean() - bq.mean()))
        out['R3_rank_gap_mean'] = float(np.mean(r3)) if r3 else np.nan
        out['R3_reached_fraction'] = float(reached.mean())
    return out


def main():
    rows = []
    for pat in [f'{ROOT}/outputs/pvb_full/cp7590/*/stride128_t32/*/seed_*/small_protein_metrics.json',
                f'{ROOT}/outputs/pvb_full/t96/bbl/stride128_t32/*/seed_*/small_protein_metrics.json']:
        for c in cells(pat):
            try:
                rows.append(analyse(c))
            except Exception as e:
                print(f'SKIP {c}: {type(e).__name__}: {e}', file=sys.stderr)
    out = f'{ROOT}/scripts/fast_folders_extended/results_phase_r0.json'
    json.dump(rows, open(out, 'w'), indent=1)
    print(f'{len(rows)} cells -> {out}')


if __name__ == '__main__':
    main()
