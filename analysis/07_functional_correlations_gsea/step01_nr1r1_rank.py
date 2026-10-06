"""Step01 (nr1r1_analysis_M5/gsea): ranked gene list for the CAR+ spots of NR1 + R1, log2FC(PR/GR).

Same pipeline as `../../nr1_tcell_gsea/step01_nr1_rank.py`, with the sample set changed from
{NR1} to {NR1, R1} (user decision: Control/Vehicle is NOT part of the GSEA - its 29 CAR+ spots
are mostly extratumoral and have no GR/PR label).

Grouping : `GRPR_quad_M5` (grpr_mem_exh/GRPR_mem_exh_final_labels.csv) - GR = MEMORY_z>0 & EXHAUSTED_z<0,
           PR = MEMORY_z<0 & EXHAUSTED_z>0, z per sample.  NR1: GR 66 / PR 50, R1: GR 37 / PR 39.

Ranking  : per section the pseudobulk CPM log2FC(PR/GR) = log2((CPM_PR+1)/(CPM_GR+1)); the two
           sections are averaged with equal weight (identical to the step50 convention: the groups are
           balanced inside every section, so the paired-by-section average is batch-free).
           A pooled-CPM variant over both sections is written as a sensitivity ranking.
           Descending: rank 1 = most PR-high; positive NES = PR-high, negative NES = GR-high.

Ties     : with 103 vs 89 spots the pseudobulk CPM is quantised and many genes share their log2FC
           exactly; `rank_metric` spreads the exactly tied values inside a band narrower than the
           smallest gap between distinct values so fgsea gets a unique, reproducible ordering.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import numpy as np
import pandas as pd
import anndata as ad
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
RCL = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/recluster_obj'
OUT = os.path.join(WD, 'nr1r1_analysis_M5', 'gsea')
SAMPLES = ['NR1', 'R1']
DETECT_MIN = 0.1
MARKER_PR = ['Havcr2', 'Entpd1', 'Ctla4', 'Lag3', 'Pdcd1', 'Tigit', 'Tox', 'Bach2']
MARKER_GR = ['Ccr7', 'Tcf7', 'Lef1', 'Sell', 'Il7r', 'Il2ra', 'Tnf', 'Gzmb', 'Ifng', 'Nkg7']

def lognorm(X):
    s = np.asarray(X.sum(1)).ravel()
    s[s == 0] = 1
    return X.multiply(10000.0 / s[:, None]).tocsr()

def section_metrics(C, lab):
    """pseudobulk CPM + mean log1p(CP10K) per group for one section"""
    g = lab == 'GR'
    p = lab == 'PR'
    if g.sum() < 5 or p.sum() < 5:
        return None
    Cg = C[g].tocsr().astype(np.float64)
    Cp = C[p].tocsr().astype(np.float64)
    cg = np.asarray(Cg.sum(0)).ravel()
    cp = np.asarray(Cp.sum(0)).ravel()
    mg = np.asarray(lognorm(Cg).mean(0)).ravel()
    mp = np.asarray(lognorm(Cp).mean(0)).ravel()
    det = np.asarray((C[g | p] > 0).sum(0)).ravel().astype(float) / int((g | p).sum())
    return dict(n_GR=int(g.sum()), n_PR=int(p.sum()), cg=cg, cp=cp, mg=mg, mp=mp, det=det, depth_GR=float(cg.sum()), depth_PR=float(cp.sum()))

def write_rank(df, path, metric='log2FC_PR_GR'):
    d = df[np.isfinite(df[metric]) & df.gene.notna() & (df.gene.astype(str).str.len() > 0)].copy()
    d = d[d.detect_frac >= DETECT_MIN]
    d = d[~d.gene.duplicated()]
    d = d.sort_values([metric, 'log2FC_meanlog'], ascending=[False, False]).reset_index(drop=True)
    d.insert(0, 'rank', np.arange(1, len(d) + 1))
    v = d[metric].values
    gaps = np.diff(np.sort(np.unique(np.round(v, 12))))
    gaps = gaps[gaps > 0]
    band = gaps.min() / 4 if len(gaps) else 1e-12
    off = np.zeros(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        if j > i:
            off[i:j + 1] = np.linspace(band, -band, j - i + 1)
        i = j + 1
    d['rank_metric'] = v + off
    print(f'  tie structure: {d[metric].nunique()} distinct / {len(d)} genes | smallest gap {(gaps.min() if len(gaps) else np.nan):.3g} | band +-{band:.3g} | unique: {d.rank_metric.nunique() == len(d)}', flush=True)
    d.to_csv(path, index=False, float_format='%.10g')
    return d

def main():
    os.makedirs(os.path.join(OUT, 'tables', 'sensitivity'), exist_ok=True)
    NE = pd.read_csv(os.path.join(WD, 'grpr_mem_exh', 'GRPR_mem_exh_final_labels.csv'), index_col=0)
    NE.index = NE.index.astype(str)
    CAR = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    car_pos = [i for i in CAR.index[CAR.car_pos.astype(bool)] if str(i).split('|')[0] in SAMPLES]
    per_sec, labels, var, pooled = ({}, None, None, {})
    for s in SAMPLES:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'))
        names = np.array([f'{s}|{b}' for b in a.obs_names])
        var = a.var_names.to_numpy().astype(str)
        pos = {n: i for i, n in enumerate(names)}
        keep = [n for n in car_pos if n in pos]
        C = a.layers['counts'].tocsr()[[pos[n] for n in keep]]
        lab = NE['GRPR_quad_M5'].reindex(keep).values
        L = pd.DataFrame({'sample': s, 'quad_M5': lab, 'MEMORY_z': NE['g_MEMORY_z'].reindex(keep).values, 'EXHAUSTED_z': NE['g_EXHAUSTED_z'].reindex(keep).values}, index=keep)
        labels = L if labels is None else pd.concat([labels, L])
        m = section_metrics(C, lab)
        if m is None:
            raise SystemExit(f'{s}: missing GR/PR group')
        per_sec[s] = m
        pooled[s] = (C, lab)
        print(f"{s}: CAR+ {len(keep)} | GR {m['n_GR']} / PR {m['n_PR']} | depth GR {m['depth_GR']:.0f} / PR {m['depth_PR']:.0f}", flush=True)
        del a
    labels.to_csv(os.path.join(OUT, 'tables', 'NR1R1_CARpos_spot_labels.csv'))
    genes = var
    cpm_g = np.mean([per_sec[s]['cg'] / max(per_sec[s]['depth_GR'], 1) * 1000000.0 for s in SAMPLES], axis=0)
    cpm_p = np.mean([per_sec[s]['cp'] / max(per_sec[s]['depth_PR'], 1) * 1000000.0 for s in SAMPLES], axis=0)
    R = pd.DataFrame({'gene': genes, 'cpm_GR': cpm_g, 'cpm_PR': cpm_p, 'log2FC_PR_GR': np.mean([np.log2((per_sec[s]['cp'] / max(per_sec[s]['depth_PR'], 1) * 1000000.0 + 1) / (per_sec[s]['cg'] / max(per_sec[s]['depth_GR'], 1) * 1000000.0 + 1)) for s in SAMPLES], axis=0), 'mean_log1p_GR': np.mean([per_sec[s]['mg'] for s in SAMPLES], axis=0), 'mean_log1p_PR': np.mean([per_sec[s]['mp'] for s in SAMPLES], axis=0), 'detect_frac': np.mean([per_sec[s]['det'] for s in SAMPLES], axis=0)})
    R['log2FC_meanlog'] = np.log2((np.expm1(R['mean_log1p_PR']) + 0.1) / (np.expm1(R['mean_log1p_GR']) + 0.1))
    R['n_GR'] = sum((per_sec[s]['n_GR'] for s in SAMPLES))
    R['n_PR'] = sum((per_sec[s]['n_PR'] for s in SAMPLES))
    D = write_rank(R, os.path.join(OUT, 'tables', 'NR1R1_rank_log2FC_PR_GR.csv'))
    print(f'primary ranking (per-section average): {len(D)} genes | range {D.log2FC_PR_GR.min():+.2f} .. {D.log2FC_PR_GR.max():+.2f}', flush=True)
    cg = np.zeros(len(genes))
    cp = np.zeros(len(genes))
    for s in SAMPLES:
        cg += per_sec[s]['cg']
        cp += per_sec[s]['cp']
    pooled_fc = np.log2((cp / cp.sum() * 1000000.0 + 1) / (cg / cg.sum() * 1000000.0 + 1))
    Rs = R.copy()
    Rs['log2FC_PR_GR'] = pooled_fc
    Ds = write_rank(Rs, os.path.join(OUT, 'tables', 'sensitivity', 'NR1R1_rank_pooledCPM.csv'))
    print(f'  sensitivity pooled-CPM: {len(Ds)} genes', flush=True)
    mk = D.set_index('gene')
    rows = []
    for side, gs in [('PR-high (expected top)', MARKER_PR), ('GR-high (expected bottom)', MARKER_GR)]:
        for g in gs:
            rows.append(dict(side=side, gene=g, rank=int(mk.loc[g, 'rank']) if g in mk.index else np.nan, n_genes=len(D), log2FC_PR_GR=float(mk.loc[g, 'log2FC_PR_GR']) if g in mk.index else np.nan, in_NR1=CAR.index.isin([]) if False else np.nan))
    M = pd.DataFrame(rows).drop(columns=['in_NR1'])
    M.to_csv(os.path.join(OUT, 'tables', 'NR1R1_rank_marker_check.csv'), index=False)
    print('\nmarker rank check (1 = most PR-high):')
    print(M.to_string(index=False))
    print('\nwrote tables/NR1R1_rank_log2FC_PR_GR.csv (+ labels, marker check, sensitivity)')
if __name__ == '__main__':
    main()
