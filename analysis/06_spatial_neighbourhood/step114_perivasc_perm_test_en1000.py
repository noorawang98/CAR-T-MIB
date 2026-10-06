"""Step114: CCR1+ myeloid x CAF neighbourhood enrichment with a UNIFIED perivascular
definition, 1000 within-slide permutations, tabular statistics only.

Unified perivascular definition
-------------------------------
Two fixed spot masks are reported for the same curve, one primary and one sensitivity:

  primary      peri_knn6_ge3 : spot whose 6 nearest neighbours contain >=3 vessel-high spots
                               (the vote rule already used on the mouse side, step61/step65-66)
  sensitivity  peri_rad60px  : spot within 60 px of any vessel-high spot
                               (VASC_PX = 60 in step41/42, ~1.2x the bin50 nearest-neighbour
                                spacing; 1 px = 1 um here)

Superseded and NOT used any more: the 'vessel-high U its 40-NN' region of step13/step40, which
covers 0.92-1.00 of every section in bin50 data and therefore cannot be distinguished from the
whole tissue. Only the CCR1+ myeloid x CAF family is affected; the MAN1A1+ macrophage x CAF
analysis (step50) keeps its own region definitions.

Statistic and the reference value 1
-----------------------------------
  E(k) = mean_{i: CAF-high} [ fraction of i's k nearest neighbours that are CCR1+ myeloid-high ]
         / global CCR1+ myeloid-high fraction,   k = 1..40
  Numerator is the CCR1+ myeloid-high fraction in the neighbourhood, denominator the whole-section
  fraction. If the two cell populations were spatially unrelated the neighbourhood fraction would
  equal the global fraction, so E(k) = 1 is the no-association expectation, and the permuted null
  is centred on exactly 1 (the label count, hence the denominator, is preserved).
  Both q75 masks and the vessel-high mask are computed within each section.

Null and p values
-----------------
  The CCR1+ myeloid-high label is permuted within each section (label count preserved),
  N_PERM = 1000 shuffles, RNG seed 0.
  Per section and k, three p values come from the same null:
    p_perm_enrich  : one-sided  (1 + #{null >= observed}) / (N_PERM + 1)
    p_perm_deplete : one-sided  (1 + #{null <= observed}) / (N_PERM + 1)
    p_perm_two     : two-sided  min(1, 2 * min(p_perm_enrich, p_perm_deplete))   <- primary
  q_BH           : BH across the 40 k values within a section, on p_perm_two
  Group level    : the section-level p values are combined per group by Fisher's method;
                   q_BH across k within the group (per-group family), and q_BH_panel across all
                   group x k combinations of the same region (panel family, the family used for
                   the 'strongest result' annotation on the figure).
  Contrast       : per k, Mann-Whitney across sections (PR vs SD/PD) + BH; curve AUC single test.

Inputs : <WD>/tables/ccr1_spots_all.csv
Outputs: <WD>/update_statistics/tables/perivasc_perm_test_1000_en.csv
         <WD>/update_statistics/tables/perivasc_perm_group_1000_en.csv
         <WD>/update_statistics/tables/perivasc_group_compare_1000_en.csv
         <WD>/update_statistics/tables/perivasc_auc_compare_1000_en.csv
         <WD>/update_statistics/tables/perivasc_definition_coverage_1000_en.csv
Usage  : ST_WD=<workdir> python step114_perivasc_perm_test_en1000.py
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import shutil
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import chi2, mannwhitneyu
WD = os.environ.get('ST_WD', PROJECT_ROOT + '/destvi_260911')
TAB = os.path.join(WD, 'tables')
OUT = os.path.join(WD, 'update_statistics')
TF = os.path.join(OUT, 'tables')
FF = os.path.join(OUT, 'figs')
for d in (TF, FF):
    os.makedirs(d, exist_ok=True)
KMAX = 40
N_PERM = 1000
RNG_SEED = 0
KNN_VOTE = 6
VOTE_MIN = 3
RAD_PX = 60.0
REG_ALL = 'all'
REG_KNN = 'peri_knn6_ge3'
REG_RAD = 'peri_rad60px'
REGIONS = [REG_ALL, REG_KNN, REG_RAD]
PRIMARY_REGIONS = [REG_ALL, REG_KNN]
SENS_REGIONS = [REG_ALL, REG_RAD]
COL = {'PR': '#8e44ad', 'SD/PD': '#f1c40f'}
COL_EDGE = {'PR': '#5b2c6f', 'SD/PD': '#b7950b'}
SUFFIX = '_1000_en'
FRAC = 0.75

def q75(v):
    v = np.asarray(v, float)
    return v >= np.quantile(v, FRAC)

def q75_thr(v):
    return np.quantile(np.asarray(v, float), FRAC)

def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    qq = np.full_like(p, np.nan)
    if ok.sum():
        o = np.argsort(p[ok])
        m = ok.sum()
        v = p[ok][o] * m / (np.arange(m) + 1)
        v = np.minimum.accumulate(v[::-1])[::-1]
        t = np.empty(m)
        t[o] = np.clip(v, 0, 1)
        qq[ok] = t
    return qq

def curve_and_null(x, y, X, Y, kmax=KMAX, n_perm=N_PERM, rng=None):
    """Observed E(k) and the permutation null matrix (n_perm, kmax); X is shuffled, Y is fixed."""
    n = len(x)
    if Y.sum() < 5 or X.sum() < 5 or n < 50:
        return (None, None)
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    tree = cKDTree(np.c_[x, y])
    kk = min(kmax + 1, n)
    idx = tree.query(np.c_[x, y], k=kk, workers=-1)[1][:, 1:]
    IDX = idx[np.nonzero(Y)[0]]
    if IDX.shape[1] < kmax:
        IDX = np.pad(IDX, ((0, 0), (0, kmax - IDX.shape[1])), mode='edge')
    pX = max(float(X.mean()), 1e-09)
    obs = np.cumsum(X[IDX], axis=1).mean(0) / np.arange(1, kmax + 1) / pX
    null = np.empty((n_perm, kmax))
    for b in range(n_perm):
        Xp = rng.permutation(X)
        null[b] = np.cumsum(Xp[IDX], axis=1).mean(0) / np.arange(1, kmax + 1) / pX
    return (obs, null)

def region_masks(x, y, vasc_hi):
    """Return the whole-tissue, primary (kNN vote) and sensitivity (radius) masks."""
    tree = cKDTree(np.c_[x, y])
    n = len(x)
    vasc = np.zeros(n, bool)
    vasc[vasc_hi] = True
    nbr = tree.query(np.c_[x, y], k=min(KNN_VOTE + 1, n), workers=-1)[1][:, 1:]
    knn = vasc[nbr].sum(1) >= VOTE_MIN
    hit = tree.query_ball_point(np.c_[x[vasc_hi], y[vasc_hi]], RAD_PX)
    rad = np.zeros(n, bool)
    if len(hit):
        rad[np.unique(np.concatenate([np.asarray(h, int) for h in hit if len(h)]))] = True
    return (np.ones(n, bool), knn, rad)

def region_title(region):
    if region == REG_ALL:
        return 'Whole tissue'
    if region == REG_KNN:
        return f'Perivascular (>={VOTE_MIN} of {KNN_VOTE} NN vessel-high)'
    return f'Perivascular (<= {RAD_PX:.0f} px of a vessel-high spot)'

def fmt_p(p):
    """Exact P for a figure annotation, mathtext, never a bare inequality."""
    if p is None or not np.isfinite(p):
        return 'NA'
    if p >= 0.01:
        return f'{p:.3f}'
    if p >= 0.001:
        return f'{p:.4f}'
    m, e = f'{p:.1e}'.split('e')
    return f'{m} x 10$^{{{int(e)}}}$'

def strongest_text(C, region, pair='mm_caf', family_col='q_BH_panel'):
    """One-line academic annotation of the most significant group result in this region."""
    z = C[(C.region == region) & (C.pair == pair)].copy()
    if not len(z):
        return ''
    r = z.loc[z[family_col].idxmin()]
    return f'strongest: {r.time} {r.resp}, k = {int(r.k)}\nE(k) = {r.mean_enrich:.2f} +/- {r.se:.2f} (mean +/- SE), {r.direction} in {int(r.n_slide_dir)}/{int(r.n_slide)} sections\n$P$ = {fmt_p(r.p_fisher_dir)} (Fisher, direction-consistent, {int(r.n_slide)} sections)\nsection $P$ (median) = {fmt_p(r.p_slide_median)}, {int(r.n_slide_sig)}/{int(r.n_slide)} sections $q_{{BH}}$ < 0.05\n$q_{{BH}}$ = {fmt_p(r.q_BH)} (40 k)   $q_{{BH,panel}}$ = {fmt_p(r.q_BH_panel)} (group x k)'
FOOTNOTE = f"Within-slide permutation test. E(k) = CCR1+ myeloid-high fraction among the k nearest neighbours of CAF-high spots, divided by the whole-section CCR1+ myeloid-high fraction; E(k) = 1 is the no-association expectation and the permuted null is centred on 1. Per section: {N_PERM:,} label shuffles (counts preserved), exact two-sided P = min(1, 2 x min(P_enrich, P_deplete)); BH-FDR q across the 40 k values. Per group: the group direction at each k is the sign of the mean section E(k) and only the one-sided section P values in that direction are combined by Fisher's method (direction-consistent), so sections pointing the other way cannot create significance; BH-FDR across k (q_BH) and across group x k within a panel (q_BH,panel). Error bars: mean +/- SE across sections; n = number of sections (independent patients). Note that a group mean close to 1 is not evidence of association however small its P."

def main():
    rng = np.random.default_rng(RNG_SEED)
    A = pd.read_csv(os.path.join(TAB, 'ccr1_spots_all.csv'))
    print(f"input: {len(A):,} spots from {A['sample'].nunique()} sections; N_PERM={N_PERM}; seed={RNG_SEED}", flush=True)
    rows, cov_rows = ([], [])
    for s, g in A.groupby('sample'):
        t0 = time.time()
        g = g.reset_index(drop=True)
        x, y = (g.x.values, g.y.values)
        Xmm, Xmy = (q75(g.score_mm), q75(g.score_my))
        Ycaf = q75(g.caf)
        vasc_hi = np.nonzero(q75(g.vasc))[0]
        nn = float(np.median(cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2, workers=-1)[0][:, 1]))
        m_all, m_knn, m_rad = region_masks(x, y, vasc_hi)
        masks = {REG_ALL: m_all, REG_KNN: m_knn, REG_RAD: m_rad}
        for region, mask in masks.items():
            cov_rows.append(dict(sample=s, patient=g.patient.iloc[0], time=g.time.iloc[0], resp=g.resp.iloc[0], region=region, n_spot=int(len(g)), n_region=int(mask.sum()), coverage=float(mask.mean()), nn=nn, n_vasc_high=int(len(vasc_hi)), vasc_high_frac=float(len(vasc_hi) / len(g))))
        print(f'{s:<36} n={len(g):>6}  coverage: knn6_ge3={m_knn.mean():.2f} rad60px={m_rad.mean():.2f}', flush=True)
        for region, mask in masks.items():
            if mask.sum() < 50:
                print(f'  skip {region}: only {int(mask.sum())} spots', flush=True)
                continue
            xr, yr = (x[mask], y[mask])
            for pair, X in [('mm_caf', Xmm), ('my_caf', Xmy)]:
                obs, null = curve_and_null(xr, yr, X[mask], Ycaf[mask], rng=rng)
                if obs is None:
                    continue
                p_ge = (1 + (null >= obs).sum(0)) / (N_PERM + 1)
                p_le = (1 + (null <= obs).sum(0)) / (N_PERM + 1)
                p_two = np.minimum(1.0, 2 * np.minimum(p_ge, p_le))
                q_two = bh(p_two)
                q_ge = bh(p_ge)
                for k in range(KMAX):
                    rows.append(dict(sample=s, patient=g.patient.iloc[0], time=g.time.iloc[0], resp=g.resp.iloc[0], region=region, pair=pair, k=k + 1, enrich=float(obs[k]), p_perm_enrich=float(p_ge[k]), p_perm_deplete=float(p_le[k]), p_perm_two=float(p_two[k]), q_BH=float(q_two[k]), q_BH_enrich=float(q_ge[k]), n_spot=int(mask.sum()), nn=nn, coverage=float(mask.mean()), vasc_high_frac=float(len(vasc_hi) / len(g))))
        print(f'  done in {time.time() - t0:.1f}s', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(TF, f'perivasc_perm_test{SUFFIX}.csv'), index=False)
    pd.DataFrame(cov_rows).to_csv(os.path.join(TF, f'perivasc_definition_coverage{SUFFIX}.csv'), index=False)
    out = []
    for (region, pair, time_p, resp), g in R.groupby(['region', 'pair', 'time', 'resp']):
        for k, gk in g.groupby('k'):
            e = gk.enrich.values
            sign = 1 if e.mean() >= 1 else -1
            p_dir = gk.p_perm_enrich.values if sign > 0 else gk.p_perm_deplete.values
            n_dir = int((e > 1).sum()) if sign > 0 else int((e < 1).sum())
            pk2 = np.clip(gk.p_perm_two.values, 1e-12, 1)
            pke = np.clip(gk.p_perm_enrich.values, 1e-12, 1)
            pkd = np.clip(gk.p_perm_deplete.values, 1e-12, 1)
            pdir = np.clip(p_dir, 1e-12, 1)
            out.append(dict(region=region, pair=pair, time=time_p, resp=resp, k=k, n_slide=len(gk), mean_enrich=e.mean(), median_enrich=float(np.median(e)), se=e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan, direction='enriched' if sign > 0 else 'depleted', n_slide_dir=n_dir, p_fisher_dir=1 - chi2.cdf(-2 * np.log(pdir).sum(), 2 * len(pdir)), p_fisher_two=1 - chi2.cdf(-2 * np.log(pk2).sum(), 2 * len(pk2)), p_fisher_enrich=1 - chi2.cdf(-2 * np.log(pke).sum(), 2 * len(pke)), p_fisher_deplete=1 - chi2.cdf(-2 * np.log(pkd).sum(), 2 * len(pkd)), p_slide_median=float(np.median(gk.p_perm_two.values)), n_slide_sig=int((gk.q_BH < 0.05).sum()), frac_slide_p05=float((gk.p_perm_two < 0.05).mean()), coverage=float(gk.coverage.iloc[0])))
    C = pd.DataFrame(out)
    for col in ('q_BH', 'q_BH_dir', 'q_BH_two', 'q_BH_panel'):
        C[col] = np.nan
    for _, g in C.groupby(['region', 'pair', 'time', 'resp']):
        C.loc[g.index, 'q_BH'] = bh(g.p_fisher_dir.values)
        C.loc[g.index, 'q_BH_dir'] = bh(g.p_fisher_dir.values)
        C.loc[g.index, 'q_BH_two'] = bh(g.p_fisher_two.values)
    for _, g in C.groupby(['region', 'pair']):
        C.loc[g.index, 'q_BH_panel'] = bh(g.p_fisher_dir.values)
    C.to_csv(os.path.join(TF, f'perivasc_perm_group{SUFFIX}.csv'), index=False)
    cmp_rows, auc_rows = ([], [])
    for (region, pair, time_p), g in R.groupby(['region', 'pair', 'time']):
        for k, gk in g.groupby('k'):
            a = gk[gk.resp == 'PR'].enrich.values
            b = gk[gk.resp == 'SD/PD'].enrich.values
            if len(a) < 2 or len(b) < 2:
                continue
            sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
            cmp_rows.append(dict(region=region, pair=pair, time=time_p, k=k, n_PR=len(a), n_SDPD=len(b), mean_PR=a.mean(), mean_SDPD=b.mean(), diff=b.mean() - a.mean(), cohens_d=(b.mean() - a.mean()) / sd if sd > 0 else np.nan, p_MWU=mannwhitneyu(a, b, alternative='two-sided').pvalue))
        auc = g.groupby(['sample', 'resp']).enrich.mean().reset_index()
        a = auc[auc.resp == 'PR'].enrich.values
        b = auc[auc.resp == 'SD/PD'].enrich.values
        if len(a) >= 2 and len(b) >= 2:
            auc_rows.append(dict(region=region, pair=pair, time=time_p, n_PR=len(a), n_SDPD=len(b), AUC_PR=a.mean(), AUC_SDPD=b.mean(), diff=a.mean() - b.mean(), p_MWU=mannwhitneyu(a, b, alternative='two-sided').pvalue))
    GC = pd.DataFrame(cmp_rows)
    GC['q_BH'] = np.nan
    GC['q_BH_panel'] = np.nan
    for _, g in GC.groupby(['region', 'pair', 'time']):
        GC.loc[g.index, 'q_BH'] = bh(g.p_MWU.values)
    for _, g in GC.groupby(['region', 'pair']):
        GC.loc[g.index, 'q_BH_panel'] = bh(g.p_MWU.values)
    GC.to_csv(os.path.join(TF, f'perivasc_group_compare{SUFFIX}.csv'), index=False)
    AC = pd.DataFrame(auc_rows)
    if len(AC):
        AC['q_BH'] = bh(AC.p_MWU.values)
    AC.to_csv(os.path.join(TF, f'perivasc_auc_compare{SUFFIX}.csv'), index=False)
    pd.set_option('display.width', 250)
    print('\n=== definition coverage (min / mean / max across sections) ===')
    print(pd.DataFrame(cov_rows).groupby('region').coverage.agg(['min', 'mean', 'max']).round(3).to_string())
    print('\n=== group-level Fisher combination (pair = mm_caf); direction-consistent Fisher is primary ===')
    z = C[C.pair == 'mm_caf'].copy()
    z['sig'] = z.q_BH < 0.05
    print(z.groupby(['region', 'time', 'resp']).agg(n_k=('k', 'size'), n_sig=('sig', 'sum'), min_p_dir=('p_fisher_dir', 'min'), min_p_two=('p_fisher_two', 'min'), min_q_BH=('q_BH', 'min'), min_q_panel=('q_BH_panel', 'min'), E_min=('mean_enrich', 'min'), E_max=('mean_enrich', 'max'), max_abs_dev=('mean_enrich', lambda s: float(np.abs(s - 1).max()))).round(4).to_string())
    print('\n=== strongest group x k result per region (panel family, direction-consistent) ===')
    for region in REGIONS:
        r = C[(C.region == region) & (C.pair == 'mm_caf')]
        r = r.loc[r.q_BH_panel.idxmin()]
        print(f'  {region:<15} {r.time} {r.resp} k={int(r.k):>2}  E={r.mean_enrich:.3f} ({r.direction}, {int(r.n_slide_dir)}/{int(r.n_slide)} sections)  P_dir={r.p_fisher_dir:.3e}  q_BH={r.q_BH:.4f}  q_panel={r.q_BH_panel:.4f}  sectionP_median={r.p_slide_median:.4f}')
    print('\n=== group contrast, curve AUC (PR vs SD/PD across sections) ===')
    if len(AC):
        print(AC.round(4).to_string(index=False))
    print('\n=== group contrast, per-k Mann-Whitney (pair = mm_caf) ===')
    gz = GC[GC.pair == 'mm_caf']
    print(gz.groupby(['region', 'time']).agg(n_k=('k', 'size'), n_sig=('q_BH', lambda s: int((s < 0.05).sum())), min_p=('p_MWU', 'min'), min_q=('q_BH', 'min'), min_q_panel=('q_BH_panel', 'min')).round(4).to_string())
    print('\n=== sections per group ===')
    print(C[C.pair == 'mm_caf'].groupby(['region', 'time', 'resp']).n_slide.first().to_string())
if __name__ == '__main__':
    main()
