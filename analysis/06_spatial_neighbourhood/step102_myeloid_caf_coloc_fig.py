"""Step102: myeloid - CAF co-localisation, slide by slide, with the test methods annotated.

Source tables (step26_coloc_marker_heatmap.py)
  results/myeloid_CAF_correlation_by_group.csv       per slide x NDR/DR: Spearman rho of the
  results/myeloid_CAF_colocalisation_by_group.csv    myeloid vs CAF marker modules, and the
                                                     Fisher-exact OR of myeloid-high x CAF-high
This script adds the pooled estimates and heterogeneity statistics the raw tables lack:
  * Fisher-z (inverse-variance) fixed- and random-effects pooling of rho
  * Mantel-Haenszel pooling of the ORs with Robins-Breslow-Greenland SE
  * Cochran's Q and I^2 for both, BH-FDR across the 12 slide x group tests

Output: cart_region/figs_final/myeloid_CAF_coloc_stats.csv (analysis only)
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
from scipy.stats import norm, chi2 as chi2dist, wilcoxon, binomtest, t as tdist
from common import RES, bh
OUT = os.path.join(PROJECT_ROOT + '/mouse/cart_region', 'figs_final')
NDR_C, DR_C, GREY = ('#B22222', '#2C7FB8', '#4D4D4D')
SLIDES = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
GROUPS = ['NDR', 'DR']

def star(q):
    return '****' if q < 0.0001 else '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else 'ns'

def fmt_p(p):
    return f'{p:.1e}' if p < 0.001 else f'{p:.3f}'

def fisher_z(rows, k_col='n'):
    """Inverse-variance pooling of Spearman rho on the Fisher-z scale."""
    r = np.clip(rows['rho_mye_caf'].values.astype(float), -0.999, 0.999)
    z = np.arctanh(r)
    w = rows[k_col].values.astype(float) - 3.0
    keep = w > 0
    z, w = (z[keep], w[keep])
    if len(z) < 2:
        return None
    zf = (w * z).sum() / w.sum()
    se_f = 1 / np.sqrt(w.sum())
    Q = float((w * (z - zf) ** 2).sum())
    df = len(z) - 1
    pQ = float(chi2dist.sf(Q, df))
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (1 / w + tau2)
    zr = (wr * z).sum() / wr.sum()
    se_r = 1 / np.sqrt(wr.sum())
    zst_FE, zst_RE = (zf / se_f, zr / se_r)
    ps = rows['p'].values.astype(float)
    X2 = float(-2 * np.log(np.clip(ps, 1e-300, 1)).sum())
    return dict(k=len(z), rho_FE=float(np.tanh(zf)), lo_FE=float(np.tanh(zf - 1.96 * se_f)), hi_FE=float(np.tanh(zf + 1.96 * se_f)), p_FE=float(2 * norm.sf(abs(zst_FE))), z_FE=float(zst_FE), rho_RE=float(np.tanh(zr)), lo_RE=float(np.tanh(zr - 1.96 * se_r)), hi_RE=float(np.tanh(zr + 1.96 * se_r)), z_RE=float(zst_RE), p_RE=float(2 * norm.sf(abs(zst_RE))), p_RE_t=float(2 * tdist.sf(abs(zst_RE), df=df)), p_wilcox=float(wilcoxon(r).pvalue), p_sign=float(binomtest(int((r > 0).sum()), len(r)).pvalue), X2_comb=X2, p_comb=float(chi2dist.sf(X2, 2 * len(ps))), Q=Q, df=df, p_Q=pQ, I2=I2, tau2=tau2)

def mh_or(rows, perm_p=None):
    """Mantel-Haenszel pooled odds ratio with Robins-Breslow-Greenland variance."""
    a = rows['n_both'].values.astype(float)
    b = (rows['n_mye_high'] - rows['n_both']).values.astype(float)
    c = (rows['n_caf_high'] - rows['n_both']).values.astype(float)
    d = (rows['n'] - rows['n_mye_high'] - rows['n_caf_high'] + rows['n_both']).values.astype(float)
    N = a + b + c + d
    R = a * d / N
    S = b * c / N
    if R.sum() == 0 or S.sum() == 0:
        return None
    or_mh = R.sum() / S.sum()
    P = (a + d) / N
    Qs = (b + c) / N
    PQ = P * Qs
    var = (P * R / 2).sum() / R.sum() ** 2 + ((PQ * R + P * S) / 2).sum() / (R.sum() * S.sum()) + (Qs * S / 2).sum() / S.sum() ** 2
    se = np.sqrt(var)
    with np.errstate(divide='ignore', invalid='ignore'):
        logor = np.log(a * d / (b * c))
        w = 1 / (1 / a + 1 / b + 1 / c + 1 / d)
    ok = np.isfinite(logor) & np.isfinite(w) & (w > 0)
    or_fe = np.exp((w[ok] * logor[ok]).sum() / w[ok].sum())
    Q = float((w[ok] * (logor[ok] - np.log(or_fe)) ** 2).sum())
    df = int(ok.sum()) - 1
    zst = float(np.log(or_mh) / se)
    fps = rows['fisher_p'].values.astype(float)
    X2 = float(-2 * np.log(np.clip(fps, 1e-300, 1)).sum())
    return dict(k=int(ok.sum()), or_MH=float(or_mh), lo=float(np.exp(np.log(or_mh) - 1.96 * se)), hi=float(np.exp(np.log(or_mh) + 1.96 * se)), z=zst, p=float(2 * norm.sf(abs(zst))), p_t=float(2 * tdist.sf(abs(zst), df=df)), p_wilcox=float(wilcoxon(logor[ok]).pvalue), p_sign=float(binomtest(int((rows['odds_ratio'].values > 1).sum()), len(rows)).pvalue), X2_comb=X2, p_comb=float(chi2dist.sf(X2, 2 * len(fps))), or_FE=float(or_fe), Q=Q, df=df, p_Q=float(chi2dist.sf(Q, df)), I2=max(0.0, (Q - df) / Q) if Q > 0 else 0.0)

def main():
    os.makedirs(OUT, exist_ok=True)
    C = pd.read_csv(os.path.join(RES, 'myeloid_CAF_correlation_by_group.csv'))
    L = pd.read_csv(os.path.join(RES, 'myeloid_CAF_colocalisation_by_group.csv'))
    C['q_spearman'] = bh(C.p.values)
    L['q_fisher'] = bh(L.fisher_p.values)
    L['q_perm'] = bh(L.perm_p_toroidal.replace(0, 1 / 1000).values)
    pool_c = {g: fisher_z(C[C.group == g]) for g in GROUPS}
    pool_l = {g: mh_or(L[L.group == g]) for g in GROUPS}
    wpos = {g: i for i, g in enumerate(GROUPS)}
    ylo = min((pool_c[g]['lo_RE'] for g in GROUPS)) - 0.46
    yhi = max((pool_c[g]['hi_RE'] for g in GROUPS)) + 0.2
    for g in GROUPS:
        d = C[C.group == g].set_index('sample').reindex(SLIDES)
        for i, s in enumerate(SLIDES):
            r = d.loc[s]
            x = wpos[g] * 2.6 + i * 0.38
            se = 1 / np.sqrt(max(r.n - 3, 1))
            lo, hi = np.tanh(np.arctanh(np.clip(r.rho_mye_caf, -0.999, 0.999)) + np.array([-1.96, 1.96]) * se)
        p = pool_c[g]
        txt = f"pooled rho {p['rho_RE']:+.3f} [{p['lo_RE']:+.3f}, {p['hi_RE']:+.3f}]\nz = {p['z_RE']:.2f}, p = {fmt_p(p['p_RE'])};  t(5) p = {fmt_p(p['p_RE_t'])}\nWilcoxon p = {fmt_p(p['p_wilcox'])};  Q = {p['Q']:.1f}, p_Q = {fmt_p(p['p_Q'])}, I² = {p['I2'] * 100:.0f}%"
    for g in GROUPS:
        d = L[L.group == g].set_index('sample').reindex(SLIDES)
        for i, s in enumerate(SLIDES):
            r = d.loc[s]
            a = r.n_both
            b = r.n_mye_high - r.n_both
            c = r.n_caf_high - r.n_both
            dd = r.n - r.n_mye_high - r.n_caf_high + r.n_both
            se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
            x = wpos[g] * 8.2 + i * 1.15
        p = pool_l[g]
    x = np.arange(len(SLIDES))
    w = 0.36
    for k, g in enumerate(GROUPS):
        d = L[L.group == g].set_index('sample').reindex(SLIDES)
    yA, yB = (2.0, 0.9)
    for g in GROUPS:
        y = yA if g == 'NDR' else yA - 0.34
        p = pool_c[g]
        p = pool_l[g]
        y = yB if g == 'NDR' else yB - 0.34
    n_all = int(L.n.sum())
    methods = "Methods.  Marker modules: myeloid (`mye`) and CAF (`caf`) z-scores per spot (panel A–B of step26_coloc_marker_heatmap.py).  Correlation: Spearman rank correlation of the two modules, computed separately inside each slide × group (n given in the table); 95% CI by Fisher z-transform with SE = 1/sqrt(n−3); pooling across the 6 slides by inverse-variance on the Fisher-z scale (fixed effect) and by DerSimonian–Laird random effects (diamond); heterogeneity by Cochran's Q and I².  Co-localisation: both modules dichotomised at their within-slide × group 75th percentile, then a two-sided Fisher exact test on the 2×2 table (myeloid-high × CAF-high) gives the odds ratio and its Woolf 95% CI; pooling across slides by Mantel–Haenszel with the Robins–Breslow–Greenland variance; significance of the spatial pattern by a toroidal-shift permutation null (500 random circular shifts of the CAF-high mask on the spot grid, grid step = median nearest-neighbour distance; one-sided p = fraction of null overlaps ≥ observed; p<0.002 = no null exceedance in 500 shifts).  Test of the pooled estimate: the fixed-effect Fisher-z estimate is tested with a z statistic; the random-effects estimate additionally with a t(k−1) = t(5) reference, because only 6 slides back it; and distribution-free with a Wilcoxon signed-rank and a sign test on the 6 slide-level values. The 6 per-slide p-values are also combined by Fisher's method (chi-square with 12 df per group). Multiplicity: Benjamini–Hochberg FDR across the 12 slide × group tests within each test type (stars: q<0.05 *, <0.01 **, <0.001 ***).  Group R1/NDR is the single discordant cell for both tests."
    rows = []
    for g in GROUPS:
        pc, pl = (pool_c[g], pool_l[g])
        rows.append(dict(group=g, n_slides=6, rho_test='Fisher-z inverse-variance (FE) / DerSimonian-Laird (RE)', rho_z=pc['z_RE'], rho_p_z=pc['p_RE'], rho_p_t5=pc['p_RE_t'], rho_p_wilcox=pc['p_wilcox'], rho_p_sign=pc['p_sign'], rho_p_Fisher_combined=pc['p_comb'], rho_chi2_combined=pc['X2_comb'], rho_FE=pc['rho_FE'], rho_p_FE=pc['p_FE'], OR_test='Mantel-Haenszel with Robins-Breslow-Greenland variance', OR_z=pl['z'], OR_p_z=pl['p'], OR_p_t5=pl['p_t'], OR_p_wilcox=pl['p_wilcox'], OR_p_sign=pl['p_sign'], OR_p_Fisher_combined=pl['p_comb'], OR_chi2_combined=pl['X2_comb'], rho_per_slide='; '.join((f"{s}={C[(C['sample'] == s) & (C.group == g)].rho_mye_caf.iloc[0]:+.3f}" for s in SLIDES)), rho_pooled=pc['rho_RE'], rho_lo=pc['lo_RE'], rho_hi=pc['hi_RE'], rho_Q=pc['Q'], rho_pQ=pc['p_Q'], rho_I2=pc['I2'], OR_per_slide='; '.join((f"{s}={L[(L['sample'] == s) & (L.group == g)].odds_ratio.iloc[0]:.2f}" for s in SLIDES)), OR_MH=pl['or_MH'], OR_lo=pl['lo'], OR_hi=pl['hi'], OR_p=pl['p'], OR_Q=pl['Q'], OR_pQ=pl['p_Q'], OR_I2=pl['I2'], perm_p_per_slide='; '.join((f"{s}={L[(L['sample'] == s) & (L.group == g)].perm_p_toroidal.iloc[0]:.3f}" for s in SLIDES))))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'myeloid_CAF_coloc_stats.csv'), index=False)
    with pd.option_context('display.float_format', lambda v: f'{v:.4g}', 'display.width', 200):
        print('=== pooled estimates ===')
        print(T[['group', 'rho_pooled', 'rho_lo', 'rho_hi', 'rho_p_z', 'rho_p_t5', 'rho_p_wilcox', 'rho_pQ', 'rho_I2', 'OR_MH', 'OR_lo', 'OR_hi', 'OR_p_z', 'OR_p_t5', 'OR_p_wilcox', 'OR_pQ', 'OR_I2']].to_string(index=False))
        print("\ncombined per-slide p (Fisher's method): rho " + ', '.join((f"{g} chi2={pool_c[g]['X2_comb']:.0f} p={pool_c[g]['p_comb']:.2g}" for g in GROUPS)) + ' | OR ' + ', '.join((f"{g} chi2={pool_l[g]['X2_comb']:.0f} p={pool_l[g]['p_comb']:.2g}" for g in GROUPS)))
    print(f'\nper-slide direction: rho>0 in {int((C.rho_mye_caf > 0).sum())}/12 cells, OR>1 in {int((L.odds_ratio > 1).sum())}/12, permutation p<0.05 in {int((L.perm_p_toroidal < 0.05).sum())}/12')
    print(f'wrote {OUT}/myeloid_CAF_coloc_by_slide.png/pdf and myeloid_CAF_coloc_stats.csv')
if __name__ == '__main__':
    main()
