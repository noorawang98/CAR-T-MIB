"""Step01 (revision_panels): data for the replacement panels C/D/E/F.

All GR/PR statements use the CURRENT definition only
    `GRPR_quad_M5` in `grpr_mem_exh/GRPR_mem_exh_final_labels.csv`
    GR = g_MEMORY_z > 0 & g_EXHAUSTED_z < 0 ;  PR = the opposite quadrant
(per-sample z of the clinical GSVA panels).  No old label source is touched.

Panel E : CAR+ spot MEMORY-vs-EXHAUSTED GSVA correlation, after normalising T-cell content.
          T-cell content = mean log1p(CP10K) of canonical T markers, regressed out of both GSVA
          scores (linear residuals); raw and residual Spearman/Pearson are both reported, plus a
          within-section label permutation p (2000x) for the residual Spearman.
Panel F : gene-level GR/PR signature.  Feature genes are taken data-driven from the existing
          NR1 pseudobulk ranking (`nr1_tcell_gsea/tables/NR1_rank_log2FC_PR_GR.csv`, detection
          >= 10% of the labelled spots): the 10 most GR-high and the 10 most PR-high genes.  For
          every section the GR and PR group means (log1p CP10K, and pseudobulk CPM) are computed
          together with a one-sided Mann-Whitney test of the expected direction.

Outputs -> tables/TableE_*.csv, TableF_*.csv
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import sys
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
H5 = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/recluster_obj'
OUT = os.path.join(WD, 'nr1r1_analysis_M5', 'panels')
TAB = os.path.join(OUT, 'tables')
os.makedirs(TAB, exist_ok=True)
SECTIONS = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
E_SECTIONS = ['NR1', 'R1']
RANK_NR1 = os.path.join(WD, 'nr1_tcell_gsea', 'tables', 'NR1_rank_log2FC_PR_GR.csv')
T_MARKERS = ['Cd3e', 'Cd3d', 'Cd3g', 'Trbc2', 'Cd2', 'Thy1', 'Ms4a4b', 'Skap1', 'Lat', 'Cd28']
N_TOP = 10
N_PERM = 2000
RNG = np.random.default_rng(20260209)

def load_adata(sample):
    a = ad.read_h5ad(os.path.join(H5, f'{sample}_sc2st_DestVI_destvi_recluster.h5ad'))
    return a

def cp10k_log1p(X):
    X = X.tocsr().astype(np.float64)
    s = np.asarray(X.sum(1)).ravel()
    s[s == 0] = 1
    X = X.multiply(10000.0 / s[:, None]).tocsr()
    X.data = np.log1p(X.data)
    return X

def residualise(y, x):
    """least-squares residual of y on [1, x]"""
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta
SCORES = ['g_MEMORY_z', 'g_EXHAUSTED_z', 'g_NAIVE_z', 'g_EFFECTOR_z']
SHORT = {'g_MEMORY_z': 'MEMORY', 'g_EXHAUSTED_z': 'EXHAUSTED', 'g_NAIVE_z': 'NAIVE', 'g_EFFECTOR_z': 'EFFECTOR'}
P_STARS = [(0.001, '***'), (0.01, '**'), (0.05, '*')]

def stars(p):
    for thr, s in P_STARS:
        if p < thr:
            return s
    return ''

def panel_e(labels):
    rows, spots, mats = ([], [], [])
    for s in SECTIONS:
        a = load_adata(s)
        idx = [f'{s}|{b}' for b in a.obs_names]
        lab = labels.reindex(idx).dropna(subset=['GRPR_quad_M5'])
        keep = np.isin(idx, lab.index.values)
        X = cp10k_log1p(a.X[keep])
        var = np.array(a.var_names)
        tset = [g for g in T_MARKERS if g in set(var)]
        tscore = np.asarray(X[:, [list(var).index(g) for g in tset]].mean(1)).ravel()
        raw = {k: lab[k].values.astype(float) for k in SCORES}
        res = {k: residualise(v, tscore) for k, v in raw.items()}
        mem, exh, mem_r, exh_r = (raw['g_MEMORY_z'], raw['g_EXHAUSTED_z'], res['g_MEMORY_z'], res['g_EXHAUSTED_z'])
        rho0, p0 = spearmanr(mem, exh)
        r0, pr0 = pearsonr(mem, exh)
        rho1, p1 = spearmanr(mem_r, exh_r)
        r1, pr1 = pearsonr(mem_r, exh_r)
        ge = 0
        for _ in range(N_PERM):
            if abs(spearmanr(mem_r, RNG.permutation(exh_r))[0]) >= abs(rho1) - 1e-12:
                ge += 1
        pperm = (ge + 1) / (N_PERM + 1)
        q0 = np.where((mem > 0) & (exh < 0), 'GR', np.where((mem < 0) & (exh > 0), 'PR', 'Other'))
        q1 = np.where((mem_r > 0) & (exh_r < 0), 'GR', np.where((mem_r < 0) & (exh_r > 0), 'PR', 'Other'))
        cur = lab['GRPR_quad_M5'].values
        agree = float(np.mean(q1 == cur))
        for i, ki in enumerate(SCORES):
            for kj in SCORES[i:]:
                rr, pp = spearmanr(res[ki], res[kj])
                mats.append(dict(sample=s, row=SHORT[ki], col=SHORT[kj], rho=float(rr), p=float(pp), n=len(lab), stars=stars(float(pp))))
        rows.append(dict(sample=s, n_carpos=len(lab), n_GR=int((cur == 'GR').sum()), n_PR=int((cur == 'PR').sum()), n_tcell_markers=len(tset), rho_raw=rho0, p_raw=p0, r_raw=r0, p_r_raw=pr0, rho_adj=rho1, p_adj=p1, r_adj=r1, p_r_adj=pr1, p_perm_adj=pperm, rho_tcell_mem=spearmanr(mem, tscore)[0], rho_tcell_exh=spearmanr(exh, tscore)[0], n_GR_adj=int((q1 == 'GR').sum()), n_PR_adj=int((q1 == 'PR').sum()), agree_with_quad_M5=agree))
        d = dict(sample=s, spot=lab.index.values, tscore=tscore, g_MEMORY_z=mem, g_EXHAUSTED_z=exh, quad_GRPR=cur, quad_adj=q1)
        for k in SCORES:
            d[SHORT[k] + '_raw'] = raw[k]
            d[SHORT[k] + '_res'] = res[k]
        spots.append(pd.DataFrame(d))
        print(f'[E] {s}: n={len(lab)} rho_raw={rho0:+.3f} (p={p0:.3g}) rho_adj={rho1:+.3f} (p={p1:.3g}, perm={pperm:.4f}) agree={agree:.3f}')
    S = pd.concat(spots, ignore_index=True)
    tsc = S['tscore'].values
    mem, exh = (S['g_MEMORY_z'].values, S['g_EXHAUSTED_z'].values)
    cur = S['quad_GRPR'].values
    rho0, p0 = spearmanr(mem, exh)
    r0, pr0 = pearsonr(mem, exh)
    mem_r = residualise(mem, tsc)
    exh_r = residualise(exh, tsc)
    rho1, p1 = spearmanr(mem_r, exh_r)
    r1, pr1 = pearsonr(mem_r, exh_r)
    ge = sum((1 for _ in range(N_PERM) if abs(spearmanr(mem_r, RNG.permutation(exh_r))[0]) >= abs(rho1) - 1e-12))
    q1 = np.where((mem_r > 0) & (exh_r < 0), 'GR', np.where((mem_r < 0) & (exh_r > 0), 'PR', 'Other'))
    rows.append(dict(sample='ALL', n_carpos=len(S), n_GR=int((cur == 'GR').sum()), n_PR=int((cur == 'PR').sum()), n_tcell_markers=len(T_MARKERS), rho_raw=rho0, p_raw=p0, r_raw=r0, p_r_raw=pr0, rho_adj=rho1, p_adj=p1, r_adj=r1, p_r_adj=pr1, p_perm_adj=(ge + 1) / (N_PERM + 1), rho_tcell_mem=spearmanr(mem, tsc)[0], rho_tcell_exh=spearmanr(exh, tsc)[0], n_GR_adj=int((q1 == 'GR').sum()), n_PR_adj=int((q1 == 'PR').sum()), agree_with_quad_M5=float(np.mean(q1 == cur))))
    res_all = {k: residualise(S[SHORT[k] + '_raw'].values, tsc) for k in SCORES}
    for i, ki in enumerate(SCORES):
        for kj in SCORES[i:]:
            rr, pp = spearmanr(res_all[ki], res_all[kj])
            mats.append(dict(sample='ALL', row=SHORT[ki], col=SHORT[kj], rho=float(rr), p=float(pp), n=len(S), stars=stars(float(pp))))
    Sa = S.copy()
    Sa['sample'] = 'ALL'
    for k in SCORES:
        Sa[SHORT[k] + '_res'] = res_all[k]
    Sa['quad_adj'] = q1
    print(f"[E] pooled over all {len(S)} CAR+ spots of the six samples: rho_raw={rho0:+.3f} (p={p0:.3g}) rho_adj={rho1:+.3f} (p={p1:.3g}, perm={rows[-1]['p_perm_adj']:.4f}) agree={rows[-1]['agree_with_quad_M5']:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'TableE_gsva_corr_tcell_adjusted.csv'), index=False)
    pd.concat([S, Sa], ignore_index=True).to_csv(os.path.join(TAB, 'TableE_spots.csv'), index=False)
    pd.DataFrame(mats).to_csv(os.path.join(TAB, 'TableE_gsva_corr_matrix.csv'), index=False)
    print(f'[E] 4x4 Spearman matrices written for {len(SECTIONS)} sections + ALL')
PANELS_JSON = os.path.join(WD, 'grpr_pred_v3', 'clinical_gsva', 'panels_final.json')
MIN_MEAN = 0.2

def panel_f(labels):
    import json
    P = json.load(open(PANELS_JSON))
    mem, exh = (list(P['MEMORY']), list(P['EXHAUSTED']))
    shared = [g for g in mem if g in exh]
    cand = {'MEMORY': mem, 'EXHAUSTED': exh}
    print(f'[F] definition panels: MEMORY n={len(mem)}, EXHAUSTED n={len(exh)}, shared={shared}')
    long = []
    for s in SECTIONS:
        a = load_adata(s)
        idx = np.array([f'{s}|{b}' for b in a.obs_names])
        lab = labels.reindex(idx)
        var = np.array(a.var_names)
        want = list(dict.fromkeys(mem + exh))
        have = [g for g in want if g in set(var)]
        miss = [g for g in want if g not in set(var)]
        gcol = [list(var).index(g) for g in have]
        grp = np.array(lab['GRPR_quad_M5'].values)
        keep = pd.notna(grp)
        X = cp10k_log1p(a.X[keep])[:, gcol].tocsc()
        grp = grp[keep]
        mg_m, pr_m = (grp == 'GR', grp == 'PR')
        for j, g in enumerate(have):
            col = np.asarray(X[:, j].todense()).ravel()
            in_mem, in_exh = (g in mem, g in exh)
            cls = 'shared' if in_mem and in_exh else 'MEMORY' if in_mem else 'EXHAUSTED'
            alt = 'two-sided' if cls == 'shared' else 'less' if cls == 'MEMORY' else 'greater'
            try:
                p = float(mannwhitneyu(col[mg_m], col[pr_m], alternative=alt).pvalue)
            except ValueError:
                p = np.nan
            long.append(dict(sample=s, gene=g, cls=cls, mean_GR=float(col[mg_m].mean()), mean_PR=float(col[pr_m].mean()), p_one_sided=p, n_GR=int(mg_m.sum()), n_PR=int(pr_m.sum())))
        print(f'    [F] {s}: {len(have)}/{len(want)} panel genes present' + (f', missing {miss}' if miss else ''))
    M = pd.DataFrame(long)
    cls_order = {'MEMORY': 0, 'shared': 1, 'EXHAUSTED': 2}
    M['cls_rank'] = M['cls'].map(cls_order)
    M = M.sort_values(['cls_rank', 'gene']).drop(columns='cls_rank')
    M.to_csv(os.path.join(TAB, 'TableF_panel_gene_matrix.csv'), index=False)
    print(f"[F] matrix rows={len(M)} (genes {M['gene'].nunique()} x sections {M['sample'].nunique()}); p<0.05 cells: {int((M['p_one_sided'] < 0.05).sum())}/{len(M)}")
    S = M.groupby(['cls', 'gene']).agg(mean_GR=('mean_GR', 'mean'), mean_PR=('mean_PR', 'mean'), diff_mean=('mean_PR', lambda v: v.mean()), n_p05=('p_one_sided', lambda v: int((v < 0.05).sum())), max_mean=('mean_GR', 'max')).reset_index()
    S['diff_mean'] = M.groupby(['cls', 'gene'])['mean_PR'].mean().values - M.groupby(['cls', 'gene'])['mean_GR'].mean().values
    S.to_csv(os.path.join(TAB, 'TableF_panel_gene_summary.csv'), index=False)
    print(S.groupby('cls')[['mean_GR', 'mean_PR', 'diff_mean', 'n_p05']].mean().round(3).to_string())
DETECT_MIN = 0.1

def panel_volcano(labels):
    """GR vs PR across all labelled CAR+ spots of the six sections (labels are per-section)."""
    import json
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    P = json.load(open(PANELS_JSON))
    panel_of = {}
    for g in P['MEMORY']:
        panel_of[g] = 'MEMORY'
    for g in P['EXHAUSTED']:
        panel_of[g] = 'EXHAUSTED' if g not in panel_of else 'shared'
    mats, grps, sums_g, sums_p, lin_g, lin_p, det, var, nl = ([], [], None, None, None, None, None, None, [0, 0])
    for s in SECTIONS:
        a = load_adata(s)
        idx = np.array([f'{s}|{b}' for b in a.obs_names])
        lab = labels.reindex(idx)['GRPR_quad_M5'].values
        keep = pd.notna(lab) & np.isin(lab, ['GR', 'PR'])
        grp = lab[keep]
        Xr = a.X[keep].tocsr().astype(np.float64)
        var = np.array(a.var_names)
        sum_g = np.asarray(Xr[grp == 'GR'].sum(0)).ravel()
        sum_p = np.asarray(Xr[grp == 'PR'].sum(0)).ravel()
        rs = np.asarray(Xr.sum(1)).ravel()
        rs[rs == 0] = 1
        Xl = Xr.multiply(10000.0 / rs[:, None]).tocsr()
        l_g = np.asarray(Xl[grp == 'GR'].sum(0)).ravel()
        l_p = np.asarray(Xl[grp == 'PR'].sum(0)).ravel()
        lin_g = l_g if lin_g is None else lin_g + l_g
        lin_p = l_p if lin_p is None else lin_p + l_p
        d_g = np.asarray((Xr[grp == 'GR'] > 0).sum(0)).ravel()
        d_p = np.asarray((Xr[grp == 'PR'] > 0).sum(0)).ravel()
        sums_g = sum_g if sums_g is None else sums_g + sum_g
        sums_p = sum_p if sums_p is None else sums_p + sum_p
        det = d_g + d_p if det is None else det + (d_g + d_p)
        nl[0] += int((grp == 'GR').sum())
        nl[1] += int((grp == 'PR').sum())
        mats.append(np.asarray(cp10k_log1p(Xr).todense(), dtype=np.float32))
        grps.append(grp)
        print(f"    [volcano] {s}: {int((grp == 'GR').sum())} GR / {int((grp == 'PR').sum())} PR")
    X = np.vstack(mats)
    is_gr = np.concatenate(grps) == 'GR'
    frac = det / (nl[0] + nl[1])
    test = frac >= DETECT_MIN
    idx_g = np.where(test)[0]
    print(f'[volcano] testing {len(idx_g)} / {len(var)} genes (detected in >= {int(100 * DETECT_MIN)}% of {nl[0] + nl[1]} labelled spots)')
    pv = np.ones(len(idx_g))
    cl = np.zeros(len(idx_g))
    for j, gi in enumerate(idx_g):
        col = X[:, gi]
        if col[is_gr].max() == col[~is_gr].max() == 0:
            continue
        try:
            u, pp = mannwhitneyu(col[is_gr], col[~is_gr], alternative='two-sided')
            pv[j] = pp
            cl[j] = 2.0 * u / (nl[0] * nl[1]) - 1.0
        except ValueError:
            pv[j] = 1.0
    q = multipletests(pv, method='fdr_bh')[1]
    cpm_g = sums_g[idx_g] / max(sums_g.sum(), 1) * 1000000.0
    cpm_p = sums_p[idx_g] / max(sums_p.sum(), 1) * 1000000.0
    mg = X[is_gr][:, idx_g].mean(0)
    mp = X[~is_gr][:, idx_g].mean(0)
    mcg = lin_g[idx_g] / nl[0]
    mcp = lin_p[idx_g] / nl[1]
    T = pd.DataFrame(dict(gene=var[idx_g], mean_CP10K_GR=mcg, mean_CP10K_PR=mcp, log2FC_spot_PR_GR=np.log2((mcp + 1) / (mcg + 1)), mean_GR=mg, mean_PR=mp, diff_spot=mp - mg, cliff_delta=cl, cpm_GR=cpm_g, cpm_PR=cpm_p, log2FC_CPM_PR_GR=np.log2((cpm_p + 1) / (cpm_g + 1)), detect_frac=frac[idx_g], p=pv, q=q, n_GR=nl[0], n_PR=nl[1]))
    T['panel'] = [panel_of.get(g, '') for g in T['gene']]
    from step76_blocks import G2B as B27
    T['block27'] = [B27.get(g, '') for g in T['gene']]
    T = T.sort_values('p')
    T.to_csv(os.path.join(TAB, 'TableF_volcano_GRvsPR.csv'), index=False)
    print(f"[volcano] spot-level (no pseudobulk): p<0.05: {int((T['p'] < 0.05).sum())}, q<0.05: {int((T['q'] < 0.05).sum())}")
    for cut in (0.25, 0.5, 1.0):
        m = (T['p'] < 0.05) & (T['log2FC_spot_PR_GR'].abs() >= cut)
        print(f"    |log2FC_spot| >= {cut} & p<0.05: {int(m.sum())} | & q<0.05: {int((m & (T['q'] < 0.05)).sum())}")
    print('    log2FC_spot 范围: %.2f .. %.2f' % (T['log2FC_spot_PR_GR'].min(), T['log2FC_spot_PR_GR'].max()))
    print(T.head(12)[['gene', 'log2FC_spot_PR_GR', 'mean_CP10K_GR', 'mean_CP10K_PR', 'cliff_delta', 'p', 'q', 'panel']].round(4).to_string(index=False))

def main():
    what = (sys.argv[1] if len(sys.argv) > 1 else 'ef').lower()
    labels = pd.read_csv(os.path.join(WD, 'grpr_mem_exh', 'GRPR_mem_exh_final_labels.csv'), index_col=0)
    if 'e' in what:
        panel_e(labels)
    if 'f' in what:
        panel_f(labels)
    if 'v' in what:
        panel_volcano(labels)
    print('done ->', TAB)
if __name__ == '__main__':
    main()
