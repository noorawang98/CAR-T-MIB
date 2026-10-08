"""Step05 (revision_panels): correlation suite for the GR/PR scores themselves.

Answers "was a correlation analysis of the scores ever done for this pipeline?".  The provenance
chain is  Scissor NDR/DR -> Lasso -> C7 T-cell x state sets -> state panels -> GSVA -> within-sample
z -> quadrant split; the four clinical panel scores behind that split are `g_MEMORY`, `g_EXHAUSTED`,
`g_NAIVE`, `g_EFFECTOR` (raw and within-sample z) in
`grpr_mem_exh/GRPR_mem_exh_final_labels.csv`.

Three blocks, all spot level, all Spearman unless noted:
  A  score x score (raw and z), pooled over the 888 labelled CAR+ spots and per section, plus the
     continuous Scissor coefficient and the binary Scissor NDR/DR label;
  B  score (z) x technical / compositional covariates (T-cell module, DestVI T-cell and
     precursor-exhausted fractions, total UMI, CAR CP10K, myeloid/endothelial/CAF/melanoma modules,
     spatial x/y), pooled and per section for the key ones, BH within each score;
  C  GR vs PR (binary, 201 vs 182) x the same covariates: Mann-Whitney p and Cliff's delta.
Output -> tables/TableG_score_correlations.csv (+ printed summary).
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import sys
import warnings
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
warnings.filterwarnings('ignore')
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
OUT = os.path.join(WD, 'nr1r1_analysis_M5', 'panels')
TAB = os.path.join(OUT, 'tables')
RCL = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/recluster_obj'
SECTIONS = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
SCORES = ['MEMORY', 'EXHAUSTED', 'NAIVE', 'EFFECTOR']
TCOL = ['CD4_Th', 'CD4_Tn/cm', 'CD4_Tpex', 'CD8_Tem', 'CD8_Tn/cm', 'CD8_Tpex', 'CD8_Trm', 'Tcycling', 'Treg', 'TNFRSF9+ T', 'γδ_Teff', 'γδ_Tex']
PEX = ['CD4_Tpex', 'CD8_Tpex', 'γδ_Tex']

def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full(len(p), np.nan)
    o = np.argsort(p[ok])
    v = p[ok][o]
    qq = np.minimum.accumulate((v * ok.sum() / (np.arange(ok.sum()) + 1))[::-1])[::-1]
    q[ok] = np.clip(qq, 0, 1)[np.argsort(o)]
    return q

def load():
    D = pd.read_csv(os.path.join(WD, 'grpr_mem_exh', 'GRPR_mem_exh_final_labels.csv'), index_col=0)
    D.index = D.index.astype(str)
    R = pd.read_csv(os.path.join(RES, 'spot_regions.csv'), index_col=0)
    R = R[[c for c in R.columns if c not in D.columns]]
    D = D.join(R, how='left')
    S = pd.read_csv(os.path.join(RES, 'scissor_coefs_long.csv'))
    S['key'] = S['sample'].astype(str) + '|' + S['barcode'].astype(str)
    P = S.pivot_table(index='key', columns='cohort', values='scissor_coef')
    P['scissor_sum'] = P.sum(1)
    D = D.join(P[['scissor_sum']], how='left')
    fr = []
    for s in SECTIONS:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'), backed='r')
        p = a.obsm['proportions']
        p = p if isinstance(p, pd.DataFrame) else pd.DataFrame(np.asarray(p))
        p.index = [f'{s}|{b}' for b in a.obs_names]
        fr.append(p)
        del a
    F = pd.concat(fr)
    tc = [c for c in F.columns if c in TCOL]
    D = D.join(pd.DataFrame({'DestVI_Tcell': F[tc].sum(1), 'DestVI_Tpex': F[[c for c in PEX if c in F.columns]].sum(1)}), how='left')
    D['DR'] = D['lab_M5'].map({'DR': 1.0, 'NDR': 0.0})
    D['GR_bin'] = D['GRPR_quad_M5'].map({'GR': 1.0, 'PR': 0.0})
    D['section'] = D['sample']
    return D

def main():
    D = load()
    rows = []

    def add(block, score, var, scope, rho, p, n):
        rows.append(dict(block=block, score=score, var=var, scope=scope, rho=rho, p=p, n=n))
    pairs = [(a, b) for i, a in enumerate(SCORES) for b in SCORES[i:]]
    for scope, d in [('pooled', D)] + [(s, D[D['section'].eq(s)]) for s in SECTIONS]:
        for a, b in pairs:
            if a == b:
                continue
            for suf in ('', '_z'):
                r, p = spearmanr(d[f'g_{a}{suf}'], d[f'g_{b}{suf}'], nan_policy='omit')
                add('A_score_score', f'g_{a}{suf}', f'g_{b}{suf}', scope, float(r), float(p), len(d))
        for var in ('scissor_sum', 'DR'):
            for a in SCORES:
                r, p = spearmanr(d[f'g_{a}_z'], d[var], nan_policy='omit')
                add('A_score_label', f'g_{a}_z', var, scope, float(r), float(p), len(d))
        r, p = spearmanr(d['g_MEMORY_z'], d['g_EXHAUSTED_z'], nan_policy='omit')
        add('A_score_score', 'g_MEMORY_z', 'g_EXHAUSTED_z', scope, float(r), float(p), len(d))
    cov = ['tcell', 'DestVI_Tcell', 'DestVI_Tpex', 'umi_total', 'car_cp10k', 'mye', 'endo', 'caf', 'ccr1mye', 'mel', 'x', 'y']
    for scope, d in [('pooled', D)] + [(s, D[D['section'].eq(s)]) for s in ('NR1', 'R1')]:
        for a in SCORES:
            for c in cov:
                if c not in d.columns:
                    continue
                r, p = spearmanr(d[f'g_{a}_z'], d[c], nan_policy='omit')
                add('B_score_covariate', f'g_{a}_z', c, scope, float(r), float(p), int(d[[f'g_{a}_z', c]].dropna().shape[0]))
    lab = D[D['GRPR_quad_M5'].isin(['GR', 'PR'])]
    for scope, d in [('pooled', lab)] + [(s, lab[lab['section'].eq(s)]) for s in SECTIONS]:
        for c in cov:
            if c not in d.columns:
                continue
            g = d.loc[d['GR_bin'] == 1, c].dropna().values
            p_ = d.loc[d['GR_bin'] == 0, c].dropna().values
            if len(g) < 3 or len(p_) < 3:
                continue
            try:
                u, pv = mannwhitneyu(g, p_, alternative='two-sided')
                delta = 2 * u / (len(g) * len(p_)) - 1
            except ValueError:
                continue
            add('C_GRvsPR_covariate', 'GR(1)_vs_PR(0)', c, scope, float(delta), float(pv), len(g) + len(p_))
    T = pd.DataFrame(rows)
    T['q'] = np.nan
    for (b, sc), idx in T.groupby(['block', 'scope']).groups.items():
        T.loc[idx, 'q'] = bh(T.loc[idx, 'p'].values)
    T.to_csv(os.path.join(TAB, 'TableG_score_correlations.csv'), index=False)
    pd.set_option('display.width', 200)
    print('=== A. 分数之间的 Spearman（pooled, 888 个 CAR+ spot）===')
    A = T[(T.block == 'A_score_score') & (T.scope == 'pooled') & T['score'].str.endswith('_z')]
    print(A[['score', 'var', 'rho', 'p', 'n']].round(4).to_string(index=False))
    print('\n=== A2. 分数(z) 与 Scissor 连续系数 / NDR-DR 标签 ===')
    print(T[(T.block == 'A_score_label') & (T.scope == 'pooled')][['score', 'var', 'rho', 'p', 'q']].round(4).to_string(index=False))
    print('\n=== B. 分数(z) 与协变量（pooled, 仅列 |rho|>0.1 或 q<0.05）===')
    B = T[(T.block == 'B_score_covariate') & (T.scope == 'pooled')]
    print(B[(B.rho.abs() > 0.1) | (B.q < 0.05)][['score', 'var', 'rho', 'p', 'q']].round(4).to_string(index=False))
    print("\n=== C. GR(201) vs PR(182) 的协变量差异（Cliff's delta / MWU p）===")
    print(T[(T.block == 'C_GRvsPR_covariate') & (T.scope == 'pooled')][['var', 'rho', 'p', 'q', 'n']].round(4).to_string(index=False))
    print('\nsaved ->', os.path.join(TAB, 'TableG_score_correlations.csv'))
if __name__ == '__main__':
    main()
