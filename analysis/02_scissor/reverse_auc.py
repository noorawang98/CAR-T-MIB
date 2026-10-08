"""Step22 (reverse of step21): AUC of the T-cell functional GR/PR axis predicting the
Scissor NDR / DR label.

Truth (positive class = DR): scissor coefficient < 0 (Scissor-), i.e. DR; NR = NDR.
Scores: marker EFF-EXH net (step18), GSVA net and its effector/exhaustion z scores,
the marker GR/PR binary axis, and the pooled scissor-free label set for reference.
Permutation test: within-slice permutation of the scissor label, 10 000 draws.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from common import grpr_labels, RES
FIG = os.path.join(RES, 'figs_tme')
TREATED = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
NPERM = 10000
rng = np.random.default_rng(20260113)

def auc(x, y, nboot=2000):
    x = np.asarray(x, float)
    y = np.asarray(y).astype(bool)
    a, b = (x[y], x[~y])
    if len(a) < 3 or len(b) < 3:
        return dict(auc=np.nan, lo=np.nan, hi=np.nan, n_pos=len(a), n_neg=len(b))
    v = mannwhitneyu(a, b, alternative='two-sided').statistic / (len(a) * len(b))
    bs = [mannwhitneyu(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))], alternative='two-sided').statistic / (len(a) * len(b)) for _ in range(nboot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(auc=float(v), lo=float(lo), hi=float(hi), n_pos=len(a), n_neg=len(b))

def perm_p(score, pos, slice_, nperm=NPERM):
    score = np.asarray(score, float)
    pos = np.asarray(pos).astype(bool)
    obs = auc(score, pos)['auc']
    idx = [np.where(np.asarray(slice_) == s)[0] for s in np.unique(slice_)]
    cnt = 0
    for _ in range(nperm):
        pp = pos.copy()
        for ii in idx:
            pp[ii] = rng.permutation(pp[ii])
        v = mannwhitneyu(score[pp], score[~pp], alternative='two-sided').statistic / (pp.sum() * (~pp).sum())
        if abs(v - 0.5) >= abs(obs - 0.5):
            cnt += 1
    return cnt / nperm

def main():
    D = pd.read_csv(os.path.join(RES, 'spot_full_table.csv'), index_col=0)
    MK = grpr_labels()
    P = pd.read_csv(os.path.join(RES, 'scissor_pooled_labels.csv'), index_col=0)
    MKc = MK[[c for c in ['GRPR', 'net', 'EFF_z', 'EXH_z'] if c not in D.columns]]
    A = D[D['car_pos'] & D['sample'].isin(TREATED)].join(MKc, how='inner').join(P[[c for c in P.columns if c in ('scissor_pooled', 'coef_sum') or c.startswith('lab_')]], how='inner')
    A['GRPR_num'] = (A['GRPR'] == 'PR').astype(float)
    A.to_csv(os.path.join(RES, 'reverse_auc_input.csv'))
    print('CAR+ spots:', len(A), '| DR', int((A['scissor_lab'] == 'R').sum()), 'NDR', int((A['scissor_lab'] == 'NR').sum()))
    scores = ['EFF_z', 'EXH_z', 'net', 'GRPR_num', 'car_cp10k']
    truths = [('scissor_lab_R_pooled', (A['scissor_lab'] == 'R').values)]
    for c in ['lab_GSE197977', 'lab_GSE248835']:
        if c in A.columns:
            truths.append((f'{c}_DR', (A[c] == -1).values))
    rows = []
    for tname, tv in truths:
        m = np.isin(tv, [True, False]) & pd.notna(tv)
        d = A[m]
        tvv = tv[m]
        if tvv.sum() < 3 or (~tvv).sum() < 3:
            continue
        for sc in scores:
            r = auc(d[sc], tvv)
            rows.append(dict(truth=tname, score=sc, n=len(d), **r, perm_p_within_slice=perm_p(d[sc], tvv, d['sample'])))
    T = pd.DataFrame(rows).sort_values(['truth', 'perm_p_within_slice'])
    T.to_csv(os.path.join(RES, 'reverse_auc_GRPR_predicts_scissor.csv'), index=False)
    pd.set_option('display.width', 220)
    print('\n=== AUC: T-cell functional scores predicting Scissor DR (DR=positive, NDR=negative) ===')
    print(T.round(4).to_string(index=False))
    d = A[A['scissor_lab'].isin(['R', 'NR'])]
    pos = (d['scissor_lab'] == 'R').values
    for sc, col in [('net', '#B22222'), ('EFF_z', '#d95f02'), ('EXH_z', '#7570b3'), ('GRPR_num', 'grey')]:
        v = np.asarray(d[sc], float)
        th = np.unique(np.r_[-np.inf, v, np.inf])
    for nm, col in [('NR', '#2C7FB8'), ('R', '#B22222')]:
        v = d.loc[d['scissor_lab'] == nm, 'net']
    print('\nwrote reverse_auc_GRPR_predicts_scissor.csv, reverse_auc_GRPR_to_scissor.*')
if __name__ == '__main__':
    main()
