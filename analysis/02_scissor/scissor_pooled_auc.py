"""Step21: pooled Scissor label (NDR=+1 / DR=-1 summed over the 5 bulk cohorts) versus the
GR / PR T-cell functional label -- AUC + permutation test.

Label construction
  for every cohort c in {CC2025, GSE153437, GSE153438, GSE197977, GSE248835}:
      lab_c = +1 if Scissor5 coefficient > 0   (Scissor+ = NDR)
              -1 if Scissor5 coefficient < 0   (Scissor- = DR)
               0 otherwise (not selected)
  scissor_pooled = sum_c lab_c        in {-5..+5}
The continuous coefficient sum (previously used) and the per-cohort labels are kept as
comparators.

AUC
  ground truth: T-cell functional label with GR = -1 (negative class) and PR = +1
                (positive class) -> AUC = P(score higher in PR than in GR)
  scores tested: scissor_pooled, each cohort label, the coefficient sum.
  Permutation test: GR/PR labels are permuted (i) inside each slice (keeps slice
  composition) and (ii) globally, 10 000 permutations each; p = P(|AUC_perm-0.5| >=
  |AUC_obs-0.5|).  Both GR/PR definitions are run (marker-based and GSVA-based).
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from common import grpr_labels, RES, scissor_labels
FIG = os.path.join(RES, 'figs_tme')
os.makedirs(FIG, exist_ok=True)
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

def perm_p(score, gr, slice_, nperm=NPERM, strat='slice'):
    score = np.asarray(score, float)
    pos = np.asarray(gr) == 'PR'
    obs = auc(score, pos)['auc']
    if not np.isfinite(obs):
        return (np.nan, obs, np.nan)
    cnt, null = (0, [])
    idx = [np.where(np.asarray(slice_) == s)[0] for s in np.unique(slice_)] if strat == 'slice' else None
    for _ in range(nperm):
        pp = pos.copy()
        if idx is None:
            pp = rng.permutation(pp)
        else:
            for ii in idx:
                pp[ii] = rng.permutation(pp[ii])
        v = mannwhitneyu(score[pp], score[~pp], alternative='two-sided').statistic / (pp.sum() * (~pp).sum())
        null.append(v)
        if abs(v - 0.5) >= abs(obs - 0.5):
            cnt += 1
    return (cnt / nperm, obs, float(np.median([abs(q - 0.5) for q in null])))

def main():
    S = pd.read_csv(os.path.join(RES, 'scissor_coefs_long.csv'))
    W = S.pivot_table(index=['sample', 'barcode'], columns='cohort', values='scissor_coef')
    W.index = [f'{s}|{b}' for s, b in W.index]
    L = pd.DataFrame({c: np.sign(W[c]).astype(int) for c in W.columns})
    L['scissor_pooled'] = L.sum(1)
    L['coef_sum'] = W.sum(1)
    L.columns = [f'lab_{c}' if c not in ('scissor_pooled', 'coef_sum') else c for c in L.columns]
    L['n_ndr'] = (L[[c for c in L.columns if c.startswith('lab_')]] > 0).sum(1)
    L['n_dr'] = (L[[c for c in L.columns if c.startswith('lab_')]] < 0).sum(1)
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    A = L.join(M[['sample', 'car_pos']], how='left')
    A = A[A['car_pos'] & A['sample'].isin(TREATED)]
    A['GRPR'] = grpr_labels()['GRPR'].reindex(A.index)
    A.to_csv(os.path.join(RES, 'scissor_pooled_labels.csv'))
    print('CAR+ spots:', len(A))
    print('scissor_pooled distribution:\n', A['scissor_pooled'].value_counts().sort_index().to_string())
    scores = ['scissor_pooled', 'coef_sum'] + [c for c in A.columns if c.startswith('lab_')]
    rows = []
    for gcol, gname in [('GRPR', 'final_GR/PR (def5)')]:
        d = A[A[gcol].isin(['GR', 'PR'])]
        for sc in scores:
            r = auc(d[sc], (d[gcol] == 'PR').values)
            p_s, obs, _ = perm_p(d[sc], d[gcol], d['sample'], NPERM, 'slice')
            p_g, _, _ = perm_p(d[sc], d[gcol], d['sample'], NPERM, 'global')
            rows.append(dict(grpr=gname, score=sc, n=len(d), n_PR=int((d[gcol] == 'PR').sum()), **r, perm_p_within_slice=p_s, perm_p_global=p_g))
        d2 = d[d['scissor_pooled'] != 0]
        r = auc(d2['scissor_pooled'], (d2[gcol] == 'PR').values)
        p_s, _, _ = perm_p(d2['scissor_pooled'], d2[gcol], d2['sample'], NPERM, 'slice')
        rows.append(dict(grpr=gname, score='scissor_pooled (|sum|>=1)', n=len(d2), n_PR=int((d2[gcol] == 'PR').sum()), **r, perm_p_within_slice=p_s, perm_p_global=np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, 'scissor_pooled_vs_GRPR_AUC.csv'), index=False)
    pd.set_option('display.width', 220)
    print('\n=== AUC of the pooled Scissor label vs GR(-1)/PR(+1) ===')
    print(T.round(4).to_string(index=False))
    for sc, col, lw in [('scissor_pooled', '#B22222', 2.4), ('coef_sum', '#2C7FB8', 2.0)] + [(c, 'grey', 0.9) for c in A.columns if c.startswith('lab_')]:
        d = A[A['GRPR'].isin(['GR', 'PR'])]
        pos = (d['GRPR'] == 'PR').values
        v = np.asarray(d[sc], float)
        th = np.unique(np.r_[-np.inf, v, np.inf])
        tpr = [np.mean(v[pos] > t) for t in th]
        fpr = [np.mean(v[~pos] > t) for t in th]
        a = auc(v, pos)['auc']
    d = A[A['GRPR'].isin(['GR', 'PR'])]
    for nm, col in [('GR', '#B22222'), ('PR', '#2C7FB8')]:
        v = d.loc[d['GRPR'] == nm, 'scissor_pooled']
    print('\nwrote scissor_pooled_labels.csv, scissor_pooled_vs_GRPR_AUC.csv, scissor_pooled_GRPR_AUC.*')
if __name__ == '__main__':
    main()
