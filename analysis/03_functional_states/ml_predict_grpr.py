"""Step58: predict the clinical-response (NDR/DR) label for ALL CAR+ spots with ML.

y = 1 for DR (durable = good response) -> GR ; 0 for NDR (non-durable = poor) -> PR
   (the Scissor BEST label set, alpha=0.1, only ~48% of spots are labelled -> the model
    predicts the remaining spots).
Features: (primary) CAR+ HVG expression (top 2000, computed on the 888 CAR+ spots);
          (sensitivity) the 7 MSigDB-consensus GSVA scores.
Validation: leave-one-slice-out CV (primary) + stratified 5-fold spot CV; permutation test
with 1000 within-slice label shuffles (parallel). Out-of-fold probabilities are used so no
spot is predicted by a model trained on itself.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, sys, json
import numpy as np, pandas as pd, anndata as ad
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from common import RES, RCL, TREATED
OUT = PROJECT_ROOT + '/mouse/cart_region/grpr_pred_v3'
FIG = os.path.join(OUT, 'figs')
SCD = PROJECT_ROOT + '/mouse/cart_region/scissor_pooled_noRelapse'
NHVG, C_FIX, NPERM = (2000, 0.1, 1000)
BRICK, BLUE, GREY = ('#B22222', '#2C7FB8', '#BFBFBF')

def build_expr():
    cache = os.path.join(OUT, 'carpos_hvg_expr.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0)
    L = pd.read_csv(os.path.join(RES, 'GRPR_labels_final.csv'), index_col=0)
    cnt, obs, var = ([], [], None)
    for s in TREATED:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'))
        names = np.array([f'{s}|{b}' for b in a.obs_names])
        pos = np.isin(names, L.index.values)
        cnt.append(a.layers['counts'][pos].tocsr())
        obs.append(pd.DataFrame({'sample': s}, index=names[pos]))
        if var is None:
            var = a.var_names.to_numpy()
        del a
        print(f'  {s}: CAR+ {int(pos.sum())}', flush=True)
    C = vstack(cnt).tocsr()
    D = np.asarray(C.todense(), dtype=np.float32)
    mu, va = (D.mean(0), D.var(0))
    ok = mu > 0.01
    bins = pd.qcut(pd.Series(mu[ok]), 20, labels=False, duplicates='drop')
    v = pd.Series(va[ok])
    z = (v - v.groupby(bins.values).transform('mean')) / v.groupby(bins.values).transform('std').replace(0, np.nan)
    genes = pd.DataFrame({'gene': var[ok], 'std_var': z.values}).sort_values('std_var', ascending=False).head(NHVG)
    gidx = [int(np.where(var == g)[0][0]) for g in genes.gene]
    tot = np.asarray(C.sum(1)).ravel()
    med = np.median(tot)
    X = np.log1p(np.asarray(C[:, gidx].todense(), dtype=np.float32) / tot[:, None] * med)
    df = pd.DataFrame(X, index=pd.concat(obs).index, columns=genes.gene.values)
    df.to_csv(cache)
    return df

def cv_auc(X, y, groups, scheme='logo', n_splits=5, seed=0):
    pipe = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(penalty='l1', solver='liblinear', C=C_FIX, max_iter=2000))
    if scheme == 'logo':
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
    aucs, probs = ([], np.full(len(y), np.nan))
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        probs[te] = p
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], p))
    return (float(np.mean(aucs)), float(np.std(aucs)), aucs, probs)

def main():
    os.makedirs(FIG, exist_ok=True)
    Xh = build_expr()
    G = pd.read_csv(PROJECT_ROOT + '/mouse/cart_region/grpr_gsva_v2/gsva_scores_z.csv', index_col=0)
    S = pd.read_csv(os.path.join(SCD, 'scissor_labels_BEST_all_samples.csv'))
    S = S.set_index('spot')
    A = Xh.join(G.reindex(Xh.index), how='left')
    A['sample'] = [i.split('|')[0] for i in A.index]
    A['scissor'] = S['label'].reindex(A.index)
    A['coef'] = S['coef'].reindex(A.index)
    tr = A[A.scissor.isin(['NDR', 'DR'])].copy()
    tr['y'] = (tr.scissor == 'DR').astype(int)
    hvg_cols = list(Xh.columns)
    gsva_cols = list(G.columns)
    print(f'CAR+ spots {len(A)} | labelled {len(tr)} (GR/DR {int(tr.y.sum())} / PR/NDR {int((1 - tr.y).sum())})')
    print('features: HVG', len(hvg_cols), '| GSVA', len(gsva_cols))
    rows = []
    labels = {}
    for fname, cols in [('HVG', hvg_cols), ('GSVA', gsva_cols)]:
        X = tr[cols].values
        y = tr.y.values
        gr = tr['sample'].values
        for scheme in ('logo', '5fold'):
            m, s, aucs, probs = cv_auc(X, y, gr, scheme)
            rows.append(dict(features=fname, cv=scheme, n=len(y), auc_mean=m, auc_sd=s, fold_aucs=';'.join((f'{a:.3f}' for a in aucs))))
            print(f'  {fname:5s} {scheme:6s}: AUC {m:.3f} +- {s:.3f}  folds {np.round(aucs, 3)}', flush=True)
            if scheme == 'logo':
                labels[fname] = probs
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(OUT, 'cv_metrics.csv'), index=False)
    X = tr[hvg_cols].values
    y = tr.y.values
    gr = tr['sample'].values
    obs_auc, _, _, _ = cv_auc(X, y, gr, 'logo')
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(y)) for _ in range(NPERM)]

    def one(pi):
        yp = y.copy()
        for sl in np.unique(gr):
            m = gr == sl
            yp[m] = y[m][np.random.default_rng(abs(hash((sl, pi[0]))) % 2 ** 31).permutation(m.sum())]
        return cv_auc(X, yp, gr, 'logo')[0]
    null = Parallel(n_jobs=10, verbose=1)((delayed(one)(pi) for pi in perms))
    null = np.array([v for v in null if np.isfinite(v)])
    p = (np.sum(null >= obs_auc) + 1) / (len(null) + 1)
    pd.DataFrame({'null_auc': null}).to_csv(os.path.join(OUT, 'permutation_null.csv'), index=False)
    print(f'permutation: observed AUC {obs_auc:.3f}, null mean {null.mean():.3f}, p = {p:.4f} (n={len(null)})')
    pd.DataFrame([dict(features='HVG', cv='logo', n=len(y), auc_observed=obs_auc, null_mean=null.mean(), null_sd=null.std(), n_perm=len(null), p_perm=p)]).to_csv(os.path.join(OUT, 'permutation_test.csv'), index=False)
    final = {}
    for fname, cols in [('HVG', hvg_cols), ('GSVA', gsva_cols)]:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', C=C_FIX, max_iter=2000))
        pipe.fit(tr[cols].values, tr.y.values)
        co = pipe[-1].coef_.ravel()
        final[fname] = pd.Series(co, index=cols).sort_values(key=np.abs, ascending=False)
        oof = pd.Series(labels[fname], index=tr.index)
        p_all = oof.reindex(A.index)
        p_final = pipe.predict_proba(A[cols].values)[:, 1]
        p_all = p_all.fillna(pd.Series(p_final, index=A.index))
        A[f'p_GR_{fname}'] = p_all.values
    A['GRPR_pred'] = np.where(A.p_GR_HVG >= 0.5, 'GR', 'PR')
    A['GRPR_pred_gsva'] = np.where(A.p_GR_GSVA >= 0.5, 'GR', 'PR')
    A.to_csv(os.path.join(OUT, 'GRPR_pred_labels.csv'))
    final['HVG'].to_csv(os.path.join(OUT, 'lasso_coef_HVG.csv'))
    final['GSVA'].to_csv(os.path.join(OUT, 'lasso_coef_GSVA.csv'))
    print('\npredicted label counts (all CAR+ spots):', A.GRPR_pred.value_counts().to_dict())
    print('per slice:\n', pd.crosstab(A['sample'], A.GRPR_pred).to_string())
    print('agreement HVG vs GSVA model labels: %.3f' % float((A.GRPR_pred == A.GRPR_pred_gsva).mean()))
    lab = A[A.scissor.isin(['NDR', 'DR'])]
    print('out-of-fold AUC on labelled spots (HVG): %.3f' % roc_auc_score((lab.scissor == 'DR').astype(int), lab.p_GR_HVG))
    for f, c in [('HVG', BRICK), ('GSVA', BLUE)]:
        sub = A[f'p_GR_{f}']
    for lab_, c in [('GR', BRICK), ('PR', BLUE)]:
        v = A.loc[A.GRPR_pred == lab_, 'p_GR_HVG']
    print('\nwrote GRPR_pred_labels.csv, cv_metrics.csv, permutation_test.csv, lasso_coef_*.csv, figs/')
if __name__ == '__main__':
    main()
