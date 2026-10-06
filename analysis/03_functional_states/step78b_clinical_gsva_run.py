"""Step78b: run GSVA on the clinical-response panels and test whether the CAR+ NDR/DR label
separates functionally; scatter Effector vs Exhaustion (NDR red / DR blue)."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, json, subprocess
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.decomposition import PCA
from common import RES, bh
OUT = PROJECT_ROOT + '/mouse/cart_region/grpr_pred_v3'
WORK = os.path.join(OUT, 'clinical_gsva')
SCD = PROJECT_ROOT + '/mouse/cart_region/scissor_pooled_noRelapse'
BRED, BBLUE, GREY = ('#B22222', '#2C7FB8', '#BFBFBF')

def rgsva():
    script = f'\n.libPaths(c(paste0(Sys.getenv("HOME_ROOT"), "/R_4.3"), .libPaths()))\nsuppressPackageStartupMessages({{library(GSVA); library(jsonlite)}})\nW <- "{WORK}"\nE <- t(as.matrix(read.csv(file.path(W, "expr_panel_genes.csv"), row.names = 1, check.names = FALSE)))\nP <- fromJSON(file.path(W, "panels_final.json"))\nP <- lapply(P, function(g) intersect(g, rownames(E))); P <- P[sapply(P, length) >= 5]\ncat("genes x spots:", dim(E), "| panels:", paste(names(P), sapply(P, length), sep="=", collapse=" "), "\\n")\nset.seed(1)\nsc <- GSVA::gsva(E, P, kcdf = "Gaussian", mx.diff = TRUE, verbose = FALSE)\nwrite.csv(as.data.frame(t(sc)), file.path(W, "gsva_clinical_scores.csv"))\ncat("done\\n")\n'
    open(os.path.join(WORK, 'run_gsva.R'), 'w').write(script)
    r = subprocess.run(['Rscript', os.path.join(WORK, 'run_gsva.R')], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-300:])

def main():
    A = pd.read_csv(os.path.join(WORK, 'assigned_genes_nonzero.csv'))
    E = pd.read_csv(os.path.join(OUT, 'carpos_hvg_expr.csv'), index_col=0)
    panels = {}
    maj = {st: [g for g in A.loc[A.state == st, 'gene'] if g in E.columns] for st in ['EFFECTOR', 'MEMORY', 'NAIVE']}
    relax = [g for g in A.loc[A.n_EXHAUSTED.fillna(0) >= 1, 'gene'] if g in E.columns]
    panels['EFFECTOR'] = maj['EFFECTOR']
    panels['EXHAUSTED'] = relax
    panels['MEMORY'] = maj['MEMORY']
    panels['NAIVE'] = maj['NAIVE']
    panels['Marker_Effector'] = [g for g in ['Gzma', 'Gzmb', 'Gzmk', 'Prf1', 'Ifng', 'Nkg7', 'Ccr7', 'Tcf7', 'Sell', 'Lef1', 'Il7r', 'Cd69', 'Il2ra', 'Il2rb', 'Mki67', 'Tnf'] if g in E.columns]
    panels['Marker_Exhaustion'] = [g for g in ['Pdcd1', 'Ctla4', 'Lag3', 'Havcr2', 'Tigit', 'Tox', 'Entpd1', 'Bach2'] if g in E.columns]
    json.dump(panels, open(os.path.join(WORK, 'panels_final.json'), 'w'), indent=0)
    print('panels:', {k: len(v) for k, v in panels.items()})
    genes = sorted(set(sum(panels.values(), [])) & set(E.columns))
    E[genes].to_csv(os.path.join(WORK, 'expr_panel_genes.csv'))
    pd.DataFrame([dict(panel=k, n_genes=len([g for g in v if g in E.columns]), genes=';'.join([g for g in v if g in E.columns])) for k, v in panels.items()]).to_csv(os.path.join(WORK, 'panels_final_summary.csv'), index=False)
    rgsva()
    S = pd.read_csv(os.path.join(SCD, 'scissor_labels_BEST_all_samples.csv')).set_index('spot')
    S.index = S.index.astype(str)
    G = pd.read_csv(os.path.join(WORK, 'gsva_clinical_scores.csv'), index_col=0).reindex(E.index)
    lab = S['label'].reindex(E.index)
    D = G.copy()
    D['sample'] = [i.split('|')[0] for i in D.index]
    D['label'] = lab
    Z = D.groupby('sample')[G.columns].transform(lambda v: (v - v.mean()) / (v.std() or 1))
    Z['sample'] = D['sample'].values
    Z['label'] = D['label'].values
    d = Z[Z.label.isin(['NDR', 'DR'])].copy()
    d['y'] = (d.label == 'DR').astype(int)
    print(f'\nCAR+ labelled spots: {len(d)} (DR {int(d.y.sum())} / NDR {int((1 - d.y).sum())})')
    e, h = (d['EFFECTOR'], d['EXHAUSTED'])
    r = np.corrcoef(e, h)[0, 1]
    p = PCA(2).fit(np.c_[e, h])
    try:
        sil = silhouette_score(np.c_[e, h], d.y.values)
    except Exception:
        sil = np.nan
    q = pd.Series(np.where((e >= 0) & (h >= 0), 'double_high', np.where((e >= 0) & (h < 0), 'eff_only', np.where((e < 0) & (h >= 0), 'exh_only', 'double_low'))))
    print(f'\n=== clinical panels separation ===\nr(EFF, EXH) = {r:.3f} | PC1 = {100 * p.explained_variance_ratio_[0]:.1f}% | silhouette(NDR/DR) = {sil:.3f}')
    print('quadrants(%):', (q.value_counts(normalize=True) * 100).round(1).to_dict())
    rows = []
    for c in [x for x in ['EFFECTOR', 'EXHAUSTED', 'MEMORY', 'NAIVE', 'Marker_Effector', 'Marker_Exhaustion'] if x in d.columns]:
        dd = d[[c, 'y', 'sample']].dropna()
        m = smf.mixedlm('v ~ y', dd.assign(v=dd[c]), groups=dd['sample']).fit(reml=True, method='lbfgs')
        auc = roc_auc_score(dd.y, dd[c])
        auc = max(auc, 1 - auc)
        rows.append(dict(panel=c, mean_DR=dd.loc[dd.y == 1, c].mean(), mean_NDR=dd.loc[dd.y == 0, c].mean(), diff=float(m.params['y']), p=float(m.pvalues['y']), auc=auc))
    T = pd.DataFrame(rows)
    T['q'] = bh(T.p.fillna(1))
    T.to_csv(os.path.join(WORK, 'clinical_panel_tests.csv'), index=False)
    print('\n=== panel-level tests (DR vs NDR, mixed model) ===')
    print(T.round(4).to_string(index=False))
    X2 = d[['EFFECTOR', 'EXHAUSTED']].values
    y = d.y.values
    g = d['sample'].values
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probs = np.full(len(y), np.nan)
    aucs = []
    for tr, te in LeaveOneGroupOut().split(X2, y, g):
        pipe.fit(X2[tr], y[tr])
        pr = pipe.predict_proba(X2[te])[:, 1]
        probs[te] = pr
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], pr))
    auc_cv = float(np.mean(aucs))
    print(f'\nLOGO-CV AUC of [Effector, Exhaustion] for DR vs NDR: {auc_cv:.3f} (folds {np.round(aucs, 3)})')
    rng = np.random.default_rng(0)
    NB = 300
    null = []
    for _ in range(NB):
        yp = y.copy()
        for s in np.unique(g):
            m = g == s
            yp[m] = rng.permutation(yp[m])
        aa = []
        for tr, te in LeaveOneGroupOut().split(X2, yp, g):
            if len(np.unique(yp[tr])) < 2:
                continue
            pp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X2[tr], yp[tr])
            if len(np.unique(yp[te])) == 2:
                aa.append(roc_auc_score(yp[te], pp.predict_proba(X2[te])[:, 1]))
        null.append(np.mean(aa) if aa else np.nan)
    null = np.array([v for v in null if np.isfinite(v)])
    pperm = (np.sum(null >= auc_cv) + 1) / (len(null) + 1)
    print(f'permutation (within-slice label shuffle, {len(null)} runs): null mean {null.mean():.3f}, p = {pperm:.4f}')
    pd.DataFrame([dict(logocv_auc=auc_cv, perm_p=pperm, null_mean=null.mean(), null_sd=null.std(), r_EFF_EXH=r, PC1=100 * p.explained_variance_ratio_[0], silhouette=sil, n=len(d))]).to_csv(os.path.join(WORK, 'clinical_separation_summary.csv'), index=False)
    pd.DataFrame({'null_auc': null}).to_csv(os.path.join(WORK, 'clinical_permutation_null.csv'), index=False)
    d.to_csv(os.path.join(WORK, 'clinical_scores_labelled.csv'))
    for lab_, c in [('NDR', BRED), ('DR', BBLUE)]:
        v = d[d.label == lab_]
    if 'Marker_Exhaustion' in d.columns and 'Marker_Effector' in d.columns:
        for lab_, c in [('NDR', BRED), ('DR', BBLUE)]:
            v = d[d.label == lab_]
        rm = np.corrcoef(d.Marker_Effector, d.Marker_Exhaustion)[0, 1]
    else:
        M = pd.read_csv(os.path.join(RES, 'marker_tcell_scores.csv'), index_col=0).reindex(d.index)
        for lab_, c in [('NDR', BRED), ('DR', BBLUE)]:
            v = M[d.label.values == lab_]
        rm = np.corrcoef(M.EFF_z, M.EXH_z)[0, 1]
    samp = sorted(d['sample'].unique())
    for i, s in enumerate(samp):
        v = d[d['sample'] == s]
        for lab_, c in [('NDR', BRED), ('DR', BBLUE)]:
            vv = v[v.label == lab_]
    print('\nwrote clinical_gsva/{clinical_panel_tests,clinical_separation_summary,clinical_scores_labelled}.csv, figs/clinical_grpr_scatter.*')
if __name__ == '__main__':
    main()
