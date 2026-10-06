"""Step78: clinical-response GR/PR attempt -
(a) Lasso on CAR+ HVG for the BEST NDR/DR label -> contributing genes;
(b) intersect with the MSigDB T-cell functional sets used in the original screening rule
    (C7 IMMUNESIGDB, T-cell x state regex), assign each gene to the state it appears in most;
(c) GSVA on those state panels -> test functional separation of NDR/DR (CAR+);
(d) if separated, scatter Effector vs Exhaustion (NDR red / DR blue)."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from scipy import stats
from common import RES, bh
OUT = PROJECT_ROOT + '/mouse/cart_region/grpr_pred_v3'
WORK = os.path.join(OUT, 'clinical_gsva')
os.makedirs(WORK, exist_ok=True)
SCD = PROJECT_ROOT + '/mouse/cart_region/scissor_pooled_noRelapse'
MSIG = HOME_ROOT + '/LJX/github_upload_test'
TC_PAT = 'T_CELL|CD4|CD8|TH1|TH2|TH17|TREG|T_LYMPH'
ST_PAT = 'EFF|EFFECTOR|ACTIV|MEMOR|EXHAUST|NAIVE|ANERG|TOLERAN'
STATES = {'EFFECTOR': 'EFF|EFFECTOR|ACTIV', 'EXHAUSTED': 'EXHAUST', 'MEMORY': 'MEMOR', 'NAIVE': 'NAIVE'}
BRED, BBLUE, GREY = ('#B22222', '#2C7FB8', '#BFBFBF')

def lasso_genes(X, y, g, genes, C=0.1, nb=200, seed=0):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=2000))
    pipe.fit(X, y)
    co = pd.Series(pipe[-1].coef_.ravel(), index=genes)
    rng = np.random.default_rng(seed)
    cnt = np.zeros(len(genes))
    for _ in range(nb):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        p2 = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=2000))
        p2.fit(X[idx], y[idx])
        cnt += p2[-1].coef_.ravel() != 0
    return (co, pd.Series(cnt / nb, index=genes))

def main():
    E = pd.read_csv(os.path.join(OUT, 'carpos_hvg_expr.csv'), index_col=0)
    S = pd.read_csv(os.path.join(SCD, 'scissor_labels_BEST_all_samples.csv')).set_index('spot')
    S.index = S.index.astype(str)
    lab = S['label'].reindex(E.index)
    tr = E[lab.isin(['NDR', 'DR'])].copy()
    tr['y'] = (lab[tr.index] == 'DR').astype(int)
    tr['sample'] = [i.split('|')[0] for i in tr.index]
    genes_hvg = list(E.columns)
    print(f'training spots {len(tr)} | genes {len(genes_hvg)}')
    co, freq = lasso_genes(tr[genes_hvg].values, tr.y.values, tr['sample'].values, genes_hvg)
    nz = co[co != 0]
    stable = freq[freq >= 0.5]
    print(f'Lasso non-zero genes: {len(nz)} | stable (freq>=0.5): {len(stable)}')
    pd.DataFrame({'coef': co, 'freq': freq}).to_csv(os.path.join(WORK, 'lasso_all_hvg.csv'))
    M = pd.read_csv(PROJECT_ROOT + '/mouse/cart_region/grpr_gsva_v2/msigdb_sets.csv.gz')
    full = pd.read_csv(PROJECT_ROOT + '/mouse/cart_region/grpr_gsva_v2/msigdb_sets.csv.gz')
    c7 = pd.read_csv(PROJECT_ROOT + '/mouse/cart_region/grpr_gsva_v2/msigdb_sets.csv.gz')
    c7file = os.path.join(WORK, 'msigdb_c7_mouse.csv.gz')
    if not os.path.exists(c7file):
        print('NOTE: need the C7 dump; run step78a_dump_c7.R first')
        return
    C7 = pd.read_csv(c7file)
    tc = C7[C7.gs_name.str.contains(TC_PAT, case=False) & C7.gs_name.str.contains(ST_PAT, case=False)]
    sets = tc.groupby('gs_name')['gene_symbol'].apply(lambda v: sorted(set(v))).to_dict()
    print(f'C7 T-cell x state sets: {len(sets)}')
    set_state = {}
    for name in sets:
        for st, pat in STATES.items():
            if pd.Series([name]).str.contains(pat, case=False).iloc[0]:
                set_state[name] = st
                break
    print('set state counts:', pd.Series(list(set_state.values())).value_counts().to_dict())

    def assign(sel):
        cnt = {}
        for name, gs in sets.items():
            st = set_state.get(name)
            if st is None:
                continue
            for g in set(gs) & set(sel):
                cnt.setdefault(g, {}).setdefault(st, 0)
                cnt[g][st] += 1
        rows = []
        for g, d in cnt.items():
            st = max(d.items(), key=lambda kv: (kv[1], -list(STATES).index(kv[0])))[0]
            rows.append(dict(gene=g, state=st, n_sets=sum(d.values()), **{f'n_{k}': v for k, v in d.items()}))
        return pd.DataFrame(rows)
    A1 = assign(nz.index)
    A2 = assign(stable.index)
    A1.to_csv(os.path.join(WORK, 'assigned_genes_nonzero.csv'), index=False)
    A2.to_csv(os.path.join(WORK, 'assigned_genes_stable.csv'), index=False)
    print('\n=== 交集与功能归属（Lasso 非零基因 ∩ MSigDB T 功能集）===')
    print(A1.state.value_counts().to_dict(), '-> 基因数', len(A1))
    print(A1.sort_values('n_sets', ascending=False).head(15).round(0).to_string(index=False))
    print('\n（稳定基因 freq>=0.5 版本）', A2.state.value_counts().to_dict())
    panels = {}
    for src, tag in [(A1, 'nonzero'), (A2, 'stable')]:
        for st in STATES:
            g = [x for x in src.loc[src.state == st, 'gene'] if x in E.columns]
            if len(g) >= 5:
                panels[f'{st}_{tag}'] = g
    json.dump(panels, open(os.path.join(WORK, 'panels_clinical.json'), 'w'), indent=0)
    print('\npanels:', {k: len(v) for k, v in panels.items()})
    exp = E[[c for c in E.columns if c in set(sum(panels.values(), []))]]
    exp.to_csv(os.path.join(WORK, 'expr_panel_genes.csv'))
    print('expr for GSVA:', exp.shape)
if __name__ == '__main__':
    main()
