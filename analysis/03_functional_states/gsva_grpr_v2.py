"""Step54: PR/GR v2 = unsupervised K-means on MSigDB-consensus gene-set GSVA scores.

Gene panels (mouse, msigdbr):
  Exhaustion / Effector / Memory / Stemness / Hypoxia / Inflammation / Cytokine-chemokine
  - only sets whose name ends with '_UP', except GO (GOBP/GOCC/GOMF) and KEGG sets
  - a gene enters a panel only if it occurs in >= 3 distinct datasets (sets) of that panel
  - final panel = panel genes INTERSECT the HVGs computed on the CAR+ spots
Scores: GSVA per panel on the CAR+ treated spots, z-scored inside each sample (batch-free),
K-means k=2..5 -> Good response (high effector/memory/stemness, low exhaustion) vs
Poor response (mirror image).
Outputs -> grpr_gsva_v2/
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, json, itertools
import numpy as np, pandas as pd, anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from common import RES, RCL, TREATED, grpr_labels
OUT = PROJECT_ROOT + '/mouse/cart_region/grpr_gsva_v2'
FIG = os.path.join(OUT, 'figs')
NHVG = 5000
CATS = {'Exhaustion': 'EXHAUST', 'Effector': 'EFFECTOR|CYTOTOX|GZMB|PRF1', 'Memory': 'MEMORY', 'Stemness': 'STEMNESS|STEM_CELL', 'Hypoxia': 'HYPOXIA', 'Inflammation': 'INFLAMMAT', 'Cytokine_chemokine': 'CYTOKINE|CHEMOKINE'}
GO_KEGG = '^(GOBP|GOCC|GOMF|KEGG|REACTOME|WP|BIOCARTA|PID)_'

def msigdb_panels():
    M = pd.read_csv(os.path.join(OUT, 'msigdb_sets.csv.gz'))
    M = M[['gs_name', 'gene_symbol']].dropna().drop_duplicates()
    M['is_gokegg'] = M.gs_name.str.match(GO_KEGG)
    up = M.gs_name.str.endswith('_UP')
    M = M[up | M.is_gokegg]
    panels, info = ({}, [])
    for cat, pat in CATS.items():
        S = M[M.gs_name.str.contains(pat, case=False, regex=True)]
        if not len(S):
            continue
        sets = S.groupby('gs_name')['gene_symbol'].apply(set)
        sets = sets[sets.apply(len) >= 5]
        cnt = {}
        for st in sets:
            for g in st:
                cnt[g] = cnt.get(g, 0) + 1
        genes = sorted([g for g, n in cnt.items() if n >= 3])
        panels[cat] = genes
        info.append(dict(category=cat, n_sets=len(sets), n_genes_ge3sets=len(genes), examples='; '.join(list(sets.index[:3]))))
        print(f'  {cat:20s} sets={len(sets):4d} genes(>=3 sets)={len(genes):5d}', flush=True)
    return (panels, pd.DataFrame(info), M)

def carpos_matrix():
    L = grpr_labels()
    cnt_all, obs, var = ([], [], None)
    for s in TREATED:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'))
        names = np.array([f'{s}|{b}' for b in a.obs_names])
        pos = np.isin(names, L.index.values)
        cnt_all.append(a.layers['counts'][pos].tocsr())
        obs.append(pd.DataFrame({'sample': s}, index=names[pos]))
        if var is None:
            var = a.var_names.to_numpy()
        del a
        print(f'  {s}: CAR+ {int(pos.sum())}', flush=True)
    from scipy.sparse import vstack
    return (vstack(cnt_all).tocsr(), var, pd.concat(obs))

def hvg_carpos(cnt, var, n=NHVG):
    C = np.asarray(cnt.todense(), dtype=np.float32)
    mu, va = (C.mean(0), C.var(0))
    ok = mu > 0.01
    bins = pd.qcut(pd.Series(mu[ok]), 20, labels=False, duplicates='drop')
    v = pd.Series(va[ok])
    z = (v - v.groupby(bins.values).transform('mean')) / v.groupby(bins.values).transform('std').replace(0, np.nan)
    tab = pd.DataFrame({'gene': var[ok], 'std_var': z.values}).sort_values('std_var', ascending=False)
    return tab.head(n).gene.tolist()

def main():
    os.makedirs(FIG, exist_ok=True)
    print('MSigDB panels:')
    panels, PINFO, M = msigdb_panels()
    cnt, var, obs = carpos_matrix()
    hv = set(hvg_carpos(cnt, var))
    print(f'CAR+ HVG: {len(hv)}')
    tot = np.asarray(cnt.sum(1)).ravel()
    med = np.median(tot)
    gidx = {g: i for i, g in enumerate(var)}
    union = sorted({g for v in panels.values() for g in v} & hv)
    print(f'union(panel ∩ HVG): {len(union)} genes')
    X = np.log1p(np.asarray(cnt[:, [gidx[g] for g in union]].todense(), dtype=np.float32) / tot[:, None] * med)
    E = pd.DataFrame(X, index=obs.index, columns=union)
    E.to_csv(os.path.join(OUT, 'carpos_hvg_panel_expr.csv'))
    panel_hvg = {c: sorted(set(g) & hv) for c, g in panels.items()}
    pd.DataFrame([dict(category=c, n_genes=len(g), genes=';'.join(g)) for c, g in panel_hvg.items()]).to_csv(os.path.join(OUT, 'panels_used.csv'), index=False)
    PINFO.to_csv(os.path.join(OUT, 'panel_source_summary.csv'), index=False)
    import subprocess
    E.to_csv(os.path.join(OUT, 'expr_for_gsva.csv'))
    with open(os.path.join(OUT, 'panels_for_gsva.json'), 'w') as fh:
        json.dump({c: g for c, g in panel_hvg.items() if len(g) >= 5}, fh)
    print('wrote expression + panels for GSVA (gsva.R)')
    print(json.dumps({c: len(g) for c, g in panel_hvg.items()}, indent=0))
if __name__ == '__main__':
    main()
