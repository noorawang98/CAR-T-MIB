"""Step105a: 基于 MIB_DestVI+ spot 的伪bulk (切片为单位) —— 供 CIBERSORTx 打分 (任务2)。
输出: prognosis/mibdestvi_pseudobulk_counts.csv.gz (切片 x 基因), mibdestvi_pseudobulk_meta.csv
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.spatial import cKDTree
WD = os.environ.get('ST_WD', PROJECT_ROOT + '/destvi_260911')
PROG = os.path.join(WD, 'prognosis')
H5 = os.path.join(WD, 'destvi_samples')
meta = pd.read_csv(os.path.join(WD, 'tables', 'st_image_mapping.csv'))

def sym(a):
    return a.var['real_gene_name'].astype(str).values if 'real_gene_name' in a.var.columns else np.asarray(a.var_names, dtype=str)
pbs, rows_meta = ({}, [])
for _, m in meta.iterrows():
    s = m['sample']
    ps = pd.read_csv(os.path.join(WD, 'update_statistics', 'per_sample', f'{s}.csv'))[['x', 'y', 'score_mm', 'caf', 'vasc']]
    ps = ps.merge(pd.read_csv(os.path.join(PROG, f'niche_spots_{s}.csv.gz'))[['x', 'y', 'mib', 'mib_pv']], on=['x', 'y'], how='left')
    xy = np.c_[ps.x.values, ps.y.values]
    tree = cKDTree(xy)
    q = lambda v: np.asarray(v) >= np.quantile(v, 0.75)
    band = q(ps.vasc.values)[tree.query(xy, k=min(7, len(ps)), workers=8)[1][:, 1:]].sum(1) >= 3
    MIBD = q(ps.score_mm.values) & q(ps.caf.values)
    a = ad.read_h5ad(os.path.join(H5, s, f'{s}.destvi_annotated.h5ad'), backed='r')
    obs = a.obs
    axy = np.asarray(a.obsm['spatial'])
    _, idx = cKDTree(axy).query(xy, k=1)
    for nm, mask in [('MIB_DestVI', MIBD), ('MIB_DestVI_band', MIBD & band), ('nonMIB', ~MIBD)]:
        n = int(mask.sum())
        if n < 5:
            print(f'  {s[:5]} {nm}: n={n} 跳过')
            continue
        sub = a[idx[mask], :].to_memory()
        X = sub.layers['counts'] if 'counts' in sub.layers else sub.X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        df = pd.DataFrame(X, columns=sym(sub))
        df = df.loc[:, ~df.columns.duplicated()]
        key = f'{s}|{nm}'
        pbs[key] = df.sum(0)
        rows_meta.append(dict(key=key, sample=s, patient=m['patient'], time=m['time'], resp='PR' if m['response'] == 'PR' else 'SD/PD', set=nm, n_spot=n, n_gene=df.shape[1]))
        print(f'  {s[:5]} {nm}: n={n} genes={df.shape[1]}', flush=True)
        del sub, X, df
    del a
PB = pd.DataFrame(pbs)
PB.index.name = 'key'
PB.to_csv(os.path.join(PROG, 'mibdestvi_pseudobulk_counts.csv.gz'), compression='gzip')
pd.DataFrame(rows_meta).to_csv(os.path.join(PROG, 'mibdestvi_pseudobulk_meta.csv'), index=False)
print('pseudobulk:', PB.shape)
