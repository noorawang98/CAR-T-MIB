"""Step100: perviMIB spot 内的 SD/PD vs PR 比较 —— 伪bulk 表达矩阵 + 解卷积比例。
口径: 主分析 = 鼠口径 perviMIB (mib_pv); 另出 DeSTVI perviMIB 作为敏感性。
做法: 每张切片把该切片内全部 perviMIB spot 的原始 counts 逐基因求和 -> 伪bulk(切片为单位, 避免 spot 伪重复);
      同时汇总这些 spot 的 DeSTVI 细胞类型比例均值与 spot 数。
输出: prognosis/pervimib_pseudobulk_counts.csv.gz (切片 x 基因), _meta.csv, pervimib_deconvolution.csv
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
CT = ['B cell', 'Plasma cell IgA', 'Plasma cell IgG', 'Macrophage TREM2+', 'Macrophage SPP1+', 'Macrophage KC', 'Monocyte classical', 'LAMP3+ mregDC', 'pDC', 'Neutrophil', 'Mast cell', 'T cell CD8+', 'T cell CD4+', 'Treg', 'T cell proliferating', 'NK cell', 'CAF COLEC11+', 'CAF SFRP2+ matrix', 'CAF PDGFRA+', 'CAF TCF21+', 'CAF proliferating', 'Pericyte vSMC', 'Endothelial vascular', 'Endothelial LSEC', 'Malignant epithelial CEACAM5+', 'Malignant epithelial proliferating', 'Malignant epithelial gastric', 'Hepatocyte', 'Steroidogenic cell']

def sym(a):
    return a.var['real_gene_name'].astype(str).values if 'real_gene_name' in a.var.columns else np.asarray(a.var_names, dtype=str)
pbs, metas, decs = ({}, [], [])
for _, m in meta.iterrows():
    s = m['sample']
    ps = pd.read_csv(os.path.join(WD, 'update_statistics', 'per_sample', f'{s}.csv'))[['x', 'y', 'score_mm', 'caf', 'vasc']]
    g = os.path.join(PROG, f'niche_spots_{s}.csv.gz')
    ps = ps.merge(pd.read_csv(g)[['x', 'y', 'mib', 'mib_pv']], on=['x', 'y'], how='left')
    xy = np.c_[ps.x.values, ps.y.values]
    tree = cKDTree(xy)
    q = lambda v: np.asarray(v) >= np.quantile(v, 0.75)
    band = q(ps.vasc.values)[tree.query(xy, k=min(7, len(ps)), workers=8)[1][:, 1:]].sum(1) >= 3
    masks = {'perviMIB_mouse': pd.to_numeric(ps['mib_pv'], errors='coerce').fillna(0).values > 0, 'perviMIB_DestVI': q(ps.score_mm.values) & q(ps.caf.values) & band}
    a = ad.read_h5ad(os.path.join(H5, s, f'{s}.destvi_annotated.h5ad'), backed='r')
    obs = a.obs
    axy = np.c_[np.asarray(obs['x'] if 'x' in obs else a.obsm['spatial'][:, 0]), np.asarray(obs['y'] if 'y' in obs else a.obsm['spatial'][:, 1])]
    t2 = cKDTree(axy)
    _, rows = t2.query(xy, k=1)
    prop_cols = [c for c in obs.columns if c.startswith('stprop_')]
    for nm, mask in masks.items():
        n = int(mask.sum())
        metas.append(dict(sample=s, patient=m['patient'], time=m['time'], resp='PR' if m['response'] == 'PR' else 'SD/PD', mib_def=nm, n_pervimib=n, tissue=m['tissue'], frac_pervimib=float(mask.mean()) * 100))
        if n < 5:
            print(f'  {s[:5]} {nm}: n={n} (<5, 跳过伪bulk)', flush=True)
            continue
        sub = a[rows[mask], :].to_memory()
        X = sub.layers['counts'] if 'counts' in sub.layers else sub.X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        genes = sym(sub)
        df = pd.DataFrame(X, columns=genes)
        df = df.loc[:, ~df.columns.duplicated()]
        pbs[s, nm] = df.sum(0)
        dd = obs.iloc[rows[mask]][prop_cols].astype(float)
        dd = dd.rename(columns=lambda c: c.replace('stprop_', '').replace('_', ' ') if False else c)
        row = {'sample': s, 'mib_def': nm, 'n_pervimib': n, 'patient': m['patient'], 'time': m['time'], 'resp': 'PR' if m['response'] == 'PR' else 'SD/PD'}
        for c in prop_cols:
            row[c] = float(dd[c].mean())
        decs.append(row)
        print(f'  {s[:5]} {nm}: n={n}, genes={df.shape[1]}', flush=True)
        del sub, X, df
    del a
M = pd.DataFrame(metas)
M.to_csv(os.path.join(PROG, 'pervimib_pseudobulk_meta.csv'), index=False)
PB = pd.DataFrame(pbs).T
PB.index = pd.MultiIndex.from_tuples(PB.index, names=['sample', 'mib_def'])
PB.to_csv(os.path.join(PROG, 'pervimib_pseudobulk_counts.csv.gz'), compression='gzip')
D = pd.DataFrame(decs)
D.to_csv(os.path.join(PROG, 'pervimib_deconvolution.csv'), index=False)
pd.set_option('display.width', 240)
print('\n=== 每切片 perviMIB spot 数 ===')
print(M.pivot_table(index=['sample', 'time', 'resp'], columns='mib_def', values='n_pervimib').fillna(0).astype(int).to_string())
print('\n伪bulk 矩阵:', PB.shape, '| 解卷积表:', D.shape)
