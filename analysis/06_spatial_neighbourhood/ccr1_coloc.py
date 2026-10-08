"""Clinical spot-table preparation for the final per-section neighbourhood analysis.
Retains the original input/feature definitions and ccr1_spots_all.csv export;
obsolete neighbourhood tests and drawing sections have been removed."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import glob
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import tifffile
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, mannwhitneyu
WD = os.environ.get('ST_WD', PROJECT_ROOT + '/destvi_260911')
SRC = os.environ.get('ST_SRC', PROJECT_ROOT + '/destvi_fine/samples')
SRC2 = os.environ.get('ST_SRC2', os.path.join(WD, 'destvi_new_samples'))

def ann_path(s):
    for d in (SRC2, SRC):
        g = glob.glob(os.path.join(d, s, '*.destvi_annotated.h5ad'))
        if g:
            return g[0]
    raise FileNotFoundError(f'no destvi_annotated h5ad for {s}')
FIG = os.path.join(WD, 'figs')
TAB = os.path.join(WD, 'tables')
CACHE = os.path.join(WD, '.cache', 'st13')
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
MONO_MAC = ['Macrophage TREM2+', 'Macrophage SPP1+', 'Macrophage KC', 'Monocyte classical']
MYELOID_ALL = MONO_MAC + ['LAMP3+ mregDC', 'pDC', 'Neutrophil', 'Mast cell']
CAF = ['CAF COLEC11+', 'CAF SFRP2+ matrix', 'CAF PDGFRA+', 'CAF TCF21+', 'CAF proliferating']
TUMOR = ['Malignant epithelial CEACAM5+', 'Malignant epithelial proliferating', 'Malignant epithelial gastric']
VASC = ['Endothelial vascular', 'Endothelial LSEC', 'Pericyte vSMC']
TNK = ['T cell CD8+', 'T cell CD4+', 'Treg', 'T cell proliferating', 'NK cell']
STROMA = CAF + VASC
CHIP_RUN = {'Y40150E9': 'A1556_4314', 'Y40321F6': 'A4919_1788', 'Y40321L7': 'A5879_1856', 'Y40150G4': 'A7229_3571', 'Y40051C9': 'A6994_7396', 'Y40150F1': 'A72455_9418', 'Y40051D6': 'A6365', 'Y40051FC': 'A0098', 'Y40321A5': 'A4632'}
KS_RADIUS = list(range(1, 11))
KS_VASC = list(range(1, 41))
KMAX = 40
KVASC = 40

def ccr1_expected_vec(cols):
    """参考中每个细胞类型的 CCR1 平均表达 (log-norm), 用于深度稳健的期望 CCR1 得分"""
    f = os.path.join(TAB, 'ccr1_mean_by_celltype.csv')
    if not os.path.exists(f):
        return pd.Series(0.0, index=cols)
    m = pd.read_csv(f, index_col=0)['ccr1_log']
    return pd.Series({c: float(m.get(c, 0.0)) for c in cols})

def load_props():
    mp = pd.read_csv(os.path.join(WD, 'cluster_celltype_map_final_with_scanvi.csv'))
    mp['cluster'] = mp['cluster'].astype(str)
    w = mp.groupby(['cluster', 'cell_type'], as_index=False)['n_total'].sum()
    w['wt'] = w.n_total / w.groupby('cluster').n_total.transform('sum')
    return w.pivot(index='cluster', columns='cell_type', values='wt').fillna(0.0)

def car_umis(chip, x, y):
    """从 saw_results/<run>/.../CUSTOM/CAR.gem 取该切片范围内的 CAR UMI 计数"""
    run = CHIP_RUN.get(chip)
    f = PROJECT_ROOT + f'/saw_results/{run}/STEREO_ANALYSIS_WORKFLOW_PROCESSING/CUSTOM/CAR.gem'
    if run is None or not os.path.exists(f):
        return (np.zeros(len(x)), 0)
    g = pd.read_csv(f, sep='\t', header=None, usecols=[2, 3, 4], names=['x', 'y', 'n'])
    m = (g.x >= x.min() - 1000) & (g.x <= x.max() + 1000) & (g.y >= y.min() - 1000) & (g.y <= y.max() + 1000)
    g = g[m]
    if len(g) == 0:
        return (np.zeros(len(x)), 0)
    tree = cKDTree(np.c_[x, y])
    d, i = tree.query(np.c_[g.x, g.y], k=1)
    out = np.zeros(len(x))
    ok = d <= 60
    np.add.at(out, i[ok], g.n.values[ok])
    return (out, int(out.sum()))

def tissue_intensity(chip, run, x, y):
    """spot 处 ssDNA 配准图亮度 (判断是否在组织内)"""
    p = PROJECT_ROOT + f'/saw_results/{run}/STEREO_ANALYSIS_WORKFLOW_PROCESSING/IMAGE/{chip}_ssDNA_regist.tif'
    if not os.path.exists(p):
        return (np.full(len(x), np.nan), np.nan)
    with tifffile.TiffFile(p) as t:
        H, W = t.pages[0].shape
        sub = t.pages[0].asarray()[::16, ::16]
        thr = max(5.0, 0.06 * np.percentile(sub, 99.5))
        img = t.pages[0].asarray()
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    inten = img[yi, xi].astype(float)
    del img
    return (inten, thr)

def _unused_knn_table(tree, n):
    """一次算出 k=1..KMAX 的近邻索引 (n x KMAX), 不含自身"""
    return None

def enrich_curve(tree, coords, X, Y, base_X, nn, ks):
    """邻域富集曲线: 半径 = k×NN (k in ks); Y-high spot 邻域内 X-high 占比 / 全局 X-high 占比"""
    out = {}
    if not Y.any():
        return {k: (np.nan, np.nan) for k in ks}
    yx = coords[Y]
    for k in ks:
        nb = tree.query_ball_point(yx, k * nn)
        f = [X[i].mean() for i in nb if len(i)]
        obs = float(np.mean(f)) if f else np.nan
        out[k] = (obs / base_X if base_X > 0 else np.nan, obs)
    return out

def main():
    wt = load_props()
    meta = pd.read_csv(os.path.join(TAB, 'st_image_mapping.csv'))
    rows = []
    perivasc_masks = {}
    for _, r in meta.iterrows():
        s = r['sample']
        cachef = os.path.join(CACHE, f'{s}.pkl')
        if os.path.exists(cachef):
            d = pd.read_pickle(cachef)
        else:
            a = ad.read_h5ad(ann_path(s))
            x, y = np.asarray(a.obsm['spatial']).T
            pcols = [c for c in [f'prop_{c}' for c in wt.index] if c in a.obs.columns]
            P = np.vstack([a.obs[c].values for c in pcols]).T.astype(np.float32)
            cl = [c.split('_')[1] for c in pcols]
            Ct = P @ wt.loc[cl].values
            Ct = Ct / np.clip(Ct.sum(1, keepdims=True), 1e-09, None)
            prop = pd.DataFrame(Ct, columns=wt.columns)
            ccr1 = np.asarray(a[:, 'CCR1'].layers['counts'].todense()).ravel().astype(float)
            run = CHIP_RUN.get(r['chip'], '')
            car, car_tot = car_umis(r['chip'], x, y)
            inten, thr = tissue_intensity(r['chip'], run, x, y) if run else (np.full(len(x), np.nan), np.nan)
            d = pd.DataFrame({'x': x, 'y': y, 'CCR1': ccr1, 'CAR': car, 'img_int': inten})
            for k, v in [('mono_mac', MONO_MAC), ('myeloid', MYELOID_ALL), ('caf', CAF), ('tumor', TUMOR), ('vasc', VASC), ('tnk', TNK), ('stroma', STROMA)]:
                d[k] = prop[[c for c in v if c in prop.columns]].sum(1).values
            ev = ccr1_expected_vec(list(prop.columns)).values.astype(np.float32)
            d['ccr1_exp'] = prop.values @ ev
            d['ccr1_exp_my'] = d['ccr1_exp'] * d['myeloid']
            d['ccr1_exp_mm'] = d['ccr1_exp'] * d['mono_mac']
            d.to_pickle(cachef)
            print(f'{s}: n={len(d)} CCR1+spots={int((ccr1 > 0).sum())} CAR={car_tot} tissue_thr={thr:.1f}', flush=True)
        d['sample'] = s
        d['group'] = r['patient'] and f"{r['time']}|{('PR' if r['response'] == 'PR' else 'SD/PD')}"
        d['patient'] = r['patient']
        d['time'] = r['time']
        d['resp'] = 'PR' if r['response'] == 'PR' else 'SD/PD'
        rows.append(d)
    A = pd.concat(rows, ignore_index=True)
    A['score_mm_raw'] = A.CCR1 * A.mono_mac
    A['score_my_raw'] = A.CCR1 * A.myeloid
    A['score_mm'] = A['ccr1_exp_mm']
    A['score_my'] = A['ccr1_exp_my']
    A.to_csv(os.path.join(TAB, 'ccr1_spots_all.csv'), index=False)
if __name__ == '__main__':
    main()
