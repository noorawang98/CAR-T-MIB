"""Step60: 与鼠侧 step03 完全一致的 niche/MIB 口径(人侧版本)。
鼠侧口径(step03_region_annot.py):
  - 模块分 = 逐基因在样本内 z 后取均值(setscore), 直接在 spot 表达上算(非 DeSTVI 比例)
  - 基因集: endo/mye/caf/acaf/ccr1mye/tcell/bcell (人同源基因), tbz=(tcell+bcell)/2
  - 分位: Q50/Q75/Q90 = 0.50/0.75/0.90 (样本内百分位); KNN=6
  - niche: Vessel(endo_q>=Q90) → Perivascular(n_endo_nb>=2) → CCR1_MIB(ccr1mye_q>=Q75 & (acaf_q>=Q50 | n_acaf_nb>=2))
           → Myeloid_enriched → CAF_enriched → TB_zone → Other
  - mib_perivasc_endo_caf = CCR1_MIB & (n_endo_nb>=2 | endo_q>=Q50)
输出: <WD>/prognosis/mib_mouse_aligned_slide.csv (+ 并入患者表供预后)
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, glob
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
from sklearn.neighbors import NearestNeighbors
WD = os.environ.get('ST_WD', PROJECT_ROOT + '/destvi_260911')
GS = {'endo': ['PECAM1', 'CDH5', 'VWF', 'KDR', 'ENG', 'LYVE1', 'FLT1', 'EMCN', 'PLVAP', 'RAMP2'], 'mye': ['LYZ', 'CSF1R', 'CD14', 'CD68', 'ADGRE1', 'C1QA', 'C1QB', 'C1QC', 'ITGAM', 'FCGR3A', 'S100A8', 'S100A9', 'ITGAX', 'MSR1', 'MRC1'], 'caf': ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'PDGFRA', 'PDGFRB', 'FAP', 'POSTN', 'LUM', 'COL6A1'], 'acaf': ['ACTA2', 'TAGLN', 'MYH11', 'CSPG4', 'RGS5', 'DES', 'NOTCH3', 'COL5A1'], 'ccr1mye': ['CCR1', 'CCL2', 'CCL7', 'CCL8', 'CCR2', 'CD14', 'LYZ', 'CSF1R', 'C1QA'], 'tcell': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD8B', 'CD4', 'TRAC', 'TRBC2', 'THY1', 'MS4A4A'], 'bcell': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'BANK1', 'CR2']}
Q50, Q75, Q90, KNN = (0.5, 0.75, 0.9, 6)

def zscore_cols(X, cols):
    """只对目标基因列做"逐基因、样本内 z"(避免 densify 全矩阵: A4632 有 9.6 万 spot)"""
    Xs = X[:, cols]
    Xd = np.asarray(Xs.todense() if sp.issparse(Xs) else Xs, dtype=np.float32)
    return (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-09)

def setscore(Z, genes, idx):
    cols = [idx[g] for g in genes if g in idx]
    return (Z[:, cols].mean(1) if cols else None, cols)

def main():
    meta = pd.read_csv(os.path.join(WD, 'tables', 'st_image_mapping.csv'))
    rows = []
    for _, r in meta.iterrows():
        s = r['sample']
        p = os.path.join(WD, 'destvi_samples', s, f'{s}.destvi_annotated.h5ad')
        a = ad.read_h5ad(p)
        X = sp.csr_matrix(a.X, dtype=np.float32)
        idx = {g: i for i, g in enumerate(a.var_names)}
        allg = sorted({g for gs in GS.values() for g in gs if g in idx})
        Z = zscore_cols(X, [idx[g] for g in allg])
        zi = {g: i for i, g in enumerate(allg)}
        d = {}
        for k, genes in GS.items():
            cols = [zi[g] for g in genes if g in zi]
            d[k] = Z[:, cols].mean(1) if cols else np.zeros(X.shape[0], np.float32)
        d['tbz'] = (d['tcell'] + d['bcell']) / 2
        D = pd.DataFrame(d)
        for k in ['endo', 'mye', 'caf', 'acaf', 'ccr1mye', 'tbz']:
            D[k + '_q'] = D[k].rank(pct=True)
        xy = np.asarray(a.obsm['spatial'])
        nn = NearestNeighbors(n_neighbors=min(KNN + 1, len(xy))).fit(xy)
        nbr = nn.kneighbors(xy, return_distance=False)[:, 1:]
        ev = D['endo_q'].values >= Q90
        av = (D['acaf'].values >= 0) & (D['acaf_q'].values >= Q50)
        n_endo, n_acaf = (ev[nbr].sum(1), av[nbr].sum(1))
        niche = np.full(len(D), 'Other', dtype=object)
        m = D['endo_q'].values >= Q90
        niche[m] = 'Vessel'
        m2 = ~m & (n_endo >= 2)
        niche[m2] = 'Perivascular'
        m3 = ~m & ~m2 & (D['ccr1mye_q'].values >= Q75) & ((D['acaf_q'].values >= Q50) | (n_acaf >= 2))
        niche[m3] = 'CCR1_MIB'
        m4 = ~m & ~m2 & ~m3 & (D['mye_q'].values >= Q75)
        niche[m4] = 'Myeloid_enriched'
        m5 = ~m & ~m2 & ~m3 & ~m4 & (D['caf_q'].values >= Q75)
        niche[m5] = 'CAF_enriched'
        m6 = ~m & ~m2 & ~m3 & ~m4 & ~m5 & (D['tbz_q'].values >= Q75)
        niche[m6] = 'TB_zone'
        mib_pv = m3 & ((n_endo >= 2) | (D['endo_q'].values >= Q50))
        nc = pd.Series(niche).value_counts(normalize=True).to_dict()
        pd.DataFrame({'sample': s, 'x': xy[:, 0], 'y': xy[:, 1], 'niche': niche, 'mib': m3, 'mib_pv': mib_pv, 'ccr1mye': d['ccr1mye'], 'acaf': d['acaf'], 'mye': d['mye'], 'caf': d['caf'], 'endo': d['endo']}).to_csv(os.path.join(WD, 'prognosis', f'niche_spots_{s}.csv.gz'), index=False, compression='gzip')
        SAMPLE_SPOT = True
        rows.append(dict(sample=s, patient=r['patient'], time=r['time'], resp='PR' if r['response'] == 'PR' else 'SD/PD', n_spot=len(D), mib_frac=float(m3.mean()), pervimib_flag_frac=float(mib_pv.mean()), mib_perivasc_share=float(mib_pv.sum() / max(m3.sum(), 1)), vessel_frac=float(nc.get('Vessel', 0)), perivascular_frac=float(nc.get('Perivascular', 0)), myeloid_frac=float(nc.get('Myeloid_enriched', 0)), caf_frac=float(nc.get('CAF_enriched', 0)), tbz_frac=float(nc.get('TB_zone', 0)), other_frac=float(nc.get('Other', 0)), mib_pv_ratio=float(mib_pv.mean()) / max(float(m3.mean() - mib_pv.mean()), 1e-09)))
        print(f"{s}: n={len(D)} CCR1_MIB={m3.mean():.3f} perviMIB={mib_pv.mean():.3f} Vessel={nc.get('Vessel', 0):.3f} Peri={nc.get('Perivascular', 0):.3f}", flush=True)
        del a, X, Z, D
    T = pd.DataFrame(rows)
    os.makedirs(os.path.join(WD, 'prognosis'), exist_ok=True)
    T.to_csv(os.path.join(WD, 'prognosis', 'mib_mouse_aligned_slide.csv'), index=False)
    pd.set_option('display.width', 240)
    print('\n=== 人侧(鼠口径)逐切片 ===')
    print(T[['sample', 'time', 'resp', 'mib_frac', 'pervimib_flag_frac', 'mib_perivasc_share', 'vessel_frac', 'perivascular_frac']].round(3).to_string(index=False))
    print('\n组均值:', T.groupby(['time', 'resp'])[['mib_frac', 'pervimib_flag_frac', 'mib_perivasc_share']].mean().round(3).to_string())
if __name__ == '__main__':
    main()
