"""Step10: 用 DeSTVI 每 spot 的细胞比例 (prop_0..prop_30, leiden fine) 为每一张切片注释细胞类型。

- leiden 簇 -> 更新命名后的细胞类型 (11/12 簇按 scANVI 比例拆分)
- 每 spot: 优势细胞类型/占比、优势谱系、区室(Tumor/Immune/Stroma/Parenchyma)、tumor_frac 等
- 注释直接写入各样本 `<sample>.destvi_annotated.h5ad` 的 obs (仅新增列, 不改动原有数据), 并输出汇总表与图
输出: tables/st_spot_celltype_annotation.csv, tables/st_slide_celltype_summary.csv,
     tables/st_slide_compartment_summary.csv, figs/st_spatial_dominant_celltype.*,
     figs/st_slide_composition_bar.*, figs/st_slide_celltype_heatmap.*
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import glob
import re
import numpy as np
import pandas as pd
import h5py
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
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
LINEAGE = {'B Lymphoid': ['B cell', 'Plasma cell IgG', 'Plasma cell IgA'], 'Myeloid': ['Macrophage TREM2+', 'Macrophage SPP1+', 'Macrophage KC', 'Monocyte classical', 'LAMP3+ mregDC', 'pDC', 'Neutrophil', 'Mast cell'], 'T/NK': ['T cell CD8+', 'T cell CD4+', 'Treg', 'T cell proliferating', 'NK cell'], 'Stroma': ['CAF COLEC11+', 'CAF SFRP2+ matrix', 'CAF PDGFRA+', 'CAF TCF21+', 'CAF proliferating', 'Pericyte vSMC', 'Endothelial vascular', 'Endothelial LSEC'], 'Mal Epithelial': ['Malignant epithelial CEACAM5+', 'Malignant epithelial proliferating', 'Malignant epithelial gastric'], 'Parenchyma': ['Hepatocyte', 'Steroidogenic cell']}
LIN_OF = {t: f for f, ts in LINEAGE.items() for t in ts}
ALL_CT = [t for f in LINEAGE for t in LINEAGE[f]]
COMPARTMENT = {'Mal Epithelial': 'Tumor', 'Myeloid': 'Immune', 'T/NK': 'Immune', 'B Lymphoid': 'Immune', 'Stroma': 'Stroma', 'Parenchyma': 'Parenchyma'}
CT2COMP = {t: COMPARTMENT[LIN_OF[t]] for t in ALL_CT}
FAMCOL = {'B Lymphoid': '#3182bd', 'Myeloid': '#31a354', 'T/NK': '#de2d26', 'Stroma': '#756bb1', 'Mal Epithelial': '#e6a817', 'Parenchyma': '#238b45'}

def palette():
    colors = {}
    famcmap = {'B Lymphoid': ('Blues', 0.45, 1.0), 'Myeloid': ('Greens', 0.45, 1.0), 'T/NK': ('Reds', 0.45, 1.0), 'Stroma': ('Purples', 0.45, 1.0), 'Mal Epithelial': ('YlOrBr', 0.3, 0.95), 'Parenchyma': ('BuGn', 0.35, 0.95)}
    for fam, ts in LINEAGE.items():
        nm, lo, hi = famcmap[fam]
    return colors

def parse(name):
    p = name.split('_')
    return dict(slide=p[0], patient=p[1], time=p[2], response=p[3], tissue='_'.join(p[4:]))

def wcol(ct):
    return 'stprop_' + re.sub('[^0-9A-Za-z]+', '_', ct).strip('_')

def write_cat(f, name, cats, codes):
    if name in f['obs']:
        del f['obs'][name]
    g = f['obs'].create_group(name)
    g.attrs['encoding-type'] = 'categorical'
    g.attrs['encoding-version'] = '0.2.0'
    g.attrs['ordered'] = False
    d = g.create_dataset('categories', data=np.array(list(cats), dtype=object), dtype=h5py.string_dtype(encoding='utf-8'))
    d.attrs['encoding-type'] = 'string-array'
    d.attrs['encoding-version'] = '0.2.0'
    c = g.create_dataset('codes', data=np.asarray(codes, dtype='int32'))
    c.attrs['encoding-type'] = 'array'
    c.attrs['encoding-version'] = '0.2.0'

def write_num(f, name, vals):
    if name in f['obs']:
        del f['obs'][name]
    d = f['obs'].create_dataset(name, data=np.asarray(vals, dtype='float32'))
    d.attrs['encoding-type'] = 'array'
    d.attrs['encoding-version'] = '0.2.0'

def main():
    mp = pd.read_csv(os.path.join(WD, 'cluster_celltype_map_final_with_scanvi.csv'))
    mp['cluster'] = mp['cluster'].astype(str)
    w = mp.groupby(['cluster', 'cell_type'], as_index=False)['n_total'].sum()
    w['wt'] = w.n_total / w.groupby('cluster').n_total.transform('sum')
    wt = w.pivot(index='cluster', columns='cell_type', values='wt').reindex(columns=ALL_CT).fillna(0.0)
    cnv = pd.read_csv(os.path.join(WD, 'tables', 'cnv_verdict_per_cell.csv'))
    ct_cnv = cnv.groupby('cell_type_detail').cnv_verdict.apply(lambda s: (s == 'CNV+').mean()).to_dict()
    colors = palette()
    rows, slide_sum, slide_comp = ([], [], [])
    for d in sorted(glob.glob(os.path.join(SRC, '*'))):
        fs = glob.glob(os.path.join(d, '*.destvi_annotated.h5ad'))
        if not fs:
            continue
        sample = os.path.basename(d)
        meta = parse(sample)
        with h5py.File(fs[0], 'r+') as f:
            o = f['obs']
            idx = [k.decode() if isinstance(k, bytes) else str(k) for k in o['_index'][:]]
            pcols = [f'prop_{c}' for c in wt.index]
            pcols = [c for c in pcols if c in o]
            P = np.vstack([o[c][:].astype(np.float32) for c in pcols]).T
            P = np.nan_to_num(P)
            S = P.sum(1, keepdims=True)
            S[S == 0] = 1
            P = P / S
            Ct = P @ wt.loc[[c.split('_')[1] for c in pcols]].values
            Ct = np.clip(Ct, 0, None)
            tot = Ct.sum(1, keepdims=True)
            tot[tot == 0] = 1
            Ct = Ct / tot
            df = pd.DataFrame(Ct, columns=ALL_CT, index=idx)
            dom = df.idxmax(1)
            domf = df.max(1)
            lin = pd.DataFrame({f: df[ts].sum(1) for f, ts in LINEAGE.items()})
            domlin = lin.idxmax(1)
            comp = pd.DataFrame({c: df[[t for t in ALL_CT if CT2COMP[t] == c]].sum(1) for c in ['Tumor', 'Immune', 'Stroma', 'Parenchyma']})
            comp['dominant_compartment'] = comp.idxmax(1)
            xyz = f['obsm']['spatial'][:] if 'spatial' in f['obsm'] else np.full((len(idx), 2), np.nan)
            leiden = o['leiden']
            if isinstance(leiden, h5py.Group):
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in leiden['categories'][:]]
                leiden = np.array(cats)[leiden['codes'][:]]
            ann = pd.DataFrame({'sample': sample, 'patient': meta['patient'], 'time': meta['time'], 'response': meta['response'], 'tissue': meta['tissue'], 'barcode': idx, 'x': xyz[:, 0], 'y': xyz[:, 1], 'leiden': leiden, 'cell_type_dominant': dom.values, 'dominant_frac': domf.values, 'lineage_dominant': domlin.values, 'lineage_dominant_frac': lin.max(1).values, 'compartment': comp['dominant_compartment'].values, 'tumor_frac': comp['Tumor'].values, 'immune_frac': comp['Immune'].values, 'stroma_frac': comp['Stroma'].values, 'parenchyma_frac': comp['Parenchyma'].values})
            ann['cnv_verdict_dominant'] = np.where(ann.cell_type_dominant.map(ct_cnv).fillna(0) > 0.5, 'CNV+', 'CNV-')
            rows.append(ann)
            cat_cols = {'cell_type_dominant': ALL_CT, 'lineage_dominant': list(LINEAGE), 'compartment': ['Tumor', 'Immune', 'Stroma', 'Parenchyma']}
            for cname, cats in cat_cols.items():
                codes = pd.Categorical(ann[cname], categories=cats).codes.astype('int32')
                write_cat(f, cname, cats, codes)
            write_cat(f, 'cnv_verdict_dominant', ['CNV+', 'CNV-'], pd.Categorical(ann.cnv_verdict_dominant, categories=['CNV+', 'CNV-']).codes)
            for cname in ['dominant_frac', 'lineage_dominant_frac', 'tumor_frac', 'immune_frac', 'stroma_frac', 'parenchyma_frac']:
                write_num(f, cname, ann[cname].values)
            for t in ALL_CT:
                write_num(f, wcol(t), df[t].values)
            order = [k.decode() if isinstance(k, bytes) else str(k) for k in o.attrs['column-order']]
            new = [c for c in list(cat_cols) + ['cnv_verdict_dominant', 'dominant_frac', 'lineage_dominant_frac', 'tumor_frac', 'immune_frac', 'stroma_frac', 'parenchyma_frac'] + [wcol(t) for t in ALL_CT] if c not in order]
            o.attrs['column-order'] = np.array(order + new, dtype=object)
            f.flush()
        slide_sum.append(dict(sample=sample, **meta, n_spot=len(df), dominant_cell_type=dom.value_counts().idxmax(), dominant_frac_mean=float(domf.mean()), top3=';'.join((f'{k}:{v / len(df):.1%}' for k, v in dom.value_counts().head(3).items())), **{f'mean_{t}': float(df[t].mean()) for t in ALL_CT}))
        comp_mean = comp[['Tumor', 'Immune', 'Stroma', 'Parenchyma']].mean()
        slide_comp.append(dict(sample=sample, **meta, n_spot=len(df), **comp_mean.to_dict(), dominant_compartment=comp['dominant_compartment'].value_counts().idxmax()))
        print(f"{sample}: spots={len(df)} dominant={dom.value_counts().idxmax()} top3={', '.join((f'{k}:{v / len(df):.0%}' for k, v in dom.value_counts().head(3).items()))}", flush=True)
    spot = pd.concat(rows, ignore_index=True)
    spot.to_csv(os.path.join(TAB, 'st_spot_celltype_annotation.csv'), index=False)
    ss = pd.DataFrame(slide_sum)
    ss.to_csv(os.path.join(TAB, 'st_slide_celltype_summary.csv'), index=False)
    sc = pd.DataFrame(slide_comp)
    sc.to_csv(os.path.join(TAB, 'st_slide_compartment_summary.csv'), index=False)
    print('\n=== 切片整体细胞类型构成 (mean proportion, 前 6) ===')
    print(ss[['sample', 'n_spot', 'dominant_cell_type', 'top3']].to_string(index=False))
    print('\n=== 区室构成 ===\n', sc[['sample', 'Tumor', 'Immune', 'Stroma', 'Parenchyma', 'dominant_compartment']].round(3).to_string(index=False))
    samples = ss['sample'].tolist()
    ncol = 4
    nrow = int(np.ceil(len(samples) / ncol))
    mean_cols = [f'mean_{t}' for t in ALL_CT]
    M = ss.set_index('sample')[mean_cols]
    M.columns = ALL_CT
    M = M.loc[ss.sort_values(['patient', 'time'])['sample'].values]
    bottom = np.zeros(len(M))
    for t in ALL_CT:
        bottom += M[t].values
    Z = (M - M.mean()) / M.std(ddof=0).replace(0, np.nan)
    print('\nDONE: 注释已写入各样本 h5ad 的 obs (新增列), 汇总见 tables/st_*.csv')
if __name__ == '__main__':
    main()
