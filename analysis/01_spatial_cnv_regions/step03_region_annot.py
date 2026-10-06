"""Step03: region annotation of every spot (Task 1A).

Two parallel label systems are produced, as requested ("两者都做"):
  * region_spot  -- per-spot rule-based labels from marker/CNV features
  * region_leiden-- the object's own unsupervised clustering (leiden_0.8) mapped to
                    regions by cluster-mean features

Tumour calling follows the destvi_260911 CNV logic (step4a-4d): a per-spot
chromosome-window deviation score ("cnv_score", inferCNV style, window = 101 genes)
is computed against a within-sample reference built from the spots with the lowest
melanoma-marker score, and the reference's 95th percentile is used as the CNV cut.
Because the model is a B16-type melanoma (Mlana/Dct/Tyr/Pmel positive, Epcam negative),
the melanocytic programme is used as the second, independent tumour feature.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, re, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import anndata as ad
warnings.filterwarnings('ignore')
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
RCL = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/recluster_obj'
GTF = PROJECT_ROOT + '/mouse/filtered_mm.gtf'
SAMPLES = ['Vehicle', 'NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
KNN = 6
WIN = 101
TUMOR_MOD = os.environ.get('TUMOR_MOD', 'mel')
SUF = os.environ.get('SUF', '')
GS = {'lym': ['Cd74', 'H2-Aa', 'H2-Ab1', 'H2-Eb1', 'Ighm', 'Igkc', 'Bcl6', 'Ptprc'], 'mel': ['Mlana', 'Dct', 'Tyr', 'Pmel', 'Tyrp1', 'Sox10', 'Mitf', 'Pmela', 'Gpr143', 'Oca2', 'Trpm1', 'Ednrb'], 'endo': ['Pecam1', 'Cdh5', 'Vwf', 'Kdr', 'Eng', 'Lyve1', 'Flt1', 'Emcn', 'Plvap', 'Ramp2'], 'mye': ['Lyz2', 'Csf1r', 'Cd14', 'Cd68', 'Adgre1', 'C1qa', 'C1qb', 'C1qc', 'Itgam', 'Fcgr3', 'S100a8', 'S100a9', 'Itgax', 'Msr1', 'Mrc1'], 'caf': ['Col1a1', 'Col1a2', 'Col3a1', 'Dcn', 'Pdgfra', 'Pdgfrb', 'Fap', 'Postn', 'Lum', 'Col6a1'], 'acaf': ['Acta2', 'Tagln', 'Myh11', 'Cspg4', 'Rgs5', 'Des', 'Notch3', 'Col5a1'], 'ccr1mye': ['Ccr1', 'Ccl2', 'Ccl7', 'Ccl12', 'Ccr2', 'Cd14', 'Lyz2', 'Csf1r', 'C1qa'], 'tcell': ['Cd3d', 'Cd3e', 'Cd3g', 'Cd8a', 'Cd8b1', 'Cd4', 'Trac', 'Trbc2', 'Thy1', 'Ms4a4b'], 'bcell': ['Cd19', 'Ms4a1', 'Cd79a', 'Cd79b', 'Bank1', 'Cr2']}
Q50, Q75, Q90 = (0.5, 0.75, 0.9)

def gene_order(var_names):
    """chromosome-ordered autosome gene list restricted to var_names."""
    keep = {}
    pat = re.compile('gene_name "([^"]+)"')
    with open(GTF) as f:
        for line in f:
            if '\tgene\t' not in line:
                continue
            p = line.split('\t')
            chrom = p[0]
            if not re.fullmatch('(\\d+|chr\\d+)', chrom.replace('chr', '')) or chrom.startswith('chrM'):
                continue
            m = pat.search(p[8] if len(p) > 8 else '')
            if not m:
                continue
            g = m.group(1)
            if g in var_names and g not in keep:
                keep[g] = (int(chrom.replace('chr', '')), int(p[3]))
    return [g for g, _ in sorted(keep.items(), key=lambda kv: kv[1])]

def zscore(A):
    mu = np.asarray(A.mean(0)).ravel()
    sq = np.asarray(A.multiply(A).mean(0)).ravel()
    sd = np.sqrt(np.maximum(sq - mu ** 2, 0))
    sd[sd == 0] = 1
    return sp.csr_matrix((A - mu) / sd)

def setscore(Z, genes, idx):
    g = [x for x in genes if x in idx]
    if not g:
        return (None, [])
    cols = [idx[x] for x in g]
    return (np.asarray(Z[:, cols].mean(1)).ravel(), g)

def smooth_ma(v, w):
    k = np.ones(w) / w
    pad = w // 2
    vp = np.pad(v, pad, mode='edge')
    return np.convolve(vp, k, mode='valid')

def main():
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    a0 = ad.read_h5ad(os.path.join(RCL, f'{SAMPLES[0]}_sc2st_DestVI_destvi_recluster.h5ad'), backed='r')
    order = gene_order(set(a0.var_names))
    del a0
    print(f'tumour module={TUMOR_MOD} suffix={SUF!r} | chromosome-ordered genes for CNV: {len(order)}', flush=True)
    print('L13 ST spot = 100 um -> 1 grid step (15.19 units) = 100 um', flush=True)
    out, leiden_rows = ([], [])
    for s in SAMPLES:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'))
        X = a.X.tocsr().astype(np.float32)
        idx = {g: i for i, g in enumerate(a.var_names)}
        Z = zscore(X)
        d = pd.DataFrame(index=[f'{s}|{b}' for b in a.obs_names])
        for k, genes in GS.items():
            v, used = setscore(Z, genes, idx)
            d[k] = v if v is not None else 0.0
            if s == SAMPLES[0]:
                print(f'  set {k}: {len(used)}/{len(genes)} genes', flush=True)
        d['tbz'] = (d['tcell'] + d['bcell']) / 2
        d['leiden'] = a.obs['leiden_0.8'].astype(str).values
        cols = [idx[g] for g in order]
        Xo = np.asarray(X[:, cols].todense(), dtype=np.float32)
        ref_sel = d[TUMOR_MOD].values <= np.quantile(d[TUMOR_MOD].values, 0.3)
        ref = np.median(Xo[ref_sel], axis=0)
        D = Xo - ref
        sm = np.vstack([smooth_ma(D[i], WIN) for i in range(D.shape[0])])
        cnv = np.abs(sm).mean(1)
        d['cnv_score'] = cnv
        thr = np.quantile(cnv[ref_sel], 0.95)
        d['cnv_thr'] = thr
        d['cnv_pos'] = cnv > thr
        del X, Z, Xo, D, sm
        d['%s_q' % TUMOR_MOD] = d[TUMOR_MOD].rank(pct=True)
        for k in ['endo', 'mye', 'caf', 'acaf', 'ccr1mye', 'tbz']:
            d[f'{k}_q'] = d[k].rank(pct=True)
        d['tumor_raw'] = d['cnv_pos'] | (d['%s_q' % TUMOR_MOD] >= Q90)
        d['tumor_cnv_only'] = d['cnv_pos']
        xy = np.asarray(a.obsm['spatial'])
        nn = NearestNeighbors(n_neighbors=min(KNN + 1, len(xy))).fit(xy)
        nbr = nn.kneighbors(xy, return_distance=False)[:, 1:]
        d['x'], d['y'] = (xy[:, 0], xy[:, 1])
        tv = d['tumor_raw'].values.copy()
        for _ in range(2):
            tv = tv[nbr].sum(1) >= 3
        d['tumor'] = tv
        ev = (d['endo_q'] >= Q90).values
        av = (d['acaf'].values >= 0) & (d['acaf_q'].values >= Q50)
        d['n_tum_nb'] = tv[nbr].sum(1)
        d['n_endo_nb'] = ev[nbr].sum(1)
        d['n_acaf_nb'] = av[nbr].sum(1)
        d['sample'] = s
        zone = np.where(tv & (d['n_tum_nb'].values >= 5), 'Tumor_core', 'Extratumor').astype(object)
        peri = tv & (d['n_tum_nb'].values < 5) | ~tv & (d['n_tum_nb'].values >= 2)
        peri = np.asarray(peri).ravel()
        zone[peri] = 'Peritumor'
        d['region_tumor'] = zone
        niche = np.full(len(d), 'Other', dtype=object)
        m = d['endo_q'].values >= Q90
        niche[m] = 'Vessel'
        m2 = ~m & (d['n_endo_nb'].values >= 2)
        niche[m2] = 'Perivascular'
        m3 = ~m & ~m2 & (d['ccr1mye_q'].values >= Q75) & ((d['acaf_q'].values >= Q50) | (d['n_acaf_nb'].values >= 2))
        niche[m3] = 'CCR1_MIB'
        m4 = ~m & ~m2 & ~m3 & (d['mye_q'].values >= Q75)
        niche[m4] = 'Myeloid_enriched'
        m5 = ~m & ~m2 & ~m3 & ~m4 & (d['caf_q'].values >= Q75)
        niche[m5] = 'CAF_enriched'
        m6 = ~m & ~m2 & ~m3 & ~m4 & ~m5 & (d['tbz_q'].values >= Q75)
        niche[m6] = 'TB_zone'
        d['region_niche'] = niche
        d['mib_perivasc_endo_caf'] = m3 & ((d['n_endo_nb'].values >= 2) | (d['endo_q'].values >= Q50))
        cl = d.groupby('leiden').agg(n=('tumor', 'size'), tumor_frac=('tumor', 'mean'), cnv=('cnv_score', 'mean'), thr=('cnv_thr', 'first'), mel=(TUMOR_MOD, 'mean'), endo=('endo', 'mean'), mye=('mye', 'mean'), caf=('caf', 'mean'), acaf=('acaf', 'mean'), ccr1mye=('ccr1mye', 'mean'), tbz=('tbz', 'mean'), n_tum_nb=('n_tum_nb', 'mean'), n_endo_nb=('n_endo_nb', 'mean'))
        cl = cl.rename(columns={'mel': TUMOR_MOD}).reset_index()
        cl['sample'] = s
        for k in [TUMOR_MOD, 'endo', 'mye', 'caf', 'acaf', 'ccr1mye', 'tbz']:
            cl[f'{k}_q'] = cl[k].rank(pct=True)
        rl = []
        for _, r in cl.iterrows():
            if r['cnv'] > r['thr'] or r['%s_q' % TUMOR_MOD] >= Q75:
                lab = 'Tumor_core' if r['tumor_frac'] >= 0.5 else 'Peritumor'
            elif r['endo_q'] >= Q90:
                lab = 'Vessel'
            elif r['n_endo_nb'] >= 2:
                lab = 'Perivascular'
            elif r['ccr1mye_q'] >= Q75 and r['acaf_q'] >= Q50:
                lab = 'CCR1_MIB'
            elif r['mye_q'] >= Q75:
                lab = 'Myeloid_enriched'
            elif r['caf_q'] >= Q75:
                lab = 'CAF_enriched'
            elif r['tbz_q'] >= Q75:
                lab = 'TB_zone'
            else:
                lab = 'Extratumor'
            rl.append(lab)
        cl['region_leiden_clus'] = rl
        leiden_rows.append(cl)
        d['region_leiden_clus'] = d['leiden'].map(dict(zip(cl['leiden'], cl['region_leiden_clus'])))
        d['region_leiden'] = np.where(d['region_leiden_clus'].isin(['Tumor_core', 'Peritumor']), d['region_leiden_clus'], np.where(d['region_tumor'] == 'Extratumor', d['region_leiden_clus'], 'Tumor_' + d['region_leiden_clus']))
        out.append(d)
        print(s, 'done | Tumor_core', int((d.region_tumor == 'Tumor_core').sum()), 'Peritumor', int((d.region_tumor == 'Peritumor').sum()), flush=True)
        del a
    R = pd.concat(out)
    R.index.name = 'spot'
    R = R.loc[M.index]
    for c in ['car_pos', 'car_tier', 'car_p', 'umi_total', 'umi_car', 'car_cp10k', 'group']:
        R[c] = M[c].values
    R.to_csv(os.path.join(RES, f'spot_regions{SUF}.csv'))
    pd.concat(leiden_rows).to_csv(os.path.join(RES, f'leiden_region_map{SUF}.csv'), index=False)
    print('\n=== region_tumor x group ===')
    print(pd.crosstab(R['region_tumor'], R['group']).to_string())
    print('\n=== region_niche x group ===')
    print(pd.crosstab(R['region_niche'], R['group']).to_string())
    print('\n=== spot vs leiden label agreement ===')
    print(pd.crosstab(R['region_leiden'], R['region_niche']).to_string())
if __name__ == '__main__':
    main()
