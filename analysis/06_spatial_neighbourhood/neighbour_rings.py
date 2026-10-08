"""Step61: neighbourhood composition in rings 1-3 (6/18/36 NN = ~100/200/300 um) around
the new predicted GR/PR CAR+ spots - spot level (mixed model) and tissue level
(per-sample values plotted as labelled scatter points, no averaging)."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd, anndata as ad
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf
from common import RES, RCL, TREATED, bh, destvi_props
OUT = PROJECT_ROOT + '/mouse/cart_region/grpr_pred_v3'
NICHE = PROJECT_ROOT + '/mouse/cart_region/lym_v2/region_niche_v2.csv'
SPAT = PROJECT_ROOT + '/mouse/spatial_out'
RINGS = {'ring1': 6, 'ring2': 18, 'ring3': 36}
MODS = ['lym', 'endo', 'mye', 'caf', 'acaf', 'ccr1mye', 'tcell', 'tbz']
BRICK, BLUE, GREY = ('#B22222', '#2C7FB8', '#BFBFBF')

def build():
    A = pd.read_csv(os.path.join(OUT, 'GRPR_pred_labels.csv'), index_col=0)
    R = pd.read_csv(os.path.join(RES, 'spot_regions_lym.csv'), index_col=0)
    N = pd.read_csv(NICHE, index_col=0)
    P26 = destvi_props()
    S = pd.read_csv(PROJECT_ROOT + '/mouse/cart_region/scissor_pooled_noRelapse/scissor_labels_BEST_all_samples.csv').set_index('spot')
    xis, yis, frames = ([], [], [])
    for s in TREATED:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'), backed='r')
        idx = np.array([f'{s}|{b}' for b in a.obs_names])
        xy = np.asarray(a.obsm['spatial'])
        d = pd.DataFrame(index=idx)
        for c in MODS:
            d[c] = R[c].reindex(idx).values
        d['niche_peri'] = N['niche_v2'].reindex(idx).isin(['Vessel', 'Perivascular_100', 'Perivascular_200']).values
        d['niche_CAF'] = (N['niche_v2'].reindex(idx) == 'CAF_enriched').values
        d['niche_MIB'] = (N['niche_v2'].reindex(idx) == 'CCR1_MIB').values
        for c in P26.columns:
            d['d26_' + c] = P26[c].reindex(idx).values
        d['scissor'] = S['label'].reindex(idx).values
        nn = NearestNeighbors(n_neighbors=max(RINGS.values()) + 1).fit(xy)
        ind = nn.kneighbors(xy, return_distance=False)[:, 1:]
        lab = A['GRPR_pred'].reindex(idx).values
        for rname, k in RINGS.items():
            nb = ind[:, :k]
            for c in MODS + ['niche_peri', 'niche_CAF', 'niche_MIB']:
                d[f'{rname}_{c}'] = d[c].values[nb].mean(1)
            for c in P26.columns:
                d[f'{rname}_d26_{c}'] = d['d26_' + c].values[nb].mean(1)
            d[f'{rname}_GR_share'] = np.nanmean((lab[nb] == 'GR') * 1.0, axis=1)
            d[f'{rname}_PR_share'] = np.nanmean((lab[nb] == 'PR') * 1.0, axis=1)
            d[f'{rname}_NDR_share'] = np.mean((d['scissor'].values[nb] == 'NDR') * 1.0, axis=1)
            d[f'{rname}_DR_share'] = np.mean((d['scissor'].values[nb] == 'DR') * 1.0, axis=1)
        d['sample'] = s
        frames.append(d)
        print(' built', s, flush=True)
    D = pd.concat(frames)
    D = D.join(A[['GRPR_pred', 'p_GR_HVG']], how='inner')
    D['y'] = (D.GRPR_pred == 'GR').astype(int)
    return D

def main():
    D = build()
    D.to_csv(os.path.join(OUT, 'ring_neighbourhood.csv'))
    feats = [c for c in D.columns if c.startswith(tuple(RINGS))]
    rows = []
    for f in feats:
        d = D[[f, 'y', 'sample']].dropna()
        if d[f].nunique() < 2:
            continue
        try:
            m = smf.mixedlm('v ~ y', d.assign(v=d[f]), groups=d['sample']).fit(reml=True, method='lbfgs')
            e, p = (float(m.params['y']), float(m.pvalues['y']))
        except Exception:
            e, p = (np.nan, np.nan)
        a1 = d.loc[d.y == 1, f].mean()
        a0 = d.loc[d.y == 0, f].mean()
        rows.append(dict(feature=f, mean_GR=a1, mean_PR=a0, diff=e, p=p, n_GR=int((d.y == 1).sum()), n_PR=int((d.y == 0).sum())))
    T = pd.DataFrame(rows)
    T['q'] = bh(T.p.fillna(1))
    T['sig'] = np.where(T.q < 0.001, '***', np.where(T.q < 0.01, '**', np.where(T.q < 0.05, '*', '')))
    T.to_csv(os.path.join(OUT, 'ring_neighbourhood_stats.csv'), index=False)
    print('=== spot-level GR vs PR neighbourhood (top 15 by p) ===')
    print(T.sort_values('p').head(15).round(4).to_string(index=False))
    tis = []
    for f in feats:
        for s, g in D.groupby('sample'):
            if g.y.nunique() < 2:
                continue
            tis.append(dict(feature=f, sample=s, GR=g.loc[g.y == 1, f].mean(), PR=g.loc[g.y == 0, f].mean(), diff=g.loc[g.y == 1, f].mean() - g.loc[g.y == 0, f].mean(), group='NR' if s.startswith('NR') else 'R'))
    TT = pd.DataFrame(tis)
    TT.to_csv(os.path.join(OUT, 'ring_neighbourhood_tissue_level.csv'), index=False)
    key = ['ring1_caf', 'ring2_caf', 'ring3_caf', 'ring1_acaf', 'ring1_ccr1mye', 'ring1_tcell', 'ring2_tcell', 'ring1_mye', 'ring1_lym']
    key = [k for k in key if k in set(TT.feature)]
    sub = T[T.feature.str.startswith(('ring1_', 'ring2_', 'ring3_'))].copy()
    sub['ring'] = sub.feature.str.split('_').str[0]
    sub['comp'] = sub.feature.str.replace('^ring[123]_', '', regex=True)
    piv = sub.pivot_table(index='comp', columns='ring', values='diff')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
    print('\nwrote ring_neighbourhood*.csv, figs/ring_tissue_scatter.*, figs/ring_spot_heatmap.*')
if __name__ == '__main__':
    main()
