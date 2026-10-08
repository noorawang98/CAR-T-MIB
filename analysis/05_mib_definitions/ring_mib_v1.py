"""Step62: MIB / perivascular share in rings 1-3 of the new predicted labels, using BOTH
niche definitions (v1 region_niche + mib_perivasc_endo_caf flag, and v2 CCR1_MIB)."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd, anndata as ad
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf
from common import RES, RCL, TREATED, bh
OUT = PROJECT_ROOT + '/mouse/cart_region/grpr_pred_v3'
RINGS = {'ring1': 6, 'ring2': 18, 'ring3': 36}
BRICK, BLUE = ('#B22222', '#2C7FB8')

def main():
    A = pd.read_csv(os.path.join(OUT, 'GRPR_pred_labels.csv'), index_col=0)
    R = pd.read_csv(os.path.join(RES, 'spot_regions_lym.csv'), index_col=0)
    N = pd.read_csv(PROJECT_ROOT + '/mouse/cart_region/lym_v2/region_niche_v2.csv', index_col=0)
    frames = []
    for s in TREATED:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'), backed='r')
        idx = np.array([f'{s}|{b}' for b in a.obs_names])
        xy = np.asarray(a.obsm['spatial'])
        v1peri = R['region_niche'].reindex(idx).isin(['Perivascular']).values
        v1mib = (R['region_niche'].reindex(idx) == 'CCR1_MIB').values
        v1flag = R['mib_perivasc_endo_caf'].reindex(idx).astype(bool).values
        v1perimib = v1peri | v1mib
        v2mib = (N['niche_v2'].reindex(idx) == 'CCR1_MIB').values
        d = pd.DataFrame({'sample': s, 'v1_peri': v1peri, 'v1_mib': v1mib, 'v1_flag': v1flag, 'v1_peri_or_mib': v1perimib, 'v2_mib': v2mib}, index=idx)
        nn = NearestNeighbors(n_neighbors=max(RINGS.values()) + 1).fit(xy)
        ind = nn.kneighbors(xy, return_distance=False)[:, 1:]
        for rname, k in RINGS.items():
            nb = ind[:, :k]
            for c in ['v1_peri', 'v1_mib', 'v1_flag', 'v1_peri_or_mib', 'v2_mib']:
                d[f'{rname}_{c}'] = d[c].values[nb].mean(1)
        frames.append(d)
    D = pd.concat(frames).join(A[['GRPR_pred', 'p_GR_HVG']], how='inner')
    D['y'] = (D.GRPR_pred == 'GR').astype(int)
    D.to_csv(os.path.join(OUT, 'ring_mib_shares.csv'))
    feats = [c for c in D.columns if c.startswith(tuple(RINGS))]
    rows = []
    for f in feats:
        d = D[[f, 'y', 'sample']].dropna()
        try:
            m = smf.mixedlm('v ~ y', d.assign(v=d[f]), groups=d['sample']).fit(reml=True, method='lbfgs')
            e, p = (float(m.params['y']), float(m.pvalues['y']))
        except Exception:
            e, p = (np.nan, np.nan)
        rows.append(dict(feature=f, mean_GR=d.loc[d.y == 1, f].mean(), mean_PR=d.loc[d.y == 0, f].mean(), diff=e, p=p))
    T = pd.DataFrame(rows)
    T['q'] = bh(T.p.fillna(1))
    T.to_csv(os.path.join(OUT, 'ring_mib_shares_stats.csv'), index=False)
    pd.set_option('display.width', 200)
    print('=== MIB / perivascular share in rings (spot level, mixed model + BH) ===')
    print(T.round(4).to_string(index=False))
    tis = []
    for f in feats:
        for s, g in D.groupby('sample'):
            if g.y.nunique() < 2:
                continue
            tis.append(dict(feature=f, sample=s, GR=g.loc[g.y == 1, f].mean(), PR=g.loc[g.y == 0, f].mean(), diff=g.loc[g.y == 1, f].mean() - g.loc[g.y == 0, f].mean(), group='NR' if s.startswith('NR') else 'R'))
    TT = pd.DataFrame(tis)
    TT.to_csv(os.path.join(OUT, 'ring_mib_shares_tissue.csv'), index=False)
    print('\n=== tissue level (per slice, no averaging): ring1 ===')
    print(TT[TT.feature == 'ring1_v2_mib'].round(4).to_string(index=False))
    print('\nring1_v1_peri_or_mib:')
    print(TT[TT.feature == 'ring1_v1_peri_or_mib'].round(4).to_string(index=False))
    print('\n每切片 diff 的符号统计（正/负）:')
    for f in ['ring1_v2_mib', 'ring1_v1_mib', 'ring1_v1_flag', 'ring1_v1_peri_or_mib', 'ring1_v1_peri']:
        v = TT[TT.feature == f]['diff'].values
        print(f'  {f:22s} {np.sum(v > 0)}/{len(v)} 正  ({np.round(v, 3)})')
    key = ['ring1_v2_mib', 'ring2_v2_mib', 'ring3_v2_mib', 'ring1_v1_mib', 'ring1_v1_flag', 'ring1_v1_peri_or_mib', 'ring1_v1_peri']
    print('\nwrote ring_mib_shares*.csv, figs/ring_mib_tissue_scatter.*')
if __name__ == '__main__':
    main()
