"""Step01: build per-spot master table (7 samples).

Columns: sample, barcode, group(R/NR/Vehicle), x, y, leiden_0.8,
         umi_total, umi_mCherry-CAR (raw), umi_mCherry, umi_EGFP,
         car_cp10k, CAR-raw X value from h5ad.
Also exports the normalized expression matrix (gene x spot, log1p) as one npz per sample
and a common gene list for downstream marker/CNV scoring.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
warnings.filterwarnings('ignore')
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
RCL = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/recluster_obj'
SAMPLES = ['Vehicle', 'NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
GROUP = {s: 'Vehicle' if s == 'Vehicle' else 'NR' if s.startswith('NR') else 'R' for s in SAMPLES}
TREATED = [s for s in SAMPLES if s != 'Vehicle']

def main():
    frames = []
    for s in SAMPLES:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'))
        raw = pd.read_csv(os.path.join(RES, f'raw_umi_{s}.csv'))
        raw = raw.set_index('barcode').reindex(a.obs_names)
        j = list(a.var_names).index('mCherry-CAR')
        carx = np.asarray(a.X[:, j].todense()).ravel()
        sp_xy = np.asarray(a.obsm['spatial'])
        d = pd.DataFrame({'sample': s, 'barcode': a.obs_names, 'group': GROUP[s], 'x': sp_xy[:, 0], 'y': sp_xy[:, 1], 'leiden': a.obs['leiden_0.8'].astype(str).values, 'umi_total': raw['umi_total'].fillna(0).astype(np.int64).values, 'umi_car': raw['umi_mCherry-CAR'].fillna(0).astype(np.int64).values, 'umi_mcherry': raw['umi_mCherry'].fillna(0).astype(np.int64).values, 'umi_egfp': raw['umi_EGFP'].fillna(0).astype(np.int64).values, 'car_x_h5ad': carx}, index=[f'{s}|{b}' for b in a.obs_names])
        d['car_cp10k'] = 10000.0 * d['umi_car'] / d['umi_total'].clip(lower=1)
        frames.append(d)
        print(s, 'n', len(d), 'medianUMI', int(d.umi_total.median()), 'car>0', int((d.umi_car > 0).sum()), flush=True)
        del a
    M = pd.concat(frames)
    M.index.name = 'spot'
    M.to_csv(os.path.join(RES, 'spot_master.csv'))
    # Export the original spot order required by scissor_coefs.R.
    M[['sample', 'barcode']].to_csv(os.path.join(RES, 'spot_order.csv'), index=False)
    print('wrote spot_master.csv', M.shape)
    print(M.groupby('sample')[['umi_total', 'umi_car', 'car_cp10k']].agg(['median', 'sum']).to_string())
if __name__ == '__main__':
    main()
