"""Step02: CAR+ spot calling with Vehicle-calibrated, UMI-depth-aware cutoff.

Background per-UMI CAR capture rate q is estimated from the Vehicle (untreated control)
spots:  q = sum(CAR umi) / sum(total umi) over Vehicle.
For every spot the null is Binomial(n = umi_total, p = q); a spot is CAR+ when the
one-sided upper-tail p-value < ALPHA.  Because the null scales with depth, spots from
low-UMI samples (NR1/NR3) need the same evidence as high-UMI samples (R2/R3), which is
the "UMI abundance" correction requested.  FPR is read out directly on Vehicle.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
from scipy.stats import binom
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
ALPHA_MAIN = 0.05
ALPHAS = [0.05, 0.01, 0.001]

def main():
    M = pd.read_csv(os.path.join(RES, 'spot_master.csv'), index_col=0)
    V = M[M.group == 'Vehicle']
    q = V.umi_car.sum() / V.umi_total.sum()
    n = M.umi_total.clip(lower=1).values
    k = M.umi_car.values
    M['car_p'] = binom.sf(k - 1, n, q)
    M.loc[k <= 0, 'car_p'] = 1.0
    for a in ALPHAS:
        M[f'car_pos_a{a}'] = (M.car_p < a) & (M.umi_car > 0)
    M['car_pos'] = M[f'car_pos_a{ALPHA_MAIN}']
    M['car_tier'] = np.select([M.umi_car == 0, M.car_p < 0.001, M.car_p < 0.01, M.car_p < 0.05], ['neg', 'strict', 'moderate', 'permissive'], default='ambient')
    rows = []
    for s, d in M.groupby('sample'):
        dec = np.round(np.quantile(d.umi_total, [0.1, 0.5, 0.9])).astype(int)
        for a in ALPHAS:
            kmin = []
            for nn in dec:
                kk = 1
                while binom.sf(kk - 1, max(nn, 1), q) > a and kk < 20:
                    kk += 1
                kmin.append(kk)
            rows.append(dict(sample=s, alpha=a, umi_deciles=list(dec), k_min=kmin))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'car_cutoff_by_depth.csv'), index=False)
    M.to_csv(os.path.join(RES, 'spot_master_car.csv'))
    print(f'q(per-UMI CAR rate from Vehicle) = {q:.4g}')
    print('CAR+ counts (alpha=%.2f):' % ALPHA_MAIN)
    print(M[M.car_pos].groupby(['group', 'sample']).size().to_string())
    print('\nVehicle FPR readout (background-positive rate):')
    for a in ALPHAS:
        vp = M[M.group == 'Vehicle'][f'car_pos_a{a}'].mean()
        print(f"  alpha={a}: V+={int(M[M.group == 'Vehicle'][f'car_pos_a{a}'].sum())}/{len(V)} = {vp * 100:.2f}%")
    print('\nCAR tier x sample:')
    print(pd.crosstab(M['sample'], M['car_tier']).to_string())
    print('\nmedian car_cp10k by tier:')
    print(M.groupby('car_tier')['car_cp10k'].median().round(4).to_string())
if __name__ == '__main__':
    main()
