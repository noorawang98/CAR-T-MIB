"""Step09: CAR+ spot proportions under several denominators.

Definitions
  pct_CARpos        = n(CAR+) / n(spots)                          x100
  exp_bg_pct        = sum_i P(Binom(n_i, q) >= 1) / n(spots)      x100   (Vehicle-calibrated
                      background expectation, depth aware, q = per-UMI CAR capture rate)
  excess_pct        = pct_CARpos - exp_bg_pct                     (background-corrected)
  fold_vs_bg        = pct_CARpos / exp_bg_pct
  car_umi_per_10k   = 1e4 * sum(CAR umi) / sum(total umi)          (depth-normalised load)

Denominators reported: whole slice, region_tumor, region_niche, region_leiden,
group (R / NR / Vehicle), scissor label, GR/PR label; plus the R-vs-NR Fisher test.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from common import load_base, scissor_labels, bh, RES
Q = 3.7379885345425275e-07
RC = ['region_tumor', 'region_niche', 'region_leiden']

def prop_block(df, key_cols, name):
    rows = []
    for keys, g in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(g)
        k = int(g['car_pos'].sum())
        exp_bg = float((1 - (1 - Q) ** g['umi_total'].clip(lower=1).values).sum())
        rows.append(dict(**dict(zip(key_cols, keys)), n_spots=n, n_CARpos=k, pct_CARpos=100 * k / n, exp_bg_pct=100 * exp_bg / n, excess_pct=100 * (k - exp_bg) / n, fold_vs_bg=k / exp_bg if exp_bg > 0 else np.nan, car_umi=g['umi_car'].sum(), car_umi_per_10k=10000.0 * g['umi_car'].sum() / g['umi_total'].sum(), median_umi=float(g['umi_total'].median())))
    d = pd.DataFrame(rows)
    d.insert(0, 'stratify', name)
    return d

def main():
    D = load_base(with_props=False)
    # Original common.load_base does not include the GSVA label sidecar needed
    # by the GRPR_sample split below. Load that same legacy column explicitly.
    G = pd.read_csv(os.path.join(RES, 'gsva_tcell_spot.csv'), index_col=0)
    D['GRPR_sample'] = G['GRPR_sample'].reindex(D.index)
    S = scissor_labels()
    D = D.join(S[['scissor_sum', 'scissor_lab']], how='left')
    T = D[D['group'].isin(['R', 'NR'])].copy()
    blocks = [prop_block(D, ['group', 'sample'], 'sample')]
    for rc in RC:
        blocks.append(prop_block(T, ['group', rc], f'group x {rc}'))
        blocks.append(prop_block(T, ['sample', rc], f'sample x {rc}'))
    blocks.append(prop_block(T, ['scissor_lab'], 'scissor_label'))
    for rc in RC:
        blocks.append(prop_block(T, ['group', 'scissor_lab', rc], f'scissor x group x {rc}'))
    P = pd.concat(blocks, ignore_index=True)
    P.to_csv(os.path.join(RES, 'task_carpos_proportions.csv'), index=False)
    comp = []
    for rc in RC:
        for keys, g in T[T['car_pos']].groupby(['group', rc]):
            comp.append(dict(region_col=rc, group=keys[0], region_name=keys[1], n_CARpos=len(g), pct_of_CARpos=100 * len(g) / (T['car_pos'] & (T['group'] == keys[0])).sum()))
        for bk, g in T[T['car_pos']].groupby(['sample', rc]):
            comp.append(dict(region_col=rc, group=bk[0], region_name=bk[1], n_CARpos=len(g), pct_of_CARpos=100 * len(g) / (T['car_pos'] & (T['sample'] == bk[0])).sum()))
    pd.DataFrame(comp).to_csv(os.path.join(RES, 'task_carpos_composition.csv'), index=False)
    gp = T[T['car_pos']].groupby(['group', 'sample', 'GRPR_sample']).size().unstack(fill_value=0)
    gp['n_CARpos'] = gp.sum(1)
    gp['pct_GR'] = 100 * gp.get('GR', 0) / gp['n_CARpos']
    gp['pct_PR'] = 100 * gp.get('PR', 0) / gp['n_CARpos']
    gp.to_csv(os.path.join(RES, 'task_carpos_GRPR_split.csv'))
    rows = []
    for rc in RC:
        for reg, g in T.groupby(rc):
            a = int(g.loc[g['group'] == 'R', 'car_pos'].sum())
            na = int((g['group'] == 'R').sum())
            b = int(g.loc[g['group'] == 'NR', 'car_pos'].sum())
            nb = int((g['group'] == 'NR').sum())
            if min(na, nb) == 0:
                continue
            orr, p = fisher_exact([[a, na - a], [b, nb - b]])
            rows.append(dict(region=rc, region_name=reg, pct_R=100 * a / na, pct_NR=100 * b / nb, n_R=na, n_NR=nb, CARpos_R=a, CARpos_NR=b, odds_ratio=orr, fisher_p=p))
    F = pd.DataFrame(rows)
    F['fisher_q'] = bh(F['fisher_p'])
    F.to_csv(os.path.join(RES, 'task_carpos_prop_R_vs_NR.csv'), index=False)
    for rc in RC:
        piv = (T.pivot_table(index=['sample', 'group'], columns=rc, values='car_pos', aggfunc='mean') * 100).round(3)
        piv.to_csv(os.path.join(RES, f'task_carpos_pct_by_sample_{rc}.csv'))
    pd.set_option('display.width', 220)
    print('=== CAR+ spot proportion by slice (background expectation = Vehicle-calibrated) ===')
    print(P[P.stratify == 'sample'][['group', 'sample', 'n_spots', 'n_CARpos', 'pct_CARpos', 'exp_bg_pct', 'excess_pct', 'fold_vs_bg', 'car_umi_per_10k']].round(3).to_string(index=False))
    for rc in RC:
        print(f'\n=== CAR+ spot proportion, group x {rc} ===')
        print(P[P.stratify == f'group x {rc}'][['group', rc, 'n_spots', 'n_CARpos', 'pct_CARpos', 'exp_bg_pct', 'excess_pct', 'fold_vs_bg']].round(3).to_string(index=False))
    print('\n=== R vs NR Fisher on CAR+ proportion ===')
    print(F.round(5).to_string(index=False))
    print('\n=== by scissor label (treated slices) ===')
    print(P[P.stratify == 'scissor_label'][['scissor_lab', 'n_spots', 'n_CARpos', 'pct_CARpos', 'exp_bg_pct', 'excess_pct', 'fold_vs_bg']].round(3).to_string(index=False))
    print('\n=== GR / PR split inside CAR+ spots ===')
    print(gp.round(3).to_string())
    print("\n=== composition of CAR+ spots (%% of that group's CAR+ spots) ===")
    CC = pd.read_csv(os.path.join(RES, 'task_carpos_composition.csv'))
    for rc in RC:
        print(f'\n-- {rc} --')
        print(CC[(CC.region_col == rc) & CC.group.isin(['R', 'NR'])][['group', 'region_name', 'n_CARpos', 'pct_of_CARpos']].round(3).to_string(index=False))
    print('\nwrote task_carpos_* tables')
if __name__ == '__main__':
    main()
