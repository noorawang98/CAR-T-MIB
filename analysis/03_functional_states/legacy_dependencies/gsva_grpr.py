"""Step04: T-cell functional GSVA scores and GR / PR labels for CAR+ spots (Task 2 input).

GSVA matrix: /home/Data/ST_20241110/spatial_input/h5ad/ST_ALL_GSVA_M3_TCell.csv
(627 MSigDB T-cell gene sets x 23867 spots of the 7 slices, mouse genes mapped to human
symbols; produced by GSVA::gsva on the reclustered slices).

Label definition (as specified):
  effector score  = mean z-score of MSigDB gene sets annotated "EFFECTOR" & Selected
  exhaustion score= mean z-score of MSigDB gene sets annotated "EXHAUSTED" & Selected
  net             = effector - exhaustion
  GR (good response) = high effector / low exhaustion  -> net above the cut
  PR (poor response) = high exhaustion / low effector  -> net below the cut
Primary cut = per-sample median of net over CAR+ spots (sample-controlled);
a pooled-median cut is also produced as sensitivity.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
GSVA = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/ST_ALL_GSVA_M3_TCell.csv'
ANN = PROJECT_ROOT + '/ST_20241110/spatial_input/h5ad/MsigDB_T_CELL_all.csv'
ANN = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/MsigDB_T_CELL_all.csv'

def main():
    ann = pd.read_csv(ANN, index_col=0)
    cols = list(pd.read_csv(GSVA, nrows=0).columns)
    eff_sets = [c for c in ann.index[ann['T cell state'].eq('EFFECTOR') & ann['IN'].eq('Selected')] if c in cols]
    exh_sets = [c for c in ann.index[ann['T cell state'].eq('EXHAUSTED') & ann['IN'].eq('Selected')] if c in cols]
    print('effector sets', len(eff_sets), '| exhaustion sets', len(exh_sets))
    use = ['slice_id', 'spot_id'] + eff_sets + exh_sets
    G = pd.read_csv(GSVA, usecols=use)
    G['spot'] = G['spot_id'].astype(str) + '|' + G['slice_id'].astype(str)
    G['sample'] = G['slice_id'].astype(str)
    G['barcode'] = G['spot_id'].astype(str).str.rsplit('-', n=1).str[0]
    G['key'] = G['sample'] + '|' + G['barcode']
    print('GSVA rows', G.shape)
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    E = G[eff_sets].apply(lambda x: (x - x.mean()) / (x.std() or 1)).mean(1)
    H = G[exh_sets].apply(lambda x: (x - x.mean()) / (x.std() or 1)).mean(1)
    out = pd.DataFrame({'key': G['key'].values, 'gsva_effector': E.values, 'gsva_exhaustion': H.values})
    out = out.drop_duplicates('key').set_index('key')
    out = out.reindex(M.index)
    print('mapped spots', out['gsva_effector'].notna().sum(), '/', len(M))
    for c in ['gsva_effector', 'gsva_exhaustion']:
        out[c + '_z'] = out.groupby(M['sample'].values)[c].transform(lambda v: (v - v.mean()) / (v.std() or 1))
    out['net'] = out['gsva_effector_z'] - out['gsva_exhaustion_z']
    out['sample'] = M['sample'].values
    out['car_pos'] = M['car_pos'].values
    car = out['car_pos'].values
    cut_s = np.full(len(out), np.nan)
    for s, g in out[car].groupby('sample'):
        cut_s[out.index.isin(g.index)] = np.median(g['net'])
    cut_all = np.median(out.loc[car, 'net'])
    out['cut_sample'] = cut_s
    out['cut_pooled'] = cut_all
    out['GRPR_sample'] = np.where(out['net'] >= cut_s, 'GR', 'PR')
    out['GRPR_pooled'] = np.where(out['net'] >= cut_all, 'GR', 'PR')
    out.loc[~car, ['GRPR_sample', 'GRPR_pooled']] = 'n/a'
    out.to_csv(os.path.join(RES, 'gsva_tcell_spot.csv'))
    print('\nper-sample net median (CAR+ spots):')
    print(out[car].groupby('sample')['net'].median().round(3).to_string())
    print('\nGR/PR among CAR+ spots (per-sample cut):')
    print(pd.crosstab(out.loc[car, 'sample'], out.loc[car, 'GRPR_sample']).to_string())
    print('\nGR/PR among CAR+ spots (pooled cut):')
    print(pd.crosstab(out.loc[car, 'sample'], out.loc[car, 'GRPR_pooled']).to_string())
if __name__ == '__main__':
    main()
