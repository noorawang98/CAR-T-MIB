"""Step48: freeze the final GR/PR label set (def5) - self-contained.

def5 = quartiles of net = EFF_z - EXH_z cut INSIDE every sample among the CAR+ treated
spots. EFF_z / EXH_z are the per-sample z-scores of the effector / exhaustion marker
blocks (see step18 definition of the blocks); net is therefore already batch-free, so no
cross-sample correction is needed (see results/README_GRPR_definition.md).

Output: results/GRPR_labels_final.csv  (GRPR + scores + block scores), the single label
source used by every downstream step (common.grpr_labels()).
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import numpy as np, pandas as pd
from common import RES, TREATED
BLOCKS = ['EFF_core', 'EFF_cyt', 'EFF_mem', 'EFF_act', 'EXH']

def build():
    S = pd.read_csv(os.path.join(RES, 'marker_tcell_scores.csv'), index_col=0)
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    S['car_pos'] = M['car_pos'].reindex(S.index).fillna(False).values
    S['sample'] = [i.split('|')[0] for i in S.index]
    A = S[S['car_pos'] & S['sample'].isin(TREATED)].copy()
    A['net'] = A['EFF_z'] - A['EXH_z']
    A['GRPR'] = 'Other'
    for s, g in A.groupby('sample'):
        A.loc[g.index[g['net'] >= g['net'].quantile(0.75)], 'GRPR'] = 'GR'
        A.loc[g.index[g['net'] <= g['net'].quantile(0.25)], 'GRPR'] = 'PR'
    A['GRPR_def'] = 'def5_persample_net_quartile'
    cols = ['sample', 'car_pos', 'GRPR', 'GRPR_def', 'EFF_z', 'EXH_z', 'net'] + BLOCKS + ['EFF']
    return A[cols]

def main():
    A = build()
    A.to_csv(os.path.join(RES, 'GRPR_labels_final.csv'))
    print('GRPR:', A.GRPR.value_counts().to_dict(), '| spots', len(A))
    print(pd.crosstab(A['sample'], A.GRPR).to_string())
    print('wrote results/GRPR_labels_final.csv')
if __name__ == '__main__':
    main()
