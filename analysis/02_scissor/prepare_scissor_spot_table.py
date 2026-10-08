"""Prepare the integrated spot_full_table.csv from original master/region tables,
DestVI proportions and Scissor coefficients. Historical GR/PR comparison
statistics and rendering are excluded from this entry point."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from common import load_base, scissor_labels, bh, RES, TREATED, CT_ORDER

def main():
    D = load_base(with_props=True)
    S = scissor_labels()
    D = pd.concat([D, S.reindex(D.index)], axis=1)
    ct = [c for c in pd.read_csv(LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/destvi_res/NR1_sc2st_DestVI.csv', index_col=0, nrows=0).columns if c in D.columns]
    D['destvi_max'] = D[ct].max(1)
    D['destvi_entropy'] = -(D[ct].clip(lower=1e-09) * np.log(D[ct].clip(lower=1e-09))).sum(1)
    D.to_csv(os.path.join(RES, 'spot_full_table.csv'))
    print('wrote spot_full_table.csv')
if __name__ == '__main__':
    main()
