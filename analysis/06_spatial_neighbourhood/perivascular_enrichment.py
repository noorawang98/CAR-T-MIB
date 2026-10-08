"""Section-wise simultaneous TAM/CAF perivascular enrichment; no pooling or plotting.
Extracted from step12_perivascular_summary_fig.py without changing collect().
"""
import os
import numpy as np
import pandas as pd
from region_define_stats import TAB
SECTIONS = ["NR1", "NR2", "NR3", "R1", "R2", "R3"]
REGIONS = ["Tumor_core", "Peritumor", "Extratumor", "Perivascular"]

def collect():
    """region-wise dp_nb / both2 per section and group, plus the perivascular fold"""
    A, B = ([], [])
    for s in SECTIONS:
        T = pd.read_csv(os.path.join(TAB, f'TableR14_GRPR_TAM_CAF_simultaneous_{s}.csv'))
        for metric, store in [('dp_nb', A), ('both2', B)]:
            D = T[T['metric'] == metric].set_index('stratum')
            for g, mcol, ncol in [('GR', 'mean_GR', 'n_GR'), ('PR', 'mean_PR', 'n_PR')]:
                vals = {}
                for r in REGIONS:
                    ok = r in D.index and D.loc[r, 'n_GR'] >= 3 and (D.loc[r, 'n_PR'] >= 3)
                    vals[r] = float(D.loc[r, mcol]) if ok else np.nan
                oth = [v for r, v in vals.items() if r != 'Perivascular' and np.isfinite(v)]
                store.append(dict(sample=s, group=g, metric=metric, **vals, n_other=len(oth), other_mean=float(np.mean(oth)) if oth else np.nan, fold=vals['Perivascular'] / np.mean(oth) if oth and np.isfinite(vals['Perivascular']) and (np.mean(oth) > 0) else np.nan))
    return (pd.DataFrame(A), pd.DataFrame(B))

if __name__ == '__main__':
    os.makedirs(TAB, exist_ok=True)
    A, B = collect()
    A.to_csv(os.path.join(TAB, 'TableR19_perivascular_simultaneous_by_section.csv'), index=False)
    B.to_csv(os.path.join(TAB, 'TableR19_both2_perivascular_by_section.csv'), index=False)
