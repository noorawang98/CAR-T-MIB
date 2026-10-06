"""Step93: AUC inside the CCR1_MIB region (niche_v1) vs outside, for GR/PR and NDR/DR."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import RES
WD = PROJECT_ROOT + '/mouse/cart_region'
OUT = os.path.join(WD, 'figs_final')
PRED = os.path.join(WD, 'grpr_pred_v3')
MEMX = os.path.join(WD, 'grpr_mem_exh')
SCD = os.path.join(WD, 'scissor_pooled_noRelapse')
BRED, BBLUE, BGREY = ('#B22222', '#2C7FB8', '#BFBFBF')

def perm_p(score, y, g, n=1000, seed=0):
    obs = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    c = 0
    for _ in range(n):
        yp = y.copy()
        for s in np.unique(g):
            m = g == s
            yp[m] = rng.permutation(yp[m])
        if roc_auc_score(yp, score) >= obs:
            c += 1
    return (obs, (c + 1) / (n + 1))

def main():
    R = pd.read_csv(os.path.join(RES, 'spot_regions_lym.csv'), index_col=0)
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    G = pd.read_csv(os.path.join(PRED, 'clinical_gsva', 'gsva_clinical_scores.csv'), index_col=0)
    G = G.groupby([i.split('|')[0] for i in G.index]).transform(lambda v: (v - v.mean()) / (v.std() or 1))
    ML = pd.read_csv(os.path.join(PRED, 'GRPR_pred_labels.csv'), index_col=0)
    NE = pd.read_csv(os.path.join(MEMX, 'GRPR_mem_exh_final_labels.csv'), index_col=0)
    M5 = pd.read_csv(os.path.join(SCD, 'scissor_labels_M5_all_samples.csv')).set_index('spot')
    M5.index = M5.index.astype(str)
    car = M['car_pos'].fillna(False) & M['sample'].isin(['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3'])
    D = pd.DataFrame(index=M.index[car])
    D['sample'] = M['sample'][car]
    D['MIB'] = (R['region_niche'].reindex(D.index) == 'CCR1_MIB').values
    D['mem_exh'] = (G['MEMORY'] - G['EXHAUSTED']).reindex(D.index).values
    D['ml_score'] = ML['p_GR_HVG'].reindex(D.index).values
    D['ML'] = ML['GRPR_pred'].reindex(D.index).values
    D['NE'] = NE['GRPR_quad_M5'].reindex(D.index).values
    D['NDRDR'] = M5['label'].reindex(D.index).values
    D['coef'] = M5['coef'].reindex(D.index).values
    print(f'CAR+ spots {len(D)} | MIB {int(D.MIB.sum())} | non-MIB {int((~D.MIB).sum())}')
    specs = [('new GR/PR (mem/exh)', 'NE', 'GR', 'mem_exh', 'MEM-EXH score -> GR'), ('ML GR/PR', 'ML', 'GR', 'ml_score', 'P(GR) -> GR'), ('NDR/DR (M5)', 'NDRDR', 'NDR', 'mem_exh', 'MEM-EXH score -> NDR'), ('NDR/DR (M5)', 'NDRDR', 'NDR', 'coef', '-Scissor coef -> NDR')]
    rows = []
    for name, gcol, g1, score, desc in specs:
        for sub_name, mask in [('MIB (CCR1_MIB)', D.MIB.values), ('non-MIB', (~D.MIB).values), ('all CAR+', np.ones(len(D), bool))]:
            d = D[mask & D[gcol].isin(['GR', 'PR', 'NDR', 'DR'])].dropna(subset=[score])
            if len(d) < 30 or d[gcol].nunique() < 2:
                continue
            y = (d[gcol] == g1).astype(int).values
            s = d[score].values * (1 if score != 'coef' else -1)
            auc = roc_auc_score(y, s)
            auc = max(auc, 1 - auc)
            _, p = perm_p(s, y, d['sample'].values, n=500)
            per = []
            for sl, g in d.groupby('sample'):
                if g[gcol].nunique() == 2:
                    a = roc_auc_score((g[gcol] == g1).astype(int), g[score].values * (1 if score != 'coef' else -1))
                    per.append(max(a, 1 - a))
            rows.append(dict(definition=name, score=desc, subset=sub_name, n=len(d), n1=int(y.sum()), n2=int((1 - y).sum()), auc=auc, perm_p=p, n_slice_auc_gt05=int(np.sum(np.array(per) > 0.5)), n_slice=len(per), slice_auc_mean=float(np.mean(per)) if per else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'auc_in_mib.csv'), index=False)
    pd.set_option('display.width', 220)
    print('\n=== AUC inside vs outside CCR1_MIB ===')
    print(T.round(4).to_string(index=False))
    T['lab'] = T.definition + '\n' + T.score
    subs = ['MIB (CCR1_MIB)', 'non-MIB', 'all CAR+']
    xs = np.arange(len(T.lab.unique()))
    w = 0.26
    for i, sub in enumerate(subs):
        v = T[T.subset == sub].set_index('lab').reindex(T.lab.unique())
    print('wrote auc_in_mib.*')
if __name__ == '__main__':
    main()
