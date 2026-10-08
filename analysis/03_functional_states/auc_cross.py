"""Step93b: AUC across INDEPENDENT label/score systems, inside the CCR1_MIB region and outside.
(Defining scores are excluded: they give AUC=1 by construction.)"""
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

def auc_rank(score, y):
    r = pd.Series(score).rank().values
    n1, n0 = (y.sum(), (1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def perm_p(score, y, g, n=500, seed=0):
    obs = auc_rank(score, y)
    rng = np.random.default_rng(seed)
    c = 0
    for _ in range(n):
        yp = y.copy()
        for s in np.unique(g):
            m = g == s
            yp[m] = rng.permutation(yp[m])
        if auc_rank(score, yp) >= obs:
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
    D['ccr1mye'] = R['ccr1mye'].reindex(D.index).values
    D['mem_exh'] = (G['MEMORY'] - G['EXHAUSTED']).reindex(D.index).values
    D['ml_score'] = ML['p_GR_HVG'].reindex(D.index).values
    D['ML'] = ML['GRPR_pred'].reindex(D.index).values
    D['NE'] = NE['GRPR_quad_M5'].reindex(D.index).values
    D['NDRDR'] = M5['label'].reindex(D.index).values
    D['coef'] = M5['coef'].reindex(D.index).values
    print(f'CAR+ {len(D)} | MIB {int(D.MIB.sum())} / non-MIB {int((~D.MIB).sum())}')
    specs = [('MEM−EXH (功能轴)', 'mem_exh', 'NDRDR', 'Scissor_NDR/DR', ['NDR', 'DR'], 1), ('P(GR) ML模型', 'ml_score', 'NDRDR', 'Scissor_NDR/DR', ['NDR', 'DR'], 1), ('MEM−EXH (功能轴)', 'mem_exh', 'NE', 'new_GR/PR(mem/exh)', ['GR', 'PR'], 1), ('MEM−EXH (功能轴)', 'mem_exh', 'ML', 'ML_GR/PR', ['GR', 'PR'], 1), ('−Scissor coef', 'coef', 'NE', 'new_GR/PR(mem/exh)', ['GR', 'PR'], -1), ('−Scissor coef', 'coef', 'ML', 'ML_GR/PR', ['GR', 'PR'], -1), ('CCR1⁺髓系模块', 'ccr1mye', 'NE', 'new_GR/PR(mem/exh)', ['GR', 'PR'], 1), ('CCR1⁺髓系模块', 'ccr1mye', 'NDRDR', 'Scissor_NDR/DR', ['NDR', 'DR'], 1)]
    rows = []
    for sname, scol, gcol, sysname, groups, sign in specs:
        for sub_name, mask in [('MIB', D.MIB.values), ('non-MIB', (~D.MIB).values), ('all CAR+', np.ones(len(D), bool))]:
            d = D[mask & D[gcol].isin(groups)].dropna(subset=[scol])
            if len(d) < 40 or d[gcol].nunique() < 2:
                continue
            y = (d[gcol] == groups[0]).astype(int).values
            s = sign * d[scol].values
            auc, p = perm_p(s, y, d['sample'].values, n=500)
            per = []
            for sl, g in d.groupby('sample'):
                if g[gcol].nunique() == 2:
                    per.append(auc_rank(sign * g[scol].values, (g[gcol] == groups[0]).astype(int).values))
            circular = scol == 'mem_exh' and sysname == 'new_GR/PR(mem/exh)' or (scol == 'ml_score' and sysname == 'ML_GR/PR') or (scol == 'coef' and sysname == 'Scissor_NDR/DR')
            rows.append(dict(score=sname, label_system=sysname, labels=f'{groups[0]} vs {groups[1]}', circular=circular, subset=sub_name, n=len(d), auc=auc, perm_p=p, slice_auc_mean=float(np.nanmean(per)) if per else np.nan, n_slice=len(per), n_slice_gt05=int(np.nansum(np.array(per) > 0.5)) if per else 0))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'auc_in_mib_cross.csv'), index=False)
    pd.set_option('display.width', 230)
    print('\n=== 跨系统 AUC：MIB 区内 vs 区外 ===')
    print(T.round(4).to_string(index=False))
    P2 = T[~T.circular]
    print('wrote auc_in_mib_cross.*')
if __name__ == '__main__':
    main()
