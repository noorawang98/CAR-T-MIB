"""Step94: per-slide AUCs (cross-system, all / MIB / non-MIB) + per-slide ROC curves."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import RES
WD = PROJECT_ROOT + '/mouse/cart_region'
OUT = os.path.join(WD, 'figs_final')
PRED = os.path.join(WD, 'grpr_pred_v3')
MEMX = os.path.join(WD, 'grpr_mem_exh')
SCD = os.path.join(WD, 'scissor_pooled_noRelapse')
BRED, BBLUE, BGREY = ('#B22222', '#2C7FB8', '#BFBFBF')
SL = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']

def auc_rank(score, y):
    r = pd.Series(score).rank().values
    n1, n0 = (y.sum(), (1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def main():
    R = pd.read_csv(os.path.join(RES, 'spot_regions_lym.csv'), index_col=0)
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    G = pd.read_csv(os.path.join(PRED, 'clinical_gsva', 'gsva_clinical_scores.csv'), index_col=0)
    G = G.groupby([i.split('|')[0] for i in G.index]).transform(lambda v: (v - v.mean()) / (v.std() or 1))
    ML = pd.read_csv(os.path.join(PRED, 'GRPR_pred_labels.csv'), index_col=0)
    NE = pd.read_csv(os.path.join(MEMX, 'GRPR_mem_exh_final_labels.csv'), index_col=0)
    M5 = pd.read_csv(os.path.join(SCD, 'scissor_labels_M5_all_samples.csv')).set_index('spot')
    M5.index = M5.index.astype(str)
    car = M['car_pos'].fillna(False) & M['sample'].isin(SL)
    D = pd.DataFrame(index=M.index[car])
    D['sample'] = M['sample'][car]
    D['MIB'] = (R['region_niche'].reindex(D.index) == 'CCR1_MIB').values
    D['ccr1mye'] = R['ccr1mye'].reindex(D.index).values
    D['mem_exh'] = (G['MEMORY'] - G['EXHAUSTED']).reindex(D.index).values
    D['MEMORY'] = G['MEMORY'].reindex(D.index).values
    D['EXHAUSTED'] = G['EXHAUSTED'].reindex(D.index).values
    D['ml_score'] = ML['p_GR_HVG'].reindex(D.index).values
    D['ML'] = ML['GRPR_pred'].reindex(D.index).values
    D['NE'] = NE['GRPR_quad_M5'].reindex(D.index).values
    D['NDRDR'] = M5['label'].reindex(D.index).values
    D['coef'] = M5['coef'].reindex(D.index).values
    specs = [('MEM−EXH', 'mem_exh', 'NDRDR', ['NDR', 'DR'], 1, False), ('P(GR) ML', 'ml_score', 'NDRDR', ['NDR', 'DR'], 1, False), ('CCR1⁺髓系', 'ccr1mye', 'NDRDR', ['NDR', 'DR'], 1, False), ('−Scissor coef', 'coef', 'NE', ['GR', 'PR'], -1, False), ('−Scissor coef', 'coef', 'ML', ['GR', 'PR'], -1, False), ('MEM−EXH', 'mem_exh', 'ML', ['GR', 'PR'], 1, False), ('CCR1⁺髓系', 'ccr1mye', 'NE', ['GR', 'PR'], 1, False)]
    rows = []
    for sname, scol, gcol, groups, sign, _c in specs:
        for sub_name, smask in [('all CAR+', np.ones(len(D), bool)), ('MIB', D.MIB.values), ('non-MIB', (~D.MIB).values)]:
            for sl in SL:
                d = D[smask & (D['sample'] == sl) & D[gcol].isin(groups)].dropna(subset=[scol])
                if len(d) < 15 or d[gcol].nunique() < 2:
                    rows.append(dict(score=sname, labels=f'{groups[0]} vs {groups[1]}', subset=sub_name, sample=sl, n=len(d), auc=np.nan))
                    continue
                y = (d[gcol] == groups[0]).astype(int).values
                rows.append(dict(score=sname, labels=f'{groups[0]} vs {groups[1]}', subset=sub_name, sample=sl, n=len(d), auc=auc_rank(sign * d[scol].values, y)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'auc_per_slide.csv'), index=False)
    pd.set_option('display.width', 240)
    print('=== 逐切片 AUC（all CAR+）===')
    print(T[T.subset == 'all CAR+'].pivot_table(index=['score', 'labels'], columns='sample', values='auc').round(3).to_string())
    print('\n=== 逐切片 AUC（MIB 区内）===')
    print(T[T.subset == 'MIB'].pivot_table(index=['score', 'labels'], columns='sample', values='auc').round(3).to_string())
    for j, sl in enumerate(SL):
        d = D[(D['sample'] != sl) & D.NDRDR.isin(['NDR', 'DR'])].dropna(subset=['MEMORY', 'EXHAUSTED'])
        t = D[(D['sample'] == sl) & D.NDRDR.isin(['NDR', 'DR'])].dropna(subset=['MEMORY', 'EXHAUSTED'])
        if len(t) > 10 and t.NDRDR.nunique() == 2:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(d[['MEMORY', 'EXHAUSTED']].values, (d.NDRDR == 'DR').astype(int).values)
            p = m.predict_proba(t[['MEMORY', 'EXHAUSTED']].values)[:, 1]
            y = (t.NDRDR == 'DR').astype(int).values
            fpr, tpr, _ = roc_curve(y, p)
            from sklearn.metrics import roc_auc_score
            a = roc_auc_score(y, p)
        d2 = D[(D['sample'] != sl) & D.NE.isin(['GR', 'PR'])].dropna(subset=['ccr1mye', 'MEMORY', 'EXHAUSTED'])
        t2 = D[(D['sample'] == sl) & D.NE.isin(['GR', 'PR'])].dropna(subset=['ccr1mye', 'MEMORY', 'EXHAUSTED'])
        if len(t2) > 10 and t2.NE.nunique() == 2:
            m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(d2[['MEMORY', 'EXHAUSTED']].values, (d2.NE == 'GR').astype(int).values)
            p2 = m2.predict_proba(t2[['MEMORY', 'EXHAUSTED']].values)[:, 1]
            y2 = (t2.NE == 'GR').astype(int).values
            from sklearn.metrics import roc_auc_score
            fpr2, tpr2, _ = roc_curve(y2, p2)
            a2 = roc_auc_score(y2, p2)
    print('\nwrote auc_per_slide.csv, roc_per_slide.*')
if __name__ == '__main__':
    main()
