"""Step82: final Memory-Exhaustion GR/PR (M5 labels as primary) + fixed spatial maps.
Uses MEMORY alone (not the MEM+NAIVE composite, which cancels the signal) and writes
GR/PR/Other for ALL 888 CAR+ spots by k-means (k=3) and by the quadrant rule."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd, anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from scipy.stats import chi2_contingency, spearmanr, pearsonr
from common import RES, RCL, TREATED, bh
OUT = PROJECT_ROOT + '/mouse/cart_region/grpr_mem_exh'
PRED = PROJECT_ROOT + '/mouse/cart_region/grpr_pred_v3'
WORK = os.path.join(PRED, 'clinical_gsva')
SCD = PROJECT_ROOT + '/mouse/cart_region/scissor_pooled_noRelapse'
BRED, BBLUE, BGREY = ('#B22222', '#2C7FB8', '#BFBFBF')

def main():
    E = pd.read_csv(os.path.join(PRED, 'carpos_hvg_expr.csv'), index_col=0)
    Gc = pd.read_csv(os.path.join(WORK, 'gsva_clinical_scores.csv'), index_col=0).reindex(E.index)
    D = pd.DataFrame(index=E.index)
    D['sample'] = [i.split('|')[0] for i in D.index]
    for c in ['MEMORY', 'EXHAUSTED', 'NAIVE', 'EFFECTOR']:
        D['g_' + c] = Gc[c]
    Z = D.groupby('sample')[[c for c in D.columns if c.startswith('g_')]].transform(lambda v: (v - v.mean()) / (v.std() or 1))
    for c in Z.columns:
        D[c + '_z'] = Z[c].values
    for tag, f in [('M5', 'scissor_labels_M5_all_samples.csv'), ('BEST', 'scissor_labels_BEST_all_samples.csv')]:
        S = pd.read_csv(os.path.join(SCD, f)).set_index('spot')
        S.index = S.index.astype(str)
        D['lab_' + tag] = S['label'].reindex(D.index).values
    print('=== 面板与标签方向（M5）===')
    lab = D[D.lab_M5.isin(['NDR', 'DR'])].copy()
    lab['y'] = (lab.lab_M5 == 'DR').astype(int)
    for c in ['g_MEMORY_z', 'g_NAIVE_z', 'g_EFFECTOR_z', 'g_EXHAUSTED_z']:
        r = pearsonr(lab[c], lab.y)[0]
        r2 = pearsonr(lab[c], D[c].reindex(lab.index))[0]
        print(f"  {c:16s} 与 DR(r) = {r:+.3f} | 与 MEMORY_z 相关 = {pearsonr(lab[c], lab['g_MEMORY_z'])[0]:+.3f}")
    print(f'  NAIVE 与 MEMORY 的相关 r = {pearsonr(D.g_NAIVE_z, D.g_MEMORY_z)[0]:+.3f} （方向相反 → (MEM+NAIVE)/2 相互抵消）')
    rows = []
    for tag in ['M5', 'BEST']:
        lab = D[D['lab_' + tag].isin(['NDR', 'DR'])].copy()
        lab['y'] = (lab['lab_' + tag] == 'DR').astype(int)
        X = lab[['g_MEMORY_z', 'g_EXHAUSTED_z']].values
        km = KMeans(n_clusters=3, n_init=50, random_state=0).fit(X)
        cen = np.vstack([X[km.labels_ == j].mean(0) for j in range(3)])
        sc = cen[:, 0] - cen[:, 1]
        gmax, gmin = (int(np.argmax(sc)), int(np.argmin(sc)))
        allX = D[['g_MEMORY_z', 'g_EXHAUSTED_z']].values
        kall = km.predict(allX)
        v = np.array(['Other'] * len(kall), dtype=object)
        v[kall == gmax] = 'GR'
        v[kall == gmin] = 'PR'
        D['GRPR_km_' + tag] = v
        q = np.array(['Other'] * len(allX), dtype=object)
        q[(allX[:, 0] > 0) & (allX[:, 1] < 0)] = 'GR'
        q[(allX[:, 0] < 0) & (allX[:, 1] > 0)] = 'PR'
        D['GRPR_quad_' + tag] = q
        ref = lab['lab_' + tag].map({'DR': 'GR', 'NDR': 'PR'})
        for name, vv in [('kmeans3', pd.Series(v, index=D.index).reindex(lab.index)), ('quadrant', pd.Series(q, index=D.index).reindex(lab.index))]:
            sel = vv.isin(['GR', 'PR'])
            agree = float((vv[sel] == ref[sel]).mean()) if sel.sum() else np.nan
            ct = pd.crosstab(vv[sel], ref[sel])
            kap = cohen_kappa_score(vv[sel], ref[sel]) if ct.shape == (2, 2) else np.nan
            chi = chi2_contingency(ct.values)[1] if ct.shape == (2, 2) else np.nan
            rows.append(dict(labels=tag, rule=name, n_labelled=len(lab), n_selected=int(sel.sum()), agreement=agree, kappa=kap, chi2_p=chi, GR_total=int((pd.Series(v, index=D.index) if name == 'kmeans3' else pd.Series(q, index=D.index)).eq('GR').sum()), PR_total=int((pd.Series(v, index=D.index) if name == 'kmeans3' else pd.Series(q, index=D.index)).eq('PR').sum())))
        print(f'\n{tag}: GR/PR counts (k-means3)= {pd.Series(v).value_counts().to_dict()} | (quadrant)= {pd.Series(q).value_counts().to_dict()}')
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, 'mem_exh_final_consistency.csv'), index=False)
    pd.set_option('display.width', 200)
    print('\n=== GR/PR(Memory-Exhaustion) vs Scissor NDR/DR ===')
    print(R.round(4).to_string(index=False))
    D.to_csv(os.path.join(OUT, 'GRPR_mem_exh_final_labels.csv'))
    lab = D[D.lab_M5.isin(['NDR', 'DR'])]
    for nm, c in [('NDR', BRED), ('DR', BBLUE)]:
        vv = lab[lab.lab_M5 == nm]
    for nm, c in [('GR', BRED), ('Other', BGREY), ('PR', BBLUE)]:
        vv = D[D.GRPR_km_M5 == nm]
    lab2 = D[D.lab_M5.isin(['NDR', 'DR'])]
    for nm, c in [('NDR', BRED), ('DR', BBLUE)]:
        vv = lab2[lab2.lab_M5 == nm]
    print('\nwrote mem_exh_final_consistency.csv, GRPR_mem_exh_final_labels.csv, figs/mem_exh_final_scatter.*, figs/GRPR_mem_exh_final_spatial.*')
if __name__ == '__main__':
    main()
