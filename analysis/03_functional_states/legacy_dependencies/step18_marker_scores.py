"""Step18: marker-based T-cell functional scores and the GR / PR labels.

Replaces the MSigDB GSVA scores (whose EFF/EXH sets share no genes and are strongly
co-linear) by a direct marker panel:

  Effector core      Gzma, Gzmb, Gzmk, Prf1, Ifng, Nkg7        (Gnly absent from the mm10 ref)
  Effector cytokine  Il12a, Il12b, Il21, Tnf                  (Il12 = a + b chains)
  Memory             Ccr7, Tcf7, Sell, Lef1, Il7r
  Activation         Cd69, Il2ra, Il2rb, Mki67
  Exhaustion         Pdcd1, Ctla4, Lag3, Havcr2, Tigit, Tox, Entpd1, Bach2

Per sample every gene is z-scored across the spots of that sample; a block score is the
mean z of its genes, EFF = mean of the four effector blocks, EXH = the exhaustion block.
GR / PR are assigned as in step17: K-means on the contrast axis (PC2 of [EFF, EXH],
oriented high = high effector / low exhaustion), k = 2 primary, k = 3/4 sensitivity.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import numpy as np, pandas as pd
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from common import RES
RCL = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/recluster_obj'
FIG = os.path.join(RES, 'figs_tme')
os.makedirs(FIG, exist_ok=True)
SAMPLES = ['Vehicle', 'NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
TREATED = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
BLOCKS = {'EFF_core': ['Gzma', 'Gzmb', 'Gzmk', 'Prf1', 'Ifng', 'Nkg7'], 'EFF_cyt': ['Il12a', 'Il12b', 'Il21', 'Tnf'], 'EFF_mem': ['Ccr7', 'Tcf7', 'Sell', 'Lef1', 'Il7r'], 'EFF_act': ['Cd69', 'Il2ra', 'Il2rb', 'Mki67'], 'EXH': ['Pdcd1', 'Ctla4', 'Lag3', 'Havcr2', 'Tigit', 'Tox', 'Entpd1', 'Bach2']}
EFF_BLOCKS = ['EFF_core', 'EFF_cyt', 'EFF_mem', 'EFF_act']
BLUE, BRICK = ('#2C7FB8', '#B22222')

def scores():
    frames, used = ([], {})
    for s in SAMPLES:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'))
        idx = {g: i for i, g in enumerate(a.var_names)}
        X = a.X.tocsr().astype(np.float32)
        mu = np.asarray(X.mean(0)).ravel()
        sd = np.sqrt(np.maximum(np.asarray(X.multiply(X).mean(0)).ravel() - mu ** 2, 0))
        sd[sd == 0] = 1
        d = pd.DataFrame(index=[f'{s}|{b}' for b in a.obs_names])
        for b, gs in BLOCKS.items():
            cols = [idx[g] for g in gs if g in idx]
            used[b] = [g for g in gs if g in idx]
            Z = (X[:, cols] - mu[cols]) / sd[cols]
            d[b] = np.asarray(Z.mean(1)).ravel()
        d['EFF'] = d[EFF_BLOCKS].mean(1)
        d['sample'] = s
        frames.append(d)
        del a, X
    return (pd.concat(frames), used)

def kmeans_marker2d(A, k, seed=0):
    """K-means k on the 2-D marker space [EFF_z, EXH_z] (no net / contrast axis).

    GR = cluster with the highest mean EFF (expected to be the low-EXH side),
    for k>2 the two extreme clusters by mean EFF are kept and the rest is 'Other'.
    """
    F = A[['EFF_z', 'EXH_z']].values
    lab = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(F)
    cen = pd.DataFrame({'EFF': [F[lab == c, 0].mean() for c in range(k)], 'EXH': [F[lab == c, 1].mean() for c in range(k)]})
    order = np.argsort(-cen['EFF'].values)
    out = np.array(['Other'] * len(lab), dtype=object)
    out[lab == order[0]] = 'GR'
    if k > 1:
        out[lab == order[1]] = 'PR'
    return (out, cen, silhouette_score(F, lab))

def kmeans_contrast(A, k, seed=0):
    lab = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(A[['contrast']].values)
    cen = pd.Series(A['contrast'].values).groupby(lab).mean().values
    order = np.argsort(-cen)
    out = np.array(['Other'] * len(lab), dtype=object)
    out[lab == order[0]] = 'GR'
    if k > 1:
        out[lab == order[1]] = 'PR'
    return (out, silhouette_score(A[['contrast']].values, lab))

def main():
    S, used = scores()
    print('genes used:', {k: f'{len(v)}' for k, v in used.items()})
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    S['car_pos'] = M['car_pos'].reindex(S.index).fillna(False).values
    for c in ['EFF', 'EXH']:
        S[c + '_z'] = S.groupby('sample')[c].transform(lambda v: (v - v.mean()) / (v.std() or 1))
    S['net_marker'] = S['EFF_z'] - S['EXH_z']
    A = S[S['car_pos'] & S['sample'].isin(TREATED)].copy()
    E = A[['EFF_z', 'EXH_z']].values
    r_all = float(np.corrcoef(E[:, 0], E[:, 1])[0, 1])
    w, V = np.linalg.eigh(np.cov(E.T))
    o = np.argsort(-w)
    w, V = (w[o], V[:, o])
    pc = E @ V
    if V[0, 1] < 0:
        V[:, 1] *= -1
        pc[:, 1] *= -1
    A['contrast'] = pc[:, 1]
    print(f'CAR+ spots: {len(A)} | marker r(EFF, EXH) = {r_all:.3f} | PC1 {100 * w[0] / w.sum():.1f}% | PC2 {100 * w[1] / w.sum():.1f}%')
    ann = []
    for k in (2, 3, 4):
        lab, cen, sil = kmeans_marker2d(A, k)
        A[f'GRPR_mk{k}'] = lab
        gr = int(np.argmax((cen.EFF - cen.EXH).values))
        pr = int(np.argmin((cen.EFF - cen.EXH).values))
        sep = float((cen.EFF - cen.EXH).values[gr] - (cen.EFF - cen.EXH).values[pr])
        ann.append(dict(k=k, silhouette=sil, n_GR=int((lab == 'GR').sum()), n_PR=int((lab == 'PR').sum()), n_Other=int((lab == 'Other').sum()), GR_EFF=cen.EFF[gr], GR_EXH=cen.EXH[gr], PR_EFF=cen.EFF[pr], PR_EXH=cen.EXH[pr], contrast_separation=sep, GR_is_highEFF_lowEXH=bool(cen.EFF[gr] > cen.EFF[pr] and cen.EXH[gr] < cen.EXH[pr]), PR_is_highEXH_lowEFF=bool(cen.EXH[pr] > cen.EXH[gr] and cen.EFF[pr] < cen.EFF[gr])))
        print(f"  [marker 2-D kmeans] k={k} silhouette {sil:.3f} {dict(zip(*np.unique(lab, return_counts=True)))} GR=(EFF {cen.EFF[gr]:.2f}, EXH {cen.EXH[gr]:.2f}) PR=(EFF {cen.EFF[pr]:.2f}, EXH {cen.EXH[pr]:.2f}) rule_ok={ann[-1]['GR_is_highEFF_lowEXH'] and ann[-1]['PR_is_highEXH_lowEFF']} sep={sep:.3f}")
    AN = pd.DataFrame(ann)
    AN.to_csv(os.path.join(RES, 'marker_kmeans_annotation_check.csv'), index=False)
    ok = AN[AN.GR_is_highEFF_lowEXH & AN.PR_is_highEXH_lowEFF & (AN.n_GR >= 50) & (AN.n_PR >= 50)]
    kbest = int(ok.sort_values('contrast_separation', ascending=False).k.iloc[0]) if len(ok) else 2
    print(f'  -> annotation rule satisfied at k={list(ok.k)}; primary k*={kbest}')
    A['GRPR_marker'] = A[f'GRPR_mk{kbest}']
    A['GRPR_marker_k'] = kbest
    A['GRPR_marker_k2'] = A['GRPR_mk2']
    A['GRPR_marker_contrast'] = kmeans_contrast(A, 2)[0]
    A['GRPR_kmeansGSVA'] = pd.read_csv(os.path.join(RES, 'gsva_kmeans_GRPR.csv'), index_col=0)['GRPR_kmeans'].reindex(A.index)
    A['GRPR_medianGSVA'] = pd.read_csv(os.path.join(RES, 'gsva_tcell_spot.csv'), index_col=0)['GRPR_sample'].reindex(A.index)
    A.to_csv(os.path.join(RES, 'marker_tcell_GRPR.csv'))
    S.to_csv(os.path.join(RES, 'marker_tcell_scores.csv'))
    B = ['EFF_core', 'EFF_cyt', 'EFF_mem', 'EFF_act', 'EXH']
    Zb = S[B + ['sample']].groupby('sample').transform(lambda v: (v - v.mean()) / (v.std() or 1))
    C = Zb[B].corr(method='spearman')
    C.to_csv(os.path.join(RES, 'marker_tcell_block_correlation.csv'))
    cor = pd.DataFrame({'sample': SAMPLES, 'r_EFF_EXH': [np.corrcoef(S.loc[S['sample'] == s, 'EFF_z'], S.loc[S['sample'] == s, 'EXH_z'])[0, 1] for s in SAMPLES]})
    cor.to_csv(os.path.join(RES, 'marker_tcell_EFF_EXH_correlation.csv'), index=False)
    for nm, col in [('PR', BLUE), ('GR', BRICK)]:
        m = (A['GRPR_marker'] == nm).values
    for i, (nm, col) in enumerate([('GR', BRICK), ('PR', BLUE)]):
        v = A.loc[A['GRPR_marker'] == nm, 'contrast'].values
    for nm, col in [('GR', BRICK), ('PR', BLUE)]:
        v = A.loc[A['GRPR_marker'] == nm, 'net_marker']
    pd.set_option('display.width', 220)
    print('\nmarker block Spearman r:\n', C.round(3).to_string())
    print('\nper-sample r(EFF, EXH):\n', cor.round(3).to_string(index=False))
    print('\nmarker GR/PR vs GSVA median label:\n', pd.crosstab(A['GRPR_medianGSVA'], A['GRPR_marker']).to_string())
    print('\nmarker GR/PR vs GSVA K-means label:\n', pd.crosstab(A['GRPR_kmeansGSVA'], A['GRPR_marker']).to_string())
    print('\nper slice:\n', pd.crosstab(A['sample'], A['GRPR_marker']).to_string())
if __name__ == '__main__':
    main()
