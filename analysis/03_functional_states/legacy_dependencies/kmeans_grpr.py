"""DEPRECATED (2026): the GR/PR label scheme used here was replaced by def5 (results/GRPR_labels_final.csv via common.grpr_labels()); its outputs were deleted in step49 - see results/README_GRPR_definition.md.
Step17: unsupervised (K-means) definition of the GR / PR labels.

Requested rule: cluster the CAR+ spots of the treated slices on the T-cell functional
space and call the two subpopulations that are *negatively correlated* in effector vs
exhaustion GR (high exhaustion / low effector) and PR (high effector / low exhaustion).

Diagnostic first (see the log): among CAR+ spots effector and exhaustion are strongly
POSITIVELY correlated (r = +0.80), so the literal 2-D K-means partition only separates
"high-both" from "low-both" spots - both centroids lie on the positive diagonal and no
cluster is high-effector/low-exhaustion.  The requested contrast therefore has to be
taken on the axis orthogonal to that shared component:

  PC1 = shared effector+exhaustion axis
  PC2 = contrast axis, oriented so that high PC2 = high effector / low exhaustion

K-means is run on the contrast axis (k = 2 primary, k = 3/4 with the intermediate
cluster kept as "Other").  Reported per cluster: size, effector/exhaustion centroids and
the within-cluster r(effector, exhaustion).  The literal 2-D K-means is kept as a
sensitivity block.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from common import RES, grpr_labels
FIG = os.path.join(RES, 'figs_tme')
os.makedirs(FIG, exist_ok=True)
BLUE, BRICK = ('#2C7FB8', '#B22222')
SAMPLES = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']

def kmeans_labels(F, k, seed=0):
    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(F)
    sil = silhouette_score(F, km.labels_) if k > 1 else np.nan
    return (km.labels_, np.asarray(km.cluster_centers_), sil)

def assign_grpr(lab, cen, k):
    order = np.argsort(-cen[:, 0])
    out = np.array(['Other'] * len(lab), dtype=object)
    out[lab == order[0]] = 'GR'
    if k > 1:
        out[lab == order[1]] = 'PR'
    return out

def main():
    G = pd.read_csv(os.path.join(RES, 'gsva_tcell_spot.csv'), index_col=0)
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    A = G.drop(columns=[c for c in ['sample'] if c in G.columns]).join(M[['sample']], how='left')
    A = A[A['car_pos'] & A['sample'].isin(SAMPLES)].copy()
    A['eff'] = A['gsva_effector_z']
    A['exh'] = A['gsva_exhaustion_z']
    A[['eff', 'exh']] = (A[['eff', 'exh']] - A[['eff', 'exh']].mean()) / A[['eff', 'exh']].std()
    E = A[['eff', 'exh']].values
    r_all = float(np.corrcoef(E[:, 0], E[:, 1])[0, 1])
    w, V = np.linalg.eigh(np.cov(E.T))
    o = np.argsort(-w)
    w, V = (w[o], V[:, o])
    pc = E @ V
    if V[0, 1] < 0:
        V[:, 1] *= -1
        pc[:, 1] *= -1
    A['PC1'], A['PC2'] = (pc[:, 0], pc[:, 1])
    A['contrast'] = pc[:, 1]
    print(f'CAR+ spots: {len(A)} | r(effector, exhaustion) = {r_all:.3f} | PC1 {100 * w[0] / w.sum():.1f}% (loadings {V[0, 0]:.2f},{V[1, 0]:.2f}) | PC2 {100 * w[1] / w.sum():.1f}% (loadings {V[0, 1]:.2f},{V[1, 1]:.2f})')
    lit = []
    for k in (2, 3, 4):
        lab, cen, sil = kmeans_labels(E, k)
        for c in range(k):
            m = lab == c
            lit.append(dict(variant='2D_kmeans', k=k, cluster=c, label='high-both' if E[m, 0].mean() > 0 else 'low-both', n=int(m.sum()), mean_eff=E[m, 0].mean(), mean_exh=E[m, 1].mean(), r_eff_exh_within=float(np.corrcoef(E[m, 0], E[m, 1])[0, 1]), silhouette=sil, r_eff_exh_all=r_all))
    lit = pd.DataFrame(lit)
    rows, labels = ([], {})
    for k in (2, 3, 4):
        lab, cen, sil = kmeans_labels(A[['contrast']].values, k)
        gl = assign_grpr(lab, cen, k)
        A[f'GRPR_km{k}'] = gl
        labels[k] = gl
        for c in range(k):
            m = lab == c
            rows.append(dict(variant='contrast_kmeans', k=k, cluster=c, label=gl[m][0] if m.sum() else 'Other', n=int(m.sum()), mean_eff=E[m, 0].mean(), mean_exh=E[m, 1].mean(), r_eff_exh_within=float(np.corrcoef(E[m, 0], E[m, 1])[0, 1]) if m.sum() > 2 else np.nan, silhouette=sil, r_eff_exh_all=r_all))
    K = pd.concat([lit, pd.DataFrame(rows)], ignore_index=True)
    K.to_csv(os.path.join(RES, 'gsva_kmeans_cluster_stats.csv'), index=False)
    A['GRPR_kmeans'] = labels[2]
    A['GRPR_median'] = A['GRPR_sample']
    A[['sample', 'car_pos', 'eff', 'exh', 'net', 'contrast', 'PC1', 'PC2', 'GRPR_kmeans', 'GRPR_km3', 'GRPR_km4', 'GRPR_median']].to_csv(os.path.join(RES, 'gsva_kmeans_GRPR.csv'))
    for name, col in [('PR', BLUE), ('GR', BRICK)]:
        m = labels[2] == name
    pd.set_option('display.width', 230)
    print('\n=== literal 2-D K-means (premise check) ===')
    print(lit[['k', 'label', 'n', 'mean_eff', 'mean_exh', 'r_eff_exh_within', 'silhouette']].round(3).to_string(index=False))
    print('\n=== contrast-axis K-means (primary) ===')
    print(K[K.variant == 'contrast_kmeans'][['k', 'cluster', 'label', 'n', 'mean_eff', 'mean_exh', 'r_eff_exh_within', 'silhouette']].round(3).to_string(index=False))
    print('\nk=2 vs previous median split:')
    print(pd.crosstab(A['GRPR_median'], A['GRPR_kmeans']).to_string())
    print('\nk=2 per slice:')
    print(pd.crosstab(A['sample'], A['GRPR_kmeans']).to_string())
    print('\nwrote gsva_kmeans_cluster_stats.csv, gsva_kmeans_GRPR.csv; figure ->', FIG)
if __name__ == '__main__':
    main()
