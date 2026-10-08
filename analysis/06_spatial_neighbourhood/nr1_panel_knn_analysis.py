"""Step01 (nr1r1_analysis_M5/niche_knn): panel I pooled over the six treated sections.

Panel I = x is the rank k of the spatial nearest neighbour of a CAR+ spot (k = 1..40), y is the %
share of each cell type among those k neighbours, one curve per cell type, GR and PR shown in two
separate panels (no GR-PR difference anywhere).

Design
    * focal spots: every CAR+ spot of NR1/NR2/NR3/R1/R2/R3 that carries a GR/PR label -> one pooled set
      (labels = GRPR_quad_M5: GR = MEMORY_z>0 & EXHAUSTED_z<0, PR = MEMORY_z<0 & EXHAUSTED_z>0).
      Neighbours are taken inside the section the focal spot belongs to (coordinates are not
      comparable across sections); pooling happens at the level of the focal spots.
    * no niche / region filter.
    * tumour out of the composition, done separately: microenvironment panels use the tumour-free
      neighbour pool (all spots except non-CAR+ tumour spots; CAR+ spots kept) and draw M2 / VEC /
      CAF / SPP1+Mphi (DestVI ratios) + the CAR-T spot share; the tumour is its own curve set
      (tumour share of the k neighbours, all-spot pool).  Reference pools: "nontumour", "all".
    * permutation: GR/PR labels shuffled among the pooled focal spots, positions fixed,
      stratified by section (each section keeps its own GR/PR counts); B = 2000.  Curve-level
      statistics S_max = max_k |GR-PR| and S_auc = mean_k (GR-PR) with BH over the curves; per-k
      p values with BH inside a curve.  Results are reported as p values only.

Outputs: tables/panelI_pooled_*.csv (curves / stats / perk / tumour), focal counts and pool sizes.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
TAB = os.path.join(WD, 'nr1r1_analysis_M5', 'niche_knn', 'tables')
NE_FILE = os.path.join(WD, 'grpr_mem_exh', 'GRPR_mem_exh_final_labels.csv')
DEF5_FILE = os.path.join(RES, 'GRPR_labels_final.csv')
SECTIONS = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
KMAX, NPERM, SEED = (40, 1000, 0)
PROP_TYPES = [('M2', 'M2'), ('VEC', 'Vein EC/Fibro'), ('CAF', 'Fibro'), ('SPP1+Mø', 'SPP1+ Mø')]
FLAG_TYPES = [('CAR-T', 'car_pos')]
CATS = [c for c, _ in PROP_TYPES] + [c for c, _ in FLAG_TYPES]
TUMOUR_CAT, TUMOUR_FLAG = ('Tumour cells', 'tumor')
METRICS = ['ratio', 'flag']
POOLS = [('microenv', 'tumour-free neighbourhood (CAR+ spots kept)'), ('nontumour', 'strictly non-tumour spots'), ('all', 'all spots, tumour included')]
GROUPS = [('M5quad', 'GRPR_quad_M5'), ('EXHzsign', 'EXHAUSTED_z sign split'), ('def5', 'superseded def5 label')]
PRIMARY_GROUP, PRIMARY_POOL = ('M5quad', 'microenv')

def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)

def load(section):
    D = pd.read_csv(os.path.join(RES, 'spot_full_table.csv'), index_col=0)
    D = D[D['sample'] == section].copy()
    NE = pd.read_csv(NE_FILE, index_col=0)
    NE.index = NE.index.astype(str)
    D['grpr_M5quad'] = NE['GRPR_quad_M5'].reindex(D.index).values
    D['mem_z'] = NE['g_MEMORY_z'].reindex(D.index).values
    D['exh_z'] = NE['g_EXHAUSTED_z'].reindex(D.index).values
    D['grpr_EXHzsign'] = np.where(D['exh_z'].values < 0, 'GR', np.where(D['exh_z'].values > 0, 'PR', None)).astype(object)
    DF = pd.read_csv(DEF5_FILE, index_col=0)
    D['grpr_def5'] = DF['GRPR'].reindex(D.index).values
    return D

def pool_indices(D, pool):
    tumour = D[TUMOUR_FLAG].values.astype(bool)
    if pool == 'microenv':
        return np.where(~tumour | D['car_pos'].values.astype(bool))[0]
    if pool == 'nontumour':
        return np.where(~tumour)[0]
    return np.arange(len(D))

def tensors(D):
    """dict pool -> {metric: (n_spots, KMAX, n_cat)} and the tumour tensor (all-spot pool)"""
    xy = D[['x', 'y']].values
    prop = {lab: D[col].values.astype(float) for lab, col in PROP_TYPES}
    flag = {lab: D[col].values.astype(bool) for lab, col in FLAG_TYPES}
    tum = D[TUMOUR_FLAG].values.astype(bool)
    nbs = {}
    for pool, _ in POOLS:
        idx = pool_indices(D, pool)
        nn = cKDTree(xy[idx]).query(xy, k=KMAX + 1, workers=8)[1]
        row = idx[nn]
        fixed = np.zeros((len(D), KMAX), int)
        for i in range(len(D)):
            r = row[i][row[i] != i][:KMAX]
            fixed[i] = r if len(r) == KMAX else np.pad(r, (0, KMAX - len(r)), constant_values=idx[0])
        nbs[pool] = fixed
    out = {}
    for pool, _ in POOLS:
        T = {m: np.zeros((len(D), KMAX, len(CATS))) for m in METRICS}
        for k in range(1, KMAX + 1):
            ix = nbs[pool][:, :k]
            for j, (lab, _) in enumerate(PROP_TYPES):
                T['ratio'][:, k - 1, j] = prop[lab][ix].mean(1) * 100
            for j, (lab, _) in enumerate(FLAG_TYPES, start=len(PROP_TYPES)):
                T['flag'][:, k - 1, j] = flag[lab][ix].mean(1) * 100
        out[pool] = T
    Tt = np.stack([tum[nbs['all'][:, :k]].mean(1) * 100 for k in range(1, KMAX + 1)], axis=1)
    return (out, Tt)

def focal(D, group):
    lab = D['grpr_' + group].values
    keep = D['car_pos'].values.astype(bool) & np.isin(lab, ['GR', 'PR'])
    return (np.where(keep)[0], lab[keep])

def curves(T, lab, cats, metric):
    rows = []
    for j, cat in enumerate(cats):
        for k in range(1, KMAX + 1):
            v = T[:, k - 1, j]
            g, p = (v[lab == 'GR'], v[lab == 'PR'])
            rows.append(dict(metric=metric, cell_type=cat, k=k, mean_GR=g.mean(), sem_GR=g.std(ddof=1) / np.sqrt(len(g)), n_GR=len(g), mean_PR=p.mean(), sem_PR=p.std(ddof=1) / np.sqrt(len(p)), n_PR=len(p), diff=g.mean() - p.mean()))
    return pd.DataFrame(rows)

def permute(T, lab, cats, strata, nperm=NPERM, seed=SEED):
    rng = np.random.default_rng(seed)
    n_f, is_g = (len(lab), lab == 'GR')
    obs = T[is_g].mean(0) - T[~is_g].mean(0)
    groups = [np.where(strata == s)[0] for s in pd.unique(strata)]
    null_k = np.zeros((nperm, KMAX, len(cats)))
    for b in range(nperm):
        sh = np.zeros(n_f, bool)
        for gr in groups:
            kg = int(is_g[gr].sum())
            if kg:
                sh[gr[rng.permutation(len(gr))[:kg]]] = True
        null_k[b] = T[sh].mean(0) - T[~sh].mean(0)
    obs_max, obs_auc = (np.abs(obs).max(0), obs.mean(0))
    p_max = (1 + (np.abs(null_k).max(1) >= obs_max).sum(0)) / (nperm + 1)
    p_auc = (1 + (np.abs(null_k.mean(1)) >= np.abs(obs_auc)).sum(0)) / (nperm + 1)
    k_at = np.argmax(np.abs(obs), axis=0) + 1
    S = pd.DataFrame([dict(cell_type=c, S_max=obs_max[j], p_max=p_max[j], q_max=bh(p_max)[j], S_auc=obs_auc[j], p_auc=p_auc[j], q_auc=bh(p_auc)[j], k_at=int(k_at[j]), diff_at_max=obs[k_at[j] - 1, j]) for j, c in enumerate(cats)])
    P = pd.DataFrame([dict(cell_type=c, k=k + 1, diff=obs[k, j], p=(1 + (np.abs(null_k[:, k, j]) >= abs(obs[k, j])).sum()) / (nperm + 1), lo=float(np.percentile(null_k[:, k, j], 2.5)), hi=float(np.percentile(null_k[:, k, j], 97.5)), abs95=float(np.percentile(np.abs(null_k[:, k, j]), 95))) for j, c in enumerate(cats) for k in range(KMAX)])
    P['q'] = P.groupby('cell_type')['p'].transform(lambda v: bh(v.values))
    return (S, P)
OUTLIER_FEATURES = ['tumour_pct', 'car_pct', 'umi_med', 'tumour_nb_k40', 'CAR_T_nb_k40', 'M2_k40', 'VEC_k40', 'CAF_k40', 'SPP1_k40', 'GR_pct']
OUTLIER_Z = 3.0

def section_features(sec, TT):
    """section-level deviation features used by the outlier flag"""
    rows = []
    for s in SECTIONS:
        D = sec[s]
        ix, lab = focal(D, PRIMARY_GROUP)
        T, Tt = TT[s]
        prof = np.concatenate([T[PRIMARY_POOL]['ratio'][ix, KMAX - 1, :4].mean(0), [T[PRIMARY_POOL]['flag'][ix, KMAX - 1, 4].mean()]])
        rows.append(dict(section=s, n_spots=len(D), car_pct=100 * D.car_pos.mean(), tumour_pct=100 * D[TUMOUR_FLAG].mean(), umi_med=float(D['umi_total'].median()), n_focal=len(ix), n_GR=int((lab == 'GR').sum()), n_PR=int((lab == 'PR').sum()), GR_pct=100 * (lab == 'GR').mean(), tumour_nb_k40=float(Tt[ix, KMAX - 1].mean()), M2_k40=float(T[PRIMARY_POOL]['ratio'][ix, KMAX - 1, 0].mean()), VEC_k40=float(T[PRIMARY_POOL]['ratio'][ix, KMAX - 1, 1].mean()), CAF_k40=float(T[PRIMARY_POOL]['ratio'][ix, KMAX - 1, 2].mean()), SPP1_k40=float(T[PRIMARY_POOL]['ratio'][ix, KMAX - 1, 3].mean()), CAR_T_nb_k40=float(T[PRIMARY_POOL]['flag'][ix, KMAX - 1, 4].mean())))
    X = pd.DataFrame(rows)
    for c in OUTLIER_FEATURES:
        v = X[c].values.astype(float)
        mad = np.median(np.abs(v - np.median(v))) * 1.4826
        X['z_' + c] = (v - np.median(v)) / mad if mad > 0 else 0.0
    Z = X[['z_' + c for c in OUTLIER_FEATURES]]
    X['max_abs_z'] = Z.abs().max(1)
    X['outlier'] = X.max_abs_z > OUTLIER_Z
    X['driver'] = [OUTLIER_FEATURES[i] for i in Z.abs().values.argmax(1)]
    return X

def pooled_run(sec, TT, keep, tag, trim_spots=0.0, write=True):
    """pooled curves + stratified permutation test over the sections in `keep`

    trim_spots : robust-z cutoff on the focal spots' own k = 40 profile (0 = no trimming); focal
                 spots beyond it in any of the five curves are dropped before pooling.
    """
    lab = np.concatenate([focal(sec[s], PRIMARY_GROUP)[1] for s in keep])
    sids = np.concatenate([[s] * len(focal(sec[s], PRIMARY_GROUP)[1]) for s in keep])
    T = {m: np.concatenate([TT[s][0][PRIMARY_POOL][m][focal(sec[s], PRIMARY_GROUP)[0]] for s in keep]) for m in METRICS}
    Tz = np.concatenate([TT[s][1][focal(sec[s], PRIMARY_GROUP)[0]][:, :, None] for s in keep])
    if trim_spots > 0:
        prof = np.c_[T['ratio'][:, -1, :], T['flag'][:, -1, 4]]
        med = np.median(prof, 0)
        mad = np.median(np.abs(prof - med), 0) * 1.4826
        mad[mad == 0] = 1e-09
        zz = np.abs((prof - med) / mad)
        good = (zz <= trim_spots).all(1)
        lab, sids, Tz = (lab[good], sids[good], Tz[good])
        for m in METRICS:
            T[m] = T[m][good]
    out = []
    for m in METRICS:
        C = curves(T[m], lab, CATS, m)
        if write:
            C.to_csv(os.path.join(TAB, f'panelI_pooled_curves_{PRIMARY_GROUP}_{PRIMARY_POOL}_{m}{tag}.csv'), index=False)
        S, P = permute(T[m], lab, CATS, sids)
        S['metric'] = m
        out.append(S)
        if write:
            P.to_csv(os.path.join(TAB, f'panelI_pooled_perk_{PRIMARY_GROUP}_{PRIMARY_POOL}_{m}{tag}.csv'), index=False)
    St = pd.concat(out, ignore_index=True)
    Ct = curves(Tz, lab, [TUMOUR_CAT], 'flag')
    Stt, _ = permute(Tz, lab, [TUMOUR_CAT], sids)
    if write:
        St.to_csv(os.path.join(TAB, f'panelI_pooled_stats_{PRIMARY_GROUP}_{PRIMARY_POOL}{tag}.csv'), index=False)
        Ct.to_csv(os.path.join(TAB, f'panelI_pooled_tumour_{PRIMARY_GROUP}{tag}.csv'), index=False)
        Stt.to_csv(os.path.join(TAB, f'panelI_pooled_stats_{PRIMARY_GROUP}_tumour{tag}.csv'), index=False)
    St['n_focal'], St['n_GR'] = (len(lab), int((lab == 'GR').sum()))
    St['n_PR'] = int((lab == 'PR').sum())
    Stt['n_focal'], Stt['n_GR'] = (len(lab), int((lab == 'GR').sum()))
    Stt['n_PR'] = int((lab == 'PR').sum())
    Stt['pool'] = 'tumour_all'
    return (St, Stt, len(lab))

def main():
    os.makedirs(TAB, exist_ok=True)
    sec = {s: load(s) for s in SECTIONS}
    TT = {s: tensors(sec[s]) for s in SECTIONS}
    frows, prows, ftabs = ([], [], [])
    for s in SECTIONS:
        D = sec[s]
        for g, _ in GROUPS:
            lab = focal(D, g)[1]
            frows.append(dict(section=s, grouping=g, n_spots=len(D), n_carpos=int(D.car_pos.sum()), n_tumour=int(D[TUMOUR_FLAG].sum()), n_focal=len(lab), n_GR=int((lab == 'GR').sum()), n_PR=int((lab == 'PR').sum())))
        for p, desc in POOLS:
            ix = pool_indices(D, p)
            prows.append(dict(section=s, pool=p, description=desc, n_pool_spots=len(ix), n_tumour_in_pool=int(D[TUMOUR_FLAG].values[ix].sum())))
        idx = focal(D, PRIMARY_GROUP)[0]
        t = D.iloc[idx][['barcode', 'x', 'y', 'region_niche', 'grpr_M5quad', 'grpr_def5', 'mem_z', 'exh_z', 'tumor', 'car_pos']].copy()
        t.insert(0, 'section', s)
        ftabs.append(t)
    pd.DataFrame(frows).to_csv(os.path.join(TAB, 'panelI_focal_counts.csv'), index=False)
    pd.DataFrame(prows).to_csv(os.path.join(TAB, 'panelI_neighbour_pools.csv'), index=False)
    pd.concat(ftabs).to_csv(os.path.join(TAB, 'panelI_pooled_focal_spots.csv'))
    X = section_features(sec, TT)
    X.to_csv(os.path.join(TAB, 'panelI_section_outliers.csv'), index=False)
    keep = [s for s in SECTIONS if not bool(X.set_index('section').outlier[s])]
    drop = [s for s in SECTIONS if s not in keep]
    print('\n=== section deviation (robust z, cutoff %.1f) ===' % OUTLIER_Z)
    print(X[['section', 'n_focal', 'max_abs_z', 'driver', 'outlier']].round(2).to_string(index=False))
    print('excluded sections:', drop, '| kept:', keep)
    stats = []
    St_c, Stt_c, n_clean = pooled_run(sec, TT, keep, '')
    St_a, Stt_a, n_all = pooled_run(sec, TT, SECTIONS, '_all6')
    St_t, Stt_t, n_trim = pooled_run(sec, TT, keep, '_trimspot', trim_spots=3.5)
    for tag, St, Stt in [('outliers_removed', St_c, Stt_c), ('all6', St_a, Stt_a), ('outliers_removed_trimspot', St_t, Stt_t)]:
        St['case'] = tag
        Stt['case'] = tag
        stats += [St, Stt]
    for g, _ in GROUPS:
        if g == PRIMARY_GROUP:
            continue
        lab = np.concatenate([focal(sec[s], g)[1] for s in keep])
        sids = np.concatenate([[s] * len(focal(sec[s], g)[1]) for s in keep])
        for pool, _ in POOLS:
            T = {m: np.concatenate([TT[s][0][pool][m][focal(sec[s], g)[0]] for s in keep]) for m in METRICS}
            rows = []
            for m in METRICS:
                S, _ = permute(T[m], lab, CATS, sids)
                S['metric'] = m
                rows.append(S)
            Sr = pd.concat(rows, ignore_index=True)
            Sr['grouping'], Sr['pool'], Sr['case'] = (g, pool, 'outliers_removed')
            Sr['n_focal'], Sr['n_GR'] = (len(lab), int((lab == 'GR').sum()))
            Sr['n_PR'] = int((lab == 'PR').sum())
            stats.append(Sr)
        Tz = np.concatenate([TT[s][1][focal(sec[s], g)[0]][:, :, None] for s in keep])
        Stt, _ = permute(Tz, lab, [TUMOUR_CAT], sids)
        Stt['grouping'], Stt['pool'], Stt['case'] = (g, 'tumour_all', 'outliers_removed')
        Stt['metric'] = 'flag'
        stats.append(Stt)
    loo = []
    for s in SECTIONS:
        kk = [x for x in SECTIONS if x != s]
        A, B, n = pooled_run(sec, TT, kk, f'_loo_{s}', write=False)
        A['case'] = f'drop_{s}'
        B['case'] = f'drop_{s}'
        loo += [A, B]
    pd.concat(loo, ignore_index=True).to_csv(os.path.join(TAB, 'panelI_pooled_LOO.csv'), index=False)
    sets = {'all6': list(SECTIONS), 'z3': keep, 'z5': [s for s in SECTIONS if float(X.set_index('section').max_abs_z[s]) <= 5.0], 'drop_NR1': [s for s in SECTIONS if s != 'NR1'], 'drop_NR1_R1': [s for s in SECTIONS if s not in ('NR1', 'R1')]}
    srows = []
    for name, kk in sets.items():
        A, B, n = pooled_run(sec, TT, kk, f'_set_{name}', write=False)
        for _, r in A[A.metric == 'ratio'].iterrows():
            srows.append(dict(set=name, kept='+'.join(kk), cell_type=r.cell_type, n_focal=n, S_max=r.S_max, p_max=r.p_max, q_max=r.q_max, p_auc=r.p_auc, q_auc=r.q_auc))
        srows.append(dict(set=name, kept='+'.join(kk), cell_type='Tumour cells', n_focal=n, S_max=B.S_max.iloc[0], p_max=B.p_max.iloc[0], q_max=B.q_max.iloc[0], p_auc=B.p_auc.iloc[0], q_auc=B.q_auc.iloc[0]))
    pd.DataFrame(srows).to_csv(os.path.join(TAB, 'panelI_outlier_sets.csv'), index=False)
    print('\n=== alternative exclusion sets (p_auc) ===')
    print(pd.DataFrame(srows).pivot_table(index=['set', 'n_focal'], columns='cell_type', values='p_auc').round(3).to_string())
    print('\n=== leave-one-section-out (p_auc, cell ratios) ===')
    L = pd.concat(loo, ignore_index=True)
    print(L[L.metric == 'ratio'].pivot_table(index='case', columns='cell_type', values='p_auc').round(3).to_string())
    print('\n=== outlier-removed vs all six (p_auc / p_max) ===')
    for c, St in [('outliers_removed', St_c), ('all6', St_a)]:
        print(f'-- {c} (n={int(St.n_focal.iloc[0])})')
        print(St[['metric', 'cell_type', 'S_max', 'p_max', 'q_max', 'p_auc', 'q_auc']].round(4).to_string(index=False))
        print(f"-- {c} tumour: p_auc={(Stt_c.p_auc.iloc[0] if c == 'outliers_removed' else Stt_a.p_auc.iloc[0]):.4f}")
    pd.concat(stats, ignore_index=True).to_csv(os.path.join(TAB, 'panelI_pooled_summary.csv'), index=False)
    print('\nwrote tables ->', TAB)
if __name__ == '__main__':
    main()
