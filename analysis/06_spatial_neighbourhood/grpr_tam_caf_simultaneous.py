"""Step09 (nr1r1_analysis_M5/region): SIMULTANEOUS co-localisation of CCR1+ TAM and FAP+ CAF
around / at the GR vs PR CAR+ spots - per section, nothing pooled.

The previous step (07) tested each marker on its own; this step asks whether the GR/PR contrast is
specific to the *joint* occurrence of the two markers:

spot level (per labelled CAR+ spot)
    self_dp       the spot itself is double positive (ccr1mye z > 0 AND caf z > 0)
    self_dp_high  the spot itself is double high   (both >= section Q75)
    both_present  >=1 CCR1+ TAM neighbour AND >=1 FAP+ CAF neighbour
    both2         >=2 CCR1+ TAM neighbours AND >=2 FAP+ CAF neighbours  (the project's MIB-like rule)
    min_frac      min(fraction of CCR1+ TAM neighbours, fraction of FAP+ CAF neighbours)
    dp_nb         fraction of the 6 neighbours that are double positive themselves
neighbour level (each of the k=6 neighbour relations of a labelled CAR+ spot is one observation)
    rate_ccr1 / rate_fap / rate_dp / rate_either   - how often a neighbour is CCR1+ TAM only,
    FAP+ CAF only, both, or either, grouped by the parent spot's GR/PR label

Tests: Mann-Whitney (continuous) / Fisher (binary) + within-section permutation of the parent labels
(N_PERM), which keeps the spatial layout and the neighbour clustering intact.  Region strata are
computed as well, so the joint effect can be checked against the region composition difference.

Outputs: tables/TableR14_GRPR_TAM_CAF_simultaneous_<S>.csv (one per section, 6 files)
         tables/TableR15_GRPR_TAM_CAF_simultaneous_summary.csv (wide)
         tables/TableR16_neighbourlevel_TAM_CAF_simultaneous.csv (neighbour-level detail)
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT + '/mouse/cart_region')
from region_define_stats import RES, TAB, WD
from common import RCL
SECTIONS = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
GROUP = {'NR1': 'NR', 'NR2': 'NR', 'NR3': 'NR', 'R1': 'R', 'R2': 'R', 'R3': 'R'}
REGIONS = ['Tumor_core', 'Peritumor', 'Extratumor', 'Perivascular']
KNN = 6
N_PERM = 2000

def load():
    R = pd.read_csv(os.path.join(RES, 'spot_regions.csv'), index_col=0)
    R['sample'] = [i.split('|')[0] for i in R.index]
    NE = pd.read_csv(os.path.join(WD, 'grpr_mem_exh', 'GRPR_mem_exh_final_labels.csv'), index_col=0)
    NE.index = NE.index.astype(str)
    R['grpr'] = NE['GRPR_quad_M5'].reindex(R.index).values
    R['car'] = R['car_pos'].astype(bool)
    return R[R['sample'].isin(SECTIONS)].copy()

def main():
    R = load()
    rows, nrows, spot_rows = ([], [], [])
    for s in SECTIONS:
        d = R[R['sample'] == s].copy()
        xy = d[['x', 'y']].values
        nbr = NearestNeighbors(n_neighbors=KNN + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
        ccr1 = d['ccr1mye'].values > 0
        fapc = d['caf'].values > 0
        dp = ccr1 & fapc
        cafh = d['caf'].values >= np.quantile(d['caf'].values, 0.75)
        ccr1h = d['ccr1mye'].values >= np.quantile(d['ccr1mye'].values, 0.75)
        nb_ccr1 = ccr1[nbr].mean(1)
        nb_fap = fapc[nbr].mean(1)
        nb_dp = dp[nbr].mean(1)
        nb_either = (ccr1 | fapc)[nbr].mean(1)
        cnt_ccr1 = ccr1[nbr].sum(1)
        cnt_fap = fapc[nbr].sum(1)
        n1_ccr1, n1_fap = (cnt_ccr1 >= 1, cnt_fap >= 1)
        n2_ccr1, n2_fap = (cnt_ccr1 >= 2, cnt_fap >= 2)
        met = {'self_dp': dp.astype(float), 'self_dp_high': (ccr1h & cafh).astype(float), 'both_present': (n1_ccr1 & n1_fap).astype(float), 'both2': (n2_ccr1 & n2_fap).astype(float), 'min_frac': np.minimum(nb_ccr1, nb_fap), 'dp_nb': nb_dp, 'either_nb': nb_either, 'ccr1_tam_nb': nb_ccr1, 'fap_caf_nb': nb_fap}
        desc = {'self_dp': 'spot itself double positive (CCR1+ TAM & FAP+ CAF)', 'self_dp_high': 'spot itself double high (both modules >= Q75)', 'both_present': '>=1 CCR1+ TAM AND >=1 FAP+ CAF neighbour', 'both2': '>=2 CCR1+ TAM AND >=2 FAP+ CAF neighbours', 'min_frac': 'min(CCR1+ TAM neighbour fraction, FAP+ CAF neighbour fraction)', 'dp_nb': 'neighbours that are double positive themselves', 'either_nb': 'neighbours that are CCR1+ TAM or FAP+ CAF', 'ccr1_tam_nb': 'CCR1+ TAM neighbour fraction', 'fap_caf_nb': 'FAP+ CAF neighbour fraction'}
        binary = {'self_dp', 'self_dp_high', 'both_present', 'both2'}
        lab = d['grpr'].values
        car = d['car'].values
        keep = car & np.isin(lab, ['GR', 'PR'])
        zone4 = np.where(d['region_niche'].values == 'Perivascular', 'Perivascular', d['region_tumor'].values)
        sub = d[keep]
        print(f"\n=== {s} ({GROUP[s]}): labelled CAR+ GR {int((lab[keep] == 'GR').sum())} / PR {int((lab[keep] == 'PR').sum())} ===")
        print(f'  section marker rate: CCR1+ TAM {ccr1.mean():.1%}, FAP+ CAF {fapc.mean():.1%}, double positive {dp.mean():.1%}, either {(ccr1 | fapc).mean():.1%}')
        spot_rows.append(pd.DataFrame({**{k: met[k][keep] for k in met}, 'sample': s, 'grpr': lab[keep], 'zone4': zone4[keep]}, index=sub.index))
        rng = np.random.default_rng(0)

        def analyse(stratum, smask):
            sel = smask
            gg, pp = (lab[keep][sel] == 'GR', lab[keep][sel] == 'PR')
            ngg, npp = (int(gg.sum()), int(pp.sum()))
            if ngg < 3 or npp < 3:
                return
            for key in met:
                v = met[key][keep][sel]
                if key in binary:
                    a = int((gg & (v > 0)).sum())
                    b = ngg - a
                    c = int((pp & (v > 0)).sum())
                    e = npp - c
                    orr, pf = fisher_exact([[a, b], [c, e]]) if a + b and c + e else (np.nan, np.nan)
                    rows.append(dict(sample=s, group=GROUP[s], stratum=stratum, metric=key, description=desc[key], n_GR=ngg, n_PR=npp, mean_GR=100 * a / max(a + b, 1), mean_PR=100 * c / max(c + e, 1), diff=100 * a / max(a + b, 1) - 100 * c / max(c + e, 1), mwu_p=pf, perm_p=np.nan, odds_ratio=orr))
                else:
                    vg, vp = (v[gg], v[pp])
                    u, pu = mannwhitneyu(vg, vp)
                    obs = float(vg.mean() - vp.mean())
                    null = np.empty(N_PERM)
                    for i in range(N_PERM):
                        pl = rng.permutation(len(v))
                        null[i] = v[pl][:ngg].mean() - v[pl][ngg:].mean()
                    rows.append(dict(sample=s, group=GROUP[s], stratum=stratum, metric=key, description=desc[key], n_GR=ngg, n_PR=npp, mean_GR=float(vg.mean()), mean_PR=float(vp.mean()), diff=obs, mwu_p=float(pu), perm_p=float((np.abs(null) >= abs(obs) - 1e-12).mean())))
            if stratum == 'all':
                for key in ('self_dp', 'both_present', 'both2', 'dp_nb', 'min_frac'):
                    r = [x for x in rows if x['sample'] == s and x['stratum'] == 'all' and (x['metric'] == key)][0]
                    print(f"  {key:13s} GR {r['mean_GR']:7.2f} vs PR {r['mean_PR']:7.2f}  diff {r['diff']:+7.2f}  p={r['mwu_p']:.3g}" + (f" (perm {r['perm_p']:.3g})" if np.isfinite(r.get('perm_p', np.nan)) else ''))
        analyse('all', np.ones(keep.sum(), bool))
        for r in REGIONS:
            analyse(r, zone4[keep] == r)
        pi, ni = np.where(np.zeros_like(nbr, bool) == False)
        pi, ni = np.meshgrid(np.arange(nbr.shape[0]), np.arange(nbr.shape[1]), indexing='ij')
        pi, ni = (pi.ravel(), ni.ravel())
        nj = nbr.ravel()
        keep_nb = keep[pi]
        if keep_nb.sum():
            nrows.append(pd.DataFrame({'sample': s, 'parent': d.index.values[pi[keep_nb]], 'parent_grpr': lab[pi[keep_nb]], 'parent_zone4': zone4[pi[keep_nb]], 'nb_ccr1': ccr1[nj[keep_nb]], 'nb_fap': fapc[nj[keep_nb]], 'nb_dp': dp[nj[keep_nb]], 'nb_cafhigh': cafh[nj[keep_nb]]}))
        pd.DataFrame([x for x in rows if x['sample'] == s]).to_csv(os.path.join(TAB, f'TableR14_GRPR_TAM_CAF_simultaneous_{s}.csv'), index=False)
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(TAB, 'TableR15_GRPR_TAM_CAF_simultaneous_summary.csv'), index=False)
    NB = pd.concat(nrows, ignore_index=True)
    NB.to_csv(os.path.join(TAB, 'TableR16_neighbourlevel_TAM_CAF_simultaneous.csv'), index=False)
    print('\n=== neighbour-level rates by parent GR/PR (each of the 6 neighbours = 1 observation) ===')
    lvl = []
    for s in SECTIONS:
        b = NB[NB['sample'] == s]
        if b.empty:
            continue
        out = {'sample': s, 'group': GROUP[s], 'n_neighbour': len(b), 'n_GR_parent_neigh': int((b['parent_grpr'] == 'GR').sum()), 'n_PR_parent_neigh': int((b['parent_grpr'] == 'PR').sum())}
        rng = np.random.default_rng(0)
        for col in ['nb_ccr1', 'nb_fap', 'nb_dp']:
            g = b.loc[b['parent_grpr'] == 'GR', col].values.astype(float)
            p = b.loc[b['parent_grpr'] == 'PR', col].values.astype(float)
            obs = g.mean() - p.mean()
            parents = b['parent_grpr'].values
            groups = b.groupby('parent').size().index.values
            arr = b[col].values.astype(float)
            par = pd.factorize(b['parent'].values)[0]
            plab = pd.Series(b['parent_grpr'].values).groupby(par).first().values
            n_gr_par = int((plab == 'GR').sum())
            n_par = len(plab)
            null = np.empty(N_PERM)
            for i in range(N_PERM):
                perm = rng.permutation(n_par)
                newlab = plab[perm]
                isg = np.zeros(len(arr), bool)
                for j in range(n_par):
                    isg[par == j] = newlab[j] == 'GR'
                null[i] = arr[isg].mean() - arr[~isg].mean()
            out[f'rate_{col}_GR'] = g.mean() * 100
            out[f'rate_{col}_PR'] = p.mean() * 100
            out[f'diff_{col}'] = obs * 100
            out[f'perm_p_{col}'] = float((np.abs(null) >= abs(obs) - 1e-12).mean())
            a_, b_ = (int(g.sum()), len(g) - int(g.sum()))
            c_, e_ = (int(p.sum()), len(p) - int(p.sum()))
            orr, pf = fisher_exact([[a_, b_], [c_, e_]]) if a_ + b_ and c_ + e_ else (np.nan, np.nan)
            out[f'OR_{col}'] = orr
            out[f'fisher_p_{col}'] = pf
        lvl.append(out)
        L = lvl[-1]
        print(f"  {s:4s} CCR1 {L['rate_nb_ccr1_GR']:5.1f}->{L['rate_nb_ccr1_PR']:5.1f}% (p={L['perm_p_nb_ccr1']:.3g}) | FAP {L['rate_nb_fap_GR']:5.1f}->{L['rate_nb_fap_PR']:5.1f}% (p={L['perm_p_nb_fap']:.3g}) | BOTH {L['rate_nb_dp_GR']:5.1f}->{L['rate_nb_dp_PR']:5.1f}% (p={L['perm_p_nb_dp']:.3g})")
    LV = pd.DataFrame(lvl)
    LV.to_csv(os.path.join(TAB, 'TableR17_neighbourlevel_rates_by_section.csv'), index=False)
    S = pd.concat(spot_rows)
    S.to_csv(os.path.join(TAB, 'TableR18_spotlevel_TAM_CAF_simultaneous.csv'))
    print(f'\nwrote TableR14 (per section, 6 files) / R15 summary / R16 neighbour-level detail / R17 neighbour-level rates / R18 spot-level ({len(S)} spots)')
if __name__ == '__main__':
    main()
