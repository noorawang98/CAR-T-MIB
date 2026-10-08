"""Step07 (nr1r1_analysis_M5/region): GR vs PR CAR+ spots and their co-localisation with
CCR1+ TAM and FAP+ CAF - per section, nothing pooled.

Marker definitions (project conventions, `results/spot_regions.csv` + `tam_caf_coloc/spot_flags.csv`)
    CCR1+ TAM : `ccr1mye` module (Ccr1/Ccl2/Ccl7/Ccl12/Ccr2/Cd14/Lyz2/Csf1r/C1qa, per-sample centred) > 0
    FAP+ CAF  : `caf` module (Col1a1/Col1a2/Col3a1/Dcn/Pdgfra/Pdgfrb/Fap/Postn/Lum/Col6a1) > 0;
                `CAF_high` = caf >= within-section Q75 (the collagen/FAP-high variant)
                NOTE: at this resolution Fap alone is detected in only 1.9-5.0% of spots, so the
                FAP-associated CAF panel is used; the raw Fap detection rate is reported as well.
    CCR1_MIB  : the project's CCR1+ myeloid x activated-CAF niche (niche_v2 == "CCR1_MIB")

Per section, for the GR- and PR-labelled CAR+ spots:
    own     : ccr1mye z, caf z, acaf z, Fap detection rate
    neighbour co-localisation (k = 6 nearest neighbours, as in step03):
              fraction of neighbours that are CCR1+ TAM / FAP+ CAF / CAF_high, and the
              ">=2 marker neighbours" binary flag
    tests   : Mann-Whitney (GR vs PR) for the continuous metrics, Fisher for the binaries,
              and a within-section label permutation (N_PERM) for the GR-PR differences
              (the labels are permuted among the labelled CAR+ spots, positions kept fixed).

Outputs: tables/TableR11_GRPR_TAM_CAF_coloc_<S>.csv (one per section, 6 files),
         tables/TableR12_GRPR_TAM_CAF_coloc_summary.csv (wide, one row per section x metric)
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
KNN = 6
N_PERM = 2000
REGIONS = ['Tumor_core', 'Peritumor', 'Extratumor', 'Perivascular']
METRICS = [('ccr1_tam_nb', 'fraction of neighbours that are CCR1+ TAM'), ('fap_caf_nb', 'fraction of neighbours that are FAP+ CAF'), ('cafhigh_nb', 'fraction of neighbours that are CAF-high (caf Q75)'), ('ccr1mye_z', 'own CCR1+ TAM module z'), ('caf_z', 'own FAP-CAF module z'), ('acaf_z', 'own activated-CAF module z')]

def load():
    R = pd.read_csv(os.path.join(RES, 'spot_regions.csv'), index_col=0)
    R['sample'] = [i.split('|')[0] for i in R.index]
    NE = pd.read_csv(os.path.join(WD, 'grpr_mem_exh', 'GRPR_mem_exh_final_labels.csv'), index_col=0)
    NE.index = NE.index.astype(str)
    R['grpr'] = NE['GRPR_quad_M5'].reindex(R.index).values
    R['car'] = R['car_pos'].astype(bool)
    return R[R['sample'].isin(SECTIONS)].copy()

def fap_expression(section):
    """Fap counts for every spot of one section (only that gene is read)."""
    import anndata as ad
    a = ad.read_h5ad(os.path.join(RCL, f'{section}_sc2st_DestVI_destvi_recluster.h5ad'), backed='r')
    names = np.array([f'{section}|{b}' for b in a.obs_names])
    out = None
    if 'Fap' in a.var_names:
        X = a.layers['counts'][:, list(a.var_names).index('Fap')]
        X = np.asarray(X.todense()).ravel() if hasattr(X, 'todense') else np.asarray(X).ravel()
        out = pd.Series(X, index=names, name='Fap')
    del a
    return out

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    R = load()
    rows, wide, spot_rows = ([], [], [])
    for s in SECTIONS:
        d = R[R['sample'] == s].copy()
        xy = d[['x', 'y']].values
        nn = NearestNeighbors(n_neighbors=KNN + 1).fit(xy)
        nbr = nn.kneighbors(xy, return_distance=False)[:, 1:]
        ccr1 = d['ccr1mye'].values > 0
        fapc = d['caf'].values > 0
        cafh = d['caf'].values >= np.quantile(d['caf'].values, 0.75)
        lab = d['grpr'].values
        car = d['car'].values
        keep = car & np.isin(lab, ['GR', 'PR'])
        nb_ccr1 = ccr1[nbr].mean(1)
        nb_fap = fapc[nbr].mean(1)
        nb_cafh = cafh[nbr].mean(1)
        n2_ccr1 = ccr1[nbr].sum(1) >= 2
        n2_fap = fapc[nbr].sum(1) >= 2
        fap = fap_expression(s)
        d['Fap'] = fap.reindex(d.index).values if fap is not None else np.nan
        zone4 = np.where(d['region_niche'].values == 'Perivascular', 'Perivascular', d['region_tumor'].values)
        spot_rows.append(pd.DataFrame({'sample': s, 'grpr': lab[keep], 'zone4': zone4[keep], 'ccr1_tam_nb': nb_ccr1[keep], 'fap_caf_nb': nb_fap[keep], 'cafhigh_nb': nb_cafh[keep], 'ccr1mye_z': d['ccr1mye'].values[keep], 'caf_z': d['caf'].values[keep], 'acaf_z': d['acaf'].values[keep], 'n2_ccr1_nb': n2_ccr1[keep], 'n2_fap_nb': n2_fap[keep], 'in_CCR1_MIB': d['region_niche'].eq('CCR1_MIB').values[keep], 'Fap_pos': (d['Fap'].values > 0)[keep]}, index=d.index[keep]))
        met = {'ccr1_tam_nb': nb_ccr1, 'fap_caf_nb': nb_fap, 'cafhigh_nb': nb_cafh, 'ccr1mye_z': d['ccr1mye'].values, 'caf_z': d['caf'].values, 'acaf_z': d['acaf'].values}
        g, p = (lab[keep] == 'GR', lab[keep] == 'PR')
        n_g, n_p = (int(g.sum()), int(p.sum()))
        print(f'\n=== {s} ({GROUP[s]}): labelled CAR+ GR {n_g} / PR {n_p} ===')
        n_ccr1 = int(ccr1.sum())
        n_fap = int(fapc.sum())
        print(f"  marker spots in section: CCR1+ TAM {n_ccr1} ({n_ccr1 / len(d):.1%}), FAP+ CAF {n_fap} ({n_fap / len(d):.1%}), CAF-high {int(cafh.sum())} | Fap detected {100 * np.nanmean(d['Fap'].values > 0):.1f}%")
        rng = np.random.default_rng(0)
        binary_masks = {'n2_ccr1_nb': n2_ccr1, 'n2_fap_nb': n2_fap, 'in_CCR1_MIB': d['region_niche'].eq('CCR1_MIB').values}
        binaries = [('n2_ccr1_nb', '>=2 CCR1+ TAM neighbours'), ('n2_fap_nb', '>=2 FAP+ CAF neighbours'), ('in_CCR1_MIB', 'CCR1_MIB niche')]

        def analyse(stratum, smask):
            sel = keep & smask
            gg, pp = (lab[sel] == 'GR', lab[sel] == 'PR')
            n_gg, n_pp = (int(gg.sum()), int(pp.sum()))
            if n_gg < 3 or n_pp < 3:
                return
            for key, desc in METRICS + binaries:
                binary = key in binary_masks
                v = np.asarray(binary_masks[key] if binary else met[key])[sel]
                if binary:
                    a = int((gg & v).sum())
                    b = int((gg & ~v).sum())
                    c = int((pp & v).sum())
                    e = int((pp & ~v).sum())
                    orr, pf = fisher_exact([[a, b], [c, e]]) if a + b and c + e else (np.nan, np.nan)
                    rows.append(dict(sample=s, group=GROUP[s], stratum=stratum, metric=key, description=desc, n_GR=n_gg, n_PR=n_pp, mean_GR=100 * a / max(a + b, 1), mean_PR=100 * c / max(c + e, 1), diff=100 * a / max(a + b, 1) - 100 * c / max(c + e, 1), mwu_p=pf, perm_p=np.nan, odds_ratio=orr, n_GR_yes=a, n_PR_yes=c))
                    continue
                vg, vp = (v[gg], v[pp])
                u, pu = mannwhitneyu(vg, vp)
                obs = float(np.mean(vg) - np.mean(vp))
                null = np.empty(N_PERM)
                for i in range(N_PERM):
                    pl = rng.permutation(len(v))
                    null[i] = v[pl][:n_gg].mean() - v[pl][n_gg:].mean()
                pperm = float((np.abs(null) >= abs(obs) - 1e-12).mean())
                rows.append(dict(sample=s, group=GROUP[s], stratum=stratum, metric=key, description=desc, n_GR=n_gg, n_PR=n_pp, mean_GR=float(np.mean(vg)), mean_PR=float(np.mean(vp)), median_GR=float(np.median(vg)), median_PR=float(np.median(vp)), diff=obs, mwu_p=float(pu), perm_p=pperm, sd_GR=float(np.std(vg, ddof=1)), sd_PR=float(np.std(vp, ddof=1))))
            w = [r for r in rows if r['sample'] == s and r['stratum'] == stratum]
            if stratum == 'all':
                print('   [all CAR+] ' + ' | '.join((f"{r['metric']} {r['diff']:+.3f} (MWU {r['mwu_p']:.3g}" + (f", perm {r['perm_p']:.3g})" if np.isfinite(r.get('perm_p', np.nan)) else ')') for r in w if r['metric'] in ('ccr1_tam_nb', 'fap_caf_nb'))))
            else:
                sig = [f"{r['metric']} {r['diff']:+.2f} (p={r['mwu_p']:.2g})" for r in w if r['metric'] in ('ccr1_tam_nb', 'fap_caf_nb') and r['mwu_p'] < 0.05]
                print(f'   [{stratum}] n GR/PR {n_gg}/{n_pp} ' + ('; '.join(sig) if sig else 'n.s.'))
        for stratum, smask in [('all', np.ones(len(d), bool))] + [(r, zone4 == r) for r in REGIONS]:
            analyse(stratum, smask)
        pd.DataFrame([r for r in rows if r['sample'] == s]).to_csv(os.path.join(TAB, f'TableR11_GRPR_TAM_CAF_coloc_{s}.csv'), index=False)
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(TAB, 'TableR12_GRPR_TAM_CAF_coloc_summary.csv'), index=False)
    S = pd.concat(spot_rows)
    S.to_csv(os.path.join(TAB, 'TableR13_spotlevel_TAM_CAF_coloc.csv'))
    print(f'per-spot metrics: {S.shape[0]} labelled CAR+ spots -> TableR13')
    print(f'\nwrote TableR11 per section (6 files) + TableR12 summary ({len(A)} rows)')
if __name__ == '__main__':
    main()
