"""Step01 (nr1r1_analysis_M5/region): region definition + CAR+ region statistics + representative FOVs.

Region scheme (4 mutually exclusive classes, documented in the README):
    Perivascular  : region_niche == "Perivascular"   (>2 endothelial-high neighbours, not a vessel itself)
    Tumor_core    : region_tumor == "Tumor_core"     (CNV+ tumour spot with >=5 tumour neighbours)
    Peritumor     : region_tumor == "Peritumor"      (tumour border)
    Extratumor    : region_tumor == "Extratumor"
Perivascular takes priority over the tumour zone, so the four classes partition the spots; the
non-exclusive crosstab (perivascular x tumour zone) is written out as well.

Statistics
    * composition of CAR+ spots and of all spots ("background") per section
    * CAR+ enrichment per region: Fisher exact, CAR+ vs background, within section, BH over the 4 regions
    * NR1 vs R1 composition test (chi-square + per-region Fisher)

Representative fields of view (panel I): for every section x region every spot of that region is
tried as the centre of a square window (FOV_PX) and the window is scored by the fold-enrichment of
the region inside it (share in window / share in section); windows within 80% of the best
enrichment are then ranked by the number of CAR+ spots of that region, then by the number of region
spots, then by proximity to the tissue centroid.  The window is kept inside the image.
"""
import json
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
OUT = os.path.join(WD, 'nr1r1_analysis_M5', 'region')
TAB = os.path.join(OUT, 'tables')
SAMPLES = ['NR1', 'R1']
REGIONS = ['Tumor_core', 'Peritumor', 'Extratumor', 'Perivascular']
FOV_PX = 260
HE = {'NR1': LEGACY_DATA_ROOT + '/ST_20241110/NR1_Level13/he_roi_small.png', 'R1': LEGACY_DATA_ROOT + '/ST_20241110/R1_Level13/he_roi_small.png', 'Vehicle': LEGACY_DATA_ROOT + '/ST_20241110/Control_Level13/he_roi_small.png'}

def load():
    R = pd.read_csv(os.path.join(RES, 'spot_regions.csv'), index_col=0)
    R['sample'] = [i.split('|')[0] for i in R.index]
    R['car'] = R['car_pos'].astype(bool)
    R['peri'] = R['region_niche'].eq('Perivascular')
    R['zone4'] = np.where(R['peri'], 'Perivascular', R['region_tumor'])
    return R[R['sample'].isin(SAMPLES)].copy()

def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)

def composition(R):
    rows = []
    for s in SAMPLES:
        d = R[R['sample'] == s]
        for grp, sub in [('CAR+', d[d['car']]), ('background (all spots)', d)]:
            n = len(sub)
            for r in REGIONS:
                k = int((sub['zone4'] == r).sum())
                rows.append(dict(sample=s, group=grp, region=r, n=k, n_total=n, pct=100 * k / max(n, 1)))
    return pd.DataFrame(rows)

def enrichment(R):
    rows = []
    for s in SAMPLES:
        d = R[R['sample'] == s]
        sec = []
        for r in REGIONS:
            a = int((d['car'] & d['zone4'].eq(r)).sum())
            b = int((d['car'] & ~d['zone4'].eq(r)).sum())
            c = int((~d['car'] & d['zone4'].eq(r)).sum())
            e = int((~d['car'] & ~d['zone4'].eq(r)).sum())
            orr, p = fisher_exact([[a, b], [c, e]])
            sec.append(dict(sample=s, region=r, n_carpos=a, n_carpos_total=a + b, pct_carpos=100 * a / max(a + b, 1), n_bg=c, n_bg_total=c + e, pct_bg=100 * c / max(c + e, 1), odds_ratio=orr, p=p))
        q = bh([x['p'] for x in sec])
        for i, x in enumerate(sec):
            x['p_BH_within_section'] = q[i]
        rows += sec
    T = pd.DataFrame(rows)
    T['log2_OR'] = np.log2(T['odds_ratio'].replace(0, np.nan))
    return T

def cross_section(R):
    car = R[R['car']]
    ct = pd.crosstab(car['zone4'], car['sample']).reindex(REGIONS).fillna(0).astype(int)
    chi, p, dof, _ = chi2_contingency(ct.values)
    rows = []
    for r in REGIONS:
        a, b = (int(ct.loc[r, 'NR1']), int(ct.loc[r, 'R1']))
        c = int(ct['NR1'].sum() - a)
        e = int(ct['R1'].sum() - b)
        orr, pf = fisher_exact([[a, c], [b, e]])
        rows.append(dict(region=r, NR1=a, R1=b, pct_NR1=100 * a / int(ct['NR1'].sum()), pct_R1=100 * b / int(ct['R1'].sum()), odds_ratio=orr, p_fisher=pf))
    T = pd.DataFrame(rows)
    T['p_BH'] = bh(T['p_fisher'])
    T.attrs['chi2'] = (chi, p, dof)
    return (T, dict(chi2=float(chi), p=float(p), dof=int(dof), n_NR1=int(ct['NR1'].sum()), n_R1=int(ct['R1'].sum())))

def main():
    os.makedirs(TAB, exist_ok=True)
    R = load()
    print('spots:', len(R), '| CAR+:', int(R['car'].sum()), '|', R.groupby('sample')['car'].sum().to_dict())
    print('zone4 counts (all spots):', R['zone4'].value_counts().to_dict())
    print('zone4 counts (CAR+):', R[R['car']]['zone4'].value_counts().to_dict())
    C = composition(R)
    C.to_csv(os.path.join(TAB, 'TableR1_region_composition.csv'), index=False)
    E = enrichment(R)
    E.to_csv(os.path.join(TAB, 'TableR2_region_CARpos_enrichment.csv'), index=False)
    X, chi = cross_section(R)
    X.to_csv(os.path.join(TAB, 'TableR3_NR1_vs_R1_composition.csv'), index=False)
    ct = pd.crosstab(R['region_niche'], R['zone4'])
    ct.to_csv(os.path.join(TAB, 'TableR5_niche_x_zone_crosstab.csv'))
    R.to_csv(os.path.join(TAB, 'TableR0_spot_region_carpos_NR1_R1.csv'))
    json.dump({'samples': SAMPLES, 'regions': REGIONS, 'fov_px': FOV_PX, 'chi2': chi, 'rule': 'Perivascular > region_tumor (Tumor_core/Peritumor/Extratumor)'}, open(os.path.join(TAB, 'region_definitions.json'), 'w'), indent=2)
    pd.set_option('display.width', 200)
    print('\n=== CAR+ region composition (%) ===')
    print(C[C.group == 'CAR+'].pivot_table(index='region', columns='sample', values=['n', 'pct']).reindex(REGIONS).round(1).to_string())
    print('\n=== CAR+ enrichment vs background (Fisher, BH within section) ===')
    print(E[['sample', 'region', 'n_carpos', 'pct_carpos', 'pct_bg', 'odds_ratio', 'p', 'p_BH_within_section']].round(4).to_string(index=False))
    print(f"\n=== NR1 vs R1 CAR+ composition: chi2={chi['chi2']:.2f} p={chi['p']:.3g} (n {chi['n_NR1']} vs {chi['n_R1']}) ===")
    print(X.round(4).to_string(index=False))
    print('\nwrote tables ->', TAB)
if __name__ == '__main__':
    main()
