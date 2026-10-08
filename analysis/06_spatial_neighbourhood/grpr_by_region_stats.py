"""Step04 (nr1r1_analysis_M5/region): GR vs PR of the CAR+ spots inside every region - all 6 sections.

Question: within a given region, are the CAR+ spots biased towards GR (memory-functional) or PR
(exhausted)?  Labels = `GRPR_quad_M5` (GR = MEMORY_z>0 & EXHAUSTED_z<0); region = the 4-class scheme
(Perivascular > tumor zone).

Outputs
    TableR6_GRvsPR_by_region_<S>.csv        one table per section (NR1, NR2, NR3, R1, R2, R3)
        per region: n_CARpos, n_labelled, n_GR, n_PR, GR_pct, OR + Fisher p (region vs rest of the
        section), q = BH over the 4 regions, binomial p vs 50:50, direction, plus the
        region x GR/PR chi-square of the section
    TableR6_GRvsPR_by_region_all_sections.csv   all rows stacked
    TableR7_GRvsPR_section_summary.csv          one row per section
    TableR8_GRvsPR_region_matrix.csv            region x section matrix of GR % (and counts)
    TableR9_GRvsPR_region_pooled.csv            cross-section comparison per region:
        pooled GR %, Mantel-Haenszel pooled OR + CMH p (stratified by section), homogeneity p of the
        ORs across sections, number of sections with OR>1, and the NR-pooled vs R-pooled Fisher test
"""
import json
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_define_stats import RES, REGIONS, TAB, WD
SECTIONS = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
GROUP = {'NR1': 'NR', 'NR2': 'NR', 'NR3': 'NR', 'R1': 'R', 'R2': 'R', 'R3': 'R'}

def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)

def load(samples):
    R = pd.read_csv(os.path.join(RES, 'spot_regions.csv'), index_col=0)
    R['sample'] = [i.split('|')[0] for i in R.index]
    R['car'] = R['car_pos'].astype(bool)
    R['zone4'] = np.where(R['region_niche'].eq('Perivascular'), 'Perivascular', R['region_tumor'])
    NE = pd.read_csv(os.path.join(WD, 'grpr_mem_exh', 'GRPR_mem_exh_final_labels.csv'), index_col=0)
    NE.index = NE.index.astype(str)
    R['grpr'] = NE['GRPR_quad_M5'].reindex(R.index).values
    return R[R['sample'].isin(samples)].copy()

def main():
    os.makedirs(TAB, exist_ok=True)
    D = load(SECTIONS)
    all_rows, summary = ([], [])
    for s in SECTIONS:
        d = D[(D['sample'] == s) & D['car']]
        lab = d[d['grpr'].isin(['GR', 'PR'])]
        n_gr, n_pr = (int((lab['grpr'] == 'GR').sum()), int((lab['grpr'] == 'PR').sum()))
        print(f'\n=== {s} ({GROUP[s]}): CAR+ {len(d)} | labelled GR {n_gr} / PR {n_pr} ({100 * n_gr / max(n_gr + n_pr, 1):.1f}% GR) ===')
        rows = []
        for r in REGIONS:
            a = int(((lab['grpr'] == 'GR') & lab['zone4'].eq(r)).sum())
            b = int(((lab['grpr'] == 'PR') & lab['zone4'].eq(r)).sum())
            c, e = (n_gr - a, n_pr - b)
            orr, p = fisher_exact([[a, b], [c, e]]) if a + b and c + e else (np.nan, np.nan)
            pb = binomtest(a, a + b, 0.5).pvalue if a + b else np.nan
            rows.append(dict(sample=s, group=GROUP[s], region=r, n_CARpos=int((d['zone4'] == r).sum()), n_labelled=a + b, n_GR=a, n_PR=b, GR_pct=100 * a / max(a + b, 1), n_GR_outside=c, n_PR_outside=e, odds_ratio=orr, p_fisher=p, p_binom_50=pb))
        T = pd.DataFrame(rows)
        T['q_BH_within_section'] = bh(T['p_fisher'].fillna(1).values)
        T['direction'] = np.where(T['p_fisher'] < 0.05, np.where(T['odds_ratio'] > 1, 'GR-enriched', 'PR-enriched'), 'n.s.')
        ct = np.array([[int(((lab['grpr'] == 'GR') & lab['zone4'].eq(r)).sum()), int(((lab['grpr'] == 'PR') & lab['zone4'].eq(r)).sum())] for r in REGIONS])
        ok = ct.sum(1) > 0
        chi2, chip = (np.nan, np.nan)
        if ok.sum() >= 2:
            chi2, chip = chi2_contingency(ct[ok])[:2]
        T['chi2_region_x_GRPR'], T['chi2_p'] = (chi2, chip)
        T.to_csv(os.path.join(TAB, f'TableR6_GRvsPR_by_region_{s}.csv'), index=False)
        all_rows.append(T)
        summary.append(dict(sample=s, group=GROUP[s], n_CARpos=len(d), n_labelled=n_gr + n_pr, n_GR=n_gr, n_PR=n_pr, GR_pct=100 * n_gr / max(n_gr + n_pr, 1), chi2=chi2, chi2_p=chip, GR_pct_min=float(T['GR_pct'].min()), GR_pct_max=float(T['GR_pct'].max()), n_region_p05=int((T['p_fisher'] < 0.05).sum()), n_region_q05=int((T['q_BH_within_section'] < 0.05).sum())))
        pd.set_option('display.width', 220)
        print(T[['region', 'n_CARpos', 'n_labelled', 'n_GR', 'n_PR', 'GR_pct', 'odds_ratio', 'p_fisher', 'q_BH_within_section', 'direction']].round(4).to_string(index=False))
        if np.isfinite(chi2):
            print(f'  region x GR/PR chi2 = {chi2:.2f}, p = {chip:.3g}')
    ALL = pd.concat(all_rows, ignore_index=True)
    ALL.to_csv(os.path.join(TAB, 'TableR6_GRvsPR_by_region_all_sections.csv'), index=False)
    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(TAB, 'TableR7_GRvsPR_section_summary.csv'), index=False)
    mat = []
    for r in REGIONS:
        row = {'region': r}
        for s in SECTIONS:
            e = ALL[(ALL['sample'] == s) & (ALL['region'] == r)].iloc[0]
            row[f'GRpct_{s}'] = e['GR_pct']
            row[f'nGR_{s}'] = int(e['n_GR'])
            row[f'nPR_{s}'] = int(e['n_PR'])
        row['pooled_n_GR'] = int(ALL[ALL['region'] == r]['n_GR'].sum())
        row['pooled_n_PR'] = int(ALL[ALL['region'] == r]['n_PR'].sum())
        row['pooled_GR_pct'] = 100 * row['pooled_n_GR'] / max(row['pooled_n_GR'] + row['pooled_n_PR'], 1)
        mat.append(row)
    M = pd.DataFrame(mat)
    M.to_csv(os.path.join(TAB, 'TableR8_GRvsPR_region_matrix.csv'), index=False)
    wide = []
    for s in SECTIONS:
        e = ALL[ALL['sample'] == s]
        row = {'sample': s, 'group': GROUP[s], 'n_CARpos': int(e['n_CARpos'].sum()), 'n_labelled': int(e['n_labelled'].sum()), 'n_GR': int(e['n_GR'].sum()), 'n_PR': int(e['n_PR'].sum()), 'GR_pct': 100 * e['n_GR'].sum() / max(e['n_labelled'].sum(), 1), 'chi2_region_x_GRPR': float(e['chi2_region_x_GRPR'].iloc[0]), 'chi2_p': float(e['chi2_p'].iloc[0])}
        for r in REGIONS:
            x = e[e['region'] == r].iloc[0]
            row[f'{r}_nCARpos'] = int(x['n_CARpos'])
            row[f'{r}_nGR'] = int(x['n_GR'])
            row[f'{r}_nPR'] = int(x['n_PR'])
            row[f'{r}_GRpct'] = float(x['GR_pct'])
            row[f'{r}_OR'] = float(x['odds_ratio']) if np.isfinite(x['odds_ratio']) else np.nan
            row[f'{r}_p'] = float(x['p_fisher']) if np.isfinite(x['p_fisher']) else np.nan
            row[f'{r}_q'] = float(x['q_BH_within_section'])
            row[f'{r}_direction'] = x['direction']
        wide.append(row)
    W = pd.DataFrame(wide)
    W.to_csv(os.path.join(TAB, 'TableR10_GRvsPR_by_region_wide_per_section.csv'), index=False)
    pooled = []
    for r in REGIONS:
        tabs, tot_gr, tot_pr, sec_gr, sec_pr = ([], 0, 0, 0, 0)
        ors = []
        for s in SECTIONS:
            e = ALL[(ALL['sample'] == s) & (ALL['region'] == r)].iloc[0]
            a, b = (int(e['n_GR']), int(e['n_PR']))
            c, d_ = (int(e['n_GR_outside']), int(e['n_PR_outside']))
            tot_gr += a
            tot_pr += b
            if GROUP[s] == 'NR':
                sec_gr += a
                sec_pr += b
            if a + b and c + d_:
                tabs.append([[a, b], [c, d_]])
                if np.isfinite(e['odds_ratio']):
                    ors.append(float(e['odds_ratio']))
        st = StratifiedTable(tabs)
        mh_or = float(st.oddsratio_pooled)
        cmh_p = float(st.test_null_odds().pvalue)
        hom_p = float(st.test_equal_odds().pvalue) if len(tabs) >= 2 else np.nan
        r_nr = fisher_exact([[sec_gr, sec_pr], [tot_gr - sec_gr, tot_pr - sec_pr]]) if sec_gr + sec_pr and tot_gr - sec_gr + tot_pr - sec_pr else (np.nan, np.nan)
        pooled.append(dict(region=r, n_sections=len(tabs), n_GR=tot_gr, n_PR=tot_pr, GR_pct=100 * tot_gr / max(tot_gr + tot_pr, 1), MH_pooled_OR=mh_or, CMH_p=cmh_p, homogeneity_p=hom_p, n_sections_OR_gt1=int(sum((o > 1 for o in ors))), n_sections_p05=int(sum(((ALL[(ALL['region'] == r) & (ALL['sample'] == s)]['p_fisher'] < 0.05).any() for s in SECTIONS))), median_section_OR=float(np.median(ors)) if ors else np.nan, NR_GR=sec_gr, NR_PR=sec_pr, R_GR=tot_gr - sec_gr, R_PR=tot_pr - sec_pr, NR_vs_R_OR=float(r_nr[0]), NR_vs_R_p=float(r_nr[1])))
    P = pd.DataFrame(pooled)
    P['CMH_q_BH'] = bh(P['CMH_p'].fillna(1).values)
    P.to_csv(os.path.join(TAB, 'TableR9_GRvsPR_region_pooled.csv'), index=False)
    json.dump({'sections': SECTIONS, 'regions': REGIONS, 'groups': GROUP, 'label': 'GRPR_quad_M5 (GR = MEMORY_z>0 & EXHAUSTED_z<0)', 'tests': {'per_section': 'Fisher (region GR/PR vs rest), q = BH over 4 regions', 'cross_section': 'Mantel-Haenszel pooled OR + Cochran-Mantel-Haenszel p stratified by section; homogeneity p = equal odds across sections; NR vs R = Fisher on pooled counts'}}, open(os.path.join(TAB, 'GRvsPR_by_region_definitions.json'), 'w'), indent=2)
    print('\n=== per-section summary (all 6) ===')
    print(S.round(2).to_string(index=False))
    print('\n=== region x section GR % (n = labelled CAR+ spots) ===')
    for r in REGIONS:
        cells = '  '.join((f"{ALL[(ALL['sample'] == s) & (ALL['region'] == r)]['GR_pct'].iloc[0]:5.1f}({int(ALL[(ALL['sample'] == s) & (ALL['region'] == r)]['n_GR'].iloc[0])}+{int(ALL[(ALL['sample'] == s) & (ALL['region'] == r)]['n_PR'].iloc[0])})" for s in SECTIONS))
        print(f'  {r:13s} {cells}')
    print('\n=== cross-section comparison per region ===')
    print(P[['region', 'n_sections', 'n_GR', 'n_PR', 'GR_pct', 'MH_pooled_OR', 'CMH_p', 'CMH_q_BH', 'homogeneity_p', 'n_sections_OR_gt1', 'median_section_OR', 'NR_vs_R_OR', 'NR_vs_R_p']].round(4).to_string(index=False))
    print('\nwrote TableR6 (per section, 6 files) / R6_all / R7 / R8 / R9')
if __name__ == '__main__':
    main()
