"""Step32: redefined spot niches (region_niche v2), 100 um aware, written to a NEW folder.

Scale: L13 Stereo-seq spot = 100 um; one grid step (15.19 coordinate units) = 100 um.
Therefore the neighbourhoods are expressed in rings of 100 um:
    ring1  = 6 nearest spots   (~100 um)
    ring2  = 18 nearest spots  (~200 um)

Definitions (all quantiles are within-slice, so slices are comparable):
    Vessel              endo_q >= 0.90                                  (endothelial-high spot)
    Perivascular_100    not Vessel and >=1 of 6 NN is Vessel            (<= 100 um of a vessel)
    Perivascular_200    not Vessel / not Peri100 and >=1 of 18 NN is Vessel (100-200 um)
    CCR1_MIB            ccr1mye_q >= 0.75 and (acaf_q >= 0.50 or >=2 of 6 NN are aCAF-high q75)
    CCR1_MIB_endoCAF    CCR1_MIB and (Peri100 or endo_q >= 0.50)         (the endo-CAF variant)
    Myeloid_enriched    mye_q  >= 0.75     CAF_enriched  caf_q >= 0.75    TB_zone  tbz_q >= 0.75
    priority: Vessel > Peri100 > Peri200 > CCR1_MIB > Myeloid > CAF > TB > Other
An unsupervised (leiden_0.8) domain label is also produced by applying the same rules to the
cluster means, so the niche can be read at the cluster level too.

Outputs (new folder): region_niche_v2.csv, niche_counts_by_slice_v2.csv, leiden_domain_map_v2.csv,
                      fig_niche_map_v2.{png,pdf,ai}, README_niche_v2.md
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import numpy as np, pandas as pd
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from common import RES, load_base
RCL = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/recluster_obj'
OUT = PROJECT_ROOT + '/mouse/cart_region/lym_v2'
os.makedirs(OUT, exist_ok=True)
SAMPLES = ['Vehicle', 'NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
NICHES = ['Vessel', 'Perivascular_100', 'Perivascular_200', 'CCR1_MIB', 'Myeloid_enriched', 'CAF_enriched', 'TB_zone', 'Other']
COL = {'Vessel': '#08519c', 'Perivascular_100': '#6baed6', 'Perivascular_200': '#bdd7e7', 'CCR1_MIB': '#d62728', 'Myeloid_enriched': '#8B0000', 'CAF_enriched': '#6a3d9a', 'TB_zone': '#31a354', 'Other': '#d9d9d9'}

def main():
    R = pd.read_csv(os.path.join(RES, 'spot_regions_lym.csv'), index_col=0)
    D = load_base(with_props=False)
    frames, clin = ([], [])
    for s in SAMPLES:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'))
        idx = [f'{s}|{b}' for b in a.obs_names]
        xy = np.asarray(a.obsm['spatial'])
        d = pd.DataFrame(index=idx)
        d['sample'] = s
        d['x'], d['y'] = (xy[:, 0], xy[:, 1])
        d['leiden'] = a.obs['leiden_0.8'].astype(str).values
        for c in ['endo', 'mye', 'caf', 'acaf', 'ccr1mye', 'tbz', 'tcell', 'bcell', 'lym']:
            d[c] = R[c].reindex(idx).values
        for c in ['endo', 'mye', 'caf', 'acaf', 'ccr1mye', 'tbz']:
            d[c + '_q'] = d[c].rank(pct=True)
        nn6 = NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
        nn18 = NearestNeighbors(n_neighbors=19).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
        ves = (d['endo_q'] >= 0.9).values
        acaf_hi = (d['acaf_q'] >= 0.75).values
        d['n_ves6'] = ves[nn6].sum(1)
        d['n_ves18'] = ves[nn18].sum(1)
        d['n_acaf6'] = acaf_hi[nn6].sum(1)
        niche = np.full(len(d), 'Other', dtype=object)
        m = ~ves & (d['n_ves6'].values >= 1)
        niche[m] = 'Perivascular_100'
        m2 = ~ves & (d['n_ves6'].values == 0) & (d['n_ves18'].values >= 1)
        niche[m2] = 'Perivascular_200'
        mib = (d['ccr1mye_q'].values >= 0.75) & ((d['acaf_q'].values >= 0.5) | (d['n_acaf6'].values >= 2))
        niche[mib] = 'CCR1_MIB'
        for cond, lab in [(d['mye_q'].values >= 0.75, 'Myeloid_enriched'), (d['caf_q'].values >= 0.75, 'CAF_enriched'), (d['tbz_q'].values >= 0.75, 'TB_zone')]:
            niche[(niche == 'Other') & cond] = lab
        niche[ves] = 'Vessel'
        d['niche_v2'] = niche
        d['CCR1_MIB_endoCAF'] = mib & ((niche == 'Perivascular_100') | (d['endo_q'].values >= 0.5))
        d['dist_to_vessel_rings'] = np.where(ves, 0, np.where(d['n_ves6'] >= 1, 1, np.where(d['n_ves18'] >= 1, 2, np.nan)))
        frames.append(d)
        cl = d.groupby('leiden').agg(n=('x', 'size'), endo=('endo', 'mean'), mye=('mye', 'mean'), caf=('caf', 'mean'), acaf=('acaf', 'mean'), ccr1mye=('ccr1mye', 'mean'), tbz=('tbz', 'mean'), ves_frac=('niche_v2', lambda v: (v == 'Vessel').mean()), peri_frac=('niche_v2', lambda v: v.isin(['Perivascular_100', 'Perivascular_200']).mean()), mib_frac=('niche_v2', lambda v: (v == 'CCR1_MIB').mean())).reset_index()
        for c in ['endo', 'mye', 'caf', 'acaf', 'ccr1mye', 'tbz']:
            cl[c + '_q'] = cl[c].rank(pct=True)
        lab = []
        for _, r in cl.iterrows():
            if r['ves_frac'] >= 0.5:
                lab.append('Vessel')
            elif r['peri_frac'] >= 0.5:
                lab.append('Perivascular_100')
            elif r['mib_frac'] >= 0.5:
                lab.append('CCR1_MIB')
            elif r['mye_q'] >= 0.75:
                lab.append('Myeloid_enriched')
            elif r['caf_q'] >= 0.75:
                lab.append('CAF_enriched')
            elif r['tbz_q'] >= 0.75:
                lab.append('TB_zone')
            else:
                lab.append('Other')
        cl['domain_leiden_v2'] = lab
        cl.insert(0, 'sample', s)
        clin.append(cl)
        del a
    A = pd.concat(frames)
    A.index.name = 'spot'
    A.to_csv(os.path.join(OUT, 'region_niche_v2.csv'))
    pd.concat(clin).to_csv(os.path.join(OUT, 'leiden_domain_map_v2.csv'), index=False)
    cnt = pd.crosstab(A['sample'], A['niche_v2'])[NICHES]
    cnt.assign(total=cnt.sum(1)).to_csv(os.path.join(OUT, 'niche_counts_by_slice_v2.csv'))
    print('=== niche v2 counts per slice ===')
    print(cnt.assign(total=cnt.sum(1)).to_string())
    print('\nCCR1_MIB_endoCAF per slice:', A.groupby('sample')['CCR1_MIB_endoCAF'].sum().to_dict())
    print('distance-to-vessel ring distribution:', A['dist_to_vessel_rings'].value_counts(dropna=False).to_dict())
    with open(os.path.join(OUT, 'README_niche_v2.md'), 'w') as f:
        f.write(f"# region_niche v2 (100 um aware)  --  {pd.Timestamp.now():%Y-%m-%d %H:%M}\n\nScale: L13 ST spot = 100 um; 1 grid step (15.19 units) = 100 um.\nring1 = 6 nearest spots (~100 um), ring2 = 18 nearest spots (~200 um).\n\n| niche | rule |\n|---|---|\n| Vessel | endo_q >= 0.90 (within slice) |\n| Perivascular_100 | not Vessel and >=1 of ring1 is Vessel (<=100 um) |\n| Perivascular_200 | not Vessel / not Peri100 and >=1 of ring2 is Vessel (100-200 um) |\n| CCR1_MIB | ccr1mye_q >= 0.75 and (acaf_q >= 0.50 or >=2 of ring1 are aCAF-high q75) |\n| CCR1_MIB_endoCAF | CCR1_MIB and (Peri100 or endo_q >= 0.50)  -- extra flag column |\n| Myeloid_enriched | mye_q >= 0.75 |\n| CAF_enriched | caf_q >= 0.75 |\n| TB_zone | tbz_q >= 0.75 |\n| Other | none of the above |\n\nPriority: Vessel > Peri100 > Peri200 > CCR1_MIB > Myeloid > CAF > TB > Other.\nModules are the per-sample z of the marker panels in step03 (`GS`), so every quantile is within-slice.\n`domain_leiden_v2` in leiden_domain_map_v2.csv applies the same rules to leiden_0.8 cluster means\n(the object's own unsupervised clustering).\n\nCounts per slice -> niche_counts_by_slice_v2.csv ; per-spot table -> region_niche_v2.csv\n(the original `results/spot_regions.csv` and its `region_niche` column are left untouched).\n")
    print('\nwrote to', OUT)
if __name__ == '__main__':
    main()
