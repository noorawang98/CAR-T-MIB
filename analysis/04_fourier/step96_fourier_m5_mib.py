"""Step96: Fourier decomposition on the M5 Scissor field, with CCR1+ MIB genes/fields added.

Reuses the step11 pipeline (rasterise -> Hann -> 2D FFT -> 4 radial bands -> band power and
feature x band correlation) but (a) the label field is the M5 pooled coefficient, (b) the
feature list adds the CCR1+ MIB definition genes (ccr1mye panel 9 genes, aCAF panel 8 genes,
per-sample z) plus the MIB niche fields, and (c) every correlation gets a toroidal-shift
permutation p (BH q) that is annotated on the figure."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd, anndata as ad
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage
from scipy.stats import mannwhitneyu
from common import RES, RCL, SAMPLES, bh
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else 'M5'
FIG = os.path.join(RES, 'figs_fourier')
os.makedirs(FIG, exist_ok=True)
OUT = PROJECT_ROOT + f'/mouse/cart_region/figs_fourier_{TAG.lower()}'
os.makedirs(OUT, exist_ok=True)
SCD = PROJECT_ROOT + '/mouse/cart_region/scissor_pooled_noRelapse'
NDR, DR = (['NR1', 'NR2', 'NR3'], ['R1', 'R2', 'R3'])
CCR1_GENES = ['Ccr1', 'Ccl2', 'Ccl7', 'Ccl12', 'Ccr2', 'Cd14', 'Lyz2', 'Csf1r', 'C1qa']
ACAF_GENES = ['Acta2', 'Tagln', 'Myh11', 'Cspg4', 'Rgs5', 'Des', 'Notch3', 'Col5a1']
BANDS = [(16, np.inf, 'B1 >1.6 mm'), (8, 16, 'B2 0.8-1.6 mm'), (4, 8, 'B3 0.4-0.8 mm'), (2, 4, 'B4 0.2-0.4 mm')]
NPERM = 300

def mib_gene_z():
    cache = os.path.join(OUT, 'mib_gene_z.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0)
    frames = []
    for s in NDR + DR:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'))
        idx = {g: i for i, g in enumerate(a.var_names)}
        gs = [g for g in CCR1_GENES + ACAF_GENES if g in idx]
        X = a.X.tocsc()[:, [idx[g] for g in gs]].tocsr().astype(np.float32)
        mu = np.asarray(X.mean(0)).ravel()
        sd = np.sqrt(np.maximum(np.asarray(X.multiply(X).mean(0)).ravel() - mu ** 2, 0))
        sd[sd == 0] = 1
        Z = np.asarray((X - mu) / sd)
        frames.append(pd.DataFrame(Z, index=[f'{s}|{b}' for b in a.obs_names], columns=gs))
        del a, X
    Zf = pd.concat(frames)
    Zf.to_csv(cache)
    return Zf

def to_grid(x, y, v, sp):
    xi = np.round((x - x.min()) / sp).astype(int)
    yi = np.round((y - y.min()) / sp).astype(int)
    G = np.full((yi.max() + 1, xi.max() + 1), np.nan)
    G[yi, xi] = v
    return G

def fill(G):
    m = np.isnan(G)
    if m.any():
        idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
        G = G[tuple(idx)]
    return G

def spec(G):
    G = G - G.mean()
    F = np.fft.fftshift(np.fft.fft2(G * np.hanning(G.shape[0])[:, None] * np.hanning(G.shape[1])[None, :]))
    fy = np.fft.fftshift(np.fft.fftfreq(G.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(G.shape[1]))[None, :]
    return (F, np.sqrt(fy ** 2 + fx ** 2))

def masks(r):
    lam = np.where(r > 0, 1.0 / np.maximum(r, 1e-09), np.inf)
    return [(lam >= lo) & (lam < hi) for lo, hi, _ in BANDS]

def recon(F, ms):
    return [np.real(np.fft.ifft2(np.fft.ifftshift(F * m))) for m in ms]

def main():
    R = pd.read_csv(os.path.join(RES, 'spot_regions_lym.csv'), index_col=0)
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    S = pd.read_csv(os.path.join(SCD, f'scissor_labels_{TAG}_all_samples.csv')).set_index('spot')
    S.index = S.index.astype(str)
    Zg = mib_gene_z()
    D = pd.DataFrame(index=M.index)
    D['sample'] = M['sample']
    D['x'] = R['x'].reindex(D.index)
    D['y'] = R['y'].reindex(D.index)
    FLD = f'scissor_coef_{TAG}'
    D[FLD] = S['coef'].reindex(D.index)
    for c in ['ccr1mye', 'acaf', 'caf', 'mye', 'endo']:
        D[c] = R[c].reindex(D.index)
    D['MIB_flag'] = (R['region_niche'].reindex(D.index) == 'CCR1_MIB').astype(float)
    D['MIB_endoCAF_flag'] = R['mib_perivasc_endo_caf'].reindex(D.index).astype(float)
    Zg = Zg.reindex(D.index)
    for g in Zg.columns:
        D['g_' + g] = Zg[g].values
    FEATS = [('ccr1mye', 'CCR1+ myeloid (module)'), ('MIB_flag', 'CCR1_MIB niche (v1)'), ('MIB_endoCAF_flag', 'MIB perivasc-endoCAF flag'), ('acaf', 'aCAF module'), ('caf', 'CAF module'), ('mye', 'myeloid module'), ('endo', 'endothelial module')] + [('g_' + g, f'MIB gene {g}') for g in CCR1_GENES] + [('g_' + g, f'aCAF gene {g}') for g in ACAF_GENES]
    D = D[D['sample'].isin(SAMPLES)]
    maps, spac = ({}, {})
    for s, g in D.groupby('sample'):
        xy = g[['x', 'y']].values
        sp = float(np.median(NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]))
        spac[s] = sp
        maps[s] = {k: fill(to_grid(xy[:, 0], xy[:, 1], g[k].values.astype(float), sp)) for k, _ in [(FLD, '')] + FEATS}
    print('spacing:', {k: round(v, 2) for k, v in spac.items()})
    pw, rec = ([], {})
    for s, m in maps.items():
        F, r = spec(m[FLD])
        ms = masks(r)
        tot = sum(((np.abs(F) ** 2 * k).sum() for k in ms))
        row = dict(sample=s, group='NDR' if s in NDR else 'DR')
        for i, (_, _, lab) in enumerate(BANDS):
            row[lab] = float((np.abs(F) ** 2 * ms[i]).sum() / tot)
        pw.append(row)
        rec[s] = {k: recon(*spec(m[k])) for k, _ in FEATS}
    PW = pd.DataFrame(pw)
    PW.to_csv(os.path.join(OUT, f'band_power_{TAG}.csv'), index=False)
    print('\n=== 各带功率占比（{TAG} 场；NR vs DR 切片 MWU）===')
    for _, _, lab in BANDS:
        a = PW.loc[PW.group == 'NDR', lab]
        b = PW.loc[PW.group == 'DR', lab]
        print(f'  {lab}: NDR {a.mean():.3f} vs DR {b.mean():.3f} | MWU p={mannwhitneyu(a, b).pvalue:.3f}')
    rng = np.random.default_rng(0)
    rows = []
    for fk, flab in FEATS:
        for bi, (_, _, bl) in enumerate(BANDS):
            cs, nul = ([], [])
            for s in SAMPLES:
                if s == 'Vehicle':
                    continue
                sc = recon(*spec(maps[s][FLD]))[bi]
                fc = rec[s][fk][bi]
                cs.append(np.corrcoef(fc.ravel(), sc.ravel())[0, 1])
                for _ in range(NPERM):
                    sh = (rng.integers(0, fc.shape[0]), rng.integers(0, fc.shape[1]))
                    fc2 = np.roll(np.roll(fc, sh[0], 0), sh[1], 1)
                    nul.append(np.corrcoef(fc2.ravel(), sc.ravel())[0, 1])
            obs = float(np.nanmean(cs))
            nl = np.array(nul)
            p = float((np.sum(np.abs(nl) >= abs(obs)) + 1) / (len(nl) + 1))
            rows.append(dict(feature=flab, band=bl, mean_r=obs, p_perm=p, r_NDR=float(np.nanmean(cs[:3])), r_DR=float(np.nanmean(cs[3:]))))
    C = pd.DataFrame(rows)
    C['q'] = bh(C.p_perm.fillna(1))
    C['sig'] = np.where(C.q < 0.001, '***', np.where(C.q < 0.01, '**', np.where(C.q < 0.05, '*', '')))
    C.to_csv(os.path.join(OUT, f'feature_band_correlation_{TAG}.csv'), index=False)
    show = C[C.feature.str.contains('MIB|CCR1|aCAF|CAF')].pivot_table(index='feature', columns='band', values='mean_r')
    print('\n=== MIB/CAF 相关字段 × 空间尺度的相关（* q<0.05）===')
    print(C[C.feature.str.contains('MIB|CCR1|aCAF|CAF')].sort_values(['band', 'q'])[['feature', 'band', 'mean_r', 'p_perm', 'q', 'sig']].round(4).to_string(index=False))
    sub = C[C.feature.str.contains('MIB|CCR1|aCAF|CAF')].copy()
    piv = sub.pivot_table(index='feature', columns='band', values='mean_r')
    sig = sub.pivot_table(index='feature', columns='band', values='sig', aggfunc='first')
    order = [b[2] for b in BANDS]
    piv = piv.reindex(columns=order)
    sig = sig.reindex(columns=order)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
    print('\nwrote', OUT, 'band_power_M5.csv, feature_band_correlation_M5.csv, fourier_M5_MIB_bands.*')
if __name__ == '__main__':
    main()
