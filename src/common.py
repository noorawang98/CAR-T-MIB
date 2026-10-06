"""Shared loaders for the CAR / region / GR-PR / scissor analysis."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import numpy as np, pandas as pd
WD = PROJECT_ROOT + '/mouse/cart_region'
RES = os.path.join(WD, 'results')
RCL = PROJECT_ROOT + '/ST_20241110/spatial_input/h5ad/recluster_obj'
RCL = LEGACY_DATA_ROOT + '/ST_20241110/spatial_input/h5ad/recluster_obj'
SAMPLES = ['Vehicle', 'NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
TREATED = ['NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
CT_ORDER = None

def destvi_props():
    """26-column DestVI proportion matrix, index = spot key sample|barcode."""
    global CT_ORDER
    import anndata as ad
    frames, cols = ([], None)
    for s in SAMPLES:
        a = ad.read_h5ad(os.path.join(RCL, f'{s}_sc2st_DestVI_destvi_recluster.h5ad'), backed='r')
        P = a.obsm['proportions']
        if not isinstance(P, pd.DataFrame):
            P = pd.DataFrame(np.asarray(P))
        cols = list(P.columns)
        P.index = [f'{s}|{b}' for b in a.obs_names]
        frames.append(P)
        del a
    D = pd.concat(frames)
    CT_ORDER = cols
    return D

def grpr_labels():
    """Final GR/PR labels (def5: net quartiles cut inside every sample) - single source."""
    return pd.read_csv(os.path.join(RES, 'GRPR_labels_final.csv'), index_col=0)

def load_base(with_props=True):
    M = pd.read_csv(os.path.join(RES, 'spot_master_car.csv'), index_col=0)
    R = pd.read_csv(os.path.join(RES, 'spot_regions.csv'), index_col=0)
    D = pd.concat([M, R[[c for c in R.columns if c not in M.columns]]], axis=1)
    if with_props:
        P = destvi_props()
        D = pd.concat([D, P[[c for c in P.columns if c not in D.columns]]], axis=1)
    return D

def scissor_labels():
    S = pd.read_csv(os.path.join(RES, 'scissor_coefs_long.csv'))
    P = S.pivot_table(index=['sample', 'barcode'], columns='cohort', values='scissor_coef')
    P.index = [f'{s}|{b}' for s, b in P.index]
    P.columns = [f'scissor_{c}' for c in P.columns]
    P['scissor_sum'] = P.sum(1)
    P['scissor_lab'] = np.where(P['scissor_sum'] < 0, 'R', np.where(P['scissor_sum'] > 0, 'NR', '0'))
    P['scissor_n_pos'] = (P[[c for c in P.columns if c.startswith('scissor_') and c not in ('scissor_sum', 'scissor_lab')]] > 0).sum(1)
    return P

def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)
