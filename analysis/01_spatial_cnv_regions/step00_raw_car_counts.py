"""Extract per-spot TRUE total UMI and CAR-transgene counts from raw Stereo-seq level_13 matrices."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, gzip, sys, time
import numpy as np, pandas as pd
OUT = PROJECT_ROOT + '/mouse/cart_region/results'
SAMPLES = {'Vehicle': LEGACY_DATA_ROOT + '/ST_20241110/Control_Level13', 'NR1': LEGACY_DATA_ROOT + '/ST_20241110/NR1_Level13', 'R1': LEGACY_DATA_ROOT + '/ST_20241110/R1_Level13', 'NR2': LEGACY_DATA_ROOT + '/ST_20241110/NR2/05.AllheStat/level_matrix/level_13', 'NR3': LEGACY_DATA_ROOT + '/ST_20241110/NR3/05.AllheStat/level_matrix/level_13', 'R2': LEGACY_DATA_ROOT + '/ST_20241110/R2/05.AllheStat/level_matrix/level_13', 'R3': LEGACY_DATA_ROOT + '/ST_20241110/R3/05.AllheStat/level_matrix/level_13'}
TARGET = ['mCherry-CAR', 'mCherry', 'EGFP']
for sid, d in SAMPLES.items():
    t0 = time.time()
    feat = pd.read_csv(os.path.join(d, 'features.tsv.gz'), sep='\t', header=None)
    bc = pd.read_csv(os.path.join(d, 'barcodes.tsv.gz'), sep='\t', header=None)[0].values
    name2row = {str(g): i + 1 for i, g in enumerate(feat[0].values)}
    rows = {g: name2row.get(g) for g in TARGET}
    inv = {v: k for k, v in rows.items() if v is not None}
    ncol = len(bc)
    total = np.zeros(ncol, dtype=np.int64)
    nnz = np.zeros(ncol, dtype=np.int64)
    target_cnt = {g: np.zeros(ncol, dtype=np.int64) for g in TARGET}
    with gzip.open(os.path.join(d, 'matrix.mtx.gz'), 'rt') as f:
        for line in f:
            if line[0] == '%':
                continue
            p = line.split()
            if len(p) == 3:
                if p[0] == str(len(feat)) and p[1] == str(ncol):
                    continue
                gi = int(p[0])
                ci = int(p[1]) - 1
                v = int(float(p[2]))
                if 0 <= ci < ncol:
                    total[ci] += v
                    nnz[ci] += 1
                    g = inv.get(gi)
                    if g is not None:
                        target_cnt[g][ci] += v
    df = pd.DataFrame({'barcode': bc, 'umi_total': total, 'n_gene': nnz})
    for g in TARGET:
        df['umi_' + g] = target_cnt[g]
    df['sample'] = sid
    df.to_csv(os.path.join(OUT, f'raw_umi_{sid}.csv'), index=False)
    print(f"{sid}: {len(df)} barcodes, median UMI {int(np.median(total))}, CAR+ spots (raw>0) {int((target_cnt['mCherry-CAR'] > 0).sum())}, {time.time() - t0:.1f}s", flush=True)
print('DONE')
