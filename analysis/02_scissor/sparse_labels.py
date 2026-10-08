"""Step53b: sparse high-confidence Scissor labels.

The alpha=0.001 run did NOT give sparser labels (it selected 96% of spots, more than
alpha=0.005 at 92%, because Scissor's alpha is the elastic-net mixing parameter: larger
alpha = sparser). Sparse high-confidence labels are therefore produced by thresholding the
coefficient magnitude per sample: keep the top 25% / top 10% by |coef| as labelled spots
(DR/NDR by sign) and set the rest to 0.
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, glob, sys
import numpy as np, pandas as pd
OUT = PROJECT_ROOT + '/mouse/cart_region/scissor_pooled_noRelapse'
SAMP = ['Vehicle', 'NR1', 'NR2', 'NR3', 'R1', 'R2', 'R3']
TAG = sys.argv[1] if len(sys.argv) > 1 else 'M5a001'

def main():
    rows, keep = ([], {})
    for q, name in [(0.75, 'sparse25'), (0.9, 'sparse10')]:
        frames = []
        for s in SAMP:
            f = os.path.join(OUT, 'labels', f'{s}.{TAG}.scissor_labels.csv')
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            thr = d.coef.abs().quantile(q)
            d['label_' + name] = np.where(d.coef.abs() >= thr, d.label, '0')
            frames.append(d[['spot', 'coef', 'label', 'label_' + name]].assign(sample=s))
            keep.setdefault(name, []).append(dict(sample=s, thr=float(thr), n_lab=int((d.label != '0').sum()), n_kept=int((d['label_' + name] != '0').sum()), NDR=int((d['label_' + name] == 'NDR').sum()), DR=int((d['label_' + name] == 'DR').sum())))
        if frames:
            A = pd.concat(frames, ignore_index=True)
            A.to_csv(os.path.join(OUT, f'scissor_labels_{TAG}_{name}.csv'), index=False)
    K = pd.concat([pd.DataFrame(v).assign(scheme=k) for k, v in keep.items()], ignore_index=True)
    K.to_csv(os.path.join(OUT, f'sparse_label_summary_{TAG}.csv'), index=False)
    print(K.round(3).to_string(index=False))
if __name__ == '__main__':
    main()
