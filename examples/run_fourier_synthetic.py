# -*- coding: utf-8 -*-
"""Lightweight, fully runnable example for the released Fourier core.

No project data is needed: two synthetic spatial fields (a "feature" field and a
"Scissor-like" label field) are rasterised, decomposed into the four real bands, and the
feature x band correlation with the toroidal-shift plus-one P and BH q is written out.

Run:  python examples/run_fourier_synthetic.py --outdir examples/output
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fourier_core import BANDS, to_grid, fill, spec, masks, recon, band_corr, toroidal_null, plus_one_p, bh

NPERM = 300          # same value as the real run (step96_fourier_m5_mib.py)
SEED = 0             # same seed as the real run


def synthetic(spacing_mm, size_mm, wavelengths, seed):
    g = np.arange(0, size_mm, spacing_mm)
    X, Y = np.meshgrid(g, g)
    rng = np.random.default_rng(seed)
    feat = sum(np.sin(2 * np.pi * X / w) for w in wavelengths) + 0.05 * rng.standard_normal(X.shape)
    lab = np.sin(2 * np.pi * X / wavelengths[0]) + 0.05 * rng.standard_normal(X.shape)
    return X.ravel(), Y.ravel(), feat.ravel(), lab.ravel(), spacing_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="examples/output")
    ap.add_argument("--sections", type=int, default=6, help="synthetic sections pooled like the real run")
    ap.add_argument("--nperm", type=int, default=NPERM)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    rows, band_power = [], []
    for s in range(args.sections):
        x, y, f, l, sp = synthetic(0.1, 3.0, [1.2, 0.5], seed=s)
        Fg = fill(to_grid(x, y, f, sp))
        Lg = fill(to_grid(x, y, l, sp))
        Ff, rf = spec(Fg)
        Fl, rl = spec(Lg)
        mf, ml = masks(rf), masks(rl)
        tot = sum((np.abs(Ff) ** 2 * m).sum() for m in mf)
        band_power.append(dict(section="syn%d" % (s + 1),
                               **{lab: float((np.abs(Ff) ** 2 * m).sum() / tot) for m, (_, _, lab) in zip(mf, BANDS)}))
        comp_f, comp_l = recon(Ff, mf), recon(Fl, ml)
        for bi, (_, _, lab) in enumerate(BANDS):
            obs = band_corr(comp_f[bi], comp_l[bi])
            null = toroidal_null(comp_f[bi], comp_l[bi], n_perm=args.nperm, rng=rng)
            rows.append(dict(section="syn%d" % (s + 1), band=lab, r=obs, p_perm=plus_one_p(obs, null)))

    C = pd.DataFrame(rows)
    C["q"] = bh(C.p_perm.fillna(1))
    C.to_csv(os.path.join(args.outdir, "feature_band_correlation_synthetic.csv"), index=False)
    pd.DataFrame(band_power).to_csv(os.path.join(args.outdir, "band_power_synthetic.csv"), index=False)
    print(C.round(4).to_string(index=False))
    print("\nP floor for a single section: 1/%d = %.6f ; pooled over %d sections: 1/%d = %.6f"
          % (args.nperm + 1, 1 / (args.nperm + 1), args.sections,
             args.sections * args.nperm + 1, 1 / (args.sections * args.nperm + 1)))
    print("wrote", args.outdir)


if __name__ == "__main__":
    main()
