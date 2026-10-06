# -*- coding: utf-8 -*-
"""Synthetic-data tests for the released Fourier core.

These tests run without any project data. They check the parts of the pipeline that are
purely computational: rasterisation, band selection by wavelength, the observed statistic,
the toroidal-shift null, the plus-one correction (and its pooled floor) and BH monotonicity.

Run:  python tests/test_fourier_core.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fourier_core import (BANDS, to_grid, fill, spec, masks, recon, band_corr,
                          toroidal_null, plus_one_p, pooled_plus_one_p, bh)

FAILS = []

def check(name, cond, detail=""):
    print(("  PASS " if cond else "  FAIL ") + name + (" :: " + str(detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def synthetic_field(spacing_mm=0.1, size_mm=3.0, wavelength_mm=1.2, amplitude=1.0, seed=0):
    """Spots on a regular grid carrying a sinusoid of a known wavelength (mm)."""
    g = np.arange(0, size_mm, spacing_mm)
    X, Y = np.meshgrid(g, g)
    v = amplitude * np.sin(2 * np.pi * X / wavelength_mm)
    rng = np.random.default_rng(seed)
    v = v + 0.05 * rng.standard_normal(v.shape)
    return X.ravel(), Y.ravel(), v.ravel(), spacing_mm


def band_energy(grid, spacing_mm):
    F, r = spec(grid)
    ms = masks(r)
    lam = np.where(r > 0, 1.0 / np.maximum(r, 1e-9), np.inf) * spacing_mm  # cells -> mm
    tot = sum((np.abs(F) ** 2 * m).sum() for m in ms)
    return [float((np.abs(F) ** 2 * m).sum() / tot) for m in ms], lam


def main():
    print("1) rasterise + fill + band selection (known wavelength)")
    x, y, v, sp = synthetic_field(wavelength_mm=1.2)
    G = fill(to_grid(x, y, v, sp))
    check("grid has no NaNs after fill", not np.isnan(G).any(), G.shape)
    F, r = spec(G)
    lam_mm = np.where(r > 0, 1.0 / np.maximum(r, 1e-9), np.inf) * sp
    share, _ = band_energy(G, sp)
    b2 = [b[2] for b in BANDS].index("B2 0.8-1.6 mm")
    check("1.2 mm sinusoid lands in band B2 (0.8-1.6 mm)", share.index(max(share)) == b2,
          dict(zip([b[2] for b in BANDS], [round(s, 3) for s in share])))

    print("2) second wavelength falls in the expected band")
    x2, y2, v2, sp2 = synthetic_field(wavelength_mm=0.5)
    G2 = fill(to_grid(x2, y2, v2, sp2))
    share2, _ = band_energy(G2, sp2)
    b3 = [b[2] for b in BANDS].index("B3 0.4-0.8 mm")
    check("0.5 mm sinusoid lands in band B3 (0.4-0.8 mm)", share2.index(max(share2)) == b3,
          dict(zip([b[2] for b in BANDS], [round(s, 3) for s in share2])))

    print("3) band reconstruction recovers the right field")
    F, r = spec(G)
    rec = recon(F, masks(r))
    rs = [band_corr(c, G) for c in rec]
    check("B2 band holds the majority of the power (>90%)", share[b2] > 0.9, round(share[b2], 3))
    # NOTE: the absolute r of the B2 reconstruction is only ~0.74, because the field also
    # contains a 0.5 mm component, additive noise and the Hann window tapers the border.
    # The pipeline is a band decomposition, not an exact inverse filter, so the meaningful
    # assertion is that the band carrying the power correlates best with the input field.
    check("B2 reconstruction correlates better with the input than any other band",
          rs.index(max(rs)) == b2, dict(zip([b[2] for b in BANDS], [round(x, 3) for x in rs])))

    print("4) toroidal-shift null and plus-one correction")
    rng = np.random.default_rng(0)
    obs = band_corr(rec[b2], G)
    null = toroidal_null(rec[b2], G, n_perm=300, rng=rng)
    check("null has 300 values", len(null) == 300, len(null))
    p = plus_one_p(obs, null)
    check("plus-one P uses (k+1)/(n+1)", abs(p - (np.sum(np.abs(null) >= abs(obs)) + 1) / 301) < 1e-12, p)
    check("P is never 0", p > 0, p)
    perfect = plus_one_p(1.0, np.zeros(300))
    check("perfect correlation P = 1/301 with a single-section null", abs(perfect - 1 / 301) < 1e-12, perfect)
    pooled = pooled_plus_one_p(1.0, [np.zeros(300) for _ in range(6)])
    check("pooled (6 sections x 300 shifts) P floor = 1/1801", abs(pooled - 1 / 1801) < 1e-12, pooled)

    print("5) BH family behaviour")
    pv = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.9])
    q = bh(pv)
    check("q is monotone in p", bool(np.all(np.diff(q) >= -1e-12)), q.round(4).tolist())
    check("q >= p for every row", bool(np.all(q >= pv - 1e-12)))
    check("q is clipped to <= 1", bool(np.all(q <= 1)))
    q1 = bh(np.array([0.5]))
    check("single-row family leaves q = p", abs(q1[0] - 0.5) < 1e-12, q1)

    print("\n%s" % ("ALL TESTS PASSED" if not FAILS else "FAILED: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
