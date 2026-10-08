# -*- coding: utf-8 -*-
"""Fourier core for the spatial-scale analysis (release copy).

Provenance (logic copied verbatim - no algorithm change):
  * ``to_grid``, ``fill``, ``spec``, ``masks``, ``recon``, ``BANDS``
      <- /home/ST_Data/mouse/cart_region/fourier_m5_mib.py (lines 27-79),
         which itself reuses the step11_fourier.py pipeline
  * ``bh`` <- /home/ST_Data/mouse/cart_region/common.py (lines 61-67)
  * ``band_corr`` / ``toroidal_null`` / ``plus_one_p`` / ``pooled_plus_one_p``
      <- the inner loop of fourier_m5_mib.py (lines 130-148), factored into
         functions so that the observed statistic, the toroidal-shift null and the
         plus-one correction can be unit-tested. The numerical operations are identical:
         np.roll shifts, np.corrcoef on the flattened band components, and
         p = (sum(|null| >= |obs|) + 1) / (len(null) + 1).

Real parameters as they appear in the source script: ``NPERM = 300``,
``rng = np.random.default_rng(0)``, bands
``B1 >1.6 mm`` / ``B2 0.8-1.6 mm`` / ``B3 0.4-0.8 mm`` / ``B4 0.2-0.4 mm``.
NOTE on the P floor: the null is accumulated over the six treated sections
(300 shifts each) and pooled before the plus-one correction, i.e. the smallest
attainable P in the real run is 1/(6*300+1) = 1/1801, not 1/301. See needs_review.tsv.
"""
import numpy as np
from scipy import ndimage

BANDS = [(16, np.inf, "B1 >1.6 mm"), (8, 16, "B2 0.8-1.6 mm"),
         (4, 8, "B3 0.4-0.8 mm"), (2, 4, "B4 0.2-0.4 mm")]

try:                                     # grid spacing that produced the bands
    SPACING_MM = {b[2]: None for b in BANDS}
except Exception:                        # pragma: no cover
    SPACING_MM = {}


def to_grid(x, y, v, sp):
    """Rasterise spot values onto a square grid of pitch ``sp`` (fourier_m5_mib lines 51-54)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); v = np.asarray(v, dtype=float)
    xi = np.round((x - x.min()) / sp).astype(int); yi = np.round((y - y.min()) / sp).astype(int)
    G = np.full((yi.max() + 1, xi.max() + 1), np.nan)
    G[yi, xi] = v
    return G


def fill(G):
    """Nearest-value filling of empty grid cells via a distance transform (lines 57-62)."""
    m = np.isnan(G)
    if m.any():
        idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
        G = G[tuple(idx)]
    return G


def spec(G):
    """Mean-centred, Hann-windowed 2D FFT with the radial frequency grid (lines 65-70)."""
    G = G - G.mean()
    F = np.fft.fftshift(np.fft.fft2(G * np.hanning(G.shape[0])[:, None] * np.hanning(G.shape[1])[None, :]))
    fy = np.fft.fftshift(np.fft.fftfreq(G.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(G.shape[1]))[None, :]
    return F, np.sqrt(fy ** 2 + fx ** 2)


def masks(r):
    """Radial band masks from the frequency radius, wavelength in grid cells (lines 73-75)."""
    lam = np.where(r > 0, 1.0 / np.maximum(r, 1e-9), np.inf)
    return [(lam >= lo) & (lam < hi) for lo, hi, _ in BANDS]


def recon(F, ms):
    """Band components by inverse FFT (lines 78-79)."""
    return [np.real(np.fft.ifft2(np.fft.ifftshift(F * m))) for m in ms]


def band_corr(a, b):
    """Pearson r between two flattened band components (fourier_m5_mib line 140)."""
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def toroidal_null(feature_band, scissor_band, n_perm=300, rng=None):
    """Null r values from toroidal (wrap-around) shifts of the feature field (lines 141-144)."""
    rng = np.random.default_rng(0) if rng is None else rng
    fb = np.asarray(feature_band)
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        sh = rng.integers(0, fb.shape[0]), rng.integers(0, fb.shape[1])
        fb2 = np.roll(np.roll(fb, sh[0], 0), sh[1], 1)
        out[i] = band_corr(fb2, scissor_band)
    return out


def plus_one_p(obs, null):
    """Two-sided plus-one permutation P (fourier_m5_mib line 146)."""
    null = np.asarray(null, dtype=float)
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))


def pooled_plus_one_p(obs, nulls_per_section):
    """P after pooling the per-section nulls, which is what the real run does (lines 141-146)."""
    return plus_one_p(obs, np.concatenate([np.asarray(n, dtype=float) for n in nulls_per_section]))


def bh(p):
    """Benjamini-Hochberg q values (common.py lines 61-67, copied verbatim)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)
