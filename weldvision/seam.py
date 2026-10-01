"""Locate the longitudinal ERW weld seam in a pipe image.

An ERW seam is a straight band running along the pipe axis. After scarfing (bead trimming),
the band differs from the parent metal in brightness (freshly cut metal or heat tint) and
texture (tool marks). Taking a 1-D profile across the pipe axis turns finding the seam into
finding a peak in that profile:

    profile(y) = |row brightness - baseline| + |row texture energy - baseline|

The cylinder's lighting falloff gives a broad gradient across the pipe, so a heavily smoothed
copy of the profile is subtracted first and only the narrow seam peak remains.

On a production line the camera is fixed relative to the seam (the mill keeps the seam at
12 o'clock), so a fixed ROI or this detector run once per coil is usually enough.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Seam:
    orientation: str  # "horizontal" (seam runs left-right) or "vertical"
    center: int  # pixel coordinate across the seam, in original image scale
    half_width: int  # covers the bead plus its heat-affected zone, the critical area of an ERW weld
    prominence: float  # peak strength in robust z-score units; higher = more confident

    def contains(self, x: float, y: float) -> bool:
        c = y if self.orientation == "horizontal" else x
        return abs(c - self.center) <= self.half_width

    def band(self, shape, pad=0):
        """(x1, y1, x2, y2) of the seam band, widened by pad px on each side, for an image of the given shape."""
        h, w = shape[:2]
        lo, hi = max(0, self.center - self.half_width - pad), self.center + self.half_width + pad
        return (0, lo, w, min(h, hi)) if self.orientation == "horizontal" else (lo, 0, min(w, hi), h)


def _robust_z(p):
    med = np.median(p)
    mad = np.median(np.abs(p - med)) + 1e-6
    return (p - med) / (1.4826 * mad)


def _profile(gray, axis):
    """Seam-likelihood profile across the image; axis=1 averages along rows (horizontal seam)."""
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    bright = gray.astype(np.float32).mean(axis=axis)
    texture = lap.mean(axis=axis)
    n = bright.size
    big = max(3, int(n * 0.15)) | 1  # remove cylinder shading / slow illumination drift
    small = max(3, int(n * 0.01)) | 1
    out = np.zeros(n, np.float32)
    for sig in (bright, texture):
        hp = sig - cv2.GaussianBlur(sig.reshape(-1, 1), (1, big), 0).ravel()
        hp = cv2.GaussianBlur(hp.reshape(-1, 1), (1, small), 0).ravel()
        out += np.abs(_robust_z(hp))
    return out


def locate_seam(img, orientation="auto", min_prominence=6.0, max_side=1024):
    """Return a Seam, or None if no clear seam-like band is present.

    orientation: "horizontal", "vertical" or "auto" (picks the stronger of the two).
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)

    candidates = ["horizontal", "vertical"] if orientation == "auto" else [orientation]
    best = None
    for ori in candidates:
        prof = _profile(small, axis=1 if ori == "horizontal" else 0)
        center, half_w, prom = _best_band(prof)
        if best is None or prom > best.prominence:
            best = Seam(ori, int(center / scale), int(half_w / scale), prom)

    return best if best is not None and best.prominence >= min_prominence else None


def _best_band(prof, widths=(0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08)):
    """Find the band [c-w, c+w] whose mean score most exceeds the flanking strips of the same width.

    A scarfed seam shows up as two edge peaks (heat-tint lines) around a textured plateau, so
    a band-vs-flank contrast lands on the centre of the bead, where a single-peak search would
    lock onto one edge. Returns (center, half_width, prominence in robust-sigma units).
    """
    n = prof.size
    cs = np.concatenate([[0.0], np.cumsum(prof, dtype=np.float64)])
    sigma = 1.4826 * np.median(np.abs(prof - np.median(prof))) + 1e-6

    def mean(a, b):  # mean of prof[a:b], vectorised over arrays a, b
        return (cs[b] - cs[a]) / np.maximum(b - a, 1)

    best = (n / 2, n * 0.02, -np.inf)
    for frac in widths:
        w = max(2, int(n * frac))
        c = np.arange(2 * w, n - 2 * w)
        if c.size == 0:
            continue
        inner = mean(c - w, c + w)
        flank = np.maximum(mean(c - 2 * w, c - w), mean(c + w, c + 2 * w))  # both sides must be quieter
        # A mild width bonus (w^0.25) prefers the whole bead over a single edge line without
        # letting the band grow into the surrounding parent metal.
        score = (inner - flank) / sigma * (w / (n * 0.01)) ** 0.25
        i = int(np.argmax(score))
        if score[i] > best[2]:
            best = (float(c[i]), float(w), float(score[i]))
    return best
