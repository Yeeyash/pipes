"""Locate the weld seam in a pipe image.

An ERW seam is a straight band running along the pipe axis. After scarfing (bead trimming),
the band differs from the parent metal in brightness (freshly cut metal or heat tint) and
texture (tool marks). Taking a 1-D profile across the pipe axis turns finding the seam into
finding a peak in that profile:

    profile(y) = |row brightness - baseline| + |row texture energy - baseline|

The cylinder's lighting falloff gives a broad gradient across the pipe, so a heavily smoothed
copy of the profile is subtracted first and only the narrow seam peak remains.

On a production line the camera is fixed relative to the seam (the mill keeps the seam at
12 o'clock), so a fixed ROI or this detector run once per coil is usually enough.

A SAW weld keeps a raised bead that is wider than a scarfed ERW seam; pass wider `widths`.
A spiral (HSAW) seam crosses a body camera's image at an angle: orientation="angled" rotates
the image over a range of angles and keeps the one where the seam band stands out most (a
brute-force Radon transform of the same band score).
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Seam:
    orientation: str  # "horizontal" (seam runs left-right), "vertical" or "angled"
    center: int  # px across the seam, in original image scale (angled: offset from the image centre)
    half_width: int  # covers the bead plus its heat-affected zone, the critical area of the weld
    prominence: float  # peak strength in robust z-score units; higher = more confident
    angle: float = 0.0  # angled seams: direction of the centre line, degrees from the image x axis (y down)
    point: tuple | None = None  # angled seams: a point (x, y) on the centre line

    def distance(self, x: float, y: float) -> float:
        """Distance in px from the seam centre line."""
        if self.orientation == "horizontal":
            return abs(y - self.center)
        if self.orientation == "vertical":
            return abs(x - self.center)
        t = math.radians(self.angle)
        return abs(-math.sin(t) * (x - self.point[0]) + math.cos(t) * (y - self.point[1]))

    def contains(self, x: float, y: float) -> bool:
        return self.distance(x, y) <= self.half_width

    def band(self, shape, pad=0):
        """(x1, y1, x2, y2) of the seam band, widened by pad px on each side, for an image of the given shape.
        For an angled seam this is the band's bounding box, which can cover most of the image."""
        h, w = shape[:2]
        if self.orientation == "angled":
            poly = self.polygon(shape, pad)
            x1, y1 = np.clip(poly.min(0), 0, [w, h])
            x2, y2 = np.clip(poly.max(0), 0, [w, h])
            return int(x1), int(y1), int(x2), int(y2)
        lo, hi = max(0, self.center - self.half_width - pad), self.center + self.half_width + pad
        return (0, lo, w, min(h, hi)) if self.orientation == "horizontal" else (lo, 0, min(w, hi), h)

    def polygon(self, shape, pad=0):
        """Corners of the seam band (widened by pad px) as a 4x2 int array, for drawing and masks."""
        if self.orientation != "angled":
            x1, y1, x2, y2 = self.band(shape, pad)
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        h, w = shape[:2]
        t = math.radians(self.angle)
        d, n = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
        p, reach, hw = np.array(self.point, float), math.hypot(h, w), self.half_width + pad
        return np.array([p - reach * d - hw * n, p + reach * d - hw * n,
                         p + reach * d + hw * n, p - reach * d + hw * n]).round().astype(np.int32)


def _robust_z(p, valid=None):
    ref = p if valid is None else p[valid]
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) + 1e-6
    z = (p - med) / (1.4826 * mad)
    if valid is not None:
        z[~valid] = 0
    return z


def _profile(gray, axis, mask=None, n_ref=None, highpass=0.15, median=False, signed=False):
    """Seam-likelihood profile across the image; axis=1 averages along rows (horizontal seam).

    mask:  optional valid-pixel mask (a rotated image has empty corners); rows with too few valid
           pixels are neutralised.
    n_ref: length that sets the smoothing windows (default: the profile length).
    highpass: window (fraction of n_ref) of the baseline that is subtracted; it must be well
          wider than the seam, or the seam's own plateau is removed with the baseline.
    median: use a running median as the baseline instead of a Gaussian. A median ignores any
          plateau narrower than half its window, so a wide SAW bead does not lift the baseline
          around itself (which would make the flanks look as active as the bead).
    signed: return the two signed z-score profiles (brightness, texture) as a 2 x n array instead
          of the sum of their magnitudes; see _best_band_signed.
    """
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    valid = None
    if mask is None:
        bright = gray.astype(np.float32).mean(axis=axis)
        texture = lap.mean(axis=axis)
    else:
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8))  # the Laplacian is unreliable at the valid area's edge
        cnt_in = inner.sum(axis=axis).astype(np.float32)
        if signed:
            # Row medians: a bead runs the full length of its row, whereas a few defect patches that
            # happen to line up only move the row mean. (Medians are slower, so only for SAW.)
            g = np.where(inner > 0, gray.astype(np.float32), np.nan)
            bright = np.nanmedian(g, axis=axis)
            texture = np.nanmedian(np.where(inner > 0, lap, np.nan), axis=axis)
        else:
            bright = (gray.astype(np.float32) * mask).sum(axis=axis) / np.maximum(mask.sum(axis=axis), 1)
            texture = (lap * inner).sum(axis=axis) / np.maximum(cnt_in, 1)
        bad = cnt_in < 0.6 * cnt_in.max()  # rows crossing only a corner of the image average too few pixels
        for sig in (bright, texture):
            sig[bad | ~np.isfinite(sig)] = np.median(sig[~bad & np.isfinite(sig)])
        valid = ~bad
    n = bright.size
    n_ref = n_ref or n
    big = max(3, int(n_ref * highpass)) | 1  # remove cylinder shading / slow illumination drift
    small = max(3, int(n_ref * 0.01)) | 1
    zs = []
    for sig in (bright, texture):
        hp = sig - (_running_median(sig, big) if median else cv2.GaussianBlur(sig.reshape(-1, 1), (1, big), 0).ravel())
        hp = cv2.GaussianBlur(hp.reshape(-1, 1), (1, small), 0).ravel()
        zs.append(_robust_z(hp, valid))
    if signed:
        return np.stack(zs).astype(np.float32)
    out = np.zeros(n, np.float32)
    for z in zs:
        out += np.abs(z)
    return out


def _running_median(sig, k):
    k = min(k, 2 * (sig.size // 2) - 1)
    padded = np.pad(sig, k // 2, mode="reflect")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)


DEFAULT_WIDTHS = (0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08)


def locate_seam(img, orientation="auto", min_prominence=6.0, max_side=1024, widths=None, angles=None,
                highpass=0.15, median=False, signed=False):
    """Return a Seam, or None if no clear seam-like band is present.

    orientation: "horizontal", "vertical", "auto" (picks the stronger of the two) or "angled"
                 (searches all directions; for spiral-welded pipe).
    widths:      band half-widths to try, as fractions of the image side. The default suits a
                 scarfed ERW seam; a SAW bead is wider (see weldvision.pipes).
    highpass:    baseline window as a fraction of the image side; wide seams need a wider one.
    median:      running-median baseline (for wide SAW beads), see _profile.
    signed:      score a band by its signed contrast (a raised bead is consistently brighter or
                 darker than the plate), see _best_band_signed. Prominence is on a different
                 scale then, so set min_prominence with it (weldvision.pipes does).
    angles:      angled only: (lo, hi) range in degrees to search, e.g. around the mill's known
                 helix angle. Default: all directions.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orientation == "angled":
        max_side = min(max_side, 640)  # the angle search rotates the image ~25 times; coarser is enough
    scale = min(1.0, max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)
    widths = widths or DEFAULT_WIDTHS
    if signed:  # flat-field: remove lighting falloff in any direction before searching all angles
        f = small.astype(np.float32)
        illum = cv2.GaussianBlur(f, (0, 0), min(small.shape) / 4)
        small = np.clip(f / (illum + 1) * 128, 0, 255).astype(np.uint8)

    if orientation == "angled":
        best = _locate_angled(small, widths, angles or (-90, 90), highpass, median, signed)
        best.center, best.half_width = int(best.center / scale), int(best.half_width / scale)
        best.point = (best.point[0] / scale, best.point[1] / scale)
    else:
        candidates = ["horizontal", "vertical"] if orientation == "auto" else [orientation]
        best = None
        for ori in candidates:
            prof = _profile(small, axis=1 if ori == "horizontal" else 0, highpass=highpass, median=median,
                            signed=signed)
            center, half_w, prom = (_best_band_signed if signed else _best_band)(prof, widths)
            if best is None or prom > best.prominence:
                best = Seam(ori, int(center / scale), int(half_w / scale), prom)

    return best if best is not None and best.prominence >= min_prominence else None


def _rotation(shape, angle_deg):
    """Affine matrix that turns a line at angle_deg into a horizontal one, on a canvas big enough
    for the whole rotated image. Returns (M, (width, height))."""
    h, w = shape[:2]
    t = math.radians(angle_deg)
    r = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) @ r.T
    lo, hi = corners.min(0), corners.max(0)
    return np.hstack([r, -lo[:, None]]), (int(math.ceil(hi[0] - lo[0])), int(math.ceil(hi[1] - lo[1])))


def straighten(img, seam, half_extent):
    """Cut a strip along an angled seam: rows run across it, columns along it, and the seam centre
    line lies on row `half_extent`.

    Returns (strip, M), where M (2x3 affine) maps image points to strip points; map strip
    points back to the image with cv2.invertAffineTransform(M).
    """
    m, size = _rotation(img.shape, seam.angle)
    c = m @ np.array([seam.point[0], seam.point[1], 1.0])
    m[1, 2] -= c[1] - half_extent
    return cv2.warpAffine(img, m, (size[0], 2 * half_extent + 1), flags=cv2.INTER_LINEAR), m


def _locate_angled(small, widths, angles, highpass, median, signed):
    n_ref = min(small.shape)
    ones = np.ones(small.shape, np.uint8)

    def score(a):
        m, size = _rotation(small.shape, a)
        rot = cv2.warpAffine(small, m, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(ones, m, size, flags=cv2.INTER_NEAREST)
        prof = _profile(rot, 1, mask, n_ref, highpass, median, signed)
        center, half_w, prom = (_best_band_signed if signed else _best_band)(prof, widths, n_ref)
        return prom, center, half_w, m, size

    lo, hi = angles
    coarse = [(score(a), a) for a in np.arange(lo, hi, 3.0)]
    a0 = max(coarse, key=lambda t: t[0][0])[1]
    fine = [(score(a), a) for a in np.arange(max(lo, a0 - 2.5), min(hi, a0 + 2.5) + 0.01, 0.5)]
    (prom, center, half_w, m, size), a = max(fine + coarse, key=lambda t: t[0][0])

    # The band's centre row in the rotated image, mapped back to a point on the seam in `small`.
    px, py = cv2.invertAffineTransform(m) @ np.array([size[0] / 2, center, 1.0])
    seam = Seam("angled", 0, int(round(half_w)), float(prom), float(a), (float(px), float(py)))
    for _ in range(2):
        seam = _refine_line(small, seam, widths, n_ref)
    h, w = small.shape
    t = math.radians(seam.angle)
    seam.center = int(round(-math.sin(t) * (seam.point[0] - w / 2) + math.cos(t) * (seam.point[1] - h / 2)))
    seam.angle = (seam.angle + 90) % 180 - 90  # report the direction in [-90, 90)
    return seam


def _refine_line(small, seam, widths, n_ref, block=24):
    """Correct an angled seam's direction and position by a line fit to the bead centre found in
    short blocks along a strip straightened at the current estimate. The band search alone is
    only good to a few degrees, because a wide band tolerates a slightly tilted bead."""
    ext = int(2.5 * seam.half_width) + 4
    strip, m = straighten(small.astype(np.float32), seam, ext)
    valid = straighten(np.ones(small.shape, np.float32), seam, ext)[0] > 0.99
    cols = np.where(valid.all(axis=0))[0]
    if cols.size < 4 * block:
        return seam
    w = max(2, int(round(seam.half_width)))
    us, cs = [], []
    for u0 in range(cols[0], cols[-1] - block, block):
        prof = strip[:, u0:u0 + block].mean(axis=1)
        prof = (prof - np.median(prof)) / (1.4826 * np.median(np.abs(prof - np.median(prof))) + 1e-6)
        c, _, _ = _best_band_signed(prof[None, :], (w / n_ref,), n_ref)
        us.append(u0 + block / 2)
        cs.append(c)
    us, cs = np.array(us), np.array(cs)
    keep = np.ones(us.size, bool)
    for _ in range(3):  # line fit with outlier rejection (defects and patches pull single blocks off)
        k, b = np.polyfit(us[keep], cs[keep], 1)
        res = np.abs(cs - (k * us + b))
        keep = res <= max(2.5 * 1.4826 * np.median(res[keep]), 2.0)
        if keep.sum() < 4:
            return seam
    inv = cv2.invertAffineTransform(m)
    u_mid = us[keep].mean()
    p0 = inv @ np.array([u_mid, k * u_mid + b, 1.0])
    p1 = inv @ np.array([u_mid + 100, k * (u_mid + 100) + b, 1.0])
    angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    return Seam("angled", seam.center, seam.half_width, seam.prominence, angle, (float(p0[0]), float(p0[1])))


def _best_band(prof, widths=DEFAULT_WIDTHS, n_ref=None):
    """Find the band [c-w, c+w] whose mean score most exceeds the flanking strips of the same width.

    A scarfed seam shows up as two edge peaks (heat-tint lines) around a textured plateau, so
    a band-vs-flank contrast lands on the centre of the bead, where a single-peak search would
    lock onto one edge. Widths are fractions of n_ref (default: the profile length).
    Returns (center, half_width, prominence in robust-sigma units).
    """
    n = prof.size
    n_ref = n_ref or n
    cs = np.concatenate([[0.0], np.cumsum(prof, dtype=np.float64)])
    sigma = 1.4826 * np.median(np.abs(prof - np.median(prof))) + 1e-6

    def mean(a, b):  # mean of prof[a:b], vectorised over arrays a, b
        return (cs[b] - cs[a]) / np.maximum(b - a, 1)

    best = (n / 2, n * 0.02, -np.inf)
    for frac in widths:
        w = max(2, int(n_ref * frac))
        c = np.arange(2 * w, n - 2 * w)
        if c.size == 0:
            continue
        inner = mean(c - w, c + w)
        flank = np.maximum(mean(c - 2 * w, c - w), mean(c + w, c + 2 * w))  # both sides must be quieter
        # A mild width bonus (w^0.25) prefers the whole bead over a single edge line without
        # letting the band grow into the surrounding parent metal.
        score = (inner - flank) / sigma * (w / (n_ref * 0.01)) ** 0.25
        i = int(np.argmax(score))
        if score[i] > best[2]:
            best = (float(c[i]), float(w), float(score[i]))
    return best


def _best_band_signed(zs, widths=DEFAULT_WIDTHS, n_ref=None):
    """Band search on signed profiles (2 x n: brightness, texture) for a raised weld bead.

    A bead is consistently brighter (or darker) and differently textured than the plate across
    its whole width, so the band mean of the *signed* z-score moves far from the flanks while
    plain surface noise averages towards zero. Score per profile:
        |inner - mean(flanks)| - |left flank - right flank|
    (the second term rejects a step edge, e.g. a shading change), summed over both profiles
    and scaled by sqrt(w) so it reads as a significance. Returns (center, half_width, prominence).
    """
    n = zs.shape[1]
    n_ref = n_ref or n
    cs = np.concatenate([np.zeros((zs.shape[0], 1)), np.cumsum(zs, axis=1, dtype=np.float64)], axis=1)

    def mean(a, b):
        return (cs[:, b] - cs[:, a]) / np.maximum(b - a, 1)

    best = (n / 2, n * 0.02, -np.inf)
    for frac in widths:
        w = max(2, int(n_ref * frac))
        c = np.arange(2 * w, n - 2 * w)
        if c.size == 0:
            continue
        inner, left, right = mean(c - w, c + w), mean(c - 2 * w, c - w), mean(c + w, c + 2 * w)
        score = (np.abs(inner - (left + right) / 2) - np.abs(left - right)).sum(0) * np.sqrt(n_ref * 0.04) * (w / (n_ref * 0.04)) ** 0.25
        i = int(np.argmax(score))
        if score[i] > best[2]:
            best = (float(c[i]), float(w), float(score[i]))
    return best
