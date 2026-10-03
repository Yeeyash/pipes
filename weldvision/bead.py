"""SAW weld-bead geometry from a 2-D image: width, tracking, undercut and surface porosity.

A submerged-arc weld keeps a raised bead with a dark shadow line at each toe. Once the seam is
located, the bead is straightened into a strip (rows across the seam, columns along it) and:

  * both toe lines are traced with dynamic programming (the darkest path that may only drift
    a little from column to column, so a defect patch next to the bead cannot drag it away);
  * width     = distance between the toes, compared with the bead's own median width;
  * tracking  = the bead centre line's deviation from a straight-line fit (bead wandering off
                the joint);
  * undercut  = toe groove clearly deeper than the rest of the toe line;
  * porosity  = small, round, dark spots inside the bead (ripple marks are long arcs, so
                roundness separates them).

Each finding becomes a Detection (on the seam by definition), so it goes through the same
verdict, drawing and line tracking as the YOLO detections.

Limits: a camera sees shading, not depth. Width and tracking are measured directly, undercut
only through its shadow, and reinforcement height not at all; a laser line profiler across
the seam gives true heights and groove depths.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection
from .seam import straighten


@dataclass
class BeadReport:
    pos_px: np.ndarray  # strip columns where the bead was measured
    width_px: np.ndarray
    offset_px: np.ndarray  # centre-line deviation from a straight fit
    toe_depth: np.ndarray  # 2 x n, darkness of the upper / lower toe below the parent metal
    nominal_width_px: float
    defects: list = field(default_factory=list)

    def summary(self, mm_per_px):
        k = mm_per_px
        return {
            "Bead width (median)": f"{self.nominal_width_px * k:.1f} mm",
            "Width range": f"{np.nanmin(self.width_px) * k:.1f} - {np.nanmax(self.width_px) * k:.1f} mm",
            "Max centre-line deviation": f"{np.nanmax(np.abs(self.offset_px)) * k:.1f} mm",
            "Bead findings": str(len(self.defects)),
        }


def _strip(gray, seam, ext):
    """Bead strip with rows across the seam (centre on row ext) and a function mapping strip
    points (u, v) back to image points."""
    h, w = gray.shape
    if seam.orientation == "angled":
        strip, m = straighten(gray, seam, ext)
        valid = straighten(np.ones_like(gray), seam, ext)[0] > 0.99
        inv = cv2.invertAffineTransform(m)
        return strip, valid.all(axis=0), lambda u, v: inv @ np.array([u, v, 1.0])
    if seam.orientation == "vertical":
        gray = gray.T
    lo = seam.center - ext
    rows = np.clip(np.arange(lo, lo + 2 * ext + 1), 0, gray.shape[0] - 1)
    strip = gray[rows]
    valid = np.full(strip.shape[1], True)
    if seam.orientation == "vertical":
        return strip, valid, lambda u, v: np.array([v + lo, u])
    return strip, valid, lambda u, v: np.array([u, v + lo])


def _trace(cost, max_step):
    """Minimum-cost path through cost[row, col], moving at most max_step rows per column."""
    n_r, n_c = cost.shape
    acc = cost[:, 0].copy()
    back = np.zeros((n_r, n_c), np.int32)
    idx = np.arange(n_r)
    for c in range(1, n_c):
        best, arg = np.full(n_r, np.inf), idx.copy()
        for s in range(-max_step, max_step + 1):
            shifted = np.roll(acc, s)
            if s > 0:
                shifted[:s] = np.inf
            elif s < 0:
                shifted[s:] = np.inf
            better = shifted < best
            best[better], arg[better] = shifted[better], (idx - s)[better]
        acc, back[:, c] = best + cost[:, c], arg
    path = np.empty(n_c, np.int32)
    path[-1] = int(np.argmin(acc))
    for c in range(n_c - 1, 0, -1):
        path[c - 1] = back[path[c], c]
    return path


def _runs(flag, min_len, max_gap=0):
    """(start, end) of runs of True at least min_len long, joining runs separated by <= max_gap."""
    d = np.diff(np.concatenate([[0], flag.astype(np.int8), [0]]))
    runs = []
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if runs and a - runs[-1][1] <= max_gap:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return [(a, b) for a, b in runs if b - a >= min_len]


def analyse_bead(img, seam, mm_per_px=0.1, width_tol=0.2, track_tol_mm=2.5, min_len_mm=10.0,
                 undercut_contrast=15.0, step=4):
    """Measure the bead around `seam` and return a BeadReport. Tolerances are relative to the
    bead's own median width and in mm; real limits come from the pipe standard (e.g. API 5L)."""
    gray = (img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    hw0 = max(10, seam.half_width)
    ext = int(2.6 * hw0)
    strip, valid, to_img = _strip(gray, seam, ext)
    cols = np.where(valid)[0]
    if cols.size < 200:
        return None
    u0, u1 = cols[0], cols[-1] + 1
    g = cv2.GaussianBlur(strip, (0, 0), sigmaX=5, sigmaY=1.2)[:, u0:u1:step]
    pos = np.arange(u0, u1, step)[:g.shape[1]]

    # Toe lines: darkest smooth path in a window either side of the centre line.
    lo_in, hi_out = int(0.15 * hw0), int(2.1 * hw0)
    up_rows = np.arange(ext - hi_out, ext - lo_in)
    dn_rows = np.arange(ext + lo_in, ext + hi_out)
    line_cost = g - cv2.GaussianBlur(g, (0, 0), sigmaX=1, sigmaY=12)  # thin dark lines, not slow shading
    up = up_rows[_trace(line_cost[up_rows], 3)]
    dn = dn_rows[_trace(line_cost[dn_rows], 3)]
    width = (dn - up).astype(np.float32)
    centre = (up + dn) / 2.0

    # Parent-metal reference just outside each toe, for toe darkness (undercut).
    cidx = np.arange(g.shape[1])
    def ref(rows_from, sign):
        r = np.clip(rows_from[:, None] + sign * np.arange(18, 40)[None, :], 0, g.shape[0] - 1)
        return np.median(g[r, cidx[:, None]], axis=1)
    toe_val = np.stack([g[np.clip(up + d, 0, g.shape[0] - 1), cidx] for d in (-1, 0, 1)]).min(0), \
        np.stack([g[np.clip(dn + d, 0, g.shape[0] - 1), cidx] for d in (-1, 0, 1)]).min(0)
    depth = np.stack([ref(up, -1) - toe_val[0], ref(dn, 1) - toe_val[1]])

    nominal = float(np.median(width))
    k = mm_per_px
    min_cols = max(2, int(min_len_mm / k / step))
    defects = []

    def add(name, a, b, v_lo, v_hi, conf):
        ua, ub = pos[a], pos[min(b, len(pos)) - 1] + step
        pts = np.array([to_img(u, v) for u in (ua, ub) for v in (v_lo, v_hi)])
        (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
        defects.append(Detection(-1, name, float(np.clip(conf, 0.3, 0.99)),
                                 (round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)), True))

    # Width out of tolerance.
    rel = cv2.medianBlur(width.reshape(1, -1).astype(np.float32), 5).ravel() / nominal - 1
    for a, b in _runs(np.abs(rel) > width_tol, min_cols, min_cols):
        add("bead_width", a, b, up[a:b].min(), dn[a:b].max(), 0.5 + np.abs(rel[a:b]).max() / (2 * width_tol))

    # Tracking: centre-line deviation from a robust straight fit.
    keep = np.ones(pos.size, bool)
    for _ in range(3):
        kk, bb = np.polyfit(pos[keep], centre[keep], 1)
        off = centre - (kk * pos + bb)
        keep = np.abs(off) <= max(2.5 * 1.4826 * np.median(np.abs(off[keep])), 3.0)
    tol_px = track_tol_mm / k
    for a, b in _runs(np.abs(off) > tol_px, 2 * min_cols, min_cols):
        add("seam_tracking", a, b, up[a:b].min(), dn[a:b].max(), 0.5 + (np.abs(off[a:b]).max() - tol_px) / (2 * tol_px))

    # Undercut: a toe groove much darker than the toe line elsewhere.
    for side, line in ((0, up), (1, dn)):
        d = cv2.GaussianBlur(depth[side].reshape(1, -1), (0, 0), 2).ravel()
        base = np.median(d)
        thr = base + max(undercut_contrast, 4 * 1.4826 * np.median(np.abs(d - base)))
        for a, b in _runs(d > thr, min_cols, min_cols):
            add("undercut", a, b, line[a:b].min() - 12, line[a:b].max() + 12, 0.5 + (d[a:b].max() - thr) / 60)

    # Porosity: small round dark spots inside the bead.
    s1 = cv2.GaussianBlur(strip, (0, 0), 1.0)
    hp = s1 - cv2.GaussianBlur(strip, (0, 0), 6.0)
    inside = np.zeros(strip.shape, bool)
    for i, u in enumerate(pos):
        m = 0.2 * (dn[i] - up[i])
        inside[int(up[i] + m):int(dn[i] - m), u:u + step] = True
    sigma = 1.4826 * np.median(np.abs(hp[inside] - np.median(hp[inside]))) + 1e-6
    dark = ((hp < -max(20.0, 5 * sigma)) & inside).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    pores = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 8 <= area <= 250 and max(bw, bh) <= 2.5 * min(bw, bh) and area >= 0.45 * bw * bh:
            pores.append((x, y, x + bw, y + bh))
    pores.sort()
    clusters = []
    for p in pores:  # group pores within ~6 mm into one finding
        for c in clusters:
            if p[0] - c[2] < 6 / k and abs((p[1] + p[3]) / 2 - (c[1] + c[3]) / 2) < 6 / k:
                c[:] = [min(c[0], p[0]), min(c[1], p[1]), max(c[2], p[2]), max(c[3], p[3]), c[4] + 1]
                break
        else:
            clusters.append([*p, 1])
    for x1, y1, x2, y2, cnt in clusters:
        pts = np.array([to_img(u, v) for u in (x1 - 2, x2 + 2) for v in (y1 - 2, y2 + 2)])
        (a1, b1), (a2, b2) = pts.min(0), pts.max(0)
        defects.append(Detection(-1, "porosity", float(min(0.99, 0.55 + 0.1 * cnt)),
                                 (round(float(a1), 1), round(float(b1), 1), round(float(a2), 1), round(float(b2), 1)), True))

    return BeadReport(pos, width, off, depth, nominal, defects)
