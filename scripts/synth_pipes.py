"""Synthetic SAW (longitudinal and spiral) and seamless pipe surfaces with ground truth.

Companion to make_demo_frames.py (ERW). Everything is in pixels at the demo resolution
(0.1 mm/px for SAW, so an 18 mm bead is 180 px wide).

SAW bead: a raised, convex weld bead with solidification ripples (chevron arcs), a specular
crown highlight and a dark shadow line at each toe. Bead defects are planted as "events" with
ground truth:
    bead_width     the bead narrows or widens over a stretch (unstable current / speed / flux)
    seam_tracking  the bead wanders sideways off its line (seam tracking lost the groove)
    undercut       a deeper groove just outside one toe
    porosity       a cluster of surface pores in the bead
Real NEU-DET test defects are pasted onto the parent metal as for ERW.

Spiral (HSAW): the same bead, rotated to the helix angle as a body camera sees it.
Seamless: no seam; mill-scale texture and faint helical guide marks, plus NEU defects.
"""
import math

import cv2
import numpy as np

from make_demo_frames import NAMES, load_labels, paste_defect, pipe_surface

BEAD_EVENTS = ("bead_width", "seam_tracking", "undercut", "porosity")


def _bump(n):
    """Smooth 0 -> 1 -> 0 window of length n with a flat top (a defect that fades in and out)."""
    t = np.linspace(0, 1, n)
    return np.clip(np.minimum(t, 1 - t) / 0.2, 0, 1) ** 2 * (3 - 2 * np.clip(np.minimum(t, 1 - t) / 0.2, 0, 1))


def bead_track(w, cy, half, rng, n_events, events=BEAD_EVENTS):
    """Per-column bead centre, half-width and toe-groove depth, with planted bead defects.

    Returns (track dict, list of events (kind, x1, x2, params))."""
    x = np.arange(w, dtype=np.float32)
    drift = cv2.GaussianBlur(rng.normal(0, 1, (1, w)).astype(np.float32), (0, 0), 120).ravel()
    track = {
        "center": cy + 4 * np.sin(x / 350 + rng.uniform(0, 6)) + 2 * drift / (drift.std() + 1e-6),
        "half": half * (1 + 0.025 * np.sin(x / 140 + rng.uniform(0, 6))),
        "undercut": np.zeros((2, w), np.float32),  # [upper toe, lower toe] groove depth
    }
    placed, out = [], []
    for k in range(n_events):
        kind = events[k % len(events)] if k < len(events) else events[rng.integers(len(events))]
        n = {"bead_width": rng.integers(350, 700), "seam_tracking": rng.integers(600, 1000),
             "undercut": rng.integers(180, 420), "porosity": 140}[kind]
        for _ in range(60):
            x1 = int(rng.integers(60, max(61, w - n - 60)))
            if all(x1 + n + 150 < a or x1 > b + 150 for a, b in placed):
                break
        else:
            continue
        placed.append((x1, x1 + n))
        win = _bump(n)
        sl = slice(x1, x1 + n)
        p = {}
        if kind == "bead_width":
            f = rng.uniform(0.6, 0.72) if rng.random() < 0.6 else rng.uniform(1.3, 1.4)
            track["half"][sl] *= 1 + (f - 1) * win
            p["factor"] = round(float(f), 2)
        elif kind == "seam_tracking":
            a = rng.uniform(0.45, 0.7) * half * rng.choice([-1, 1])
            track["center"][sl] += a * win
            p["offset_px"] = round(float(a), 1)
        elif kind == "undercut":
            side = int(rng.integers(2))
            track["undercut"][side, sl] = rng.uniform(40, 60) * win
            p["side"] = ("upper", "lower")[side]
        out.append((kind, x1, x1 + n, p))
    return track, out


def render_bead(img, track, rng, toe_depth=22.0):
    """Draw the bead described by `track` into the float image (in place). Only the rows near the
    bead are touched, so long line-simulation strips stay cheap."""
    h, w = img.shape
    c, hw = track["center"], track["half"]
    r0 = int(max(0, np.floor((c - 2.2 * hw).min())))
    r1 = int(min(h, np.ceil((c + 2.2 * hw).max())))
    yy = np.arange(r0, r1, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    d = (yy - c[None, :]) / hw[None, :]
    a = np.abs(d)
    e = (a - 1) * hw[None, :]  # px outside the bead edge (negative inside)
    region = img[r0:r1]

    m = np.clip(0.5 - e / 3, 0, 1)  # bead mask with a ~3 px soft edge
    crown = np.clip(1 - d ** 2, 0, 1)
    period = rng.uniform(11, 16)
    ripple = 7 * np.sin(2 * np.pi * (xx + 0.35 * hw[None, :] * d ** 2) / period)  # chevron ripples
    highlight = 30 * np.exp(-((d + 0.3) / 0.14) ** 2)  # specular stripe on the crown
    grain = rng.normal(0, 3, region.shape).astype(np.float32)
    tone = cv2.GaussianBlur(region, (0, 0), 25) + 12  # SAW bead: slightly brighter oxide skin than the plate
    bead = tone + 16 * crown + highlight + ripple * (0.5 + 0.5 * crown) + grain

    uc = np.where(d < 0, track["undercut"][0][None, :], track["undercut"][1][None, :])  # groove depth per toe
    toe = toe_depth * np.exp(-((e - 1.5) ** 2) / (2 * 2.5 ** 2))  # shadow line at the toe
    groove = uc * np.exp(-((e - 5) ** 2) / (2 * 3.0 ** 2))  # undercut groove just outside the toe
    img[r0:r1] = region * (1 - m) + bead * m - toe - groove
    return img


def add_pores(img, track, x1, x2, rng):
    """Cluster of surface pores in the bead between x1 and x2; returns the cluster's GT box."""
    n = int(rng.integers(2, 7))
    boxes = []
    for _ in range(n):
        x = int(rng.integers(x1 + 10, x2 - 10))
        d = rng.uniform(-0.55, 0.55)
        y = int(track["center"][x] + d * track["half"][x])
        r = rng.uniform(3, 7)
        yy, xx = np.mgrid[-12:13, -12:13]
        rr = np.hypot(xx, yy)
        pore = -55 * np.clip(r + 0.8 - rr, 0, 1) + 12 * np.exp(-((rr - r - 1.5) ** 2) / 2)  # dark hole, bright rim
        ys, xs = slice(y - 12, y + 13), slice(x - 12, x + 13)
        if img[ys, xs].shape == pore.shape:
            img[ys, xs] += pore
            boxes.append((x - r - 2, y - r - 2, x + r + 2, y + r + 2))
    b = np.array(boxes)
    return [int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())]


def event_box(kind, x1, x2, track, p):
    """Ground-truth box of a bead event: the stretch where it is clearly present."""
    n = x2 - x1
    a, b = int(x1 + 0.15 * n), int(x2 - 0.15 * n)
    c, hw = track["center"][a:b], track["half"][a:b]
    if kind == "undercut":
        y = c - hw if p["side"] == "upper" else c + hw
        return [a, int(y.min() - 12), b, int(y.max() + 12)]
    return [a, int((c - hw).min()), b, int((c + hw).max())]


def paste_neu(img, n, rng, pool, avoid=(), seam_dist=None, band=0.0, bead_half=0.0, seam_share=0.0, near_seam=None):
    """Paste n NEU test patches onto the parent metal and return their ground truth.

    Patches do not overlap each other or the `avoid` boxes (planted bead defects). With a seam,
    seam_dist(x, y) gives the distance from its centre line: a patch may not sit on the bead
    itself (|dist| < bead_half at its centre), a share of them is placed beside it with
    near_seam(ph, pw) -> (x0, y0), and boxes within `band` count as on the seam."""
    h, w = img.shape
    gt, placed = [], list(avoid)
    for _ in range(n):
        patch, boxes = load_labels(pool[rng.integers(len(pool))])
        ph, pw = patch.shape
        for _ in range(100):
            if near_seam is not None and rng.random() < seam_share:
                x0, y0 = near_seam(ph, pw)
            else:
                x0, y0 = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
            x0, y0 = int(np.clip(x0, 0, w - pw)), int(np.clip(y0, 0, h - ph))
            free = all(x0 + pw + 10 < a or x0 > c + 10 or y0 + ph + 10 < b or y0 > d + 10 for a, b, c, d in placed)
            if free and (seam_dist is None or seam_dist(x0 + pw / 2, y0 + ph / 2) >= bead_half):
                break
        else:
            continue
        placed.append((x0, y0, x0 + pw, y0 + ph))
        for c, x1, y1, x2, y2 in paste_defect(img, patch, boxes, x0, y0, rng):
            on = seam_dist is not None and seam_dist((x1 + x2) / 2, (y1 + y2) / 2) <= band
            gt.append({"cls": NAMES[c], "box": [x1, y1, x2, y2], "on_seam": bool(on), "source": "surface"})
    return gt


def saw_surface(w, h, rng, cy, half, n_bead, n_neu, pool, events=BEAD_EVENTS, shading=True):
    """Float image of plate with a horizontal SAW bead at row cy, plus ground truth."""
    img = pipe_surface(h, w, rng, shading)
    track, evs = bead_track(w, cy, half, rng, n_bead, events)
    render_bead(img, track, rng)
    gt = []
    for kind, x1, x2, p in evs:
        box = add_pores(img, track, x1, x2, rng) if kind == "porosity" else event_box(kind, x1, x2, track, p)
        gt.append({"cls": kind, "box": box, "on_seam": True, "source": "bead", **p})
    band = 1.35 * half  # bead plus toes
    if n_neu:
        def near(ph, pw):  # beside the bead, overlapping the toe / heat-affected zone
            x0 = int(rng.integers(0, w - pw))
            return x0, int(track["center"][x0 + pw // 2] + rng.choice([-1, 1]) * (half + 0.45 * ph) - ph / 2)

        def dist(x, y):
            return abs(y - track["center"][int(np.clip(x, 0, w - 1))])
        gt += paste_neu(img, n_neu, rng, pool, [tuple(g["box"]) for g in gt], dist, band, half, 0.3, near)
    return img, track, gt


def to_bgr(img):
    return np.clip(np.dstack([img * 1.04, img, img * 0.97]), 0, 255).astype(np.uint8)


def lsaw_frame(w, h, n_bead, n_neu, rng, pool, half=90):
    cy = int(h * rng.uniform(0.42, 0.58))
    img, track, gt = saw_surface(w, h, rng, cy, half, n_bead, n_neu, pool)
    meta = {"pipe": "lsaw", "mm_per_px": 0.1,
            "seam": {"orientation": "horizontal", "center": cy, "half_width": int(1.35 * half)}, "defects": gt}
    return to_bgr(img), meta


def hsaw_frame(w, h, n_bead, n_neu, rng, pool, angle_deg, half=90):
    """Body-camera view of a spiral pipe: the bead crosses the frame at angle_deg to the pipe axis (x)."""
    side = int(math.ceil(math.hypot(w, h))) + 64
    cy = side // 2 + int(rng.integers(-60, 60))
    canvas, track, gt_bead = saw_surface(side, side, rng, cy, half, n_bead, 0, pool, shading=False)
    t = math.radians(angle_deg)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    c_in, c_out = np.array([side / 2, side / 2]), np.array([w / 2, h / 2])
    m = np.hstack([r, (c_out - r @ c_in)[:, None]])
    img = camera_shading(cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR))  # lighting is fixed to the camera

    gt = []
    for g in gt_bead:
        x1, y1, x2, y2 = g["box"]
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float) @ r.T + m[:, 2]
        bx1, by1 = pts.min(0)
        bx2, by2 = pts.max(0)
        cx, cyy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if 0 <= cx < w and 0 <= cyy < h:
            gt.append({**g, "box": [int(max(0, bx1)), int(max(0, by1)), int(min(w, bx2)), int(min(h, by2))]})
    point = r @ np.array([0.0, cy - side / 2]) + c_out  # a point on the seam centre line
    band = 1.35 * half

    def dist(x, y):
        return abs(-math.sin(t) * (x - point[0]) + math.cos(t) * (y - point[1]))

    def near(ph, pw):
        s = rng.uniform(-0.4, 0.4) * math.hypot(w, h)
        off = rng.choice([-1, 1]) * (half + 0.45 * ph)
        p = point + s * np.array([math.cos(t), math.sin(t)]) + off * np.array([-math.sin(t), math.cos(t)])
        return int(p[0] - pw / 2), int(p[1] - ph / 2)

    gt += paste_neu(img, n_neu, rng, pool, [tuple(g["box"]) for g in gt], dist, band, half, 0.3, near)
    meta = {"pipe": "hsaw", "mm_per_px": 0.1,
            "seam": {"orientation": "angled", "angle": angle_deg, "point": [float(point[0]), float(point[1])],
                     "half_width": int(band)}, "defects": gt}
    return to_bgr(img), meta


def seamless_surface(h, w, rng, guide_marks=True):
    """Hot-finished seamless tube: darker scaled surface, flaky scale texture, faint helical
    guide marks from the piercer / reeler. No cylinder shading (cameras add their own)."""
    streaks = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=30, sigmaY=2)
    mottling = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 50)
    flakes = np.tanh(2.5 * cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6) /
                     0.05)  # patchy scale skin
    grain = rng.normal(0, 1, (h, w)).astype(np.float32)
    img = (100 + 4 * streaks / (streaks.std() + 1e-6) + 8 * mottling / (mottling.std() + 1e-6)
           + 6 * flakes + 4 * grain)
    if guide_marks:
        phi = math.radians(rng.uniform(70, 80))
        period = rng.uniform(70, 120)
        xx = np.arange(w, dtype=np.float32)[None, :] * math.cos(phi)
        yy = np.arange(h, dtype=np.float32)[:, None] * math.sin(phi)
        img += 9 * np.sin(np.pi * (xx + yy) / period) ** 16
    return img


def camera_shading(frame):
    """Light falls off towards the edges of each camera's arc of the curved surface."""
    h = frame.shape[0]
    y = np.linspace(-1, 1, h, dtype=np.float32).reshape(-1, *[1] * (frame.ndim - 1))
    return frame * (0.75 + 0.3 * np.cos(y * 1.2))


def seamless_frame(w, h, n_neu, rng, pool):
    img = camera_shading(seamless_surface(h, w, rng))
    gt = paste_neu(img, n_neu, rng, pool)
    return to_bgr(img), {"pipe": "seamless", "mm_per_px": 0.2, "seam": None, "defects": gt}
