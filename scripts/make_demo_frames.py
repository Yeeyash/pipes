"""Synthesise high-resolution "line camera" frames of an ERW pipe for the end-to-end demo.

Each frame is a procedurally generated pipe surface with:
  * longitudinal rolling texture and cylinder shading,
  * a scarfed weld-seam band (bright cut metal, transverse tool marks, heat-tint edges),
  * real defect patches from the NEU-DET *test* split (never seen in training), alpha-blended
    in at known positions, some of them on the seam.

Ground truth goes to <out>/<frame>.json so the pipeline can be scored on these frames.

    python scripts/make_demo_frames.py --n 6
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEU = ROOT / "data" / "NEU-DET"
NAMES = (NEU / "classes.txt").read_text().split()


def pipe_surface(h, w, rng):
    """Grey steel with rolling streaks along x, mottling and cylinder shading along y."""
    streaks = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=40, sigmaY=1.2)
    mottling = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 60)
    grain = rng.normal(0, 1, (h, w)).astype(np.float32)
    base = 118 + 5 * streaks / (streaks.std() + 1e-6) + 7 * mottling / (mottling.std() + 1e-6) + 4 * grain
    y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    base *= 0.75 + 0.3 * np.cos(y * 1.2)  # light falls off towards the pipe flanks
    return base


def add_seam(img, cy, half, rng, bright=28.0, edge_depth=22.0, chatter=6.0):
    """Scarfed weld bead: brighter cut metal with periodic tool marks and heat-tint borders."""
    h, w = img.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    wobble = 3 * np.sin(xx / 180.0 + rng.uniform(0, 6))  # seam is never perfectly straight
    d = np.abs(yy - cy - wobble) / half
    inside = np.clip(1.4 - d, 0, 1)
    tool = chatter * np.sin(2 * np.pi * xx / rng.uniform(9, 14)) * inside  # scarfing chatter
    edge = np.exp(-((d - 1.0) ** 2) / 0.02)  # heat-affected-zone lines either side
    return img + bright * inside + tool - edge_depth * edge, edge


def paste_defect(frame, patch, box, x0, y0, rng):
    """Blend a defect patch into the frame; return the defect bounding boxes in frame coords."""
    ph, pw = patch.shape
    region = frame[y0:y0 + ph, x0:x0 + pw]
    p = patch.astype(np.float32)
    p = p - p.mean() + region.mean()  # match local brightness, keep the defect's own contrast
    mask = np.zeros((ph, pw), np.float32)
    cv2.rectangle(mask, (12, 12), (pw - 13, ph - 13), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 7)
    frame[y0:y0 + ph, x0:x0 + pw] = region * (1 - mask) + p * mask
    return [(c, x0 + bx1, y0 + by1, x0 + bx2, y0 + by2) for c, bx1, by1, bx2, by2 in box]


def load_labels(img_path):
    lbl = NEU / "labels" / "test" / (img_path.stem + ".txt")
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    h, w = im.shape
    boxes = []
    for line in lbl.read_text().splitlines():
        c, cx, cy, bw, bh = map(float, line.split())
        boxes.append((int(c), int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                      int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return im, boxes


def make_frame(i, w, h, n_defects, rng, pool):
    cy = int(h * rng.uniform(0.4, 0.6))
    half = int(h * 0.035)
    img, _ = add_seam(pipe_surface(h, w, rng), cy, half, rng)
    gt, placed = [], []
    for k in range(n_defects):
        patch, boxes = load_labels(pool[rng.integers(len(pool))])
        ph, pw = patch.shape
        for _ in range(50):  # rejection-sample a non-overlapping spot
            on_seam = k == 0 or rng.random() < 0.25
            y0 = int(np.clip(cy - ph / 2 + rng.integers(-20, 20), 0, h - ph)) if on_seam \
                else int(rng.integers(0, h - ph))
            x0 = int(rng.integers(0, w - pw))
            if all(abs(x0 - px) > pw or abs(y0 - py) > ph for px, py in placed):
                break
        placed.append((x0, y0))
        for c, x1, y1, x2, y2 in paste_defect(img, patch, boxes, x0, y0, rng):
            gt.append({"cls": NAMES[c], "box": [x1, y1, x2, y2], "on_seam": bool(abs((y1 + y2) / 2 - cy) <= half * 1.5)})
    # Light blue-grey tint so the frame looks like a colour camera image of steel.
    bgr = np.clip(np.dstack([img * 1.04, img, img * 0.97]), 0, 255).astype(np.uint8)
    return bgr, {"seam": {"orientation": "horizontal", "center": cy, "half_width": half}, "defects": gt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--out", default=str(ROOT / "samples" / "hd_frames"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    pool = sorted((NEU / "images" / "test").glob("*.jpg"))
    for i in range(args.n):
        n_def = 0 if i == 0 else int(rng.integers(2, 6))  # frame 0 is a clean pipe -> expect PASS
        img, meta = make_frame(i, args.width, args.height, n_def, rng, pool)
        name = f"pipe_frame_{i:02d}"
        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(f"{name}: {len(meta['defects'])} GT boxes")


if __name__ == "__main__":
    main()
