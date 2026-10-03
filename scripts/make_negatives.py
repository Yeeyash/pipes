"""Generate defect-free "background" tiles (pipe surface and normal weld seam) for training.

NEU-DET has a defect in every image, so a model trained on it alone has never seen clean
steel and reports defects on normal texture, especially the scarfed seam's tool marks and
heat-tint lines. YOLO treats images with an empty label file as pure background, which
teaches the model what "normal" looks like.

Surfaces covered: ERW plate and scarfed seam; SAW weld beads (ripples, toe shadows, normal
width and tracking variation) at any angle, since a spiral seam crosses the camera image at an
angle; plate texture at any angle; seamless tube scale and guide marks.

On a real line, replace these synthetic tiles with crops from pipes that passed inspection.
That is the most valuable data you can collect, and it costs nothing to label.

    python scripts/make_negatives.py --train 700 --val 140
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_demo_frames import add_seam, pipe_surface  # noqa: E402
from synth_pipes import camera_shading, saw_surface, seamless_surface  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "negatives"


def rotated(img, rng, h, w):
    """Rotate a larger canvas by a random angle and cut an h x w window from its centre.
    Returns (image, angle in degrees)."""
    angle = rng.uniform(0, 180)
    m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
    m[:, 2] += [w / 2 - img.shape[1] / 2, h / 2 - img.shape[0] / 2]
    return camera_shading(cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR)), angle


def random_frame(rng, h=800, w=1200):
    """A defect-free surface and, for a SAW bead, its direction through the centre (else None)."""
    kind = rng.choice(["erw", "saw", "plate", "seamless"], p=[0.4, 0.25, 0.15, 0.2])
    side = int(np.hypot(h, w)) + 8
    bead_angle = None
    if kind == "saw":  # bead with normal geometry variation (not surface defects)
        half = int(rng.uniform(50, 120))
        img, _, _ = saw_surface(side, side, rng, side // 2, half, int(rng.integers(0, 3)), 0, None,
                                events=("bead_width", "seam_tracking"), shading=False)
        img, bead_angle = rotated(img, rng, h, w)
    elif kind == "plate":
        img, _ = rotated(pipe_surface(side, side, rng, shading=False), rng, h, w)
    elif kind == "seamless":
        img = camera_shading(seamless_surface(h, w, rng))
    else:
        img = pipe_surface(h, w, rng)
        if rng.random() < 0.85:
            img, _ = add_seam(img, int(h * rng.uniform(0.3, 0.7)), int(h * rng.uniform(0.02, 0.06)), rng,
                              bright=rng.uniform(10, 40), edge_depth=rng.uniform(8, 30), chatter=rng.uniform(2, 10))
    # Exposure / contrast variation between cameras and shifts.
    img = (img - img.mean()) * rng.uniform(0.7, 1.4) + img.mean() + rng.uniform(-25, 25)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if kind == "erw" and rng.random() < 0.3:
        return np.ascontiguousarray(np.rot90(img)), None
    return img, bead_angle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=700)
    ap.add_argument("--val", type=int, default=140)
    ap.add_argument("--tile", type=int, default=200, help="match the NEU-DET image size")
    ap.add_argument("--seed", type=int, default=1000, help="differs from the demo-frame seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for split, n in (("train", args.train), ("val", args.val)):
        for d in ("images", "labels"):  # replace any older set
            for f in (OUT / d / split).glob("background_*"):
                f.unlink()
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        frame = None
        for i in range(n):
            if i % 10 == 0:
                frame, bead_angle = random_frame(rng)
            h, w = frame.shape
            if bead_angle is not None and rng.random() < 0.7:  # most crops on the bead (cv2 angle: counter-clockwise)
                t, r = np.radians(bead_angle), rng.uniform(-300, 300)
                cx, cy = w / 2 + r * np.cos(t), h / 2 - r * np.sin(t)
                x = int(np.clip(cx - args.tile / 2 + rng.uniform(-40, 40), 0, w - args.tile))
                y = int(np.clip(cy - args.tile / 2 + rng.uniform(-40, 40), 0, h - args.tile))
            else:
                y, x = int(rng.integers(0, h - args.tile)), int(rng.integers(0, w - args.tile))
            name = f"background_{split}_{i:04d}"
            cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), frame[y:y + args.tile, x:x + args.tile])
            (OUT / "labels" / split / f"{name}.txt").write_text("")  # empty = no objects
        print(f"{split}: {n} background tiles -> {OUT / 'images' / split}")


if __name__ == "__main__":
    main()
