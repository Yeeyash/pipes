"""Generate defect-free "background" tiles (pipe surface and normal weld seam) for training.

NEU-DET has a defect in every image, so a model trained on it alone has never seen clean
steel and reports defects on normal texture, especially the scarfed seam's tool marks and
heat-tint lines. YOLO treats images with an empty label file as pure background, which
teaches the model what "normal" looks like.

On a real line, replace these synthetic tiles with crops from pipes that passed inspection.
That is the most valuable data you can collect, and it costs nothing to label.

    python scripts/make_negatives.py --train 300 --val 60
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_demo_frames import add_seam, pipe_surface  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "negatives"


def random_frame(rng, h=800, w=1200):
    img = pipe_surface(h, w, rng)
    if rng.random() < 0.85:
        img, _ = add_seam(img, int(h * rng.uniform(0.3, 0.7)), int(h * rng.uniform(0.02, 0.06)), rng,
                          bright=rng.uniform(10, 40), edge_depth=rng.uniform(8, 30), chatter=rng.uniform(2, 10))
    # Exposure / contrast variation between cameras and shifts.
    img = (img - img.mean()) * rng.uniform(0.7, 1.4) + img.mean() + rng.uniform(-25, 25)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.rot90(img)) if rng.random() < 0.3 else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--tile", type=int, default=200, help="match the NEU-DET image size")
    ap.add_argument("--seed", type=int, default=1000, help="differs from the demo-frame seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for split, n in (("train", args.train), ("val", args.val)):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        frame = None
        for i in range(n):
            if i % 10 == 0:
                frame = random_frame(rng)
            h, w = frame.shape
            y, x = int(rng.integers(0, h - args.tile)), int(rng.integers(0, w - args.tile))
            name = f"background_{split}_{i:04d}"
            cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), frame[y:y + args.tile, x:x + args.tile])
            (OUT / "labels" / split / f"{name}.txt").write_text("")  # empty = no objects
        print(f"{split}: {n} background tiles -> {OUT / 'images' / split}")


if __name__ == "__main__":
    main()
