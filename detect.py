"""Batch inspection: run the defect pipeline on an image or a folder and write a report.

    python detect.py samples/hd_frames                     # tiled, seam auto-detected
    python detect.py data/NEU-DET/images/test --tile 0 --seam off
    python detect.py frame.jpg --conf 0.25 --tile 224
    python detect.py samples/hd_frames --seam-only          # dedicated seam camera, ~3x faster
    python detect.py samples/lsaw_frames --pipe lsaw        # SAW: wide bead + bead geometry checks
    python detect.py samples/hsaw_frames --pipe hsaw        # spiral: angled seam
    python detect.py samples/seamless_frames --pipe seamless

Outputs (in --out, default results/):
    annotated/<name>.jpg   image with seam band and defect boxes
    detections.csv         one row per detected defect
    summary.csv            one row per image with the PASS / REVIEW / FAIL verdict
"""
import argparse
import csv
from pathlib import Path

import cv2

from weldvision import PIPES, DefectDetector, draw, info, inspect_image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image file or folder")
    ap.add_argument("--pipe", default="erw", choices=list(PIPES), help="pipe type: seam model and bead checks")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--tile", type=int, default=224,
                    help="tile size in px (0 = whole image in one pass). Pick it so a tile covers roughly "
                         "the same physical area as a training image.")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--seam", default=None, choices=["auto", "horizontal", "vertical", "angled", "off"],
                    help="seam orientation in the image (default: auto for ERW, the pipe type's otherwise)")
    ap.add_argument("--mm-per-px", type=float, default=None, help="for SAW bead measurements (default: the pipe type's)")
    ap.add_argument("--seam-only", action="store_true", help="analyse only the seam band (faster)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    src = Path(args.source)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    out = Path(args.out)
    (out / "annotated").mkdir(parents=True, exist_ok=True)

    det = DefectDetector(args.weights) if args.weights else DefectDetector()
    with open(out / "detections.csv", "w", newline="") as fd, open(out / "summary.csv", "w", newline="") as fs:
        wd, ws = csv.writer(fd), csv.writer(fs)
        wd.writerow(["image", "defect", "confidence", "severity", "on_seam", "x1", "y1", "x2", "y2"])
        ws.writerow(["image", "verdict", "reason", "n_defects", "seam_found", "tiles", "ms"])
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"skip (unreadable): {f}")
                continue
            seam_mode = args.seam or ("auto" if args.pipe == "erw" else None)
            r = inspect_image(det, img, args.pipe, conf=args.conf, tile=args.tile, overlap=args.overlap,
                              seam_mode=seam_mode, seam_only=args.seam_only,
                              mm_per_px=args.mm_per_px or PIPES[args.pipe].mm_per_px)
            res, seam, ms, v, reason = r.res, r.seam, r.ms, r.verdict, r.reason

            for d in res.detections:
                wd.writerow([f.name, d.cls_name, f"{d.conf:.3f}", info(d.cls_name).severity, d.on_seam, *map(int, d.box)])
            ws.writerow([f.name, v, reason, len(res.detections), seam is not None, res.n_tiles, f"{ms:.0f}"])
            cv2.imwrite(str(out / "annotated" / f"{f.stem}.jpg"), draw(img, res))
            print(f"{f.name:32s} {v:6s} {len(res.detections):3d} defects  {res.n_tiles:3d} tiles  {ms:6.0f} ms  {reason}")
    print(f"\nReports written to {out.resolve()}")


if __name__ == "__main__":
    main()
