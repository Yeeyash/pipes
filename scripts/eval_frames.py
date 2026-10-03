"""Score the full pipeline (seam + tiled detection + bead geometry + verdict) on frames with ground truth.

    python scripts/eval_frames.py                      # ERW frames (samples/hd_frames)
    python scripts/eval_frames.py --pipe lsaw          # also hsaw, seamless
    python scripts/eval_frames.py --pipe all

A ground-truth box counts as found if a prediction overlaps it with IoU >= --iou. "Class
correct" also requires the same label. Predictions matching no ground truth are false alarms.
Surface defects (the YOLO detector) and SAW bead findings (weldvision.bead) are scored separately.
"""
import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from weldvision import DefectDetector, inspect_image  # noqa: E402
from weldvision.defects import DEFECTS  # noqa: E402
from weldvision.detector import DEFAULT_WEIGHTS  # noqa: E402
from weldvision.pipes import PIPES  # noqa: E402

BEAD_CLASSES = {"undercut", "porosity", "bead_width", "seam_tracking"}


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def seam_error(seam, gt):
    """Distance in px from the true seam centre (a point on it, for angled seams) to the found line."""
    if gt is None:
        return float("nan") if seam is None else float("inf")  # inf: a seam reported on seamless pipe
    if seam is None:
        return float("nan")
    if gt["orientation"] == "angled":
        return seam.distance(*gt["point"])
    return abs(seam.center - gt["center"])


def evaluate(det, pipe, frames, conf, tile, iou_thr):
    print(f"\n=== {PIPES[pipe].name}: {frames}")
    tot = {k: dict(gt=0, found=0, cls_ok=0, pred=0, fa=0) for k in ("surface", "bead")}
    seam_err, by_class = [], {}
    print(f"{'frame':18s} {'GT':>3s} {'pred':>4s} {'found':>5s} {'false':>5s} {'seam_err':>8s}  verdict")
    for jp in sorted(Path(frames).glob("*.json")):
        gt = json.loads(jp.read_text())
        img = cv2.imread(str(jp.with_suffix(".jpg")))
        r = inspect_image(det, img, pipe, conf=conf, tile=tile, mm_per_px=gt.get("mm_per_px", 0.1))
        found_f = fa_f = 0
        for part in ("surface", "bead"):
            g_part = [g for g in gt["defects"] if (g["cls"] in BEAD_CLASSES) == (part == "bead")]
            p_part = [p for p in r.res.detections if (p.cls_name in BEAD_CLASSES) == (part == "bead")]
            for g in g_part:
                hits = [p for p in p_part if iou(g["box"], p.box) >= iou_thr]
                ok = any(p.cls_name == g["cls"] for p in hits)
                t = tot[part]
                t["gt"] += 1
                t["found"] += bool(hits)
                t["cls_ok"] += ok
                by_class.setdefault(g["cls"], [0, 0])
                by_class[g["cls"]][0] += bool(hits)
                by_class[g["cls"]][1] += 1
            fa = sum(all(iou(p.box, g["box"]) < iou_thr for g in g_part) for p in p_part)
            tot[part]["pred"] += len(p_part)
            tot[part]["fa"] += fa
            found_f += sum(any(iou(g["box"], p.box) >= iou_thr for p in p_part) for g in g_part)
            fa_f += fa
        err = seam_error(r.seam, gt.get("seam"))
        seam_err.append(err)
        expect = "PASS" if not gt["defects"] else "FAIL/REVIEW"
        print(f"{jp.stem:18s} {len(gt['defects']):3d} {len(r.res.detections):4d} {found_f:5d} {fa_f:5d} "
              f"{err:8.0f}  {r.verdict} (expected {expect})")

    for part, t in tot.items():
        if t["gt"] or t["pred"]:
            print(f"{part:8s}: found {t['found']}/{t['gt']} = {t['found'] / max(t['gt'], 1):.0%}"
                  f"  correct class {t['cls_ok'] / max(t['gt'], 1):.0%}   false alarms {t['fa']} of {t['pred']}")
    print("by class: " + ", ".join(f"{DEFECTS[c].label if c in DEFECTS else c} {a}/{b}" for c, (a, b) in sorted(by_class.items())))
    finite = [e for e in seam_err if math.isfinite(e)]
    if finite:
        print(f"median seam centre error: {np.median(finite):.0f}px")
    if any(math.isinf(e) for e in seam_err):
        print("seam reported on a seamless pipe!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe", default="erw", choices=[*PIPES, "all"])
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--frames", default=None, help="frame folder (default: the pipe type's sample folder)")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    det = DefectDetector(args.weights)
    for pipe in (list(PIPES) if args.pipe == "all" else [args.pipe]):
        evaluate(det, pipe, args.frames or ROOT / PIPES[pipe].sample_dir, args.conf, args.tile, args.iou)


if __name__ == "__main__":
    main()
