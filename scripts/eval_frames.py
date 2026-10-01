"""Score the full pipeline (seam + tiled detection + verdict) on HD frames with ground truth.

    python scripts/eval_frames.py [--weights models/weld_defects.pt] [--frames samples/hd_frames]

A ground-truth box counts as found if a prediction overlaps it with IoU >= --iou. "Class
correct" also requires the same label. Predictions matching no ground truth are false alarms.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from weldvision import DefectDetector, locate_seam, verdict  # noqa: E402
from weldvision.detector import DEFAULT_WEIGHTS  # noqa: E402


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--frames", default=str(ROOT / "samples" / "hd_frames"))
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()

    det = DefectDetector(args.weights)
    tot = dict(gt=0, found=0, cls_ok=0, pred=0, fa=0, seam_err=[])
    print(f"{'frame':16s} {'GT':>3s} {'pred':>4s} {'found':>5s} {'cls_ok':>6s} {'false':>5s} {'seam_err':>8s}  verdict")
    for jp in sorted(Path(args.frames).glob("*.json")):
        gt = json.loads(jp.read_text())
        img = cv2.imread(str(jp.with_suffix(".jpg")))
        seam = locate_seam(img)
        res = det.predict(img, conf=args.conf, tile=args.tile, seam=seam)
        preds = res.detections
        g_boxes = [g["box"] for g in gt["defects"]]

        found = cls_ok = 0
        for g in gt["defects"]:
            ious = [(iou(g["box"], p.box), p) for p in preds]
            hits = [p for v, p in ious if v >= args.iou]
            found += bool(hits)
            cls_ok += any(p.cls_name == g["cls"] for p in hits)
        fa = sum(all(iou(p.box, gb) < args.iou for gb in g_boxes) for p in preds)
        serr = abs(seam.center - gt["seam"]["center"]) if seam else float("nan")
        v, _ = verdict(preds, seam)

        for k, x in (("gt", len(g_boxes)), ("found", found), ("cls_ok", cls_ok), ("pred", len(preds)), ("fa", fa)):
            tot[k] += x
        tot["seam_err"].append(serr)
        expect = "PASS" if not g_boxes else "FAIL/REVIEW"
        print(f"{jp.stem:16s} {len(g_boxes):3d} {len(preds):4d} {found:5d} {cls_ok:6d} {fa:5d} {serr:8.0f}  {v} (expected {expect})")

    print(f"\nGT boxes found (IoU>={args.iou}): {tot['found']}/{tot['gt']} = {tot['found'] / max(tot['gt'], 1):.0%}"
          f"   with correct class: {tot['cls_ok'] / max(tot['gt'], 1):.0%}")
    print(f"False alarms: {tot['fa']} of {tot['pred']} predictions"
          f"   |  median seam centre error: {np.nanmedian(tot['seam_err']):.0f}px")


if __name__ == "__main__":
    main()
