"""Simulate the production line: a pipe moving past a seam camera at line speed.

A long synthetic pipe (with NEU test defects at known positions) is cut into encoder-triggered,
overlapping frames exactly as the camera would capture them. Each frame goes through seam
localisation and detection, and defects are logged in pipe coordinates (mm from the pipe head).
The script checks two things:

  * real time: per-frame processing time against the line's frame budget,
  * the log:   every defect reported once at the right position, despite the frame overlap.

    python scripts/simulate_line.py                         # 8 m/min, 0.1 mm/px, 2 m of pipe
    python scripts/simulate_line.py --speed 8 --roi seam    # analyse only the seam band
    python scripts/simulate_line.py --realtime              # pace frames at real line speed

The functions below are also used by the "Line simulation" tab of app.py.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_demo_frames import NEU, make_frame  # noqa: E402
from weldvision import DefectDetector, info, locate_seam, verdict  # noqa: E402
from weldvision.line import LineSetup, PipeTracker  # noqa: E402


def build_pipe(length_mm, mm_per_px, across_px, n_defects, seed):
    """Synthetic pipe as one long image (pipe axis along x) plus ground-truth boxes in mm."""
    rng = np.random.default_rng(seed)
    pool = sorted((NEU / "images" / "test").glob("*.jpg"))
    strip, meta = make_frame(0, int(length_mm / mm_per_px), across_px, n_defects, rng, pool)
    k = mm_per_px
    gt = [(g["box"][0] * k, g["box"][1] * k, g["box"][2] * k, g["box"][3] * k, g["cls"]) for g in meta["defects"]]
    return strip, gt


def frame_starts(length_px, line):
    """Pixel x where each encoder-triggered frame starts, including a final frame for the pipe tail."""
    advance_px = int(round(line.advance_mm / line.mm_per_px))
    starts = list(range(0, length_px - line.frame_px_along + 1, advance_px))
    if starts[-1] + line.frame_px_along < length_px:
        starts.append(length_px - line.frame_px_along)
    return starts


def inspect_frame(det, frame, conf, tile, seam_only):
    """Seam localisation + detection on one camera frame. Returns (result, seconds)."""
    t0 = time.perf_counter()
    seam = locate_seam(frame, "horizontal")
    roi = seam.band(frame.shape, pad=tile // 2) if (seam and seam_only) else None
    res = det.predict(frame, conf=conf, tile=tile, seam=seam, roi=roi)
    return res, time.perf_counter() - t0


def iou_mm(d, g):
    iw = max(0, min(d.end_mm, g[2]) - max(d.start_mm, g[0]))
    ih = max(0, min(d.bottom_mm, g[3]) - max(d.top_mm, g[1]))
    inter = iw * ih
    a = (d.end_mm - d.start_mm) * (d.bottom_mm - d.top_mm)
    b = (g[2] - g[0]) * (g[3] - g[1])
    return inter / (a + b - inter + 1e-6)


def score(defects, gt, thr=0.3):
    """(ground-truth boxes found, false alarms) at IoU >= thr."""
    found = sum(any(iou_mm(d, g) >= thr for d in defects) for g in gt)
    false = sum(all(iou_mm(d, g) < thr for g in gt) for d in defects)
    return found, false


def write_log(defects, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["defect", "severity", "confidence", "start_mm", "end_mm", "length_mm", "across_mm", "on_seam", "frames"])
        for d in sorted(defects, key=lambda d: d.start_mm):
            w.writerow([d.cls_name, info(d.cls_name).severity, f"{d.conf:.3f}", f"{d.start_mm:.1f}", f"{d.end_mm:.1f}",
                        f"{d.length_mm:.1f}", f"{d.across_mm:.1f}", d.on_seam, " ".join(map(str, d.frames))])


def render_overview(strip, gt, defects, mm_per_px, width=2000):
    """Whole pipe wrapped into 0.5 m rows: white = ground truth, colour = detected, ticks every 250 mm."""
    k = mm_per_px
    ov = strip.copy()
    for g in gt:
        cv2.rectangle(ov, (int(g[0] / k), int(g[1] / k)), (int(g[2] / k), int(g[3] / k)), (255, 255, 255), 4)
    for d in defects:
        cv2.rectangle(ov, (int(d.start_mm / k), int(d.top_mm / k)), (int(d.end_mm / k), int(d.bottom_mm / k)),
                      info(d.cls_name).color, 8)
    for m in range(0, int(strip.shape[1] * k) + 1, 250):
        x = int(m / k)
        cv2.line(ov, (x, 0), (x, 60), (0, 255, 255), 6)
        cv2.putText(ov, f"{m / 1000:.2f} m", (x + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
    row_px = int(500 / k)
    rows = [ov[:, x:x + row_px] for x in range(0, ov.shape[1], row_px)]
    rows[-1] = cv2.copyMakeBorder(rows[-1], 0, 0, 0, row_px - rows[-1].shape[1], cv2.BORDER_CONSTANT)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 40, 0, 0, cv2.BORDER_CONSTANT) for r in rows])
    scale = width / sheet.shape[1]
    return cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=8.0, help="line speed, m/min")
    ap.add_argument("--mm-per-px", type=float, default=0.1)
    ap.add_argument("--length", type=float, default=2000, help="simulated pipe length, mm")
    ap.add_argument("--across", type=int, default=1024, help="frame height in px (sensor ROI across the seam)")
    ap.add_argument("--defects", type=int, default=12)
    ap.add_argument("--roi", choices=["seam", "full"], default="full", help="analyse whole frame or seam band only")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--realtime", action="store_true", help="sleep so frames arrive at true line speed")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(ROOT / "results" / "line_sim"))
    args = ap.parse_args()

    line = LineSetup(speed_m_min=args.speed, mm_per_px=args.mm_per_px, frame_px_across=args.across)
    print("Line setup:")
    for k, v in line.summary():
        print(f"  {k:24s} {v}")

    print(f"\nGenerating {args.length / 1000:g} m of pipe...")
    strip, gt = build_pipe(args.length, args.mm_per_px, args.across, args.defects, args.seed)

    det = DefectDetector()
    det.predict(strip[:, :line.frame_px_along], conf=args.conf, tile=args.tile)  # CUDA warm-up
    tracker = PipeTracker(args.mm_per_px)
    times = []
    print(f"\n{'frame':>5s} {'pipe pos':>12s} {'defects':>7s} {'proc':>8s} {'budget':>8s}")
    t_line0 = time.perf_counter()
    for i, x0 in enumerate(frame_starts(strip.shape[1], line)):
        if args.realtime:  # wait until the encoder would fire for this frame
            time.sleep(max(0.0, t_line0 + x0 * args.mm_per_px / line.speed_mm_s - time.perf_counter()))
        res, dt = inspect_frame(det, strip[:, x0:x0 + line.frame_px_along], args.conf, args.tile, args.roi == "seam")
        times.append(dt)
        tracker.add(i, x0 * args.mm_per_px, res.detections)
        flag = "" if dt <= line.frame_budget_s else "  << OVER BUDGET"
        print(f"{i:5d} {x0 * args.mm_per_px / 1000:9.3f} m {len(res.detections):7d} {dt * 1000:6.0f}ms "
              f"{line.frame_budget_s * 1000:6.0f}ms{flag}")

    found, false = score(tracker.defects, gt)
    raw = sum(len(d.frames) for d in tracker.defects)
    times_ms = np.array(times) * 1000
    v, reason = verdict(tracker.defects, seam_band=True)

    print(f"\nProcessing: mean {times_ms.mean():.0f} ms, max {times_ms.max():.0f} ms per frame "
          f"vs budget {line.frame_budget_s * 1000:.0f} ms -> GPU load {times_ms.mean() / (line.frame_budget_s * 1000):.0%}"
          f"  ({'keeps up' if times_ms.max() <= line.frame_budget_s * 1000 else 'FALLS BEHIND'})")
    print(f"Defect log: {len(tracker.defects)} defects ({raw} raw detections, "
          f"{raw - len(tracker.defects)} overlap duplicates merged)")
    print(f"Ground truth: {found}/{len(gt)} found, {false} false alarms   |   Pipe verdict: {v} - {reason}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_log(tracker.defects, out / "defect_log.csv")
    cv2.imwrite(str(out / "overview.jpg"), render_overview(strip, gt, tracker.defects, args.mm_per_px))
    print(f"Wrote {out / 'defect_log.csv'} and {out / 'overview.jpg'} (white = ground truth, colour = detected)")


if __name__ == "__main__":
    main()
