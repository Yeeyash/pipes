"""Simulate the production line: a pipe moving past the inspection camera(s) at line speed.

A long synthetic pipe (with NEU test defects, and planted weld-bead defects for SAW, at known
positions) is cut into encoder-triggered, overlapping frames exactly as the camera would
capture them. Each frame goes through the same inspection as a still image, and defects are
logged in pipe coordinates. The script checks two things:

  * real time: processing time per trigger against the line's frame budget,
  * the log:   every defect reported once at the right position, despite the frame overlap.

Camera layout per pipe type (weldvision.pipes):
  erw       one seam camera after the scarfing tool, 8 m/min
  lsaw      one seam camera behind the outside welding head, ~1.5 m/min (welding speed)
  hsaw      one seam-following camera behind the outside welding head. The pipe turns and
            advances, so the seam slides along itself under the camera at welding speed; the
            log converts positions along the helix to axial position and clock position.
  seamless  no seam: a ring of cameras covers the whole circumference at the inspection station.

    python scripts/simulate_line.py                          # ERW, 8 m/min, 0.1 mm/px, 2 m of pipe
    python scripts/simulate_line.py --roi seam               # analyse only the seam band
    python scripts/simulate_line.py --pipe lsaw
    python scripts/simulate_line.py --pipe hsaw --od 1016 --strip-width 1500
    python scripts/simulate_line.py --pipe seamless --od 168.3 --speed 30
    python scripts/simulate_line.py --realtime               # pace frames at real line speed

The functions below are also used by the "Line simulation" tab of app.py.
"""
import argparse
import csv
import math
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_demo_frames import NEU, make_frame  # noqa: E402
from synth_pipes import camera_shading, paste_neu, saw_surface, seamless_surface, to_bgr  # noqa: E402
from weldvision import DefectDetector, info, inspect_image, verdict  # noqa: E402
from weldvision.line import LineSetup, PipeTracker  # noqa: E402
from weldvision.pipes import PIPES, clock_label, helix_angle_deg, seam_to_pipe, weld_per_pipe_metre  # noqa: E402

DEFAULT_OD = {"hsaw": 1016.0, "seamless": 168.3}
DEFAULT_STRIP_WIDTH = 1500.0


def build_pipe(length_mm, mm_per_px, across_px, n_defects, seed, pipe="erw"):
    """Synthetic pipe as one long image (pipe axis along x) plus ground-truth boxes in mm.

    For seamless pipe the image is the unrolled surface: across_px must cover the whole
    circumference. For SAW, half of the defects are planted bead defects. Spiral pipe is made
    from thinner coil than LSAW plate, so its bead is narrower (14 vs 18 mm)."""
    rng = np.random.default_rng(seed + {"erw": 0, "lsaw": 1, "hsaw": 2, "seamless": 3}[pipe])
    pool = sorted((NEU / "images" / "test").glob("*.jpg"))
    w = int(length_mm / mm_per_px)
    if pipe == "erw":
        strip, meta = make_frame(0, w, across_px, n_defects, rng, pool)
        defects = meta["defects"]
    elif pipe in ("lsaw", "hsaw"):
        cy = across_px // 2 + int(rng.integers(-40, 40))
        n_bead = n_defects // 2
        half = int((9.0 if pipe == "lsaw" else 7.0) / mm_per_px)
        img, _, defects = saw_surface(w, across_px, rng, cy, half, n_bead, n_defects - n_bead, pool)
        strip = to_bgr(img)
    else:
        img = seamless_surface(across_px, w, rng)
        defects = paste_neu(img, n_defects, rng, pool)
        strip = to_bgr(img)
    k = mm_per_px
    gt = [(g["box"][0] * k, g["box"][1] * k, g["box"][2] * k, g["box"][3] * k, g["cls"]) for g in defects]
    return strip, gt


def frame_starts(length_px, line):
    """Pixel x where each encoder-triggered frame starts, including a final frame for the pipe tail."""
    advance_px = int(round(line.advance_mm / line.mm_per_px))
    starts = list(range(0, length_px - line.frame_px_along + 1, advance_px))
    if starts[-1] + line.frame_px_along < length_px:
        starts.append(length_px - line.frame_px_along)
    return starts


def ring_layout(od_mm, mm_per_px, overlap=0.2, n_cams=None):
    """Cameras round a seamless pipe: (number of cameras, circumference px, window start px per
    camera, window height px). Each camera covers its share of the circumference plus overlap."""
    circ_px = int(round(math.pi * od_mm / mm_per_px))
    n = n_cams or LineSetup().cameras_for_full_circumference(od_mm)[0]
    share = circ_px / n
    return n, circ_px, [int(round(i * share)) for i in range(n)], int(math.ceil(share * (1 + overlap)))


def camera_frame(strip, x0, along_px, row0=0, rows=None):
    """What one camera sees: a window of the strip, wrapping round the circumference for a ring."""
    if rows is None:
        return strip[:, x0:x0 + along_px]
    idx = np.arange(row0, row0 + rows) % strip.shape[0]
    shaded = camera_shading(strip[idx, x0:x0 + along_px].astype(np.float32))  # lighting per camera
    return np.clip(shaded, 0, 255).astype(np.uint8)


def inspect_frame(det, frame, conf, tile, seam_only, pipe="erw"):
    """Seam localisation + detection (+ bead geometry for SAW) on one camera frame.
    The seam runs along the frame (horizontal) on every seam camera, including the spiral
    seam-following one. Returns (Inspection, seconds)."""
    t0 = time.perf_counter()
    mode = "horizontal" if PIPES[pipe].seam != "none" else "off"
    r = inspect_image(det, frame, pipe, conf=conf, tile=tile, seam_mode=mode, seam_only=seam_only,
                      mm_per_px=PIPES[pipe].mm_per_px)
    return r, time.perf_counter() - t0


def run_line(det, pipe, strip, line, conf, tile, seam_only=False, ring=None, realtime=False):
    """Play the strip past the camera(s). Yields one dict per encoder trigger:
    i, x0 (px), frames [(camera, frame, Inspection)], dt (s, all cameras), tracker, bead (list of
    (position mm, width mm, offset mm) arrays along the seam so far).

    ring: (n, circ_px, starts, rows) from ring_layout() for a camera ring, else one camera."""
    k = line.mm_per_px
    tracker = PipeTracker(k, wrap_mm=ring[1] * k if ring else None)
    bead = []
    t_line0 = time.perf_counter()
    for i, x0 in enumerate(frame_starts(strip.shape[1], line)):
        if realtime:  # wait until the encoder would fire for this frame
            time.sleep(max(0.0, t_line0 + x0 * k / line.speed_mm_s - time.perf_counter()))
        cams = [(c, r0) for c, r0 in enumerate(ring[2])] if ring else [(0, 0)]
        frames, dt = [], 0.0
        for cam, r0 in cams:
            frame = camera_frame(strip, x0, line.frame_px_along, r0, ring[3] if ring else None)
            r, t = inspect_frame(det, frame, conf, tile, seam_only, pipe)
            dt += t
            tracker.add(i, x0 * k, r.res.detections, across_offset_mm=r0 * k, camera=cam)
            if r.bead is not None:
                bead.append(np.stack([(x0 + r.bead.pos_px) * k, r.bead.width_px * k, r.bead.offset_px * k]))
            frames.append((cam, frame, r))
        yield dict(i=i, x0=x0, frames=frames, dt=dt, tracker=tracker, bead=bead)


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


def score_by_class(defects, gt, thr=0.3):
    """{true class: [found, total]}."""
    out = {}
    for g in gt:
        out.setdefault(g[4], [0, 0])
        out[g[4]][0] += any(iou_mm(d, g) >= thr for d in defects)
        out[g[4]][1] += 1
    return out


def log_rows(defects, pipe="erw", od_mm=None, strip_width_mm=None):
    """Defect log rows (dicts), sorted along the pipe. Seamless adds the clock position round the
    pipe; spiral converts the position along the helical seam to axial position and clock."""
    rows = []
    for d in sorted(defects, key=lambda d: d.start_mm):
        row = {"defect": d.cls_name, "severity": info(d.cls_name).severity, "confidence": f"{d.conf:.3f}",
               "start_mm": f"{d.start_mm:.1f}", "end_mm": f"{d.end_mm:.1f}", "length_mm": f"{d.length_mm:.1f}",
               "across_mm": f"{d.across_mm:.1f}", "on_seam": d.on_seam, "frames": " ".join(map(str, d.frames))}
        if pipe == "seamless":
            deg = d.across_mm / (math.pi * od_mm) * 360
            row["clock"] = clock_label(deg)
            row["cameras"] = " ".join(map(str, sorted(d.cameras)))
        if pipe == "hsaw":
            axial, deg = seam_to_pipe((d.start_mm + d.end_mm) / 2, od_mm, strip_width_mm)
            row["axial_mm"] = f"{axial:.0f}"
            row["clock"] = clock_label(deg)
        rows.append(row)
    return rows


def write_log(defects, path, pipe="erw", od_mm=None, strip_width_mm=None):
    rows = log_rows(defects, pipe, od_mm, strip_width_mm)
    fields = list(rows[0]) if rows else ["defect", "severity", "confidence", "start_mm", "end_mm", "length_mm",
                                         "across_mm", "on_seam", "frames"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def render_overview(strip, gt, defects, mm_per_px, width=2000, ring=None):
    """Whole pipe wrapped into 0.5 m rows: white = ground truth, colour = detected, ticks every
    250 mm. For a camera ring (unrolled surface) the camera boundaries are drawn as dashed lines."""
    k = mm_per_px
    ov = strip.copy()
    lw = max(2, int(round(0.4 / k)))
    for g in gt:
        cv2.rectangle(ov, (int(g[0] / k), int(g[1] / k)), (int(g[2] / k), int(g[3] / k)), (255, 255, 255), lw)
    for d in defects:
        cv2.rectangle(ov, (int(d.start_mm / k), int(d.top_mm / k)), (int(d.end_mm / k), int(d.bottom_mm / k)),
                      info(d.cls_name).color, 2 * lw)
    if ring:
        for r0 in ring[2][1:]:
            for x in range(0, ov.shape[1], 80):
                cv2.line(ov, (x, r0), (x + 40, r0), (0, 200, 255), lw)
    fs = 1.8 * 0.1 / k * (ov.shape[0] / 1024) ** 0.5
    for m in range(0, int(strip.shape[1] * k) + 1, 250):
        x = int(m / k)
        cv2.line(ov, (x, 0), (x, int(6 / k)), (0, 255, 255), lw + 2)
        cv2.putText(ov, f"{m / 1000:.2f} m", (x + int(1 / k), int(5.5 / k)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 255, 255), lw)
    row_px = int(500 / k)
    rows = [ov[:, x:x + row_px] for x in range(0, ov.shape[1], row_px)]
    rows[-1] = cv2.copyMakeBorder(rows[-1], 0, 0, 0, row_px - rows[-1].shape[1], cv2.BORDER_CONSTANT)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, int(4 / k), 0, 0, cv2.BORDER_CONSTANT) for r in rows])
    scale = width / sheet.shape[1]
    return cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def line_setup(pipe, speed=None, mm_per_px=None, across=None, od_mm=None, n_cams=None):
    """LineSetup (and camera ring for seamless) with the pipe type's defaults."""
    p = PIPES[pipe]
    k = mm_per_px or p.mm_per_px
    ring = None
    if pipe == "seamless":
        ring = ring_layout(od_mm or DEFAULT_OD["seamless"], k, n_cams=n_cams)
        across = ring[3]
    line = LineSetup(speed_m_min=speed or p.speed_m_min, mm_per_px=k, frame_px_across=across or p.across_px)
    return line, ring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe", default="erw", choices=list(PIPES))
    ap.add_argument("--speed", type=float, default=None, help="line speed, m/min (default: the pipe type's)")
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--length", type=float, default=2000, help="simulated pipe length, mm")
    ap.add_argument("--across", type=int, default=None, help="frame height in px (sensor ROI across the seam)")
    ap.add_argument("--defects", type=int, default=12)
    ap.add_argument("--od", type=float, default=None, help="pipe OD in mm (hsaw, seamless)")
    ap.add_argument("--strip-width", type=float, default=DEFAULT_STRIP_WIDTH, help="hsaw coil width, mm")
    ap.add_argument("--cameras", type=int, default=None, help="seamless: cameras in the ring (default: 60 deg each)")
    ap.add_argument("--roi", choices=["seam", "full"], default="full", help="analyse whole frame or seam band only")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--realtime", action="store_true", help="sleep so frames arrive at true line speed")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--weights", default=None, help="model weights (default: models/weld_defects.pt)")
    ap.add_argument("--out", default=None, help="output folder (default: results/line_sim[_<pipe>])")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    pipe = args.pipe
    od = args.od or DEFAULT_OD.get(pipe)
    line, ring = line_setup(pipe, args.speed, args.mm_per_px, args.across, od, args.cameras)
    k = line.mm_per_px
    print(f"{PIPES[pipe].name} line setup:")
    for name, v in line.summary():
        print(f"  {name:24s} {v}")
    if ring:
        print(f"  {'Camera ring':24s} {ring[0]} cameras round OD {od:g} mm, {ring[3]} px ({ring[3] * k:.0f} mm) each, "
              f"budget shared by all {ring[0]}")
    if pipe == "hsaw":
        a = helix_angle_deg(od, args.strip_width)
        print(f"  {'Helix':24s} OD {od:g} mm from {args.strip_width:g} mm strip: seam at {a:.1f} deg to the axis, "
              f"{weld_per_pipe_metre(od, args.strip_width):.2f} m of weld per m of pipe, "
              f"pipe advances at {line.speed_m_min * math.cos(math.radians(a)):.2f} m/min")

    print(f"\nGenerating {args.length / 1000:g} m of {'seam' if pipe == 'hsaw' else 'pipe'}...")
    strip, gt = build_pipe(args.length, k, ring[1] if ring else line.frame_px_across, args.defects, args.seed, pipe)

    det = DefectDetector(args.weights) if args.weights else DefectDetector()
    det.predict(strip[:line.frame_px_across, :line.frame_px_along], conf=args.conf, tile=args.tile)  # CUDA warm-up
    times = []
    print(f"\n{'frame':>5s} {'position':>12s} {'defects':>7s} {'proc':>8s} {'budget':>8s}")
    for ev in run_line(det, pipe, strip, line, args.conf, args.tile, args.roi == "seam", ring, args.realtime):
        times.append(ev["dt"])
        n = sum(len(r.res.detections) for _, _, r in ev["frames"])
        flag = "" if ev["dt"] <= line.frame_budget_s else "  << OVER BUDGET"
        print(f"{ev['i']:5d} {ev['x0'] * k / 1000:9.3f} m {n:7d} {ev['dt'] * 1000:6.0f}ms "
              f"{line.frame_budget_s * 1000:6.0f}ms{flag}")
    tracker = ev["tracker"]

    found, false = score(tracker.defects, gt)
    times_ms = np.array(times) * 1000
    v, reason = verdict(tracker.defects, seam_band=True)
    who = f"GPU/CPU load for all {ring[0]} cameras" if ring else "GPU/CPU load"
    print(f"\nProcessing: mean {times_ms.mean():.0f} ms, max {times_ms.max():.0f} ms per trigger "
          f"vs budget {line.frame_budget_s * 1000:.0f} ms -> {who} {times_ms.mean() / (line.frame_budget_s * 1000):.0%}"
          f"  ({'keeps up' if times_ms.max() <= line.frame_budget_s * 1000 else 'FALLS BEHIND'})")
    print(f"Defect log: {len(tracker.defects)} defects ({tracker.raw} raw detections, "
          f"{tracker.raw - len(tracker.defects)} overlap duplicates merged)")
    print(f"Ground truth: {found}/{len(gt)} found, {false} false alarms   |   Pipe verdict: {v} - {reason}")
    per = score_by_class(tracker.defects, gt)
    print("Found by class: " + ", ".join(f"{info(c).label} {a}/{b}" for c, (a, b) in sorted(per.items())))
    if ev["bead"]:
        b = np.concatenate(ev["bead"], axis=1)
        print(f"Bead along the seam: width {np.median(b[1]):.1f} mm median "
              f"({b[1].min():.1f}-{b[1].max():.1f}), centre-line deviation up to {np.abs(b[2]).max():.1f} mm")

    out = Path(args.out or ROOT / "results" / ("line_sim" if pipe == "erw" else f"line_sim_{pipe}"))
    out.mkdir(parents=True, exist_ok=True)
    write_log(tracker.defects, out / "defect_log.csv", pipe, od, args.strip_width)
    cv2.imwrite(str(out / "overview.jpg"), render_overview(strip, gt, tracker.defects, k, ring=ring))
    print(f"Wrote {out / 'defect_log.csv'} and {out / 'overview.jpg'} (white = ground truth, colour = detected)")


if __name__ == "__main__":
    main()
