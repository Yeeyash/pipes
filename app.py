"""Pipe weld & surface defect inspection - demo UI for ERW, LSAW, HSAW (spiral) and seamless pipe.

    streamlit run app.py

The pipe type (sidebar) selects the seam model, the SAW bead measurements, the sample images
and the camera layout of the line simulation. Tabs:
    Production       whole-frame inspection of the selected images
    Seam-only        the same images, analysing only the weld-seam band (dedicated seam camera)
    Line simulation  a pipe moving past the camera(s) at line speed, logged by position on the pipe
    Pipe type guide  how the selected pipe is made, where the camera goes, what it can and cannot see
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from weldvision import DEFECTS, DefectDetector, draw, info, inspect_image, verdict
from weldvision.detector import DEFAULT_WEIGHTS
from weldvision.line import LineSetup
from weldvision.pipes import PIPES, helix_angle_deg, weld_per_pipe_metre

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from download_data import ensure_test_split  # noqa: E402

ensure_test_split()  # a fresh deployment has no data/ folder; sample patches and the line simulation need NEU test images
import simulate_line as sim  # noqa: E402

NEU_TEST = ROOT / "data" / "NEU-DET" / "images" / "test"
VERDICT_STYLE = {"PASS": ("#1b7f3b", "✅"), "REVIEW": ("#b26a00", "⚠️"), "FAIL": ("#b3261e", "⛔")}
SEAM_MODES = {"erw": ["auto", "horizontal", "vertical", "off"], "lsaw": ["horizontal", "auto", "vertical", "off"],
              "hsaw": ["angled", "horizontal", "vertical", "off"], "seamless": ["off"]}
SAW = ("lsaw", "hsaw")
BEAD_CLASSES = ("undercut", "porosity", "bead_width", "seam_tracking")  # measured by weldvision.bead

st.set_page_config(page_title="Pipe Weld Defect Inspection", page_icon="🔍", layout="wide")


@st.cache_resource
def load_detector(code_mtime, weights_mtime):
    # The arguments are only cache keys: editing the code or retraining the model invalidates
    # the cached instance instead of serving an object built from stale code.
    det = DefectDetector(DEFAULT_WEIGHTS)
    det.predict(np.zeros((448, 448, 3), np.uint8), tile=224)  # warm-up, so the first timing shown is real
    return det


@st.cache_data(max_entries=2, show_spinner="Generating the pipe...")
def cached_pipe(pipe, length_mm, n_defects, seed, mm_per_px, across_px):
    return sim.build_pipe(length_mm, mm_per_px, across_px, n_defects, seed, pipe)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_ground_truth(img, gt):
    out = img.copy()
    lw = max(1, round(max(img.shape[:2]) / 500))
    for g in gt.get("defects", []):
        x1, y1, x2, y2 = g["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), lw)
    return out


def dim_outside(img, r):
    """Darken everything outside the analysed region so it is obvious what was not inspected."""
    out = (img * 0.35).astype(np.uint8)
    if r.roi is not None:
        x1, y1, x2, y2 = r.roi
        out[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    else:  # angled seam: the band (plus half a tile) as a polygon
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [r.seam.polygon(img.shape, pad=tile // 2)], 1)
        out[mask > 0] = img[mask > 0]
    return out


def verdict_banner(v, name, reason):
    color, icon = VERDICT_STYLE[v]
    st.markdown(
        f"<div style='padding:10px 16px;border-radius:8px;background:{color};color:white;font-size:1.15rem'>"
        f"{icon} <b>{v}</b> &nbsp;·&nbsp; {name} &nbsp;·&nbsp; {reason}</div>", unsafe_allow_html=True)


def seam_label(seam):
    if seam is None:
        return "none" if pipe.seam == "none" else "not found"
    if seam.orientation == "angled":
        return f"{abs(seam.angle):.0f}° to the axis"
    return f"{seam.orientation[0].upper()} @ {seam.center}px"


@st.cache_data(max_entries=64, show_spinner="Inspecting...")
def _inspect(_det, code_key, img, pipe_key, seam_only, conf, mode, tile, overlap, seam_mode, bead, mm_per_px):
    # Cached so that clicking any widget does not re-run every selected image (seconds per frame on a CPU host).
    h, w = img.shape[:2]
    use_tile = {"Single pass": 0, "Tiled": tile, "Auto": tile if max(h, w) > tile * 1.25 else 0}[mode]
    return inspect_image(_det, img, pipe_key, conf=conf, tile=use_tile, overlap=overlap, seam_mode=seam_mode,
                         seam_only=seam_only, bead=bead, mm_per_px=mm_per_px)


def inspect(img, seam_only, gt=None):
    """Run the pipeline on one image. Returns an Inspection with everything the tabs display."""
    k = (gt or {}).get("mm_per_px", pipe.mm_per_px)
    return _inspect(detector, code_key, img, pipe_key, seam_only, conf, mode, tile, overlap, seam_mode, bead_on, k)


def bead_chart(pos_mm, width_mm, offset_mm, nominal_mm=None):
    """Bead width and centre-line deviation along the seam."""
    df = pd.DataFrame({"Position along seam (mm)": np.round(pos_mm, 1), "Bead width (mm)": width_mm,
                       "Centre-line deviation (mm)": offset_mm}).groupby("Position along seam (mm)").mean()
    c1, c2 = st.columns(2)
    c1.line_chart(df[["Bead width (mm)"]], height=220)
    c2.line_chart(df[["Centre-line deviation (mm)"]], height=220)
    if nominal_mm:
        st.caption(f"Flagged where the width differs from the median ({nominal_mm:.1f} mm) by more than 20 %, "
                   "or the centre line leaves its straight fit by more than 2.5 mm. Real limits come from "
                   "the pipe standard and need calibrated mm/px.")


def show_result(name, img, gt, r, baseline=None):
    """Verdict banner, metrics, annotated image and detection table for one inspected image.
    baseline: the production result for the same image, to show the seam-only speed-up."""
    res, seam = r.res, r.seam
    st.divider()
    verdict_banner(r.verdict, name, r.reason)

    m = st.columns(5)
    m[0].metric("Defects", len(res.detections))
    m[1].metric("On weld seam", sum(d.on_seam for d in res.detections))
    m[2].metric("Seam", seam_label(seam))
    if baseline is None:
        m[3].metric("Tiles", res.n_tiles)
        m[4].metric("Time", f"{r.ms:.0f} ms")
    else:
        b = baseline.res
        m[3].metric("Tiles", res.n_tiles, delta=f"{res.n_tiles - b.n_tiles} vs full frame", delta_color="inverse")
        m[4].metric("Time", f"{r.ms:.0f} ms", delta=f"{r.ms - baseline.ms:.0f} ms vs full frame",
                    delta_color="inverse")

    annotated = draw(img, res)
    if r.roi is not None or r.seam_only_band:
        annotated = dim_outside(annotated, r)
    if show_gt and gt:
        annotated = draw_ground_truth(annotated, gt)
    h, w = img.shape[:2]
    left, right = st.columns([3, 2]) if w <= h * 1.3 else (st.container(), st.container())
    left.image(to_rgb(annotated), caption=f"{w}×{h}px", width="stretch")

    if res.detections:
        right.dataframe(pd.DataFrame([{
            "Defect": info(d.cls_name).label,
            "Found by": "bead measurement" if d.cls_id < 0 else "detector",
            "Confidence": round(d.conf, 3),
            "Severity": info(d.cls_name).severity,
            "On seam": "yes" if d.on_seam else "",
            "Box (x1,y1,x2,y2)": tuple(int(v) for v in d.box),
        } for d in res.detections]), width="stretch", hide_index=True)
    if gt is not None:
        right.caption(f"Ground truth for this frame: {len(gt['defects'])} defect boxes "
                      f"({sum(g['on_seam'] for g in gt['defects'])} on seam).")
    if r.bead is not None and baseline is None:
        k = (gt or {}).get("mm_per_px", pipe.mm_per_px)
        with st.expander("SAW bead geometry", expanded=False):
            s = r.bead.summary(k)
            cols = st.columns(len(s))
            for c, (label, v) in zip(cols, s.items()):
                c.metric(label, v)
            bead_chart(r.bead.pos_px * k, r.bead.width_px * k, r.bead.offset_px * k, r.bead.nominal_width_px * k)


def frame_table(rows, budget_ms):
    return st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
        "Budget used": st.column_config.ProgressColumn(
            "Budget used", help=f"Processing time as a share of the {budget_ms:.0f} ms between triggers",
            min_value=0, max_value=100, format="%.0f%%"),
    })


def ring_montage(frames):
    """Stack the frames of a camera ring, each labelled, into one image."""
    out = []
    for cam, frame, r in frames:
        f = draw(frame, r.res)
        cv2.putText(f, f"camera {cam}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        out.append(cv2.copyMakeBorder(f, 0, 8, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    return np.vstack(out)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
pipe_key = st.sidebar.selectbox("Pipe type", list(PIPES), format_func=lambda k: PIPES[k].name,
                                help="Selects the seam model, bead measurements, sample images and camera layout.")
pipe = PIPES[pipe_key]
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.10, 0.05,
                         help="Lower = catch more defects but more false alarms.")
mode = st.sidebar.radio("Inference mode", ["Auto", "Tiled", "Single pass"],
                        help="Auto tiles images that are much larger than the tile size. "
                             "Tiling keeps small defects visible on high-resolution frames.")
tile = st.sidebar.select_slider("Tile size (px)", [160, 192, 224, 256, 320, 400, 512, 640], value=224,
                                help="Choose so one tile covers about the same physical area as one training image.")
overlap = st.sidebar.slider("Tile overlap", 0.0, 0.5, 0.25, 0.05)
seam_mode = st.sidebar.selectbox(
    "Weld seam localisation", SEAM_MODES[pipe_key], disabled=pipe.seam == "none",
    help="How the seam lies in the image: horizontal = pipe runs left-right, vertical = top-bottom, "
         "angled = a spiral seam crossing the image at the helix angle, auto = try horizontal and vertical. "
         "Defects on the seam are escalated." + (" A seamless pipe has no seam." if pipe.seam == "none" else ""))
bead_on = st.sidebar.checkbox("Measure the SAW bead (width, tracking, undercut, porosity)", value=True,
                              disabled=pipe_key not in SAW)
show_gt = st.sidebar.checkbox("Overlay ground truth (white) when available", value=False)

with st.sidebar.expander("Line sizing calculator"):
    speed = st.number_input("Line speed (m/min)", 0.5, 200.0, float(pipe.speed_m_min), 0.5, key=f"calc_{pipe_key}")
    mpp = st.number_input("Resolution (mm/px)", 0.01, 1.0, float(pipe.mm_per_px), 0.01, format="%.2f",
                          key=f"calc_mpp_{pipe_key}")
    c_al, c_ac = st.columns(2)
    px_al = c_al.number_input("Sensor px along", 256, 16384, 2448, 64)
    px_ac = c_ac.number_input("Sensor px across", 256, 16384, 2048, 64)
    od = st.number_input("Pipe OD (mm, optional)", 0.0, 3500.0, 0.0, 10.0)
    ls = LineSetup(speed_m_min=speed, mm_per_px=mpp, frame_px_along=int(px_al), frame_px_across=int(px_ac))
    st.table(pd.DataFrame(ls.summary(od_mm=od or None), columns=["", "value"]).set_index(""))

# ---------------------------------------------------------------- header
st.title(f"{pipe.name} Pipe Inspection")
st.caption(pipe.summary)

if not DEFAULT_WEIGHTS.exists():
    st.error(f"Model weights not found at `{DEFAULT_WEIGHTS}`. Run `python train.py` first.")
    st.stop()
code_key = max(f.stat().st_mtime for f in (ROOT / "weldvision").glob("*.py"))
detector = load_detector(code_key, DEFAULT_WEIGHTS.stat().st_mtime)
code_key = (code_key, DEFAULT_WEIGHTS.stat().st_mtime)
ON_GPU = detector.device != "cpu"
if not ON_GPU:
    st.caption("Running on CPU: processing times are several times slower than on a GPU, "
               "so the line simulation's time budget is not representative of production hardware.")

tab_prod, tab_seam, tab_line, tab_guide = st.tabs(
    ["Production (full frame)", "Seam-only", "Line simulation", "Pipe type guide"])

# ---------------------------------------------------------------- image selection (first two tabs)
images = []  # (name, bgr, gt or None)
with tab_prod:
    src = st.radio("Image source", ["Sample images", "Upload"], horizontal=True,
                   help="The same images are inspected in the Seam-only tab.")
    if src == "Upload":
        for f in st.file_uploader("Pipe images", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                                  accept_multiple_files=True) or []:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                images.append((f.name, img, None))
    else:
        sample_dirs = {f"{pipe.name} frames (synthetic line camera)": ROOT / pipe.sample_dir,
                       "NEU-DET test patches (held-out)": NEU_TEST}
        c1, c2 = st.columns([1, 2])
        folder = sample_dirs[c1.selectbox("Sample set", list(sample_dirs), key=f"set_{pipe_key}")]
        files = sorted(folder.glob("*.jpg")) if folder.exists() else []
        if not files:
            st.info(f"No samples in `{folder}`. Run `python scripts/make_demo_frames.py --pipe {pipe_key}`.")
        else:
            for name in c2.multiselect("Images", [f.name for f in files], default=[files[min(1, len(files) - 1)].name],
                                       key=f"img_{pipe_key}_{folder.name}"):
                p = folder / name
                gj = p.with_suffix(".json")
                images.append((name, cv2.imread(str(p)), json.loads(gj.read_text()) if gj.exists() else None))

full = [inspect(img, False, gt) for _, img, gt in images]

# ---------------------------------------------------------------- tab 1: production
with tab_prod:
    for (name, img, gt), r in zip(images, full):
        show_result(name, img, gt, r)

# ---------------------------------------------------------------- tab 2: seam-only
with tab_seam:
    if pipe.seam == "none":
        st.info("A seamless pipe has no weld seam, so there is no band to narrow the search to: the whole "
                "surface, all the way round, has to be inspected. See the Line simulation tab for the camera ring.")
    else:
        st.caption("Only the weld-seam band (plus half a tile of margin) is analysed; the dimmed area is not "
                   "inspected. This is the mode for a camera dedicated to the seam. Images are chosen in the "
                   "Production tab." + (" On a spiral seam the tiles are picked along the slanted band." if pipe_key == "hsaw" else ""))
        if seam_mode == "off":
            st.warning("Seam localisation is off in the sidebar, so the whole frame is analysed here too.")
        if not images:
            st.info("Select images in the Production tab.")
        for (name, img, gt), r_full in zip(images, full):
            r = inspect(img, True, gt)
            if r.seam is None and seam_mode != "off":
                st.warning(f"{name}: no seam found, so the whole frame was analysed.")
            show_result(name, img, gt, r, baseline=r_full)

# ---------------------------------------------------------------- tab 3: line simulation
CAMERA_TEXT = {
    "erw": "A synthetic ERW pipe with real NEU-DET test defects moves past a seam camera (1024 px across, 0.1 mm/px).",
    "lsaw": "A synthetic LSAW seam (raised SAW bead with planted bead defects, plus NEU-DET surface defects) moves "
            "past a camera behind the outside welding head at welding speed (1024 px across, 0.1 mm/px).",
    "hsaw": "A spiral pipe turns and advances, so the helical seam slides along itself under a fixed camera behind "
            "the outside welding head. The camera therefore sees a straight bead moving at welding speed; positions "
            "along the helix are converted to axial position and clock position on the pipe.",
    "seamless": "No seam: a ring of cameras images the whole circumference as the pipe moves through the inspection "
                "station. The synthetic surface is the unrolled pipe; each camera sees its arc plus overlap, and all "
                "cameras share one GPU's time budget.",
}
LINE_SIM_DIR = ROOT / "results" / ("line_sim" if pipe_key == "erw" else f"line_sim_{pipe_key}")

with tab_line:
    st.caption(CAMERA_TEXT[pipe_key] + " Frames are triggered by an encoder with 20% overlap; each defect is "
               "logged once. Confidence and tile size come from the sidebar.")
    c = st.columns([1, 1, 1, 1.4, 1])
    l_speed = c[0].number_input("Line speed (m/min)" if pipe_key != "hsaw" else "Welding speed (m/min)",
                                0.5, 120.0, float(pipe.speed_m_min), 0.5, key=f"sim_speed_{pipe_key}",
                                help=pipe.speed_note)
    l_length = c[1].select_slider("Pipe length (m)" if pipe_key != "hsaw" else "Seam length (m)", [1, 2, 3, 4],
                                  value=2 if pipe_key != "seamless" else 1, key=f"sim_length_{pipe_key}")
    l_defects = c[2].slider("Defects placed", 0, 20, 12, key=f"sim_defects_{pipe_key}")
    l_area = c[3].radio("Analyse", ["Full frame", "Seam band only"], horizontal=True, key=f"sim_area_{pipe_key}",
                        disabled=pipe.seam == "none")
    l_seed = c[4].number_input("Random seed", 0, 9999, 11, key=f"sim_seed_{pipe_key}")
    l_od, l_strip, l_cams = sim.DEFAULT_OD.get(pipe_key), sim.DEFAULT_STRIP_WIDTH, None
    if pipe_key == "hsaw":
        g = st.columns(4)
        l_od = g[0].number_input("Pipe OD (mm)", 400.0, 3500.0, 1016.0, 10.0)
        l_strip = g[1].number_input("Coil (strip) width (mm)", 500.0, 2500.0, 1500.0, 10.0)
        try:
            a = helix_angle_deg(l_od, l_strip)
            g[2].metric("Seam angle to the axis", f"{a:.1f}°")
            g[3].metric("Weld per metre of pipe", f"{weld_per_pipe_metre(l_od, l_strip):.2f} m",
                        help=f"Pipe advances at {l_speed * math.cos(math.radians(a)):.2f} m/min")
        except ValueError as e:
            st.error(str(e))
            st.stop()
    if pipe_key == "seamless":
        g = st.columns(3)
        l_od = g[0].number_input("Pipe OD (mm)", 20.0, 660.0, 168.3, 0.1, help="168.3 mm = 6 in NPS")
        l_cams = g[1].slider("Cameras in the ring", 3, 12, LineSetup().cameras_for_full_circumference(l_od)[0],
                             help="About 60° of arc per camera keeps the viewing angle reasonable.")
        g[2].metric("Circumference", f"{math.pi * l_od:.0f} mm")
    l_realtime = st.checkbox("Pace frames at real line speed (watch it as the line would see it)", value=False)

    if st.button("Run line simulation", type="primary"):
        line, ring = sim.line_setup(pipe_key, l_speed, None, None, l_od, l_cams)
        k = line.mm_per_px
        strip, gt = cached_pipe(pipe_key, l_length * 1000, l_defects, l_seed, k,
                                ring[1] if ring else line.frame_px_across)
        n_frames = len(sim.frame_starts(strip.shape[1], line))
        budget_ms = line.frame_budget_s * 1000

        progress = st.progress(0.0, text="Starting the line...")
        live_img, live_table = st.empty(), st.empty()
        rows = []
        for ev in sim.run_line(detector, pipe_key, strip, line, conf, tile, l_area == "Seam band only", ring,
                               l_realtime):
            i, x0 = ev["i"], ev["x0"]
            rows.append({"Trigger": i, "Position (m)": round(x0 * k / 1000, 3),
                         "Defects": sum(len(r.res.detections) for _, _, r in ev["frames"]),
                         "Processing (ms)": round(ev["dt"] * 1000), "Budget used": ev["dt"] * 1000 / budget_ms * 100})
            progress.progress((i + 1) / n_frames, text=f"Trigger {i + 1} of {n_frames} · at {x0 * k / 1000:.2f} m")
            view = ring_montage(ev["frames"]) if ring else draw(ev["frames"][0][1], ev["frames"][0][2].res)
            live_img.image(to_rgb(view), width="stretch",
                           caption=f"{'All cameras' if ring else 'Camera frame'} at trigger {i} · "
                                   f"{x0 * k / 1000:.3f}–{(x0 + line.frame_px_along) * k / 1000:.3f} m")
            with live_table.container():
                frame_table(rows, budget_ms)
        progress.empty()
        live_img.empty()
        live_table.empty()

        tracker = ev["tracker"]
        found, false = sim.score(tracker.defects, gt)
        overview = sim.render_overview(strip, gt, tracker.defects, k, ring=ring)
        LINE_SIM_DIR.mkdir(parents=True, exist_ok=True)
        sim.write_log(tracker.defects, LINE_SIM_DIR / "defect_log.csv", pipe_key, l_od, l_strip)
        cv2.imwrite(str(LINE_SIM_DIR / "overview.jpg"), overview)
        st.session_state[f"line_sim_{pipe_key}"] = dict(
            line=line, ring=ring, rows=rows, defects=tracker.defects, raw=tracker.raw, gt=gt, found=found,
            false=false, by_class=sim.score_by_class(tracker.defects, gt), overview=overview,
            bead=np.concatenate(ev["bead"], axis=1) if ev["bead"] else None,
            log=sim.log_rows(tracker.defects, pipe_key, l_od, l_strip),
            verdict=verdict(tracker.defects, seam_band=True),
            settings=f"{l_speed:g} m/min · {l_length} m · {l_area.lower() if pipe.seam != 'none' else 'full surface'} · "
                     f"conf {conf:.2f} · tile {tile}px" + (f" · {ring[0]} cameras, OD {l_od:g} mm" if ring else ""))

    s = st.session_state.get(f"line_sim_{pipe_key}")
    if s is not None:
        ms = np.array([r["Processing (ms)"] for r in s["rows"]])
        budget_ms = s["line"].frame_budget_s * 1000
        st.divider()
        verdict_banner(s["verdict"][0], "Simulated pipe", s["verdict"][1])
        st.caption(s["settings"])
        m = st.columns(6)
        m[0].metric("Triggers", len(s["rows"]), help=f"One every {s['line'].advance_mm:.0f} mm of travel"
                    + (f"; {s['ring'][0]} frames per trigger" if s["ring"] else ""))
        m[1].metric("Mean / max time", f"{ms.mean():.0f} / {ms.max():.0f} ms",
                    help="Per trigger, all cameras together" if s["ring"] else None)
        m[2].metric("Time per trigger allowed", f"{budget_ms:.0f} ms")
        m[3].metric("GPU busy" if ON_GPU else "CPU busy", f"{ms.mean() / budget_ms:.0%}",
                    help="Keeps up" if ms.max() <= budget_ms else "Falls behind the line")
        m[4].metric("Defects logged", len(s["defects"]), help=f"{s['raw'] - len(s['defects'])} overlap duplicates merged")
        m[5].metric("True defects found", f"{s['found']}/{len(s['gt'])}", help=f"{s['false']} false alarms (IoU < 0.3)")
        if ms.max() > budget_ms:
            st.error("Processing is slower than the line: frames would queue up. "
                     + ("A camera ring multiplies the work: use a faster GPU or TensorRT, a coarser resolution, "
                        "or one GPU per few cameras." if s["ring"] else "Use seam-only mode or a faster GPU."))

        st.subheader("Whole pipe" + (" (unrolled: circumference top to bottom, camera boundaries dashed)" if s["ring"] else ""))
        st.image(to_rgb(s["overview"]), width="stretch",
                 caption="0.5 m per row · white = true defect position · colour = detected")
        if s["bead"] is not None:
            st.subheader("Bead along the seam")
            b = s["bead"]
            order = np.argsort(b[0])
            bead_chart(b[0][order], b[1][order], b[2][order], float(np.median(b[1])))
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Defect log")
            log = pd.DataFrame(s["log"])
            if not log.empty:
                log["defect"] = log["defect"].map(lambda c: info(c).label)
            st.dataframe(log, width="stretch", hide_index=True)
            st.download_button("Download defect log (CSV)", log.to_csv(index=False), "defect_log.csv", "text/csv")
        with right:
            st.subheader("Found by class")
            st.dataframe(pd.DataFrame([{"Defect": info(c).label, "Found": f"{a}/{n}",
                                        "Source": "bead measurement" if c in BEAD_CLASSES else "detector"}
                                       for c, (a, n) in sorted(s["by_class"].items())]),
                         width="stretch", hide_index=True)
            st.subheader("Per-trigger timing")
            frame_table(s["rows"], budget_ms)
    elif (LINE_SIM_DIR / "overview.jpg").exists():
        st.info(f"Showing the last saved run from `{LINE_SIM_DIR.relative_to(ROOT).as_posix()}` (e.g. from "
                f"`python scripts/simulate_line.py --pipe {pipe_key}`). Press **Run line simulation** for a live run.")
        st.image(to_rgb(cv2.imread(str(LINE_SIM_DIR / "overview.jpg"))), width="stretch",
                 caption="0.5 m per row · white = true defect position · colour = detected")
        if (LINE_SIM_DIR / "defect_log.csv").exists():
            st.subheader("Defect log")
            st.dataframe(pd.read_csv(LINE_SIM_DIR / "defect_log.csv"), width="stretch", hide_index=True)

# ---------------------------------------------------------------- tab 4: pipe type guide
with tab_guide:
    st.subheader(pipe.name)
    st.write(pipe.summary)
    g = st.columns(4)
    g[0].metric("Typical OD", pipe.od_range)
    g[1].metric("Typical wall", pipe.wt_range)
    g[2].metric("Seam", pipe.seam)
    g[3].metric("Demo line speed", f"{pipe.speed_m_min:g} m/min", help=pipe.speed_note)
    left, right = st.columns(2)
    with left:
        st.markdown("**How it is made**")
        st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(pipe.process, 1)))
        st.markdown(f"**Camera in this demo:** {pipe.camera}.")
        for note in pipe.notes:
            st.markdown(f"- {note}")
    with right:
        st.markdown("**Type-specific defects a camera can see** (target classes for a real dataset)")
        st.dataframe(pd.DataFrame([{"Defect": k.replace("_", " "), "What it is": v}
                                   for k, v in pipe.visible_defects.items()]), width="stretch", hide_index=True)
        st.markdown("**Not visible to a camera** (X-ray, ultrasonic, eddy-current or flux-leakage testing)")
        st.markdown("\n".join(f"- {d}" for d in pipe.hidden_defects))
    st.markdown("**Compare the four types**")
    st.dataframe(pd.DataFrame([{"Type": p.name, "Seam": p.seam, "OD": p.od_range, "Wall": p.wt_range,
                                "Demo speed (m/min)": p.speed_m_min, "Camera": p.camera} for p in PIPES.values()]),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------- reference
st.divider()
with st.expander("Defect catalogue used by the demo"):
    st.dataframe(pd.DataFrame([{"Class": d.label, "Found by": "bead measurement" if d.name in BEAD_CLASSES
                                else "detector (NEU-DET classes)",
                                "Severity": d.severity, "What it is": d.description,
                                "Why it matters": d.erw_relevance} for d in DEFECTS.values()]),
                 width="stretch", hide_index=True)
    st.caption("Severity drives the verdict: a reject-class defect fails the pipe, minor ones go to review, "
               "and any defect on the weld seam fails it.")
