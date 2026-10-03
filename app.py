"""ERW pipe weld & surface defect inspection - demo UI.

    streamlit run app.py

Tabs:
    Production       whole-frame inspection of the selected images
    Seam-only        the same images, analysing only the weld-seam band (dedicated seam camera)
    Line simulation  a pipe moving past the camera at line speed, logged by position along the pipe
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from weldvision import DEFECTS, ERW_SEAM_DEFECTS, DefectDetector, draw, info, locate_seam, verdict
from weldvision import detector as detector_module
from weldvision.detector import DEFAULT_WEIGHTS
from weldvision.line import LineSetup, PipeTracker

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from download_data import ensure_test_split  # noqa: E402

ensure_test_split()  # a fresh deployment has no data/ folder; sample patches and the line simulation need NEU test images
import simulate_line as sim  # noqa: E402

SAMPLE_DIRS = {
    "HD pipe frames (synthetic line camera)": ROOT / "samples" / "hd_frames",
    "NEU-DET test patches (held-out)": ROOT / "data" / "NEU-DET" / "images" / "test",
}
LINE_SIM_DIR = ROOT / "results" / "line_sim"
SIM_MM_PER_PX, SIM_ACROSS_PX = 0.1, 1024  # seam camera: 2448 x 1024 px sensor window at 0.1 mm/px
VERDICT_STYLE = {"PASS": ("#1b7f3b", "✅"), "REVIEW": ("#b26a00", "⚠️"), "FAIL": ("#b3261e", "⛔")}

st.set_page_config(page_title="ERW Weld Defect Inspection", page_icon="🔍", layout="wide")


@st.cache_resource
def load_detector(code_mtime, weights_mtime):
    # The arguments are only cache keys: editing the detector code or retraining the model
    # invalidates the cached instance instead of serving an object built from stale code.
    det = DefectDetector(DEFAULT_WEIGHTS)
    det.predict(np.zeros((448, 448, 3), np.uint8), tile=224)  # warm-up, so the first timing shown is real
    return det


@st.cache_data(max_entries=2, show_spinner="Generating the pipe...")
def cached_pipe(length_mm, n_defects, seed):
    return sim.build_pipe(length_mm, SIM_MM_PER_PX, SIM_ACROSS_PX, n_defects, seed)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_ground_truth(img, gt):
    out = img.copy()
    lw = max(1, round(max(img.shape[:2]) / 500))
    for g in gt.get("defects", []):
        x1, y1, x2, y2 = g["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), lw)
    return out


def dim_outside(img, roi):
    """Darken everything outside the analysed region so it is obvious what was not inspected."""
    x1, y1, x2, y2 = roi
    out = (img * 0.35).astype(np.uint8)
    out[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return out


def verdict_banner(v, name, reason):
    color, icon = VERDICT_STYLE[v]
    st.markdown(
        f"<div style='padding:10px 16px;border-radius:8px;background:{color};color:white;font-size:1.15rem'>"
        f"{icon} <b>{v}</b> &nbsp;·&nbsp; {name} &nbsp;·&nbsp; {reason}</div>", unsafe_allow_html=True)


@st.cache_data(max_entries=64, show_spinner="Inspecting...")
def _inspect(_det, model_key, img, seam_only, conf, mode, tile, overlap, seam_mode):
    # Cached so that clicking any widget does not re-run every selected image (seconds per frame on a CPU host).
    h, w = img.shape[:2]
    use_tile = {"Single pass": 0, "Tiled": tile, "Auto": tile if max(h, w) > tile * 1.25 else 0}[mode]
    t0 = time.perf_counter()
    seam = None if seam_mode == "off" else locate_seam(img, seam_mode)
    roi = seam.band(img.shape, pad=tile // 2) if (seam_only and seam) else None
    res = _det.predict(img, conf=conf, tile=use_tile, overlap=overlap, seam=seam, roi=roi)
    ms = (time.perf_counter() - t0) * 1000
    v, reason = verdict(res.detections, seam)
    return dict(res=res, seam=seam, roi=roi, ms=ms, verdict=v, reason=reason)


def inspect(img, seam_only):
    """Run the pipeline on one image. Returns a dict with everything the tabs display."""
    return _inspect(detector, model_key, img, seam_only, conf, mode, tile, overlap, seam_mode)


def show_result(name, img, gt, r, baseline=None):
    """Verdict banner, metrics, annotated image and detection table for one inspected image.
    baseline: the production result for the same image, to show the seam-only speed-up."""
    res, seam = r["res"], r["seam"]
    st.divider()
    verdict_banner(r["verdict"], name, r["reason"])

    m = st.columns(5)
    m[0].metric("Defects", len(res.detections))
    m[1].metric("On weld seam", sum(d.on_seam for d in res.detections))
    m[2].metric("Seam", f"{seam.orientation[0].upper()} @ {seam.center}px" if seam else "not found")
    if baseline is None:
        m[3].metric("Tiles", res.n_tiles)
        m[4].metric("Time", f"{r['ms']:.0f} ms")
    else:
        b = baseline["res"]
        m[3].metric("Tiles", res.n_tiles, delta=f"{res.n_tiles - b.n_tiles} vs full frame", delta_color="inverse")
        m[4].metric("Time", f"{r['ms']:.0f} ms", delta=f"{r['ms'] - baseline['ms']:.0f} ms vs full frame",
                    delta_color="inverse")

    annotated = draw(img, res)
    if r["roi"] is not None:
        annotated = dim_outside(annotated, r["roi"])
    if show_gt and gt:
        annotated = draw_ground_truth(annotated, gt)
    h, w = img.shape[:2]
    left, right = st.columns([3, 2]) if w <= h * 1.3 else (st.container(), st.container())
    left.image(to_rgb(annotated), caption=f"{w}×{h}px", width="stretch")

    if res.detections:
        right.dataframe(pd.DataFrame([{
            "Defect": info(d.cls_name).label,
            "Confidence": round(d.conf, 3),
            "Severity": info(d.cls_name).severity,
            "On seam": "yes" if d.on_seam else "",
            "Box (x1,y1,x2,y2)": tuple(int(v) for v in d.box),
        } for d in res.detections]), width="stretch", hide_index=True)
    if gt is not None:
        right.caption(f"Ground truth for this frame: {len(gt['defects'])} defect boxes "
                      f"({sum(g['on_seam'] for g in gt['defects'])} on seam).")


def defect_log_frame(defects):
    return pd.DataFrame([{
        "Defect": info(d.cls_name).label,
        "Severity": info(d.cls_name).severity,
        "Confidence": round(d.conf, 3),
        "Start (mm)": round(d.start_mm, 1),
        "End (mm)": round(d.end_mm, 1),
        "Length (mm)": round(d.length_mm, 1),
        "Across (mm)": round(d.across_mm, 1),
        "On seam": "yes" if d.on_seam else "",
        "Frames": " ".join(map(str, d.frames)),
    } for d in sorted(defects, key=lambda d: d.start_mm)])


def frame_table(rows, budget_ms):
    return st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
        "Budget used": st.column_config.ProgressColumn(
            "Budget used", help=f"Processing time as a share of the {budget_ms:.0f} ms between frames",
            min_value=0, max_value=100, format="%.0f%%"),
    })


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.10, 0.05,
                         help="Lower = catch more defects but more false alarms.")
mode = st.sidebar.radio("Inference mode", ["Auto", "Tiled", "Single pass"],
                        help="Auto tiles images that are much larger than the tile size. "
                             "Tiling keeps small defects visible on high-resolution frames.")
tile = st.sidebar.select_slider("Tile size (px)", [160, 192, 224, 256, 320, 400, 512, 640], value=224,
                                help="Choose so one tile covers about the same physical area as one training image.")
overlap = st.sidebar.slider("Tile overlap", 0.0, 0.5, 0.25, 0.05)
seam_mode = st.sidebar.selectbox("Weld seam localisation", ["auto", "horizontal", "vertical", "off"],
                                 help="Finds the longitudinal seam band; defects on it are escalated.")
show_gt = st.sidebar.checkbox("Overlay ground truth (white) when available", value=False)

with st.sidebar.expander("Line sizing calculator"):
    speed = st.number_input("Line speed (m/min)", 0.5, 200.0, 8.0, 0.5)
    mpp = st.number_input("Resolution (mm/px)", 0.01, 1.0, 0.1, 0.01, format="%.2f")
    c_al, c_ac = st.columns(2)
    px_al = c_al.number_input("Sensor px along", 256, 16384, 2448, 64)
    px_ac = c_ac.number_input("Sensor px across", 256, 16384, 2048, 64)
    od = st.number_input("Pipe OD (mm, optional)", 0.0, 2000.0, 0.0, 10.0)
    ls = LineSetup(speed_m_min=speed, mm_per_px=mpp, frame_px_along=int(px_al), frame_px_across=int(px_ac))
    st.table(pd.DataFrame(ls.summary(od_mm=od or None), columns=["", "value"]).set_index(""))

# ---------------------------------------------------------------- header
st.title("ERW Pipe Weld & Surface Defect Inspection")
st.caption("Pipeline: weld-seam localisation, tiled YOLO defect detection, rule-based PASS / REVIEW / FAIL decision.")

if not DEFAULT_WEIGHTS.exists():
    st.error(f"Model weights not found at `{DEFAULT_WEIGHTS}`. Run `python train.py` first.")
    st.stop()
model_key = (Path(detector_module.__file__).stat().st_mtime, DEFAULT_WEIGHTS.stat().st_mtime)
detector = load_detector(*model_key)
ON_GPU = detector.device != "cpu"
if not ON_GPU:
    st.caption("Running on CPU: processing times are several times slower than on a GPU, "
               "so the line simulation's time budget is not representative of production hardware.")

tab_prod, tab_seam, tab_line = st.tabs(["Production (full frame)", "Seam-only", "Line simulation"])

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
        c1, c2 = st.columns([1, 2])
        folder = SAMPLE_DIRS[c1.selectbox("Sample set", list(SAMPLE_DIRS))]
        files = sorted(folder.glob("*.jpg")) if folder.exists() else []
        if not files:
            st.info(f"No samples in `{folder}`. For HD frames run `python scripts/make_demo_frames.py`.")
        else:
            for name in c2.multiselect("Images", [f.name for f in files], default=[files[min(1, len(files) - 1)].name]):
                p = folder / name
                gj = p.with_suffix(".json")
                images.append((name, cv2.imread(str(p)), json.loads(gj.read_text()) if gj.exists() else None))

full = [inspect(img, seam_only=False) for _, img, _ in images]

# ---------------------------------------------------------------- tab 1: production
with tab_prod:
    for (name, img, gt), r in zip(images, full):
        show_result(name, img, gt, r)

# ---------------------------------------------------------------- tab 2: seam-only
with tab_seam:
    st.caption("Only the weld-seam band (plus half a tile of margin) is analysed; the dimmed area is not inspected. "
               "This is the mode for a camera dedicated to the seam. Images are chosen in the Production tab.")
    if seam_mode == "off":
        st.warning("Seam localisation is off in the sidebar, so the whole frame is analysed here too.")
    if not images:
        st.info("Select images in the Production tab.")
    for (name, img, gt), r_full in zip(images, full):
        r = inspect(img, seam_only=True)
        if r["seam"] is None and seam_mode != "off":
            st.warning(f"{name}: no seam found, so the whole frame was analysed.")
        show_result(name, img, gt, r, baseline=r_full)

# ---------------------------------------------------------------- tab 3: line simulation
with tab_line:
    st.caption("A synthetic pipe with real NEU-DET test defects moves past a seam camera "
               f"({SIM_ACROSS_PX} px across, {SIM_MM_PER_PX} mm/px). Frames are triggered by an encoder with 20% "
               "overlap; each defect is logged once, by its position in mm from the pipe head. "
               "Confidence and tile size come from the sidebar.")
    c = st.columns([1, 1, 1, 1.4, 1])
    l_speed = c[0].number_input("Line speed (m/min)", 1.0, 60.0, 8.0, 1.0, key="sim_speed")
    l_length = c[1].select_slider("Pipe length (m)", [1, 2, 3, 4], value=2, key="sim_length")
    l_defects = c[2].slider("Defects placed", 0, 20, 12, key="sim_defects")
    l_area = c[3].radio("Analyse", ["Full frame", "Seam band only"], horizontal=True, key="sim_area")
    l_seed = c[4].number_input("Random seed", 0, 9999, 11, key="sim_seed")
    l_realtime = st.checkbox("Pace frames at real line speed (watch it as the line would see it)", value=False)

    if st.button("Run line simulation", type="primary"):
        line = LineSetup(speed_m_min=l_speed, mm_per_px=SIM_MM_PER_PX, frame_px_across=SIM_ACROSS_PX)
        strip, gt = cached_pipe(l_length * 1000, l_defects, l_seed)
        starts = sim.frame_starts(strip.shape[1], line)
        budget_ms = line.frame_budget_s * 1000

        progress = st.progress(0.0, text="Starting the line...")
        live_img, live_table = st.empty(), st.empty()
        tracker, rows = PipeTracker(SIM_MM_PER_PX), []
        t_line0 = time.perf_counter()
        for i, x0 in enumerate(starts):
            if l_realtime:  # wait until the encoder would fire for this frame
                time.sleep(max(0.0, t_line0 + x0 * SIM_MM_PER_PX / line.speed_mm_s - time.perf_counter()))
            frame = strip[:, x0:x0 + line.frame_px_along]
            res, dt = sim.inspect_frame(detector, frame, conf, tile, l_area == "Seam band only")
            tracker.add(i, x0 * SIM_MM_PER_PX, res.detections)
            rows.append({"Frame": i, "Pipe position (m)": round(x0 * SIM_MM_PER_PX / 1000, 3),
                         "Defects": len(res.detections), "Processing (ms)": round(dt * 1000),
                         "Budget used": dt * 1000 / budget_ms * 100})
            progress.progress((i + 1) / len(starts), text=f"Frame {i + 1} of {len(starts)} · "
                                                           f"pipe at {x0 * SIM_MM_PER_PX / 1000:.2f} m")
            live_img.image(to_rgb(draw(frame, res)), width="stretch",
                           caption=f"Camera frame {i} · {x0 * SIM_MM_PER_PX / 1000:.3f}–"
                                   f"{(x0 + line.frame_px_along) * SIM_MM_PER_PX / 1000:.3f} m")
            with live_table.container():
                frame_table(rows, budget_ms)
        progress.empty()
        live_img.empty()
        live_table.empty()

        found, false = sim.score(tracker.defects, gt)
        overview = sim.render_overview(strip, gt, tracker.defects, SIM_MM_PER_PX)
        LINE_SIM_DIR.mkdir(parents=True, exist_ok=True)
        sim.write_log(tracker.defects, LINE_SIM_DIR / "defect_log.csv")
        cv2.imwrite(str(LINE_SIM_DIR / "overview.jpg"), overview)
        st.session_state["line_sim"] = dict(
            line=line, rows=rows, defects=tracker.defects, gt=gt, found=found, false=false,
            overview=overview, verdict=verdict(tracker.defects, seam_band=True),
            settings=f"{l_speed:g} m/min · {l_length} m of pipe · {l_area.lower()} · conf {conf:.2f} · tile {tile}px")

    s = st.session_state.get("line_sim")
    if s is not None:
        ms = np.array([r["Processing (ms)"] for r in s["rows"]])
        budget_ms = s["line"].frame_budget_s * 1000
        raw = sum(len(d.frames) for d in s["defects"])
        st.divider()
        verdict_banner(s["verdict"][0], "Simulated pipe", s["verdict"][1])
        st.caption(s["settings"])
        m = st.columns(6)
        m[0].metric("Frames", len(s["rows"]), help=f"One every {s['line'].advance_mm:.0f} mm of travel")
        m[1].metric("Mean / max time", f"{ms.mean():.0f} / {ms.max():.0f} ms")
        m[2].metric("Time per frame allowed", f"{budget_ms:.0f} ms")
        m[3].metric("GPU busy" if ON_GPU else "CPU busy", f"{ms.mean() / budget_ms:.0%}",
                    help="Keeps up" if ms.max() <= budget_ms else "Falls behind the line")
        m[4].metric("Defects logged", len(s["defects"]), help=f"{raw - len(s['defects'])} overlap duplicates merged")
        m[5].metric("True defects found", f"{s['found']}/{len(s['gt'])}", help=f"{s['false']} false alarms (IoU < 0.3)")
        if ms.max() > budget_ms:
            st.error("Processing is slower than the line: frames would queue up. Use seam-only mode or a faster GPU.")

        st.subheader("Whole pipe")
        st.image(to_rgb(s["overview"]), width="stretch",
                 caption="0.5 m per row · white = true defect position · colour = detected")
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Defect log")
            log = defect_log_frame(s["defects"])
            st.dataframe(log, width="stretch", hide_index=True)
            st.download_button("Download defect log (CSV)", log.to_csv(index=False), "defect_log.csv", "text/csv")
        with right:
            st.subheader("Per-frame timing")
            frame_table(s["rows"], budget_ms)
    elif (LINE_SIM_DIR / "overview.jpg").exists():
        st.info("Showing the last saved run from `results/line_sim` (e.g. from `python scripts/simulate_line.py`). "
                "Press **Run line simulation** for a live run with the settings above.")
        st.image(to_rgb(cv2.imread(str(LINE_SIM_DIR / "overview.jpg"))), width="stretch",
                 caption="0.5 m per row · white = true defect position · colour = detected")
        if (LINE_SIM_DIR / "defect_log.csv").exists():
            st.subheader("Defect log")
            st.dataframe(pd.read_csv(LINE_SIM_DIR / "defect_log.csv"), width="stretch", hide_index=True)

# ---------------------------------------------------------------- reference
st.divider()
with st.expander("Defect catalogue used by the demo model"):
    st.dataframe(pd.DataFrame([{"Class": d.label, "Severity": d.severity, "What it is": d.description,
                                "Why it matters on ERW pipe": d.erw_relevance} for d in DEFECTS.values()]),
                 width="stretch", hide_index=True)
with st.expander("Next step: ERW seam-specific classes (need images from your line)"):
    st.dataframe(pd.DataFrame([{"Class": k, "Description": v} for k, v in ERW_SEAM_DEFECTS.items()]),
                 width="stretch", hide_index=True)
    st.caption("Internal defects such as cold weld, penetrators and hook cracks are not visible to a camera; "
               "those stay with ultrasonic / eddy-current testing.")
