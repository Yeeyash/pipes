"""ERW pipe weld & surface defect inspection - demo UI.

    streamlit run app.py
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from weldvision import DEFECTS, ERW_SEAM_DEFECTS, DefectDetector, draw, info, locate_seam, verdict
from weldvision import detector as detector_module
from weldvision.detector import DEFAULT_WEIGHTS
from weldvision.line import LineSetup

ROOT = Path(__file__).resolve().parent
SAMPLE_DIRS = {
    "HD pipe frames (synthetic line camera)": ROOT / "samples" / "hd_frames",
    "NEU-DET test patches (held-out)": ROOT / "data" / "NEU-DET" / "images" / "test",
}
VERDICT_STYLE = {"PASS": ("#1b7f3b", "✅"), "REVIEW": ("#b26a00", "⚠️"), "FAIL": ("#b3261e", "⛔")}

st.set_page_config(page_title="ERW Weld Defect Inspection", page_icon="🔍", layout="wide")


@st.cache_resource
def load_detector(code_mtime, weights_mtime):
    # The arguments are only cache keys: editing the detector code or retraining the model
    # invalidates the cached instance instead of serving an object built from stale code.
    return DefectDetector(DEFAULT_WEIGHTS)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_ground_truth(img, gt):
    out = img.copy()
    lw = max(1, round(max(img.shape[:2]) / 500))
    for g in gt.get("defects", []):
        x1, y1, x2, y2 = g["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), lw)
    return out


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
seam_only = st.sidebar.checkbox("Inspect seam band only", value=False,
                                help="Analyse just the weld band (plus half a tile margin). About 3x faster; "
                                     "use for a dedicated seam camera.")
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
detector = load_detector(Path(detector_module.__file__).stat().st_mtime, DEFAULT_WEIGHTS.stat().st_mtime)

# ---------------------------------------------------------------- input
src = st.radio("Image source", ["Sample images", "Upload"], horizontal=True)
images = []  # (name, bgr, gt or None)
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

# ---------------------------------------------------------------- run
for name, img, gt in images:
    h, w = img.shape[:2]
    use_tile = {"Single pass": 0, "Tiled": tile, "Auto": tile if max(h, w) > tile * 1.25 else 0}[mode]

    t0 = time.perf_counter()
    seam = None if seam_mode == "off" else locate_seam(img, seam_mode)
    roi = seam.band(img.shape, pad=tile // 2) if (seam_only and seam) else None
    res = detector.predict(img, conf=conf, tile=use_tile, overlap=overlap, seam=seam, roi=roi)
    ms = (time.perf_counter() - t0) * 1000
    v, reason = verdict(res.detections, seam)
    color, icon = VERDICT_STYLE[v]

    st.divider()
    st.markdown(
        f"<div style='padding:10px 16px;border-radius:8px;background:{color};color:white;font-size:1.15rem'>"
        f"{icon} <b>{v}</b> &nbsp;·&nbsp; {name} &nbsp;·&nbsp; {reason}</div>", unsafe_allow_html=True)

    m = st.columns(5)
    m[0].metric("Defects", len(res.detections))
    m[1].metric("On weld seam", sum(d.on_seam for d in res.detections))
    m[2].metric("Seam", f"{seam.orientation[0].upper()} @ {seam.center}px" if seam else "not found")
    m[3].metric("Tiles", res.n_tiles)
    m[4].metric("Time", f"{ms:.0f} ms")

    annotated = draw(img, res)
    if show_gt and gt:
        annotated = draw_ground_truth(annotated, gt)
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
