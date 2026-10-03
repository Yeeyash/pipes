# ERW Pipe Weld & Surface Defect Inspection (computer-vision demo)

Detects weld and surface defects on ERW (electric-resistance-welded) pipe images:

1. **Seam localisation** (`weldvision/seam.py`) finds the longitudinal weld-seam band so that
   defects on the seam, the critical zone of an ERW pipe, can be escalated.
2. **Tiled defect detection** (`weldvision/detector.py`) runs a YOLO11 detector over
   overlapping tiles. High-resolution line-camera frames are never shrunk to 320 px, which
   would erase pinholes and fine cracks. Pieces of one defect that cross tile borders are merged.
3. **Decision** (`weldvision/defects.py`) gives **PASS / REVIEW / FAIL** per image:
   a reject-class defect gives FAIL, a minor defect gives REVIEW, and any defect on the seam gives FAIL.

```
image ─► seam band ─► tiles ─► YOLO ─► NMS + cross-tile merge ─► on-seam flag ─► verdict
```

## Quick start

```bash
pip install -r requirements.txt          # install CUDA PyTorch first, see file
python scripts/download_data.py          # NEU-DET -> data/NEU-DET (--test-only: just what the app needs)
python scripts/make_negatives.py         # defect-free background tiles -> data/negatives
python train.py                          # ~20 min on an RTX 3050 4 GB -> models/weld_defects.pt
python scripts/make_demo_frames.py       # synthetic 2400x1200 pipe frames -> samples/hd_frames

streamlit run app.py                     # demo UI
python detect.py samples/hd_frames       # batch mode -> results/summary.csv, detections.csv, annotated/
python detect.py samples/hd_frames --seam-only   # seam band only, ~3x faster
python scripts/eval_frames.py            # score the full pipeline against ground truth
python scripts/simulate_line.py          # 2 m of pipe at 8 m/min: real-time check + defect log by position
```

## Deploying the demo

The repository holds everything the app needs except the NEU-DET test images, which the app
downloads (~3 MB) on first start. Training data is not needed. `requirements.txt` installs
CPU-only PyTorch, because free hosts have no GPU.

**Streamlit Community Cloud** (free; deploys straight from GitHub):
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and choose **Create app**.
3. Select the repository and branch, main file `app.py`, and under *Advanced settings* choose Python 3.12.
4. Deploy. The first build takes a few minutes. `packages.txt` adds the system libraries OpenCV needs.

**Docker** (Hugging Face Spaces with the Docker SDK, Render, Fly.io, or any VM):
```bash
docker build -t weldvision .
docker run -p 8501:8501 weldvision        # http://localhost:8501
```
On Hugging Face Spaces, set `app_port: 8501` in the Space's README header.

**Speed on a CPU host:** a 2400×1200 frame takes about 15 s for the full frame and about 5 s
for seam-only on 2 cores, against 0.4 s on the laptop GPU. Results are cached, so changing
another setting does not re-run them. The line-simulation tab therefore shows the CPU running
far over the frame budget. That is a property of the free host, not of the method. For a faster
CPU demo, export to OpenVINO or ONNX (`model.export(format="openvino")`), which is typically
2–3× faster on CPU.

## Data

| Source | What | Used for |
|---|---|---|
| [NEU-DET](https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected) | 1,797 hot-rolled steel images, 200×200 px, 6 classes, box labels | Training / val / test |
| `scripts/make_negatives.py` | 360 synthetic defect-free surface and seam tiles | Background (negative) examples |
| `scripts/make_demo_frames.py` | 2400×1200 synthetic pipe frames with a scarfed seam, containing NEU **test** defects at known positions | End-to-end demo and evaluation |

ERW pipe is formed from hot-rolled strip, so NEU-DET's classes (crazing, inclusion, patches,
pitted surface, rolled-in scale, scratches) are real defects found on pipe. The demo model
does **not** yet know seam-specific ERW defects. Those need images from your mill; see the roadmap.

## Results

YOLO11s, 320 px input, 100 epochs on an RTX 3050 Laptop GPU (4 GB), about 20 minutes.

**NEU-DET held-out test split (180 images, 420 boxes):**

| Class | AP50 (NEU only) | | Class | AP50 (NEU only) |
|---|---|---|---|---|
| scratches | 0.959 | | inclusion | 0.844 |
| patches | 0.922 | | rolled-in scale | 0.577 |
| pitted surface | 0.803 | | crazing | 0.341 |

| Model | test mAP50 | test mAP50-95 |
|---|---|---|
| NEU-DET only (`models/weld_defects_neu_only.pt`) | 0.741 | 0.436 |
| NEU-DET + background tiles (`models/weld_defects.pt`, **default**) | 0.733 | 0.432 |

These are in line with published YOLO results on NEU-DET. Crazing and rolled-in scale are
low-contrast and hard for every published method. On a real line, raking light helps both.

**End-to-end on the 6 synthetic HD frames (60 defect boxes, tile 224, IoU ≥ 0.3):**

| Model | conf | defects found | false alarms | clean frame |
|---|---|---|---|---|
| NEU only | 0.25 | 72% | 2 | PASS |
| NEU only | 0.10 | 77% | 16 | REVIEW |
| + background tiles | 0.25 | 68% | 3 | PASS |
| + background tiles | **0.10 (default)** | **77%** | **3** | **PASS** |

Every found defect also had the correct class. The median seam-centre error is 7 px, and
inference takes about 1 s per 2400×1200 frame (98 tiles) on the laptop GPU.

**Read these numbers with care.** The background tiles and the demo frames come from the same
synthetic surface generator (with different random seeds), so the false-alarm improvement is
optimistic. It shows the *mechanism* (clean examples let you lower the threshold and catch more
defects) but not the size of the gain on real pipe. Measure that on images from your line.

## Line integration at 8 m/min

At 8 m/min the pipe moves at **133 mm/s**, which is slow for machine vision. The camera, frame
rate and GPU are not the limiting factors. Lighting, heat and surface coverage are.

**Camera sizing** (5 MP area-scan, 2448×2048, 20% frame overlap; from `weldvision/line.py`, also in the app sidebar):

| Resolution | Field of view | Trigger every | Frame rate | Time per frame | Max strobe / exposure |
|---|---|---|---|---|---|
| 0.05 mm/px | 122 × 102 mm | 98 mm | 1.36 fps | 734 ms | 188 µs |
| **0.10 mm/px** | **245 × 205 mm** | **196 mm** | **0.68 fps** | **1,469 ms** | **375 µs** |
| 0.20 mm/px | 490 × 410 mm | 392 mm | 0.34 fps | 2,938 ms | 750 µs |

A line-scan camera at 0.1 mm/px needs only a 1.3 kHz line rate. Any industrial camera
manages these rates. Trigger from a **line encoder** (not a timer) so that every frame has
a known pipe position and speed changes don't leave gaps.

**Measured compute on the RTX 3050 laptop GPU** (tile 224):

| Workload | Time / frame | GPU load at 0.1 mm/px |
|---|---|---|
| Full 5 MP frame (180 tiles) | 596 ms | 41% |
| Seam band only (45 tiles) | 186 ms | 13% |
| 2448×1024 seam camera, full frame (`simulate_line.py`) | 290 ms | 20% |

One seam camera is comfortably real-time on this laptop. **Full-circumference** coverage of a
large pipe (e.g. OD 324 mm needs about 6 cameras at 0.1 mm/px, ~20 MP/s) needs a desktop GPU
or a TensorRT FP16 export (`model.export(format="engine", half=True)`, typically 2–3× faster).

**Line simulation:** `python scripts/simulate_line.py` plays 2 m of synthetic pipe past the
camera at 8 m/min with encoder-triggered, overlapping frames. It writes
`results/line_sim/defect_log.csv`, which gives each defect's position in mm from the pipe
head, its extent, and whether it is on the seam; this is what a paint marker or reject gate
consumes. Defects seen in two overlapping frames are merged (11 duplicates in the default run).

### The pipe is hot where it leaves the welder

Directly after the squeeze rolls and external scarfing, the seam is still glowing (several
hundred °C). This drives most of the hardware design:

* **Glow drowns ordinary lighting.** Hot steel emits strongly in red and near-infrared. Use
  **blue (~450 nm) strobed LED or laser light with a matching narrow band-pass filter** on the
  lens. A short strobe (≤ 375 µs, as above) also limits how much glow the sensor collects.
* **Protect the camera.** Use a heat shield, a water- or air-cooled housing, and an **air knife**
  over the window against steam, coolant mist and scale. Choose a long standoff and lens.
* **The seam drifts.** It wanders a few degrees around the pipe. Locating the seam in every
  frame (`locate_seam`) handles small drift; give the field of view enough margin across the seam.

**Where to mount cameras:**

| Position | Sees | Pros / cons |
|---|---|---|
| A. Just after external scarfing (hot) | Bead trim height, gouges, open seam, contact burns | Immediate operator feedback (e.g. a worn scarfing tool); hardest environment |
| B. After seam annealer and cooling, before cut-off (cold) | All surface defects | Cleanest images, best final quality gate; defects reported some metres after they were made |
| C. Thermal camera at the weld vee / squeeze point | Seam temperature profile | Indirect monitoring of **cold weld / lack of fusion**, which no visible camera can see |

The usual combination is camera A for fast process feedback plus camera B as the final
quality gate. At 8 m/min, a defect 5 m upstream of the marker arrives 37.5 s later, which is
plenty of time to flag it by encoder position.

## What a camera can and cannot inspect

| Visible (this system) | Not visible (keep UT / eddy-current) |
|---|---|
| Bead under-/over-trim, scarfing gouges & chatter | Cold weld / lack of fusion |
| Edge mismatch, open seam, stitch weld | Penetrators (oxide in the bond line) |
| Surface cracks, pinholes, pits | Hook cracks |
| Contact burns, burn-through, spatter, dents | Laminations |
| Scale, scratches, heat-tint discolouration | |

## Roadmap to a production system

1. **Collect images from the line.** Mount the camera after the scarfing station and use
   **low-angle (raking) or structured-light illumination** across the seam. Raking light turns
   trim height and mismatch into shadows a camera can see; diffuse light hides them.
   An area-scan camera with encoder triggering, or a line-scan camera, gives frames with
   known mm/pixel.
2. **Save good pipes as background.** Replace `data/negatives` with crops from pipes that
   passed. This is the biggest false-alarm reduction available, and it needs no labelling.
3. **Label the ERW seam classes** (`ERW_SEAM_DEFECTS` in `weldvision/defects.py`): under/over
   trim, edge mismatch, open seam, stitch weld, contact burn, surface crack. A few hundred
   boxes per class is a workable start; label with CVAT or Label Studio in YOLO format and add
   the classes to `train.py`'s dataset yaml.
4. **Add an anomaly-detection branch** (e.g. PatchCore, trained only on good pipes) to flag
   defect types nobody has labelled yet. The two branches complement each other: the
   detector names known defects, the anomaly model catches unknown ones.
5. **Measure, don't just classify.** With calibrated mm/pixel, or a laser profilometer
   across the seam, report trim height, mismatch and defect length against your standard's
   acceptance limits (e.g. API 5L / IS 3589) instead of a fixed severity table.
6. **Deploy:** export with `model.export(format="onnx")` or TensorRT, tie the verdict to the
   pipe ID and position from the encoder, and drive a paint marker or reject gate.

## Layout

```
app.py                  Streamlit demo (production, seam-only, line-simulation tabs)
Dockerfile, packages.txt, .streamlit/   deployment
detect.py               batch inspection CLI -> CSV reports + annotated images
train.py                training (NEU-DET + background tiles), copies best weights to models/
weldvision/
  seam.py               weld-seam band localisation
  detector.py           tiled YOLO inference, cross-tile merge, drawing
  defects.py            defect taxonomy, severities, PASS/REVIEW/FAIL rules
  line.py               line-speed camera sizing, defect tracking in pipe coordinates
scripts/
  download_data.py      fetch NEU-DET
  make_negatives.py     defect-free background tiles
  make_demo_frames.py   synthetic HD pipe frames with ground truth
  eval_frames.py        end-to-end scoring on those frames
  simulate_line.py      moving-pipe simulation with encoder-triggered frames
```
