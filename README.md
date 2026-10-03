# Pipe Weld & Surface Defect Inspection (computer-vision demo)

Detects weld and surface defects on images of **ERW** (electric-resistance-welded), **LSAW**
(longitudinal submerged-arc-welded), **HSAW** (spiral submerged-arc-welded) and **seamless**
pipe:

1. **Seam localisation** (`weldvision/seam.py`) finds the weld-seam band so that defects on
   the seam, the critical zone of a welded pipe, can be escalated. It handles a narrow scarfed
   ERW seam, a wide raised SAW bead, and a spiral seam crossing the image at an angle.
2. **Tiled defect detection** (`weldvision/detector.py`) runs a YOLO11 detector over
   overlapping tiles. High-resolution line-camera frames are never shrunk to 320 px, which
   would erase pinholes and fine cracks. Pieces of one defect that cross tile borders are merged.
3. **SAW bead measurement** (`weldvision/bead.py`, LSAW and HSAW only) traces both weld toes
   and reports bead width out of tolerance, the bead wandering off the joint, undercut and
   surface porosity, with classical image analysis rather than a trained model.
4. **Decision** (`weldvision/defects.py`) gives **PASS / REVIEW / FAIL** per image:
   a reject-class defect gives FAIL, a minor defect gives REVIEW, and any defect on the seam gives FAIL.

```
image ─► seam band ─► tiles ─► YOLO ─► NMS + cross-tile merge ─► on-seam flag ─┐
              └──► (SAW) bead strip ─► toe tracing ─► width / tracking / undercut / pores ─┴► verdict
```

`weldvision/pipes.py` holds what differs between the pipe types (process, seam model, camera
position, line speed, type-specific defects); `weldvision/inspection.py` runs the pipeline for
any of them, so the app, the CLIs and the line simulator all behave the same way.

## Quick start

```bash
pip install -r requirements.txt          # install CUDA PyTorch first, see file
python scripts/download_data.py          # NEU-DET -> data/NEU-DET (--test-only: just what the app needs)
python scripts/make_negatives.py         # defect-free background tiles -> data/negatives
python train.py                          # ~20 min on an RTX 3050 4 GB -> models/weld_defects.pt
python scripts/make_demo_frames.py --pipe all   # synthetic 2400x1200 frames -> samples/*_frames

streamlit run app.py                     # demo UI
python detect.py samples/hd_frames       # batch mode -> results/summary.csv, detections.csv, annotated/
python detect.py samples/hd_frames --seam-only   # seam band only, ~3x faster
python detect.py samples/lsaw_frames --pipe lsaw # also hsaw, seamless
python scripts/eval_frames.py --pipe all # score the full pipeline against ground truth
python scripts/simulate_line.py          # 2 m of ERW pipe at 8 m/min: real-time check + defect log by position
python scripts/simulate_line.py --pipe lsaw      # also hsaw, seamless (see "Pipe types")
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

## Pipe types

Choose the type in the app's sidebar, or pass `--pipe` to the scripts.

| | ERW / HFW | LSAW | HSAW (spiral) | Seamless |
|---|---|---|---|---|
| Made from | hot-rolled coil, roll-formed | heavy plate, JCOE / UOE pressed | hot-rolled coil, wound into a helix | solid billet, pierced and rolled |
| Weld | HF-heated edges forged together, no filler | submerged arc, inside + outside | submerged arc, inside + outside | none |
| Seam in the image | narrow scarfed band | raised bead, 15-25 mm wide | raised bead **at the helix angle** | - |
| Typical OD | ~21-610 mm | ~406-1626 mm | ~406-3048 mm | ~10-660 mm |
| Demo camera | seam camera after scarfing, 8 m/min | behind the outside welding head, ~1.5 m/min | seam-following, behind the outside welding head, ~1.2 m/min | ring round the pipe, 30 m/min (assumed) |
| What the type adds to the pipeline | - | bead measurement | angled seam search, helix geometry | full-circumference coverage |

**SAW seams.** A submerged-arc bead is not scarfed flush: it stays raised, 15-25 mm wide, with
solidification ripples and a shadow line at each toe. The seam finder uses wider bands, a
running-median baseline (so the wide bead does not lift the baseline around itself) and a
signed contrast score with row medians (so a few defect patches that happen to line up cannot
outscore the bead). `weldvision/bead.py` then straightens the bead into a strip and traces both
toes with dynamic programming. From the toes it measures width (against the bead's own median),
tracking (centre line against a robust straight fit), undercut (toe groove much darker than
the rest of the toe line) and porosity (small, round, dark spots; the ripples are long arcs).
A 2-D image shows shading, not height: bead height and undercut depth need a laser line profiler.

**Spiral seams.** Strip width B, outside diameter D and the seam angle to the pipe axis α are
related by **B = π·D·cos α**: OD 1016 mm from 1500 mm strip gives α ≈ 62°, and π·D/B ≈ 2.1 m
of weld per metre of pipe. A body camera sees the seam at that angle. `locate_seam(img,
"angled")` rotates the image over the angle range a mill can produce, takes the best band,
then refines the angle with a line fit to the bead centre along the seam. A camera mounted
behind the outside welding head sees the seam straight, because the pipe turns and advances
so that the seam slides along itself; the line simulation uses that camera, and converts
positions along the helix to axial position and clock position (`weldvision.pipes.seam_to_pipe`).

**Seamless.** With no seam there is nothing to narrow the search: defects can be anywhere
round the circumference, and the dominant seamless quality issue (wall-thickness variation
from an off-centre piercing) is invisible to a camera and belongs to ultrasonic testing. The
line simulation unrolls the pipe surface and covers it with a ring of cameras (about 60° of
arc each, 20% overlap). Defects are logged with a clock position, and detections of one
defect by two neighbouring cameras are merged. All cameras share the time budget of one
trigger, which is what makes seamless inspection expensive.

## Data

| Source | What | Used for |
|---|---|---|
| [NEU-DET](https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected) | 1,797 hot-rolled steel images, 200×200 px, 6 classes, box labels | Training / val / test |
| `scripts/make_negatives.py` | 840 synthetic defect-free tiles: ERW plate and scarfed seam, SAW beads and plate texture at any angle, seamless scale | Background (negative) examples |
| `scripts/make_demo_frames.py` + `synth_pipes.py` | 2400×1200 synthetic frames per pipe type (ERW scarfed seam, LSAW bead, spiral bead at the helix angle, seamless surface) with NEU **test** defects and planted SAW bead defects at known positions | End-to-end demo and evaluation |

ERW and HSAW pipe are formed from hot-rolled strip, LSAW from hot-rolled plate, and seamless
pipe is hot-rolled too, so NEU-DET's classes (crazing, inclusion, patches, pitted surface,
rolled-in scale, scratches) are real defects found on all four. The demo model does **not**
know seam-specific defects (ERW trim faults, SAW slag or cracks, seamless laps and slivers).
Those need images from your mill; see the roadmap. SAW bead geometry and porosity are measured
by `weldvision/bead.py` and need no training data, but are tuned on synthetic beads.

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
| NEU-DET + 360 ERW background tiles (previous default) | 0.733 | 0.432 |
| NEU-DET + 840 background tiles for all pipe types (`models/weld_defects.pt`, **default**) | 0.722 | 0.418 |

These are in line with published YOLO results on NEU-DET. Crazing and rolled-in scale are
low-contrast and hard for every published method. On a real line, raking light helps both.
More background tiles cost a little test mAP (a larger share of the training set is
"nothing here"), and buy far fewer false alarms on SAW and spiral pipe (below).

**End-to-end on the 6 synthetic HD frames (60 defect boxes, tile 224, IoU ≥ 0.3):**

| Model | conf | defects found | false alarms | clean frame |
|---|---|---|---|---|
| NEU only | 0.25 | 72% | 2 | PASS |
| NEU only | 0.10 | 77% | 16 | REVIEW |
| + background tiles | 0.25 | 68% | 3 | PASS |
| + ERW background tiles | 0.10 | 77% | 3 | PASS |
| + background tiles for all types | **0.10 (default)** | **72%** | **2** | **PASS** |

Every found defect also had the correct class. The median seam-centre error is 7 px, and
inference takes about 1 s per 2400×1200 frame (98 tiles) on the laptop GPU.

**All pipe types** (default model, conf 0.10, tile 224, IoU ≥ 0.3; `python scripts/eval_frames.py --pipe all`):

| Frames | Surface defects found (detector) | False alarms | SAW bead findings | Clean frame | Seam centre error |
|---|---|---|---|---|---|
| ERW, 6 frames | 43/60 (72%) | 2 | - | PASS | 7 px |
| LSAW, 4 frames | 12/16 (75%) | 10 | 10/10, 1 false | PASS | 8 px |
| HSAW body camera, 4 frames | 13/16 (81%) | 4 | 3/5, 0 false | PASS | 12 px |
| Seamless, 4 frames | 22/27 (81%) | 11 | - | PASS | (no seam) |

Effect of the extra background tiles (SAW beads, plate at any angle, seamless scale) on
detector false alarms: LSAW 21 → 10, **HSAW 218 → 4**, seamless 10 → 11, ERW 3 → 2. The
old model had never seen rolling texture running diagonally, which is how plate looks on a
spiral pipe, and called it scratches and crazing everywhere. Most remaining LSAW and seamless
false alarms are unlabelled parts of the pasted NEU patches.

The two bead findings missed on HSAW frames are seam-tracking events: a body camera sees only
~140 mm of the helical bead, and when a planted wander covers most of it there is no straight
reference left. A seam-following camera sees a long baseline, which is why the HSAW line
simulation uses one (all bead events found there).

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
| 2448×1024 seam camera, full frame (`simulate_line.py`) | 312 ms | 21% |

One seam camera is comfortably real-time on this laptop. **Full-circumference** coverage of a
large pipe (e.g. OD 324 mm needs about 6 cameras at 0.1 mm/px, ~20 MP/s) needs a desktop GPU
or a TensorRT FP16 export (`model.export(format="engine", half=True)`, typically 2–3× faster).

**Line simulation:** `python scripts/simulate_line.py` plays 2 m of synthetic pipe past the
camera at 8 m/min with encoder-triggered, overlapping frames. It writes
`results/line_sim/defect_log.csv`, which gives each defect's position in mm from the pipe
head, its extent, and whether it is on the seam; this is what a paint marker or reject gate
consumes. Defects seen in two overlapping frames are merged (9 duplicates in the default run).

**Line simulation for every pipe type** (2 m, 12 defects, default model, laptop GPU;
`python scripts/simulate_line.py --pipe <type>`):

| Pipe | Camera(s) | Speed | Time budget per trigger | Processing | Found | False alarms |
|---|---|---|---|---|---|---|
| ERW | 1 seam camera, 0.1 mm/px | 8 m/min | 1,469 ms | 312 ms (21%) | 32/38 | 8 |
| LSAW | 1 seam camera, 0.1 mm/px | 1.5 m/min | 7,834 ms | 663 ms (8%) | 14/17 | 2 |
| HSAW | 1 seam-following camera, 0.1 mm/px | 1.2 m/min along the seam | 9,792 ms | 657 ms (7%) | 16/20 | 6 |
| Seamless | ring of 6, OD 168.3 mm, 0.2 mm/px | 30 m/min | 783 ms | 899 ms (**115%**) | 21/31 | 6 |

* **SAW lines are slow** (welding speed), so even with bead measurement on top of the
  detector there are seconds to spare per frame; a CPU would keep up. All planted bead events
  were found on both SAW seams. The bead's width and centre line along the whole seam are
  plotted in the app.
* **Spiral pipe:** OD 1016 mm from 1500 mm strip puts the seam at 62° to the axis, with
  2.13 m of weld per metre of pipe; at 1.2 m/min of welding the pipe advances only 0.56 m/min.
  The log gives each defect's position along the seam plus its axial position and clock position.
* **Seamless pipe falls behind** at 30 m/min: six cameras share one GPU's time budget. It keeps
  up below about 26 m/min, or with TensorRT FP16, a desktop GPU, or one GPU per two or three
  cameras. Detections of one defect by two neighbouring cameras are merged (19 in this run),
  and the log gives each defect's clock position and the cameras that saw it.

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

| Pipe | Visible (this system) | Not visible (keep UT / X-ray / eddy-current) |
|---|---|---|
| ERW | Bead under-/over-trim, scarfing gouges & chatter, edge mismatch, open seam, stitch weld, contact burns | Cold weld / lack of fusion, penetrators, hook cracks |
| LSAW, HSAW | Undercut, surface porosity, bead width and tracking, surface cracks, slag left on the bead, burn-through, arc strikes | Lack of fusion / penetration between the inside and outside passes, internal porosity and slag, centre-line cracks, inside/outside weld offset |
| Seamless | Laps, slivers / scabs, seams (billet cracks), guide-shoe marks, rolled-in scale, dents | Wall-thickness variation (eccentricity), inside-surface defects (need an ID camera), laminations |
| All | Surface cracks, pinholes, pits, scale, scratches, heat-tint discolouration | Laminations, internal inclusions |

## Roadmap to a production system

1. **Collect images from the line.** Mount the camera after the scarfing station and use
   **low-angle (raking) or structured-light illumination** across the seam. Raking light turns
   trim height and mismatch into shadows a camera can see; diffuse light hides them.
   An area-scan camera with encoder triggering, or a line-scan camera, gives frames with
   known mm/pixel.
2. **Save good pipes as background.** Replace `data/negatives` with crops from pipes that
   passed. This is the biggest false-alarm reduction available, and it needs no labelling.
3. **Label the type-specific classes** (`visible_defects` of each type in `weldvision/pipes.py`):
   ERW trim faults, mismatch, open seam, stitch weld, contact burns; SAW surface cracks, slag,
   burn-through, arc strikes; seamless laps, slivers, seams and guide marks. A few hundred
   boxes per class is a workable start; label with CVAT or Label Studio in YOLO format and add
   the classes to `train.py`'s dataset yaml. Calibrate the SAW bead tolerances on real beads.
4. **Add an anomaly-detection branch** (e.g. PatchCore, trained only on good pipes) to flag
   defect types nobody has labelled yet. The two branches complement each other: the
   detector names known defects, the anomaly model catches unknown ones.
5. **Measure, don't just classify.** With calibrated mm/pixel, or a laser profilometer
   across the seam, report trim height, mismatch, SAW reinforcement height and undercut depth,
   and defect length against your standard's acceptance limits (e.g. API 5L / IS 3589)
   instead of a fixed severity table.
6. **Deploy:** export with `model.export(format="onnx")` or TensorRT, tie the verdict to the
   pipe ID and position from the encoder, and drive a paint marker or reject gate.

## Layout

```
app.py                  Streamlit demo (pipe type selector; production, seam-only, line-simulation, guide tabs)
Dockerfile, packages.txt, .streamlit/   deployment
detect.py               batch inspection CLI -> CSV reports + annotated images (--pipe)
train.py                training (NEU-DET + background tiles), copies best weights to models/
weldvision/
  pipes.py              pipe types (ERW, LSAW, HSAW, seamless), spiral-seam geometry
  inspection.py         one inspection step for any pipe type
  seam.py               weld-seam band localisation (straight, wide SAW bead, angled spiral)
  bead.py               SAW bead width, tracking, undercut, porosity
  detector.py           tiled YOLO inference, cross-tile merge, drawing
  defects.py            defect taxonomy, severities, PASS/REVIEW/FAIL rules
  line.py               line-speed camera sizing, defect tracking in pipe coordinates (incl. camera rings)
scripts/
  download_data.py      fetch NEU-DET
  make_negatives.py     defect-free background tiles
  make_demo_frames.py   synthetic HD pipe frames with ground truth (--pipe)
  synth_pipes.py        SAW bead, spiral and seamless surface synthesis
  eval_frames.py        end-to-end scoring on those frames (--pipe)
  simulate_line.py      moving-pipe simulation with encoder-triggered frames (--pipe)
```
