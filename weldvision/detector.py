"""Defect detector with tiled inference for high-resolution line-camera frames.

The model is trained on small patches (NEU-DET: 200x200 px). A defect occupies a similar
number of pixels in a production frame only if the frame is analysed at a matching scale,
so a 4K or 12 MP frame is cut into overlapping tiles. Each tile is detected separately and
the boxes are mapped back and merged. Downscaling the whole frame to the model input size
instead would make pinholes and fine cracks disappear.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

from .defects import info
from .seam import Seam

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "weld_defects.pt"


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    box: tuple  # x1, y1, x2, y2 in image pixels
    on_seam: bool = False

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class Result:
    detections: list
    seam: Seam | None = None
    n_tiles: int = 1
    extras: dict = field(default_factory=dict)


def _tiles(h, w, tile, overlap):
    stride = max(1, int(tile * (1 - overlap)))
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    return [(x, y) for y in ys for x in xs]


def _merge_across_tiles(boxes, scores, classes, tile_ids, ios_thr=0.2, nested_thr=0.8):
    """Fuse same-class boxes that are pieces of one defect (intersection over the smaller box,
    IoS). Plain IoU-NMS keeps such pieces because they have little IoU.

    * From different tiles: fuse when IoS > ios_thr (a defect crossing a tile border).
    * From the same tile: fuse only when one box is nested in the other (IoS > nested_thr), so
      separate neighbouring defects the model distinguished stay separate.
    Returns merged (boxes, scores, classes)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    b = boxes.numpy()
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    for i in range(n):
        for j in range(i + 1, n):
            if classes[i] != classes[j]:
                continue
            iw = min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0])
            ih = min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1])
            thr = nested_thr if tile_ids[i] == tile_ids[j] else ios_thr
            if iw > 0 and ih > 0 and iw * ih / (min(area[i], area[j]) + 1e-6) > thr:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out_b, out_s, out_c = [], [], []
    for idx in groups.values():
        g = b[idx]
        out_b.append([g[:, 0].min(), g[:, 1].min(), g[:, 2].max(), g[:, 3].max()])
        out_s.append(float(scores[idx].max()))
        out_c.append(int(classes[idx[0]]))
    return torch.tensor(out_b), torch.tensor(out_s), torch.tensor(out_c)


class DefectDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, imgsz=320, device=None):
        if not Path(weights).exists():
            raise FileNotFoundError(f"{weights} not found - run `python train.py` first.")
        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.imgsz = imgsz
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")

    def _run(self, images, conf, iou):
        return self.model.predict(images, imgsz=self.imgsz, conf=conf, iou=iou,
                                  device=self.device, verbose=False)

    def predict(self, img, conf=0.1, iou=0.5, tile=None, overlap=0.25, seam=None, roi=None, batch=16):
        """Detect defects in a BGR image.

        tile: tile size in px; None/0 = single pass on the whole image. Use tiles when the
              frame is much larger than the scale the model was trained on.
        seam: optional Seam; detections whose centre lies in the seam band get on_seam=True.
        roi:  optional (x1, y1, x2, y2); only this region is analysed (e.g. the seam band from
              Seam.band(), which cuts compute several-fold on a seam-inspection camera).
        """
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        rx, ry = 0, 0
        if roi is not None:
            rx, ry, x2, y2 = (int(v) for v in roi)
            img = img[ry:y2, rx:x2]
        h, w = img.shape[:2]

        if not tile or max(h, w) <= tile * 1.25:
            origins, crops = [(0, 0)], [img]
        else:
            tile = min(tile, h, w)
            origins = _tiles(h, w, tile, overlap)
            crops = [img[y:y + tile, x:x + tile] for x, y in origins]

        all_boxes, all_scores, all_cls, all_tiles = [], [], [], []
        for i in range(0, len(crops), batch):
            results = self._run(crops[i:i + batch], conf, iou)
            for t, ((ox, oy), r) in enumerate(zip(origins[i:i + batch], results), start=i):
                if len(r.boxes) == 0:
                    continue
                b = r.boxes.xyxy.cpu()
                b[:, [0, 2]] += ox + rx
                b[:, [1, 3]] += oy + ry
                all_boxes.append(b)
                all_scores.append(r.boxes.conf.cpu())
                all_cls.append(r.boxes.cls.cpu().long())
                all_tiles.append(torch.full((len(b),), t))

        dets = []
        if all_boxes:
            boxes, scores, classes = torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_cls)
            tiles = torch.cat(all_tiles)
            k = batched_nms(boxes, scores, classes, iou)
            boxes, scores, classes = _merge_across_tiles(boxes[k], scores[k], classes[k], tiles[k].tolist())
            for b, s, c in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
                d = Detection(c, self.names[c], s, tuple(round(v, 1) for v in b))
                if seam is not None:
                    d.on_seam = seam.contains(*d.center)
                dets.append(d)
        dets.sort(key=lambda d: -d.conf)
        return Result(dets, seam, len(crops))


def draw(img, result, show_seam=True):
    """Return an annotated copy of a BGR image."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    h, w = out.shape[:2]
    lw = max(1, round(max(h, w) / 400))
    fs = max(0.35, max(h, w) / 1400)

    if show_seam and result.seam is not None:
        x1, y1, x2, y2 = result.seam.band(out.shape)
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), -1)
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (255, 200, 0), lw)

    for d in result.detections:
        x1, y1, x2, y2 = map(int, d.box)
        color = info(d.cls_name).color
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw + (1 if d.on_seam else 0))
        text = f"{info(d.cls_name).label} {d.conf:.2f}" + (" [SEAM]" if d.on_seam else "")
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 4
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 4, ty + bl - 1), color, -1)
        cv2.putText(out, text, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out
