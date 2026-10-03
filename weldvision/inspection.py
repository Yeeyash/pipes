"""One inspection step for any pipe type: seam -> defect detector -> bead geometry -> verdict.

Shared by the app, the batch CLI, the evaluation script and the line simulator, so all of them
apply the same pipe-type rules (which seam model, whether to measure a SAW bead).
"""
import time
from dataclasses import dataclass

from .bead import analyse_bead
from .defects import verdict
from .pipes import PIPES
from .seam import locate_seam


@dataclass
class Inspection:
    res: object  # detector Result; .detections also holds the bead findings
    seam: object
    roi: tuple | None  # analysed rectangle in seam-only mode (straight seams)
    seam_only_band: bool  # seam-only mode on an angled seam (tiles selected along the band)
    bead: object  # BeadReport or None
    ms: float
    verdict: str
    reason: str


def find_seam(img, pipe_key="erw", seam_mode=None):
    """Seam for this pipe type. seam_mode overrides the type's default orientation
    ("auto", "horizontal", "vertical", "angled" or "off")."""
    pipe = PIPES[pipe_key]
    mode = seam_mode or pipe.seam_orientation or "off"
    if mode == "off" or pipe.seam == "none":
        return None
    return locate_seam(img, mode, **pipe.seam_args)


def inspect_image(det, img, pipe_key="erw", conf=0.1, tile=224, overlap=0.25, seam_mode=None,
                  seam_only=False, bead=True, mm_per_px=0.1, seam=None):
    """Inspect one image. tile=0 analyses it in a single pass. seam: skip localisation and use this one."""
    pipe = PIPES[pipe_key]
    t0 = time.perf_counter()
    if seam is None:
        seam = find_seam(img, pipe_key, seam_mode)
    pad = (tile or 0) // 2
    roi, tile_filter = None, None
    if seam_only and seam is not None:
        if seam.orientation == "angled":
            reach = seam.half_width + pad + 0.71 * (tile or 0)

            def tile_filter(x, y, t):
                return seam.distance(x + t / 2, y + t / 2) <= reach
        else:
            roi = seam.band(img.shape, pad=pad)
    res = det.predict(img, conf=conf, tile=tile, overlap=overlap, seam=seam, roi=roi, tile_filter=tile_filter)
    report = None
    if bead and seam is not None and pipe.seam in ("bead", "helical bead"):
        report = analyse_bead(img, seam, mm_per_px)
        if report is not None:
            res.detections += report.defects
    v, reason = verdict(res.detections, seam)
    return Inspection(res, seam, roi, tile_filter is not None, report, (time.perf_counter() - t0) * 1000, v, reason)
