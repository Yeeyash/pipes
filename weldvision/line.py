"""Production-line integration: camera sizing for a given line speed, and tracking defects in
pipe coordinates (mm along the pipe) across overlapping, encoder-triggered frames.

Convention: the pipe axis runs along the image x axis (seam horizontal in the image), and the
camera is triggered by a line encoder every `advance_mm` of pipe travel.
"""
import math
from dataclasses import dataclass, field


@dataclass
class LineSetup:
    speed_m_min: float = 8.0
    mm_per_px: float = 0.1  # optical resolution on the pipe surface
    frame_px_along: int = 2448  # sensor pixels along the pipe axis (5 MP area-scan: 2448 x 2048)
    frame_px_across: int = 2048
    overlap: float = 0.2  # fraction of each frame re-seen in the next (needed for tile merging)
    max_blur_px: float = 0.5  # allowed motion blur during exposure

    @property
    def speed_mm_s(self):
        return self.speed_m_min * 1000 / 60

    @property
    def fov_along_mm(self):
        return self.frame_px_along * self.mm_per_px

    @property
    def fov_across_mm(self):
        return self.frame_px_across * self.mm_per_px

    @property
    def advance_mm(self):
        """Pipe travel between triggers."""
        return self.fov_along_mm * (1 - self.overlap)

    @property
    def fps(self):
        return self.speed_mm_s / self.advance_mm

    @property
    def frame_budget_s(self):
        """Time available to process one frame and keep up with the line."""
        return 1 / self.fps

    @property
    def max_exposure_us(self):
        return self.max_blur_px * self.mm_per_px / self.speed_mm_s * 1e6

    @property
    def line_scan_rate_hz(self):
        """Equivalent line rate if a line-scan camera is used instead (square pixels)."""
        return self.speed_mm_s / self.mm_per_px

    @property
    def pixel_rate_mps(self):
        """Megapixels per second that must be processed for full-frame inspection."""
        return self.frame_px_along * self.frame_px_across * self.fps / 1e6

    def cameras_for_full_circumference(self, od_mm, usable_arc_deg=60):
        """Cameras needed round the pipe. A camera sees a curved surface usefully only within
        about +/-30 deg of normal incidence; beyond that resolution and lighting degrade."""
        arc_mm = math.pi * od_mm * usable_arc_deg / 360
        return math.ceil(360 / usable_arc_deg), arc_mm, arc_mm <= self.fov_across_mm

    def summary(self, od_mm=None):
        rows = [
            ("Line speed", f"{self.speed_m_min:g} m/min = {self.speed_mm_s:.1f} mm/s"),
            ("Resolution", f"{self.mm_per_px:g} mm/px"),
            ("Field of view", f"{self.fov_along_mm:.0f} mm along x {self.fov_across_mm:.0f} mm across"),
            ("Trigger every", f"{self.advance_mm:.0f} mm of travel ({self.overlap:.0%} overlap)"),
            ("Frame rate needed", f"{self.fps:.2f} fps"),
            ("Time budget per frame", f"{self.frame_budget_s * 1000:.0f} ms"),
            ("Max exposure / strobe", f"{self.max_exposure_us:.0f} us (for <= {self.max_blur_px} px blur)"),
            ("Line-scan alternative", f"{self.line_scan_rate_hz / 1000:.2f} kHz line rate"),
            ("Pixel throughput", f"{self.pixel_rate_mps:.1f} MP/s per camera (full frame)"),
        ]
        if od_mm:
            n, arc, fits = self.cameras_for_full_circumference(od_mm)
            rows.append(("Full circumference", f"OD {od_mm:g} mm: {n} cameras, {arc:.0f} mm arc each"
                         + ("" if fits else " (exceeds FOV - lower mm/px or add cameras)")))
        return rows


@dataclass
class PipeDefect:
    cls_name: str
    conf: float
    start_mm: float  # along the pipe, from the encoder zero (pipe head)
    end_mm: float
    top_mm: float  # extent across the frame (around the circumference)
    bottom_mm: float
    on_seam: bool
    frames: list = field(default_factory=list)

    @property
    def length_mm(self):
        return self.end_mm - self.start_mm

    @property
    def across_mm(self):
        return (self.top_mm + self.bottom_mm) / 2


class PipeTracker:
    """Collects per-frame detections into one de-duplicated defect log per pipe.

    The same defect appears in two consecutive frames when it sits in the overlap zone; such
    pairs (same class, overlapping along the pipe, close across it) are merged.
    """

    def __init__(self, mm_per_px, across_tol_mm=5.0):
        self.mm_per_px = mm_per_px
        self.across_tol_mm = across_tol_mm
        self.defects: list[PipeDefect] = []

    def add(self, frame_idx, frame_start_mm, detections):
        k = self.mm_per_px
        for d in detections:
            x1, y1, x2, y2 = d.box
            new = PipeDefect(d.cls_name, d.conf, frame_start_mm + x1 * k, frame_start_mm + x2 * k,
                             y1 * k, y2 * k, d.on_seam, [frame_idx])
            for old in self.defects:
                if (old.cls_name == new.cls_name and frame_idx - old.frames[-1] <= 1
                        and new.start_mm <= old.end_mm and new.end_mm >= old.start_mm
                        and abs(new.across_mm - old.across_mm) <= self.across_tol_mm):
                    old.start_mm, old.end_mm = min(old.start_mm, new.start_mm), max(old.end_mm, new.end_mm)
                    old.top_mm, old.bottom_mm = min(old.top_mm, new.top_mm), max(old.bottom_mm, new.bottom_mm)
                    old.conf, old.on_seam = max(old.conf, new.conf), old.on_seam or new.on_seam
                    old.frames.append(frame_idx)
                    break
            else:
                self.defects.append(new)
