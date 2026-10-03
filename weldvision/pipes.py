"""Pipe types: how each is made, what its seam looks like to a camera, and the demo's line defaults.

The inspection pipeline is the same for every type (seam localisation, tiled defect detection,
verdict). What changes is the seam: a scarfed ERW seam, a raised SAW weld bead that runs either
along the pipe (LSAW) or round it as a helix (HSAW), or no seam at all (seamless). The seam
decides where to look, what extra checks make sense, and how cameras are placed on the line.

Line speeds below are typical figures used as demo defaults, not data from a specific mill.
"""
import math
from dataclasses import dataclass, field

from .defects import ERW_SEAM_DEFECTS


@dataclass(frozen=True)
class PipeType:
    key: str
    name: str
    summary: str
    process: tuple  # manufacturing steps, in order
    seam: str  # "scarfed", "bead", "helical bead" or "none"
    seam_orientation: str | None  # how the seam appears in the demo frames: "horizontal", "angled" or None
    seam_args: dict  # extra locate_seam() arguments for this seam (band widths, high-pass window)
    od_range: str
    wt_range: str
    camera: str  # where the demo's camera sits on the line
    speed_m_min: float  # demo line default
    speed_note: str
    mm_per_px: float
    across_px: int  # camera window across the seam (or the whole circumference for seamless)
    visible_defects: dict  # seam- or type-specific defects a camera can see (target classes)
    hidden_defects: tuple  # need X-ray / UT / eddy current
    sample_dir: str
    notes: tuple = field(default_factory=tuple)


SAW_BEAD_DEFECTS = {
    "undercut": "Groove melted into the parent metal along the weld toe and left unfilled.",
    "porosity": "Gas pores breaking the bead surface, single or in clusters.",
    "bead_width": "Bead locally too narrow or too wide: unstable current, voltage, speed or flux feed.",
    "seam_tracking": "Bead wanders off the joint line: the seam-tracking system lost the groove.",
    "excess_reinforcement": "Bead too high (needs a laser profile to measure, not a 2-D image).",
    "surface_crack": "Longitudinal or transverse crack in the bead or at the toe.",
    "slag_on_surface": "Flux/slag not removed, or trapped in the bead surface.",
    "burn_through": "Weld pool falls through the root, leaving a hole or sag.",
    "arc_strike": "Stray arc mark on the parent metal next to the weld.",
}

SEAMLESS_DEFECTS = {
    "lap": "Metal folded over onto the surface during piercing or rolling, not fused.",
    "sliver_scab": "Thin flake of metal partly attached to the surface, from billet surface defects.",
    "seam_crack": "Longitudinal crack-like 'seam' carried over from the billet (not a weld).",
    "guide_marks": "Helical scratches or gouges from piercer guide shoes or reeler rolls.",
    "mandrel_marks": "Marks on the inside surface from the mandrel bar (needs an ID camera).",
    "rolled_in_scale": "Scale pressed into the surface during hot rolling.",
    "dents_bulges": "Local shape faults from handling or uneven cooling.",
}

PIPES = {
    "erw": PipeType(
        "erw", "ERW / HFW",
        "Strip roll-formed into a tube; the edges are heated by high-frequency current and forged "
        "together. No filler metal; the weld bead is scarfed flush.",
        ("Hot-rolled coil, slit to width", "Edges milled", "Roll-formed into an open tube",
         "Edges heated by HF current (contact or induction, ~200-450 kHz)",
         "Squeeze rolls forge the edges together (no filler)", "Inside and outside flash scarfed off",
         "Seam annealed by induction", "Sizing, cut-off, hydrotest, UT/eddy-current of the weld"),
        "scarfed", "horizontal", {},
        "~21-610 mm", "up to ~20 mm",
        "Seam camera after the outside scarfing tool, seam at 12 o'clock",
        8.0, "Your mill: the pipe leaves the welder at about 8 m/min.",
        0.1, 1024, ERW_SEAM_DEFECTS,
        ("Cold weld / lack of fusion", "Penetrators (oxide in the bond line)", "Hook cracks", "Laminations"),
        "samples/hd_frames",
        ("The seam is only a few mm wide after scarfing, so seam-only inspection saves most of the compute.",)),
    "lsaw": PipeType(
        "lsaw", "LSAW (longitudinal SAW)",
        "Heavy plate pressed into a tube (JCOE or UOE) and welded along one straight seam by "
        "submerged arc, inside and then outside. The weld keeps a raised bead.",
        ("Plate edges milled and bevelled", "Edges crimped", "Formed by JCOE (stepwise press) or UOE (U-press, O-press)",
         "Continuous tack weld (GMAW)", "Inside SAW weld, then outside SAW weld (multi-wire, under flux)",
         "Mechanical expansion (~1 %) to round and size", "Hydrotest", "UT and X-ray of the weld, visual inspection"),
        "bead", "horizontal", {"widths": (0.04, 0.06, 0.08, 0.10), "highpass": 0.6, "median": True, "signed": True, "min_prominence": 20.0},
        "~406-1626 mm (16-64 in)", "~6-40 mm",
        "Seam camera behind the outside welding head, after flux recovery",
        1.5, "Outside SAW welding speed is typically about 1-2.5 m/min.",
        0.1, 1024, SAW_BEAD_DEFECTS,
        ("Lack of fusion / penetration between inside and outside passes", "Internal porosity and slag",
         "Centre-line cracks", "Inside/outside weld misalignment (seen on X-ray or a macro section)"),
        "samples/lsaw_frames",
        ("At ~1.5 m/min there is a frame budget of several seconds: even a CPU keeps up.",
         "The bead's geometry (width, wander, undercut) is measured directly; height needs a laser profile.")),
    "hsaw": PipeType(
        "hsaw", "HSAW / SSAW (spiral SAW)",
        "Hot-rolled coil wound into a helix at an angle and welded by submerged arc along the "
        "helical seam. Large diameters from narrow strip; the seam runs round the pipe at an angle.",
        ("Coil uncoiled, levelled, edges milled", "Strip fed at the forming angle into a 3-roll bending cage",
         "Inside SAW at about 6 o'clock, outside SAW about half a turn later",
         "(Two-step mills: tack weld while forming, SAW on separate stations)",
         "Flying plasma cut-off", "Hydrotest", "UT and X-ray of the weld, visual inspection"),
        "helical bead", "angled",
        # A mill makes one helix hand at an angle set by strip width and OD, so search near it.
        {"widths": (0.04, 0.06, 0.08, 0.10), "highpass": 0.6, "median": True, "signed": True, "min_prominence": 20.0,
         "angles": (40, 80)},
        "~406-3048 mm", "~6-25 mm",
        "Seam-following camera behind the outside welding head (the seam slides along itself under it)",
        1.2, "SAW welding speed along the helix, typically about 1-2.5 m/min.",
        0.1, 1024, SAW_BEAD_DEFECTS,
        ("Lack of fusion / penetration", "Internal porosity and slag", "Centre-line cracks",
         "Inside/outside weld offset"),
        "samples/hsaw_frames",
        ("Seam angle: strip width B = pi x OD x cos(angle to the axis). OD 1016 mm from 1500 mm strip -> ~62 deg.",
         "A spiral pipe carries pi x OD / B metres of weld per metre of pipe (~2.1 m in that example).",
         "A body camera sees the seam at an angle; a seam-following camera sees it straight.")),
    "seamless": PipeType(
        "seamless", "Seamless",
        "Solid round billet pierced into a hollow shell and rolled down to size. No weld, so "
        "defects can be anywhere round the circumference.",
        ("Round billet heated in a rotary hearth furnace (~1250 C)", "Rotary piercing (Mannesmann cross-roll piercer + plug)",
         "Elongation: mandrel mill, plug mill, Assel or pilger mill", "Reheating",
         "Sizing or stretch-reducing mill", "Cooling bed, straightening",
         "UT for wall thickness and laminations, eddy current / flux leakage, visual"),
        "none", None, {},
        "~10-660 mm", "~2-100 mm",
        "Ring of cameras round the pipe at the finishing-line inspection station",
        30.0, "Assumed conveyor speed at a cold finishing-line inspection station (hot mills run far faster).",
        0.2, 0, SEAMLESS_DEFECTS,
        ("Wall-thickness variation / eccentricity (the main seamless quality issue - UT)", "Laminations",
         "Inside-surface defects (need an ID camera or UT)", "Internal inclusions"),
        "samples/seamless_frames",
        ("Without a seam there is nothing to narrow the search: the whole circumference must be imaged.",
         "That multiplies cameras and compute; the line simulation shows the GPU load.")),
}


# ---------------------------------------------------------------- spiral (HSAW) geometry
def helix_angle_deg(od_mm, strip_width_mm):
    """Angle between the helical seam and the pipe axis, from B = pi * OD * cos(angle)."""
    c = strip_width_mm / (math.pi * od_mm)
    if not 0 < c < 1:
        raise ValueError("strip width must be less than the pipe circumference")
    return math.degrees(math.acos(c))


def weld_per_pipe_metre(od_mm, strip_width_mm):
    """Metres of helical weld per metre of pipe: pi * OD / B."""
    return math.pi * od_mm / strip_width_mm


def seam_to_pipe(s_mm, od_mm, strip_width_mm):
    """Position s along a helical seam -> (axial position in mm, clock angle in degrees)."""
    a = math.radians(helix_angle_deg(od_mm, strip_width_mm))
    axial = s_mm * math.cos(a)
    clock = (s_mm * math.sin(a) / (math.pi * od_mm) * 360) % 360
    return axial, clock


def clock_label(deg):
    """0 deg = 12 o'clock, clockwise, as a clock-face string such as '3:30'."""
    h = (deg / 30) % 12
    hh, mm = int(h), int(round((h % 1) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{(hh % 12) or 12}:{mm:02d}"
