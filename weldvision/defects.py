"""Defect taxonomy and the quality-decision rules built on top of it.

The demo model is trained on NEU-DET (hot-rolled steel strip surface defects). The strip that
an ERW mill forms into pipe is exactly this kind of hot-rolled coil, so these classes are real
surface defects that show up on ERW pipe. Seam-specific ERW defects (bead-trim faults, edge
mismatch, open seam, stitching) need images from your own line and are listed in
ERW_SEAM_DEFECTS as the target classes for the next dataset.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DefectInfo:
    name: str
    label: str
    description: str
    erw_relevance: str
    severity: str  # "reject" -> FAIL the pipe, "minor" -> send to manual REVIEW
    color: tuple  # BGR for OpenCV drawing


DEFECTS = {
    "crazing": DefectInfo(
        "crazing", "Crazing",
        "Network of fine surface cracks.",
        "Can open into through-wall cracks during forming/sizing or hydro-test; critical near the weld.",
        "reject", (0, 0, 255)),
    "inclusion": DefectInfo(
        "inclusion", "Inclusion",
        "Non-metallic particle (oxide/slag) embedded in the surface.",
        "Inclusions at the strip edges get squeezed into the bond line and cause penetrators / lack of fusion.",
        "reject", (0, 128, 255)),
    "patches": DefectInfo(
        "patches", "Patches",
        "Irregular discoloured / scaled areas.",
        "Often heat tint or scale; cosmetic unless coinciding with the seam or HAZ.",
        "minor", (255, 128, 0)),
    "pitted_surface": DefectInfo(
        "pitted_surface", "Pitted surface",
        "Clusters of small pits / pinholes.",
        "Reduces local wall thickness; pinholes in the bead indicate weld porosity.",
        "reject", (255, 0, 255)),
    "rolled-in_scale": DefectInfo(
        "rolled-in_scale", "Rolled-in scale",
        "Mill scale pressed into the surface during rolling.",
        "Surface finish / coating adhesion issue; usually repairable by grinding.",
        "minor", (0, 200, 200)),
    "scratches": DefectInfo(
        "scratches", "Scratches",
        "Linear mechanical marks.",
        "Along the seam this is the signature of scarfing-tool gouges or chatter; elsewhere roll/handling marks.",
        "minor", (0, 220, 0)),
}

# Target classes for a line-specific dataset (not in the demo model).
ERW_SEAM_DEFECTS = {
    "under_trim": "Weld flash/bead left above the pipe surface after scarfing.",
    "over_trim": "Scarfing tool cut below the pipe surface (wall-thickness loss).",
    "edge_mismatch": "Strip edges offset radially at the seam.",
    "open_seam": "Edges not bonded - visible gap along the seam.",
    "stitch_weld": "Periodic bonded/unbonded pattern from unstable welding power.",
    "contact_burn": "Arc burns from the welding contacts/coil.",
    "surface_crack": "Cracks in or beside the weld line.",
}


def info(name: str) -> DefectInfo:
    return DEFECTS.get(name) or DefectInfo(name, name, "", "", "minor", (200, 200, 200))


def verdict(detections, seam_band=None):
    """Return ("PASS" | "REVIEW" | "FAIL", reason).

    Any reject-class defect -> FAIL. Minor defects -> REVIEW, except that a minor defect
    lying on the weld seam is escalated to FAIL, since the seam is the critical zone of an ERW pipe.
    """
    if not detections:
        return "PASS", "No defects detected."
    rejects = [d for d in detections if info(d.cls_name).severity == "reject"]
    on_seam = [d for d in detections if seam_band is not None and d.on_seam]
    if rejects:
        names = sorted({info(d.cls_name).label for d in rejects})
        return "FAIL", f"Reject-class defect(s): {', '.join(names)}."
    if on_seam:
        return "FAIL", f"{len(on_seam)} defect(s) on the weld seam."
    return "REVIEW", f"{len(detections)} minor defect(s) - manual inspection recommended."
