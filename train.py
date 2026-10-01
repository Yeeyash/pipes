"""Train the weld/steel surface defect detector (YOLO) on NEU-DET.

Usage:
    python scripts/make_negatives.py     # once: defect-free background tiles
    python train.py                      # defaults tuned for a 4 GB GPU
    python train.py --model yolo11m.pt --epochs 150 --imgsz 416
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "NEU-DET"
NEG_DIR = ROOT / "data" / "negatives"
WEIGHTS_OUT = ROOT / "models" / "weld_defects.pt"


def write_data_yaml(negatives: bool) -> Path:
    """Write a dataset yaml with absolute paths so training works from any cwd.

    With negatives=True, defect-free background tiles (scripts/make_negatives.py) are added to
    train and val so the model learns that normal surface and seam texture is not a defect.
    The test split stays pure NEU-DET so results compare with published numbers.
    """
    names = (DATA_DIR / "classes.txt").read_text().split()
    train, val = [str(DATA_DIR / "images/train")], [str(DATA_DIR / "images/val")]
    if negatives:
        if not (NEG_DIR / "images" / "train").exists():
            raise SystemExit("No background tiles found - run `python scripts/make_negatives.py` first.")
        train.append(str(NEG_DIR / "images/train"))
        val.append(str(NEG_DIR / "images/val"))
    cfg = {
        "path": str(DATA_DIR),
        "train": train,
        "val": val,
        "test": str(DATA_DIR / "images/test"),
        "names": dict(enumerate(names)),
    }
    out = ROOT / "configs" / "neu_det.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained checkpoint to fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=320, help="NEU images are 200px; mild upscaling helps small defects")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2, help="keep low on Windows")
    ap.add_argument("--name", default="neu_det")
    ap.add_argument("--no-negatives", action="store_true", help="train on NEU-DET only")
    args = ap.parse_args()

    data_yaml = write_data_yaml(not args.no_negatives)
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=30,
        cos_lr=True,
        # Steel surface texture has no canonical orientation, so vertical flips are valid too.
        flipud=0.5,
        fliplr=0.5,
        # Grayscale imagery: hue/saturation jitter is meaningless, keep brightness jitter
        # to simulate lighting variation on the line.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        plots=True,
    )

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    WEIGHTS_OUT.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS_OUT)
    print(f"\nBest weights copied to {WEIGHTS_OUT}")

    # Final, unbiased numbers on the held-out test split.
    metrics = YOLO(WEIGHTS_OUT).val(data=str(data_yaml), split="test", imgsz=args.imgsz,
                                    project=str(ROOT / "runs"), name=f"{args.name}_test", exist_ok=True)
    print(f"TEST  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
