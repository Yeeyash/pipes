"""Download NEU-DET (cleaned, YOLO format) from Hugging Face into data/NEU-DET.

Source: https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected
Original dataset: He, Song, Meng & Yan, IEEE Trans. Instrum. Meas. 69(4), 2020.
The upstream authors never stated a licence; check before commercial use.

    python scripts/download_data.py               # full dataset, for training
    python scripts/download_data.py --test-only   # test split only (~3 MB), enough for the demo app
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
NEU = ROOT / "data" / "NEU-DET"


def download(test_only=False):
    splits = ["images/test/*", "labels/test/*"] if test_only else ["images/*", "labels/*"]
    return snapshot_download(
        "KeenForgeAI/NEU-DET-corrected", repo_type="dataset", local_dir=str(NEU),
        allow_patterns=splits + ["data.yaml", "classes.txt", "LICENSE", "README.md"])


def ensure_test_split():
    """Fetch the test split if it is missing (the deployed app does this at startup)."""
    if not (NEU / "classes.txt").exists() or not any((NEU / "images" / "test").glob("*.jpg")):
        download(test_only=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-only", action="store_true")
    print(f"NEU-DET downloaded to {download(ap.parse_args().test_only)}")
