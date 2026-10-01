"""Download NEU-DET (cleaned, YOLO format) from Hugging Face into data/NEU-DET.

Source: https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected
Original dataset: He, Song, Meng & Yan, IEEE Trans. Instrum. Meas. 69(4), 2020.
The upstream authors never stated a licence; check before commercial use.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    path = snapshot_download(
        "KeenForgeAI/NEU-DET-corrected", repo_type="dataset", local_dir=str(ROOT / "data" / "NEU-DET"),
        allow_patterns=["images/*", "labels/*", "data.yaml", "classes.txt", "LICENSE", "README.md"])
    print(f"NEU-DET downloaded to {path}")
