"""Download raw images from Roboflow into ``bottle-cap.yolov8/train/images/``.

Two ways to get the images:

1. **Roboflow API** (automatic). Set environment variable ``ROBOFLOW_API_KEY``
   and run ``python data/download_dataset.py``. Requires `roboflow` package.

2. **Manual**. If you already have the images, put them into
   ``bottle-cap.yolov8/train/images/`` so that every ``.txt`` in
   ``train/labels/`` has a matching ``.jpg`` with identical stem.

The Roboflow project info is taken from ``bottle-cap.yolov8/data.yaml``.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DATASET_DIR, RAW_IMAGES_DIR, RAW_LABELS_DIR  # noqa: E402


def count_label_image_pairs() -> tuple[int, int, list[str]]:
    """Return (labels_count, images_count, missing_image_stems)."""
    if not RAW_LABELS_DIR.exists():
        return 0, 0, []

    label_stems = {p.stem for p in RAW_LABELS_DIR.glob("*.txt")}
    image_stems = (
        {p.stem for p in RAW_IMAGES_DIR.glob("*.jpg")}
        | {p.stem for p in RAW_IMAGES_DIR.glob("*.jpeg")}
        | {p.stem for p in RAW_IMAGES_DIR.glob("*.png")}
        if RAW_IMAGES_DIR.exists()
        else set()
    )
    missing = sorted(label_stems - image_stems)
    return len(label_stems), len(image_stems), missing


def download_from_roboflow() -> bool:
    """Download the dataset using the Roboflow Python SDK.

    Returns ``True`` on success, ``False`` if prerequisites are missing.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY not set - skipping automatic download.")
        return False

    try:
        from roboflow import Roboflow
    except ImportError:
        print("The `roboflow` package is not installed. Run: pip install roboflow")
        return False

    data_yaml_path = RAW_DATASET_DIR / "data.yaml"
    if not data_yaml_path.exists():
        print(f"data.yaml not found at {data_yaml_path}")
        return False

    with open(data_yaml_path, "r", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    rf_meta = meta.get("roboflow", {})
    workspace = rf_meta.get("workspace")
    project_id = rf_meta.get("project")
    version = rf_meta.get("version", "1")

    if not workspace or not project_id:
        print("Roboflow workspace/project missing from data.yaml.")
        return False

    print(f"Downloading from Roboflow: {workspace}/{project_id} v{version}")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_id)
    version_str = str(version)
    dataset = project.version(version_str).download("yolov8")

    # The Roboflow SDK drops a folder like 'adams-workspace-hi2ph-1' in cwd.
    # Merge its train/images into our expected location.
    downloaded_root = Path(dataset.location)
    src_images = downloaded_root / "train" / "images"
    if not src_images.exists():
        print(f"Downloaded folder has no train/images: {downloaded_root}")
        return False

    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img in src_images.iterdir():
        if img.is_file():
            shutil.copy2(img, RAW_IMAGES_DIR / img.name)
            copied += 1
    print(f"Copied {copied} images to {RAW_IMAGES_DIR}")
    return True


def main() -> None:
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    labels, images, missing = count_label_image_pairs()
    print(f"Labels found: {labels}")
    print(f"Images found: {images}")

    if images >= labels and labels > 0:
        print("All images appear to be in place - nothing to do.")
        return

    ok = download_from_roboflow()
    if not ok:
        print()
        print("=" * 70)
        print("MANUAL FALLBACK")
        print("=" * 70)
        print(f"Place the 639 .jpg images into:\n  {RAW_IMAGES_DIR}")
        print(
            "Each image file stem must match a .txt file stem in "
            f"{RAW_LABELS_DIR}"
        )
        if missing:
            print(f"\nExample missing image stems (first 5): {missing[:5]}")
        return

    labels, images, missing = count_label_image_pairs()
    print(f"\nAfter download: labels={labels}, images={images}, missing={len(missing)}")


if __name__ == "__main__":
    main()
