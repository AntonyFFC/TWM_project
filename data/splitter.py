"""Split all cap crops into train / val / test and persist them to disk.

Output layout:
    data/processed/crops/<split>/<class_name>/<id>.png
    data/processed/splits/index.csv  (image_path, class_id, class_name, split)

The split is **stratified** over class labels to keep the per-class proportion
consistent between train / val / test.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CLASS_NAMES,
    CROPS_DIR,
    SEED,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    ensure_dirs,
)
from data.dataset_loader import CapSample, load_all_samples  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")


def _sanitize(name: str) -> str:
    return name.replace(" ", "_").lower()


def _stratified_three_way_split(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks (train, val, test) stratified by class."""
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    indices = np.arange(len(y))
    train_idx, rest_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=y,
        random_state=seed,
    )
    rel_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=rel_val_ratio,
        stratify=y[rest_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def _write_crop(sample: CapSample, split: str, sample_id: int) -> Path:
    class_folder = CROPS_DIR / split / _sanitize(sample.class_name)
    class_folder.mkdir(parents=True, exist_ok=True)
    out_path = class_folder / f"{sample_id:05d}.png"
    cv2.imwrite(str(out_path), sample.crop)
    return out_path


def run_split(clean: bool = True) -> dict[str, int]:
    """Create train/val/test directories on disk.

    Args:
        clean: remove existing crops before writing new ones.

    Returns a dict ``{split: count}``.
    """
    ensure_dirs()

    if clean and CROPS_DIR.exists():
        for sub in CROPS_DIR.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)

    print("Loading samples from raw dataset...")
    samples = load_all_samples()
    if not samples:
        raise RuntimeError(
            "No samples loaded. Check that images are present in "
            "bottle-cap.yolov8/train/images/ (run data/download_dataset.py)."
        )

    y = np.array([s.class_id for s in samples])
    classes, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for cls, cnt in zip(classes, counts):
        print(f"  {CLASS_NAMES[cls]:15s} {cnt}")

    # Minimum 2 per class for stratified split to work.
    too_small = [CLASS_NAMES[c] for c, cnt in zip(classes, counts) if cnt < 3]
    if too_small:
        print(f"WARNING: very small classes may cause split issues: {too_small}")

    train_idx, val_idx, test_idx = _stratified_three_way_split(
        y, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
    )

    assignments: list[tuple[int, str]] = []
    for i in train_idx:
        assignments.append((int(i), "train"))
    for i in val_idx:
        assignments.append((int(i), "val"))
    for i in test_idx:
        assignments.append((int(i), "test"))
    assignments.sort()

    counts_per_split = {s: 0 for s in SPLIT_NAMES}
    rows = []
    for global_id, (idx, split) in enumerate(tqdm(assignments, desc="Saving crops")):
        sample = samples[idx]
        out_path = _write_crop(sample, split, global_id)
        rows.append(
            {
                "image_path": str(out_path.relative_to(CROPS_DIR.parent.parent)),
                "class_id": sample.class_id,
                "class_name": sample.class_name,
                "source_image": sample.source_image,
                "split": split,
            }
        )
        counts_per_split[split] += 1

    index_path = SPLITS_DIR / "index.csv"
    with open(index_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_path", "class_id", "class_name", "source_image", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote split index to {index_path}")
    print(f"Split counts: {counts_per_split}")
    return counts_per_split


if __name__ == "__main__":
    run_split()
