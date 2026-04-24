"""Load (image, YOLO labels) pairs and crop individual bottle caps.

Key function:
    iter_samples() -> generator of CapSample(image_path, class_id, bbox, crop)

A "sample" here means a single cropped cap (one row per bounding box),
which is the input unit our classifiers consume.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    BBOX_PADDING,
    CLASS_NAMES,
    IMAGE_SIZE,
    RAW_IMAGES_DIR,
    RAW_LABELS_DIR,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class CapSample:
    """A single cropped cap ready for classification."""

    source_image: str  # filename of the original photo
    class_id: int
    class_name: str
    bbox_xyxy: tuple[int, int, int, int]  # absolute pixel coords (x1, y1, x2, y2)
    crop: np.ndarray  # HxWx3 uint8 (BGR) of fixed IMAGE_SIZE


def _find_image_for_label(label_path: Path) -> Path | None:
    """Find the image file that belongs to a given label .txt."""
    for ext in IMAGE_EXTENSIONS:
        candidate = RAW_IMAGES_DIR / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    padding: float = BBOX_PADDING,
) -> tuple[int, int, int, int]:
    """Convert YOLO (cx, cy, w, h) normalized to absolute (x1, y1, x2, y2)."""
    w = w * (1.0 + 2 * padding)
    h = h * (1.0 + 2 * padding)
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(img_w - 1, round(x2)))
    y2 = int(min(img_h - 1, round(y2)))
    return x1, y1, x2, y2


def _resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize a crop into a fixed square without preserving aspect ratio."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return list of (class_id, cx, cy, w, h) for all boxes in a label file."""
    rows: list[tuple[int, float, float, float, float]] = []
    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
            rows.append((cls, cx, cy, w, h))
    return rows


def iter_samples(
    image_size: int = IMAGE_SIZE,
    skip_missing_images: bool = True,
) -> Iterator[CapSample]:
    """Yield every ``CapSample`` from the raw dataset (one per bbox)."""
    if not RAW_LABELS_DIR.exists():
        raise FileNotFoundError(f"No labels directory: {RAW_LABELS_DIR}")

    label_files = sorted(RAW_LABELS_DIR.glob("*.txt"))
    for label_path in label_files:
        img_path = _find_image_for_label(label_path)
        if img_path is None:
            if skip_missing_images:
                continue
            raise FileNotFoundError(f"Missing image for label {label_path.name}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            if skip_missing_images:
                continue
            raise RuntimeError(f"Cannot read image {img_path}")
        img_h, img_w = img.shape[:2]

        for cls, cx, cy, w, h in parse_label_file(label_path):
            x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = _resize_to_square(crop, image_size)
            yield CapSample(
                source_image=img_path.name,
                class_id=cls,
                class_name=CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls),
                bbox_xyxy=(x1, y1, x2, y2),
                crop=crop,
            )


def load_all_samples(image_size: int = IMAGE_SIZE) -> list[CapSample]:
    """Load everything into memory (fine for 639-image prototype)."""
    return list(iter_samples(image_size=image_size))


if __name__ == "__main__":
    samples = load_all_samples()
    print(f"Loaded {len(samples)} cap crops from raw dataset.")
    if samples:
        from collections import Counter

        dist = Counter(s.class_name for s in samples)
        for name, cnt in dist.most_common():
            print(f"  {name:15s} {cnt}")
