"""Common image loading / preprocessing used by classifiers.

The convention across the project:

- ``load_split_as_arrays(split)``     -> (X, y, paths) with X uint8 BGR.
- ``to_gray(X)``                      -> (N, H, W) uint8.
- ``to_float_normalized(X)``          -> (N, H, W, 3) float32 in [0, 1].
- ``to_imagenet_tensor(X)``           -> torch.Tensor (N, 3, H, W) normalized.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CROPS_DIR, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def _iter_class_folders(split_dir: Path):
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split folder not found: {split_dir}. "
            "Run `python data/splitter.py` first."
        )
    for class_folder in sorted(split_dir.iterdir()):
        if class_folder.is_dir():
            yield class_folder


def _class_name_to_id(class_name: str) -> int:
    """Map sanitized folder name back to class_id using config.CLASS_NAMES."""
    from config import CLASS_NAMES

    for i, name in enumerate(CLASS_NAMES):
        if name.replace(" ", "_").lower() == class_name:
            return i
    raise KeyError(f"Unknown class folder: {class_name}")


def load_split_as_arrays(
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Load every image for a split into one big uint8 BGR array.

    Returns (X, y, paths):
        X: (N, H, W, 3) uint8
        y: (N,) int
        paths: list of Path for each sample
    """
    split_dir = CROPS_DIR / split
    imgs: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []
    for class_folder in _iter_class_folders(split_dir):
        class_id = _class_name_to_id(class_folder.name)
        for img_path in sorted(class_folder.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            imgs.append(img)
            labels.append(class_id)
            paths.append(img_path)
    if not imgs:
        raise RuntimeError(f"No images loaded for split '{split}'.")
    X = np.stack(imgs, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, paths


def to_gray(X: np.ndarray) -> np.ndarray:
    """BGR (N, H, W, 3) -> grayscale (N, H, W) uint8."""
    return np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X], axis=0)


def to_float_normalized(X: np.ndarray) -> np.ndarray:
    """uint8 -> float32 in [0, 1]."""
    return X.astype(np.float32) / 255.0


def to_imagenet_tensor(X: np.ndarray):
    """BGR uint8 (N, H, W, 3) -> torch tensor (N, 3, H, W) ImageNet-normalized."""
    import torch  # lazy import to keep classical methods torch-free

    rgb = X[..., ::-1].astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    rgb = (rgb - mean) / std
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
