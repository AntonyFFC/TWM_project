"""Augmentation pipelines.

Two flavors:

- ``train_augmentations()`` -- random augmentation used during training
  to increase effective training size and diversity.
- ``robustness_corruptions()`` -- deterministic corruptions used in
  the ``evaluation/robustness.py`` test set (blur, noise, brightness, etc.).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import IMAGE_SIZE  # noqa: E402


def _gauss_noise_kwargs() -> dict:
    """Return kwargs for ``A.GaussNoise`` compatible with the installed version.

    Albumentations 2.0+ replaced ``var_limit`` with ``std_range`` (relative to
    [0, 1]). We pick the right kwarg based on the installed version.
    """
    version = tuple(int(p) for p in A.__version__.split(".")[:2] if p.isdigit())
    if version >= (2, 0):
        return {"std_range": (0.02, 0.1)}
    return {"var_limit": (5.0, 25.0)}


def train_augmentations(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Random augmentation used when building the training set."""
    return A.Compose(
        [
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.6
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),
            A.GaussNoise(p=0.3, **_gauss_noise_kwargs()),
            A.Resize(image_size, image_size),
        ]
    )


def _blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (7, 7), sigmaX=1.5)


def _motion_blur(image: np.ndarray) -> np.ndarray:
    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def _gauss_noise(image: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _brightness_down(image: np.ndarray, factor: float = 0.6) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _brightness_up(image: np.ndarray, factor: float = 1.4) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def robustness_corruptions() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Deterministic corruptions used in the robustness benchmark."""
    return {
        "gauss_blur": _blur,
        "motion_blur": _motion_blur,
        "gauss_noise": _gauss_noise,
        "bright_down": _brightness_down,
        "bright_up": _brightness_up,
    }


def apply_corruption(X: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a single-image corruption to a batch (N, H, W, 3)."""
    return np.stack([fn(img) for img in X], axis=0)


def apply_train_augmentation_batch(
    X: np.ndarray, y: np.ndarray, n_copies: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Produce ``n_copies`` augmented versions of each image (concatenated with
    the originals). Labels are duplicated accordingly.
    """
    np.random.seed(seed)
    aug = train_augmentations()
    all_X: list[np.ndarray] = [X]
    all_y: list[np.ndarray] = [y]
    for _ in range(n_copies):
        augmented = np.stack([aug(image=img)["image"] for img in X], axis=0)
        all_X.append(augmented)
        all_y.append(y)
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)
