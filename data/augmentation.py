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
    transforms = _default_train_transforms()
    transforms.append(A.Resize(image_size, image_size))
    return A.Compose(transforms)


def _default_train_transforms() -> list:
    """Training transforms without resize (shared by crop and full-image pipelines)."""
    return [
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
    ]


def train_augmentations_full_image(config: dict | None = None) -> A.Compose:
    """Same training-style augmentations without resizing (for full-frame CV)."""
    if config is None:
        return A.Compose(_default_train_transforms())
    return build_augmentation_compose_from_config(config)


def build_augmentation_compose_from_config(config: dict) -> A.Compose:
    """Build an Albumentations pipeline from GUI augmentation config dict."""
    transforms: list = []

    if config.get("rotate_enabled", True):
        limit = int(config.get("rotate_limit", 20))
        transforms.append(
            A.Rotate(
                limit=limit,
                border_mode=cv2.BORDER_REFLECT_101,
                p=float(config.get("rotate_prob", 0.7)),
            )
        )

    if config.get("h_flip_enabled", True):
        transforms.append(A.HorizontalFlip(p=float(config.get("h_flip_prob", 0.5))))

    if config.get("brightness_enabled", True):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(config.get("brightness_limit", 0.2)),
                contrast_limit=float(config.get("contrast_limit", 0.2)),
                p=float(config.get("brightness_prob", 0.6)),
            )
        )

    blur_enabled = config.get("blur_enabled", True)
    m_blur_enabled = config.get("m_blur_enabled", True)
    if blur_enabled or m_blur_enabled:
        blur_ops: list = []
        if blur_enabled:
            b_limit = int(config.get("blur_limit", 5))
            b_limit = max(3, b_limit if b_limit % 2 != 0 else b_limit + 1)
            blur_ops.append(A.GaussianBlur(blur_limit=(3, b_limit), p=1.0))
        if m_blur_enabled:
            mb_limit = int(config.get("m_blur_limit", 5))
            mb_limit = max(3, mb_limit if mb_limit % 2 != 0 else mb_limit + 1)
            blur_ops.append(A.MotionBlur(blur_limit=(3, mb_limit), p=1.0))
        blur_prob = float(config.get("blur_prob", 0.3))
        if len(blur_ops) == 1:
            transforms.append(A.OneOf(blur_ops, p=blur_prob))
        else:
            transforms.append(
                A.OneOf(
                    blur_ops,
                    p=max(blur_prob, float(config.get("m_blur_prob", 0.3))),
                )
            )

    if config.get("noise_enabled", True):
        n_std = float(config.get("noise_std", 0.1))
        version = tuple(int(p) for p in A.__version__.split(".")[:2] if p.isdigit())
        if version >= (2, 0):
            noise = A.GaussNoise(
                std_range=(0.0, n_std),
                p=float(config.get("noise_prob", 0.3)),
            )
        else:
            noise = A.GaussNoise(
                var_limit=(5.0, max(25.0, n_std * 255)),
                p=float(config.get("noise_prob", 0.3)),
            )
        transforms.append(noise)

    if not transforms:
        return A.Compose(_default_train_transforms())
    return A.Compose(transforms)


def augment_bgr_image(
    image_bgr: np.ndarray,
    compose: A.Compose,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Apply ``compose`` to a BGR image and return BGR."""
    if seed is not None:
        np.random.seed(seed)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out_rgb = compose(image=image_rgb)["image"]
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def generate_augmented_copies_bgr(
    image_bgr: np.ndarray,
    n_copies: int,
    *,
    compose: A.Compose | None = None,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return ``n_copies`` independently augmented BGR images."""
    pipeline = compose or train_augmentations_full_image()
    copies: list[np.ndarray] = []
    for idx in range(n_copies):
        copies.append(augment_bgr_image(image_bgr, pipeline, seed=seed + idx))
    return copies


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
