"""Visualize the augmentation process on a single photo.

This script loads a sample image and displays the augmentation pipeline
step-by-step in a matplotlib grid, showing:
  - Original image
  - Each augmentation applied individually
  - Multiple augmented versions stacked together

Usage:
    python data/visualize_augmentation.py
    python data/visualize_augmentation.py --image path/to/image.png
    python data/visualize_augmentation.py --n-copies 5
    python data/visualize_augmentation.py --save output.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import IMAGE_SIZE  # noqa: E402
from data.augmentation import train_augmentations  # noqa: E402
from data.dataset_loader import iter_samples  # noqa: E402


def _get_sample_image(image_path: str | None = None) -> np.ndarray:
    """Load a single sample image (BGR format).

    If image_path is None, loads the first available image from the dataset.
    """
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return img

    # Load first available sample from dataset
    for sample in iter_samples():
        return sample.crop

    raise FileNotFoundError(
        "No images found in dataset. "
        "Run `python data/download_dataset.py` first."
    )


def _apply_individual_augmentations(
    image: np.ndarray,
) -> dict[str, tuple[np.ndarray, str]]:
    """Apply augmentations individually to show effect of each.

    Returns dict mapping names to (image, description) tuples.
    """
    results = {
        "original": (image, "Original Image")
    }

    # Get the list of transforms in the pipeline with descriptions
    transforms_to_test = [
        (
            "rotate",
            A.Rotate(
                limit=20,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            "Rotate: ±20°",
        ),
        (
            "h_flip",
            A.HorizontalFlip(p=1.0),
            "Horizontal Flip",
        ),
        (
            "brightness_contrast",
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),
            "Brightness/Contrast: ±20%",
        ),
        (
            "gauss_blur",
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            "Gaussian Blur: kernel (3-5)×(3-5)",
        ),
        (
            "motion_blur",
            A.MotionBlur(blur_limit=5, p=1.0),
            "Motion Blur: kernel size 5",
        ),
        (
            "gauss_noise",
            A.GaussNoise(p=1.0, std_range=(0.02, 0.1)),
            "Gaussian Noise: σ ∈ [0.02, 0.1]",
        ),
        (
            "resize",
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            f"Resize: {IMAGE_SIZE}×{IMAGE_SIZE}",
        ),
    ]

    for name, transform, description in transforms_to_test:
        try:
            augmented = transform(image=image)["image"]
            results[name] = (augmented, description)
        except Exception as e:
            print(f"Warning: failed to apply {name}: {e}")

    return results


def _apply_full_pipeline(
    image: np.ndarray, n_copies: int = 5
) -> list[np.ndarray]:
    """Apply the full augmentation pipeline multiple times."""
    aug = train_augmentations()
    results = [image]  # Original
    for _ in range(n_copies):
        augmented = aug(image=image)["image"]
        results.append(augmented)
    return results


def _plot_individual_augmentations(
    image: np.ndarray, output_path: str | None = None
) -> None:
    """Create a figure showing each augmentation separately."""
    augmentations = _apply_individual_augmentations(image)
    n_augs = len(augmentations)

    # Calculate grid dimensions
    cols = 4
    rows = (n_augs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for idx, (name, (img, description)) in enumerate(
        augmentations.items()
    ):
        ax = axes[idx]
        # Convert BGR to RGB for display
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        ax.imshow(img_rgb)
        ax.set_title(description, fontsize=11, fontweight="bold")
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n_augs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def _plot_multiple_copies(
    image: np.ndarray,
    n_copies: int = 5,
    output_path: str | None = None,
) -> None:
    """Create a figure showing multiple augmented copies."""
    augmented_copies = _apply_full_pipeline(image, n_copies)

    cols = n_copies + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

    for idx, img in enumerate(augmented_copies):
        ax = axes[idx]
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        ax.imshow(img_rgb)
        if idx == 0:
            ax.set_title("Original", fontsize=12, fontweight="bold")
        else:
            ax.set_title(f"Augmented #{idx}", fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the augmentation process on a single photo."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file. If not provided, uses first "
        "available dataset image.",
    )
    parser.add_argument(
        "--n-copies",
        type=int,
        default=5,
        help="Number of augmented copies to generate (default: 5).",
    )
    parser.add_argument(
        "--mode",
        choices=("individual", "copies", "both"),
        default="both",
        help="Visualization mode: 'individual' "
        "(each augmentation separately), "
        "'copies' (multiple full-pipeline outputs), or 'both' (default).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save figure to file instead of displaying interactively.",
    )

    args = parser.parse_args()

    print("Loading image...")
    image = _get_sample_image(args.image)
    print(f"Image shape: {image.shape}")

    if args.mode in ("individual", "both"):
        print("\nGenerating individual augmentation view...")
        output_path = f"{args.save[:-4]}_individual.png" if args.save else None
        _plot_individual_augmentations(image, output_path)

    if args.mode in ("copies", "both"):
        print("\nGenerating multiple copies view...")
        output_path = f"{args.save[:-4]}_copies.png" if args.save else None
        _plot_multiple_copies(image, args.n_copies, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
