"""End-to-end pipeline: data -> training -> evaluation -> comparison.

Usage:
    python run_all.py                           # everything (raw + aug)
    python run_all.py --augmentation raw        # only raw training
    python run_all.py --augmentation aug        # only augmented training
    python run_all.py --skip-classical          # only ML + compare
    python run_all.py --skip-ml                 # only classical + compare
    python run_all.py --skip-split              # reuse existing splits
    python run_all.py --skip-eda                # don't regenerate EDA plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import CROPS_DIR, ensure_dirs  # noqa: E402
from evaluation.evaluator import TRAINED_ON_CHOICES  # noqa: E402


def _has_splits() -> bool:
    return (
        CROPS_DIR.exists()
        and any((CROPS_DIR / split).exists() for split in ("train", "val", "test"))
        and any(CROPS_DIR.rglob("*.png"))
    )


def _ensure_images_present() -> bool:
    """Fail fast with an informative message if the raw images are missing."""
    from data.download_dataset import count_label_image_pairs

    labels, images, _missing = count_label_image_pairs()
    if images == 0:
        print("No images found in bottle-cap.yolov8/train/images/.")
        print("Run `python data/download_dataset.py` or add them manually.")
        return False
    if images < labels:
        print(
            f"WARNING: fewer images ({images}) than labels ({labels}). "
            "Some samples will be skipped."
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation",
        choices=("raw", "aug", "both"),
        default="both",
        help="Which training-data variants to run (default: both).",
    )
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-ml", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    if not _ensure_images_present():
        sys.exit(1)

    if not args.skip_eda:
        print("\n### EDA ###")
        from data import eda

        eda.main()

    if not args.skip_split:
        if _has_splits():
            print("\n### Splits already present -- skipping splitter.")
        else:
            print("\n### SPLIT ###")
            from data import splitter

            splitter.run_split()
    else:
        print("\n### SPLIT (skipped) ###")

    variants = (
        list(TRAINED_ON_CHOICES) if args.augmentation == "both" else [args.augmentation]
    )
    print(f"\n### Training variants: {variants} ###")

    if not args.skip_classical:
        print("\n### CLASSICAL METHODS ###")
        from classical import run_classical

        run_classical.run_all(variants)

    if not args.skip_ml:
        print("\n### ML METHODS ###")
        from ml import run_ml

        run_ml.run_all(variants)

    print("\n### COMPARE ###")
    from evaluation import compare

    compare.main()

    print("\nDone. See the 'results/' directory for all artifacts.")


if __name__ == "__main__":
    main()
