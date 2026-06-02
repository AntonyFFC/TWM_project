"""Trening i ewaluacja metody klasycznej (HOG + SVM).

Domyslnie odpalamy raz na danych raw i raz na augmentowanych, zeby pokazac
ile daje augmentacja (``--augmentation raw|aug|both``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from classical.hog_svm import HogSvmClassifier  # noqa: E402
from classical.classical_2 import Classical2  # noqa: E402
from evaluation.evaluator import TRAINED_ON_CHOICES, run_method  # noqa: E402


def run_all(trained_on_variants: list[str]) -> None:
    print(
        f"Klasyczne: HOG+SVM x {len(trained_on_variants)} wariant(y) "
        f"= {len(trained_on_variants)} run(y)."
    )
    for trained_on in trained_on_variants:
        method = HogSvmClassifier()
        try:
            run_method(method, method_kind="classical", trained_on=trained_on)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"  !! {method.name}_{trained_on} ZEPSULO SIE: {exc}")


def run_classical2_demo(image_paths: list[str]) -> None:
    """Run Classical2 analyzer on image paths for demo/testing (no visualization)."""
    print(f"Classical2 Demo: analyzing {len(image_paths)} image(s)...")
    analyzer = Classical2()

    for img_path in image_paths:
        try:
            result = analyzer.analyze(img_path)
            print(f"\n{img_path}:")
            print(f"  Status: {result['status_list']}")
            print(f"  Measurements: {result['measurements']}")
        except FileNotFoundError:
            print(f"  ERROR: File not found: {img_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation",
        choices=("raw", "aug", "both"),
        default="both",
        help="Ktore warianty treningowe odpalic (default: both).",
    )
    parser.add_argument(
        "--classical2",
        nargs="*",
        help="Run Classical2 demo on image(s). If no args, uses sample from dataset.",
    )
    # Note: visualization/display flags removed; demo runs without GUI
    args = parser.parse_args()
    
    # Run Classical2 if requested
    if args.classical2 is not None:
        if not args.classical2:
            # Use sample images from dataset if no args provided
            sample_dir = Path(__file__).resolve().parents[1] / "bottle-cap.yolov8" / "train" / "images"
            if sample_dir.exists():
                sample_images = list(sample_dir.glob("*.jpg"))[:3]  # First 3 images
                if sample_images:
                    run_classical2_demo([str(p) for p in sample_images])
                else:
                    print(f"ERROR: No images found in {sample_dir}")
            else:
                print(f"ERROR: Sample directory not found: {sample_dir}")
        else:
            run_classical2_demo(args.classical2)
        return
    
    # Run classical HOG+SVM as default
    variants = list(TRAINED_ON_CHOICES) if args.augmentation == "both" else [args.augmentation]
    run_all(variants)


if __name__ == "__main__":
    main()
