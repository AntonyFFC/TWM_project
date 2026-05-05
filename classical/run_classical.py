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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation",
        choices=("raw", "aug", "both"),
        default="both",
        help="Ktore warianty treningowe odpalic (default: both).",
    )
    args = parser.parse_args()
    variants = list(TRAINED_ON_CHOICES) if args.augmentation == "both" else [args.augmentation]
    run_all(variants)


if __name__ == "__main__":
    main()
