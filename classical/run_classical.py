"""Train and evaluate every classical method.

Each method is by default run twice -- once on raw training data and once on
augmented training data -- so we can measure how much augmentation helps.
Use ``--augmentation raw|aug|both`` to change that.

Adding a new method is as simple as appending a factory to ``METHOD_FACTORIES``.
Every factory must return a fresh :class:`BaseClassifier` instance, because
running ``raw`` and ``aug`` requires two independently trained models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from classical.base_classifier import BaseClassifier  # noqa: E402
from classical.edge_contour import EdgeContourClassifier  # noqa: E402
from classical.hog_svm import HogSvmClassifier  # noqa: E402
from classical.threshold_morphology import ThresholdMorphologyClassifier  # noqa: E402
from evaluation.evaluator import TRAINED_ON_CHOICES, run_method  # noqa: E402


METHOD_FACTORIES: list[type[BaseClassifier]] = [
    HogSvmClassifier,
    EdgeContourClassifier,
    ThresholdMorphologyClassifier,
]


def run_all(trained_on_variants: list[str]) -> None:
    total = len(METHOD_FACTORIES) * len(trained_on_variants)
    print(
        f"Running {len(METHOD_FACTORIES)} classical method(s) "
        f"x {len(trained_on_variants)} variant(s) = {total} run(s)."
    )
    failures: list[tuple[str, str, str]] = []
    for factory in METHOD_FACTORIES:
        for trained_on in trained_on_variants:
            method = factory()
            try:
                run_method(method, method_kind="classical", trained_on=trained_on)
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                failures.append((method.name, trained_on, str(exc)))
                print(
                    f"  !! {method.name}_{trained_on} FAILED -- continuing with next run."
                )
    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for name, variant, err in failures:
            print(f"  - {name}_{variant}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation",
        choices=("raw", "aug", "both"),
        default="both",
        help="Which training-data variant(s) to run (default: both).",
    )
    args = parser.parse_args()
    variants = list(TRAINED_ON_CHOICES) if args.augmentation == "both" else [args.augmentation]
    run_all(variants)


if __name__ == "__main__":
    main()
