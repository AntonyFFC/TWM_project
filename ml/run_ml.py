"""Train and evaluate every ML model.

Each model is by default run twice (raw + aug). Use ``--augmentation`` to change
that. See ``classical/run_classical.py`` for the same pattern.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluator import TRAINED_ON_CHOICES, run_method  # noqa: E402
from ml.base_model import BaseModel  # noqa: E402
from ml.feature_ml import CNNFeatureClassifier  # noqa: E402
from ml.transfer_learning import TransferLearningModel  # noqa: E402


MODEL_FACTORIES: list[Callable[[], BaseModel]] = [
    lambda: TransferLearningModel(backbone="resnet18"),
    lambda: TransferLearningModel(backbone="mobilenet_v2"),
    lambda: CNNFeatureClassifier(classifier="xgboost"),
    lambda: CNNFeatureClassifier(classifier="random_forest"),
]


def run_all(trained_on_variants: list[str]) -> None:
    total = len(MODEL_FACTORIES) * len(trained_on_variants)
    print(
        f"Running {len(MODEL_FACTORIES)} ML model(s) "
        f"x {len(trained_on_variants)} variant(s) = {total} run(s)."
    )
    failures: list[tuple[str, str, str]] = []
    for factory in MODEL_FACTORIES:
        for trained_on in trained_on_variants:
            model = factory()
            try:
                run_method(model, method_kind="ml", trained_on=trained_on)
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                failures.append((model.name, trained_on, str(exc)))
                print(
                    f"  !! {model.name}_{trained_on} FAILED -- continuing with next run."
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
