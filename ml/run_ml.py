"""Trening i ewaluacja modelu ML (ResNet18 transfer learning).

Domyslnie odpalamy raz na danych raw i raz na augmentowanych
(``--augmentation raw|aug|both``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluator import TRAINED_ON_CHOICES, run_method  # noqa: E402
from ml.transfer_learning import TransferLearningModel  # noqa: E402


def run_all(trained_on_variants: list[str]) -> None:
    print(
        f"ML: ResNet18 x {len(trained_on_variants)} wariant(y) "
        f"= {len(trained_on_variants)} run(y)."
    )
    for trained_on in trained_on_variants:
        model = TransferLearningModel()
        try:
            run_method(model, method_kind="ml", trained_on=trained_on)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"  !! {model.name}_{trained_on} ZEPSULO SIE: {exc}")


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
