"""End-to-end pipeline: dane -> trening -> ewaluacja -> porownanie -> demo.

Uzycie:
    python run_all.py                      # raw + aug, oba modele, demo
    python run_all.py --augmentation aug   # tylko augmented (najszybciej)
    python run_all.py --skip-classical     # tylko ResNet18
    python run_all.py --skip-ml            # tylko HOG+SVM
    python run_all.py --skip-demo          # bez generowania demo PNG
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from pipeline.orchestrator import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation",
        choices=("raw", "aug", "both"),
        default="both",
        help="Ktore warianty treningowe (default: both).",
    )
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    args = parser.parse_args()

    ok = run_pipeline(
        augmentation=args.augmentation,
        skip_eda=args.skip_eda,
        skip_split=args.skip_split,
        skip_classical=args.skip_classical,
        skip_ml=args.skip_ml,
        skip_demo=args.skip_demo,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
