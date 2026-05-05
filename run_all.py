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

from config import CROPS_DIR, ensure_dirs  # noqa: E402
from evaluation.evaluator import TRAINED_ON_CHOICES  # noqa: E402


def _has_splits() -> bool:
    return (
        CROPS_DIR.exists()
        and any((CROPS_DIR / split).exists() for split in ("train", "val", "test"))
        and any(CROPS_DIR.rglob("*.png"))
    )


def _ensure_images_present() -> bool:
    """Twardy stop, gdy brakuje surowych obrazow."""
    from data.download_dataset import count_label_image_pairs

    labels, images, _missing = count_label_image_pairs()
    if images == 0:
        print("Brak obrazow w bottle-cap.yolov8/train/images/.")
        print("Odpal `python data/download_dataset.py` lub wrzuc je recznie.")
        return False
    if images < labels:
        print(
            f"UWAGA: mniej obrazow ({images}) niz etykiet ({labels}). "
            "Czesc probek bedzie pominieta."
        )
    return True


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

    ensure_dirs()

    if not _ensure_images_present():
        sys.exit(1)

    if not args.skip_eda:
        print("\n### EDA ###")
        from data import eda

        eda.main()

    if not args.skip_split:
        if _has_splits():
            print("\n### Splity juz istnieja -- pomijam splitter.")
        else:
            print("\n### SPLIT ###")
            from data import splitter

            splitter.run_split()
    else:
        print("\n### SPLIT (pominiety) ###")

    variants = (
        list(TRAINED_ON_CHOICES) if args.augmentation == "both" else [args.augmentation]
    )
    print(f"\n### Warianty treningowe: {variants} ###")

    if not args.skip_classical:
        print("\n### METODA KLASYCZNA: HOG + SVM ###")
        from classical import run_classical

        run_classical.run_all(variants)

    if not args.skip_ml:
        print("\n### MODEL ML: ResNet18 (transfer learning) ###")
        from ml import run_ml

        run_ml.run_all(variants)

    print("\n### PORÓWNANIE WYNIKÓW ###")
    from evaluation import compare

    compare.main()

    if not args.skip_demo:
        print("\n### DEMO PREDYKCJI ###")
        try:
            import demo

            demo.main()
        except Exception as exc:  # noqa: BLE001
            print(f"Demo nie wystartowalo: {exc}")

    print("\nGotowe. Zobacz folder 'results/' po wykresy i metryki.")


if __name__ == "__main__":
    main()
