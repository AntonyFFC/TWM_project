"""Evaluate Classical2 on the dataset and optionally export misclassified examples."""
from __future__ import annotations

import argparse
from pathlib import Path

from classical.classical_2 import Classical2
from classical.evaluate_classical2 import EVAL_ON_CHOICES, evaluate_dataset


def _print_eval_result(label: str, result) -> None:
    if result is None or result.total == 0:
        print(f"\n=== {label}: no samples ===")
        return
    print(f"\n=== {label} ===")
    print(f"Tested samples: {result.total}")
    print(f"Fully correct: {result.full_correct}")
    print(f"Partially correct: {result.partial_correct}")
    print(f"Accuracy: {result.accuracy:.2f}%")
    print()
    print("Error matrix (expected rows, predicted columns):")
    classes = [0, 1, 2, 3, 4]
    header = "    " + " ".join(f"{pred:>5}" for pred in classes)
    print(header)
    for exp_label in classes:
        row = (
            f"{exp_label:>2} "
            + " ".join(f"{result.confusion[exp_label].get(pred, 0):>5}" for pred in classes)
        )
        print(row)
    if result.missing_labels:
        print(f"Skipped images with missing or invalid labels: {len(result.missing_labels)}")
    print(f"Misclassified / partial: {len(result.errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Classical2 and export error galleries.")
    parser.add_argument(
        "--export-errors",
        action="store_true",
        default=True,
        help="Save visualizations for misclassified images (default: on).",
    )
    parser.add_argument(
        "--no-export-errors",
        action="store_false",
        dest="export_errors",
        help="Skip saving error image folders.",
    )
    parser.add_argument(
        "--errors-dir",
        type=Path,
        default=None,
        help="Output folder for error cases (default: classical/result/errors).",
    )
    parser.add_argument(
        "--export-partial",
        action="store_true",
        help="Also export partially correct cases (score 0.5).",
    )
    parser.add_argument(
        "--keep-errors-dir",
        action="store_true",
        help="Do not clear the errors output folder before running.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        help="JSON preset name from classical/presets/ (default: default).",
    )
    parser.add_argument(
        "--eval-on",
        choices=EVAL_ON_CHOICES,
        default="raw",
        help="Evaluate on original images (raw), augmented copies (aug), or both.",
    )
    parser.add_argument(
        "--aug-copies",
        type=int,
        default=2,
        help="Augmented copies per image when eval-on is aug or both (default: 2).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    img_dir = root / "bottle-cap.yolov8" / "train" / "images"
    label_dir = root / "bottle-cap.yolov8" / "train" / "labels"
    errors_dir = args.errors_dir or (root / "classical" / "result" / "errors")

    try:
        from gui.services.classical2_preset_store import load_preset

        params = load_preset(args.preset)
    except Exception:
        params = Classical2.get_default_params()

    analyzer = Classical2(params)
    compare = evaluate_dataset(
        analyzer,
        img_dir,
        label_dir,
        eval_on=args.eval_on,
        export_errors_dir=errors_dir if args.export_errors else None,
        export_partial=args.export_partial,
        clear_export_dir=args.export_errors and not args.keep_errors_dir,
        results_csv_path=root / "results.csv",
        aug_copies=max(1, args.aug_copies),
        preset_name=args.preset,
        log=print,
    )

    primary = compare.primary
    if primary is None or primary.total == 0:
        print("No tested images found or no valid labels.")
        return

    if args.eval_on == "both":
        _print_eval_result("Raw images", compare.raw)
        _print_eval_result("Augmented images", compare.augmented)
        if compare.raw and compare.augmented:
            delta = compare.augmented.accuracy - compare.raw.accuracy
            print(f"\nAccuracy delta (aug - raw): {delta:+.2f}%")
    else:
        _print_eval_result(args.eval_on, primary)

    print(f"\nMismatches written to: {root / 'results.csv'} (and _aug when applicable)")
    if args.export_errors:
        print(f"Exported error case(s) to: {errors_dir}")
    for label, result in (("raw", compare.raw), ("aug", compare.augmented)):
        if result and result.metrics_json_path:
            print(f"Metrics ({label}): {result.metrics_json_path}")
        if result and result.confusion_plot_path:
            print(f"Confusion matrix ({label}): {result.confusion_plot_path}")


if __name__ == "__main__":
    main()
