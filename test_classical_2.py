"""Evaluate Classical2 on the dataset and optionally export misclassified examples."""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from classical.classical_2 import (
    Classical2,
    save_analysis_visualizations,
    write_analysis_report,
)

CLASS_NAMES = {
    0: "broken_cap",
    1: "broken_ring",
    2: "good_cap",
    3: "loose_cap",
    4: "no_cap",
}


def load_labels(label_path: Path) -> list[int] | None:
    if not label_path.exists():
        return None
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return None
    labels = []
    for line in lines:
        match = re.match(r"\s*(\d)", line)
        if match:
            labels.append(int(match.group(1)))
    return labels if labels else None


def score_output(output: int, expected: list[int]) -> float:
    if not expected:
        return 0.0
    matches = sum(1 for label in expected if label == output)
    if matches == 0:
        return 0.0
    if matches == len(expected):
        return 1.0
    return 0.5


def _error_folder_name(stem: str, expected: list[int], predicted: int) -> str:
    exp = "-".join(str(x) for x in expected)
    exp_label = CLASS_NAMES.get(expected[0], str(expected[0])) if len(expected) == 1 else "multi"
    pred_label = CLASS_NAMES.get(predicted, str(predicted))
    return f"exp{exp}_{exp_label}__pred{predicted}_{pred_label}__{stem}"


def export_error_case(
    result: dict,
    image_path: Path,
    out_root: Path,
    expected: list[int],
    predicted: int,
    score: float,
) -> Path:
    folder = out_root / _error_folder_name(image_path.stem, expected, predicted)
    save_analysis_visualizations(result, folder, original_path=image_path)
    write_analysis_report(
        result,
        folder / "report.txt",
        expected=expected,
        score=score,
    )
    return folder


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
        help="Also export partially correct cases (score 0.5), not only full errors.",
    )
    parser.add_argument(
        "--keep-errors-dir",
        action="store_true",
        help="Do not clear the errors output folder before running.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    img_dir = root / "bottle-cap.yolov8" / "train" / "images"
    label_dir = root / "bottle-cap.yolov8" / "train" / "labels"
    errors_dir = args.errors_dir or (root / "classical" / "result" / "errors")

    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            "Required folders bottle-cap.yolov8/train/images or "
            "bottle-cap.yolov8/train/labels not found"
        )

    if args.export_errors and not args.keep_errors_dir and errors_dir.exists():
        shutil.rmtree(errors_dir)

    model = Classical2()
    files = sorted(img_dir.glob("WIN_*.jpg"))
    total = 0
    full_correct = 0
    partial_correct = 0
    exported = 0
    missing_labels = []
    classes = [0, 1, 2, 3, 4]
    error_matrix = {exp: {pred: 0 for pred in classes} for exp in classes}

    results_path = root / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["photo", "correct_classification", "classical_2", "score"])

        for image_path in files:
            label_path = label_dir / (image_path.stem + ".txt")
            expected = load_labels(label_path)
            if expected is None:
                missing_labels.append(image_path.name)
                continue

            result = model.analyze(str(image_path))
            status_code = result.get("status_code")
            if status_code is None:
                raise ValueError(f"No status_code returned for {image_path}")

            total += 1
            score = score_output(status_code, expected)
            if score != 1.0:
                writer.writerow(
                    [
                        image_path.name,
                        ";".join(str(x) for x in expected),
                        status_code,
                        score,
                    ]
                )
                should_export = args.export_errors and (
                    score < 0.5 or args.export_partial
                )
                if should_export:
                    export_error_case(
                        result,
                        image_path,
                        errors_dir,
                        expected,
                        status_code,
                        score,
                    )
                    exported += 1

            if score == 1.0:
                full_correct += 1
            elif score == 0.5:
                partial_correct += 1

            for exp_label in set(expected):
                if exp_label not in error_matrix:
                    error_matrix[exp_label] = {pred: 0 for pred in classes}
                if status_code not in error_matrix[exp_label]:
                    error_matrix[exp_label][status_code] = 0
                error_matrix[exp_label][status_code] += 1

    if total == 0:
        print("No tested images found or no valid labels.")
        return

    accuracy = ((full_correct + 0.5 * partial_correct) / total) * 100.0
    print(f"Tested images: {total}")
    print(f"Fully correct: {full_correct}")
    print(f"Partially correct: {partial_correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    print("Error matrix (expected rows, predicted columns):")
    header = "    " + " ".join(f"{pred:>5}" for pred in classes)
    print(header)
    for exp_label in classes:
        row = (
            f"{exp_label:>2} "
            + " ".join(f"{error_matrix[exp_label].get(pred, 0):>5}" for pred in classes)
        )
        print(row)

    if missing_labels:
        print(f"Skipped images with missing or invalid labels: {len(missing_labels)}")

    print(f"\nMismatches written to: {results_path}")
    if args.export_errors:
        print(f"Exported {exported} error case(s) to: {errors_dir}")
        print("Each folder contains: original.jpg, annotated.jpg, background.jpg,")
        print("bottle.jpg, cap.jpg, report.txt")


if __name__ == "__main__":
    main()
