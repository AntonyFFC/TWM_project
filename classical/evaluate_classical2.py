"""Batch evaluation for Classical2 against labeled dataset."""
from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from classical.classical2_labels import CLASSICAL2_SLUGS, score_output
from classical.classical_2 import (
    Classical2,
    save_analysis_visualizations,
    write_analysis_report,
)


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


def find_expected_labels(image_path: Path, labels_dir: Path) -> list[int] | None:
    """Look up YOLO class id(s) for an image from a labels directory."""
    candidates = [labels_dir / f"{image_path.stem}.txt"]
    if image_path.parent.name == "images":
        candidates.insert(0, image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
    for label_path in candidates:
        labels = load_labels(label_path)
        if labels is not None:
            return labels
    return None


def error_folder_name(stem: str, expected: list[int], predicted: int) -> str:
    exp = "-".join(str(x) for x in expected)
    exp_label = (
        CLASSICAL2_SLUGS.get(expected[0], str(expected[0]))
        if len(expected) == 1
        else "multi"
    )
    pred_label = CLASSICAL2_SLUGS.get(predicted, str(predicted))
    return f"exp{exp}_{exp_label}__pred{predicted}_{pred_label}__{stem}"


@dataclass
class ErrorCase:
    photo: str
    image_path: Path
    expected: list[int]
    predicted: int
    score: float
    status_list: list[str]
    result: dict[str, Any]
    export_folder: Path | None = None


@dataclass
class EvalResult:
    total: int = 0
    full_correct: int = 0
    partial_correct: int = 0
    missing_labels: list[str] = field(default_factory=list)
    confusion: dict[int, dict[int, int]] = field(default_factory=dict)
    errors: list[ErrorCase] = field(default_factory=list)
    results_csv_rows: list[tuple[str, str, int, float]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return ((self.full_correct + 0.5 * self.partial_correct) / self.total) * 100.0


def evaluate_dataset(
    analyzer: Classical2,
    img_dir: Path,
    label_dir: Path,
    *,
    glob_pattern: str = "WIN_*.jpg",
    export_errors_dir: Path | None = None,
    export_partial: bool = False,
    clear_export_dir: bool = False,
    results_csv_path: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> EvalResult:
    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Images or labels directory not found")

    if export_errors_dir and clear_export_dir and export_errors_dir.exists():
        shutil.rmtree(export_errors_dir)

    out = EvalResult()
    classes = [0, 1, 2, 3, 4]
    out.confusion = {exp: {pred: 0 for pred in classes} for exp in classes}

    files = sorted(img_dir.glob(glob_pattern))
    for image_path in files:
        label_path = label_dir / (image_path.stem + ".txt")
        expected = load_labels(label_path)
        if expected is None:
            out.missing_labels.append(image_path.name)
            continue

        result = analyzer.analyze(str(image_path))
        status_code = result.get("status_code")
        if status_code is None:
            raise ValueError(f"No status_code returned for {image_path}")

        out.total += 1
        score = score_output(status_code, expected)

        for exp_label in set(expected):
            out.confusion.setdefault(exp_label, {pred: 0 for pred in classes})
            out.confusion[exp_label][status_code] = out.confusion[exp_label].get(status_code, 0) + 1

        if score == 1.0:
            out.full_correct += 1
        elif score == 0.5:
            out.partial_correct += 1

        if score != 1.0:
            out.results_csv_rows.append(
                (
                    image_path.name,
                    ";".join(str(x) for x in expected),
                    status_code,
                    score,
                )
            )
            should_export = export_errors_dir is not None and (
                score < 0.5 or export_partial
            )
            export_folder = None
            if should_export:
                export_folder = export_errors_dir / error_folder_name(
                    image_path.stem, expected, status_code
                )
                save_analysis_visualizations(result, export_folder, original_path=image_path)
                write_analysis_report(
                    result,
                    export_folder / "report.txt",
                    expected=expected,
                    score=score,
                )

            out.errors.append(
                ErrorCase(
                    photo=image_path.name,
                    image_path=image_path,
                    expected=expected,
                    predicted=status_code,
                    score=score,
                    status_list=list(result.get("status_list", [])),
                    result=result,
                    export_folder=export_folder,
                )
            )

        if log and out.total % 50 == 0:
            log(f"Processed {out.total} images...")

    if results_csv_path is not None:
        with open(results_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["photo", "correct_classification", "classical_2", "score"])
            writer.writerows(out.results_csv_rows)

    return out
