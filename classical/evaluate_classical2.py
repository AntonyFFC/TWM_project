"""Batch evaluation for Classical2 against labeled dataset."""
from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np

from classical.classical2_labels import CLASSICAL2_SLUGS, score_output
from classical.classical_2 import (
    Classical2,
    save_analysis_visualizations,
    write_analysis_report,
)
from data.augmentation import (
    augment_bgr_image,
    train_augmentations_full_image,
)

EVAL_ON_CHOICES = ("raw", "aug", "both")
EvalOnChoice = Literal["raw", "aug", "both"]
DEFAULT_AUG_COPIES = 2


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
    return f"exp{exp}-{exp_label}__pred{predicted}_{pred_label}__{stem}"


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
    variant: str = "raw"
    sample_label: str = ""
    aug_copy_index: int | None = None


@dataclass
class EvalResult:
    total: int = 0
    full_correct: int = 0
    partial_correct: int = 0
    missing_labels: list[str] = field(default_factory=list)
    confusion: dict[int, dict[int, int]] = field(default_factory=dict)
    errors: list[ErrorCase] = field(default_factory=list)
    results_csv_rows: list[tuple[str, str, int, float]] = field(default_factory=list)
    variant: str = "raw"
    confusion_plot_path: Path | None = None

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return ((self.full_correct + 0.5 * self.partial_correct) / self.total) * 100.0


def confusion_dict_to_matrix(
    confusion: dict[int, dict[int, int]], n_classes: int = 5
) -> list[list[int]]:
    return [
        [confusion.get(exp, {}).get(pred, 0) for pred in range(n_classes)]
        for exp in range(n_classes)
    ]


def classical2_confusion_plot_path(preset_name: str, variant: str) -> Path:
    from config import PLOTS_DIR

    safe = re.sub(r"[^\w\-]", "_", preset_name)
    return PLOTS_DIR / f"confusion_classical2_{safe}_{variant}.png"


def plot_classical2_confusion_matrix(
    confusion: dict[int, dict[int, int]],
    *,
    title: str,
    out_path: Path,
) -> Path:
    """Save a seaborn heatmap confusion matrix (same style as ML/classical runs)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    from config import CLASS_NAMES

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.asarray(confusion_dict_to_matrix(confusion))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def format_eval_summary(result: EvalResult, *, preset_name: str) -> str:
    lines = [
        f"Method: classical2",
        f"Preset: {preset_name}",
        f"Variant: {result.variant}",
        f"Samples: {result.total}",
        f"Accuracy: {result.accuracy / 100.0:.4f}",
        f"Full correct: {result.full_correct}",
        f"Partial correct: {result.partial_correct}",
        f"Errors / partial: {len(result.errors)}",
    ]
    if result.missing_labels:
        lines.append(f"Skipped (no label): {len(result.missing_labels)}")
    return "\n".join(lines)


@dataclass
class CompareEvalResult:
    """Evaluation on raw and/or augmented images (for side-by-side comparison)."""

    raw: EvalResult | None = None
    augmented: EvalResult | None = None
    eval_on: EvalOnChoice = "raw"
    aug_copies: int = DEFAULT_AUG_COPIES

    @property
    def primary(self) -> EvalResult | None:
        if self.eval_on == "aug":
            return self.augmented
        return self.raw or self.augmented


def stable_eval_seed(base_seed: int, stem: str, copy_index: int) -> int:
    return int(base_seed) + (hash(stem) % 100_000) + copy_index * 997


def _evaluate_dataset_variant(
    analyzer: Classical2,
    img_dir: Path,
    label_dir: Path,
    *,
    variant: EvalOnChoice,
    glob_pattern: str = "WIN_*.jpg",
    export_errors_dir: Path | None = None,
    export_partial: bool = False,
    clear_export_dir: bool = False,
    results_csv_path: Path | None = None,
    aug_copies: int = DEFAULT_AUG_COPIES,
    aug_config: dict | None = None,
    seed: int = 42,
    preset_name: str = "default",
    log: Callable[[str], None] | None = None,
) -> EvalResult:
    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Images or labels directory not found")

    use_augmentation = variant == "aug"
    if export_errors_dir and clear_export_dir and export_errors_dir.exists():
        shutil.rmtree(export_errors_dir)

    out = EvalResult(variant=variant)
    classes = [0, 1, 2, 3, 4]
    out.confusion = {exp: {pred: 0 for pred in classes} for exp in classes}

    aug_compose = train_augmentations_full_image(aug_config) if use_augmentation else None
    files = sorted(img_dir.glob(glob_pattern))

    for image_path in files:
        label_path = label_dir / (image_path.stem + ".txt")
        expected = load_labels(label_path)
        if expected is None:
            out.missing_labels.append(image_path.name)
            continue

        if use_augmentation:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                out.missing_labels.append(image_path.name)
                continue
            samples: list[tuple[str, np.ndarray | None, str | None, int | None]] = []
            for copy_idx in range(aug_copies):
                sample_label = f"{image_path.name} [aug#{copy_idx + 1}]"
                aug_img = augment_bgr_image(
                    img_bgr,
                    aug_compose,
                    seed=stable_eval_seed(seed, image_path.stem, copy_idx),
                )
                samples.append(
                    (sample_label, aug_img, f"{image_path.stem}__aug{copy_idx + 1}", copy_idx)
                )
        else:
            samples = [(image_path.name, None, None, None)]

        for sample_label, sample_img, export_stem, copy_idx in samples:
            if sample_img is None:
                result = analyzer.analyze(str(image_path))
            else:
                result = analyzer.analyze(sample_img)

            status_code = result.get("status_code")
            if status_code is None:
                raise ValueError(f"No status_code returned for {sample_label}")

            out.total += 1
            score = score_output(status_code, expected)

            for exp_label in set(expected):
                out.confusion.setdefault(exp_label, {pred: 0 for pred in classes})
                out.confusion[exp_label][status_code] = (
                    out.confusion[exp_label].get(status_code, 0) + 1
                )

            if score == 1.0:
                out.full_correct += 1
            elif score == 0.5:
                out.partial_correct += 1

            if score != 1.0:
                out.results_csv_rows.append(
                    (
                        sample_label,
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
                    folder_stem = export_stem or image_path.stem
                    export_folder = export_errors_dir / error_folder_name(
                        folder_stem, expected, status_code
                    )
                    save_analysis_visualizations(
                        result,
                        export_folder,
                        original_path=image_path if sample_img is None else None,
                    )
                    if sample_img is not None:
                        cv2.imwrite(str(export_folder / "augmented_input.jpg"), sample_img)
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
                        variant=variant,
                        sample_label=sample_label,
                        aug_copy_index=copy_idx if use_augmentation else None,
                    )
                )

        if log and out.total % 50 == 0:
            log(f"[{variant}] Processed {out.total} samples...")

    if results_csv_path is not None:
        suffix = "" if variant == "raw" else f"_{variant}"
        csv_path = results_csv_path.with_name(
            f"{results_csv_path.stem}{suffix}{results_csv_path.suffix}"
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["photo", "correct_classification", "classical_2", "score"])
            writer.writerows(out.results_csv_rows)

    if out.total > 0:
        plot_path = classical2_confusion_plot_path(preset_name, variant)
        plot_classical2_confusion_matrix(
            out.confusion,
            title=f"Confusion matrix: classical2 ({preset_name}, {variant})",
            out_path=plot_path,
        )
        out.confusion_plot_path = plot_path

    return out


def evaluate_dataset(
    analyzer: Classical2,
    img_dir: Path,
    label_dir: Path,
    *,
    eval_on: EvalOnChoice = "raw",
    glob_pattern: str = "WIN_*.jpg",
    export_errors_dir: Path | None = None,
    export_partial: bool = False,
    clear_export_dir: bool = False,
    results_csv_path: Path | None = None,
    aug_copies: int = DEFAULT_AUG_COPIES,
    aug_config: dict | None = None,
    seed: int = 42,
    preset_name: str = "default",
    log: Callable[[str], None] | None = None,
) -> CompareEvalResult:
    """Evaluate Classical2 on raw images, augmented copies, or both for comparison."""
    if eval_on not in EVAL_ON_CHOICES:
        raise ValueError(f"eval_on must be one of {EVAL_ON_CHOICES}, got {eval_on!r}")

    compare = CompareEvalResult(eval_on=eval_on, aug_copies=aug_copies)
    variants: list[EvalOnChoice] = (
        ["raw", "aug"] if eval_on == "both" else [eval_on]  # type: ignore[list-item]
    )

    for idx, variant in enumerate(variants):
        errors_dir = export_errors_dir
        if export_errors_dir is not None and eval_on == "both":
            errors_dir = export_errors_dir / variant
        csv_path = results_csv_path
        if results_csv_path is not None and eval_on == "both":
            csv_path = results_csv_path.with_name(
                f"{results_csv_path.stem}_{variant}{results_csv_path.suffix}"
            )

        result = _evaluate_dataset_variant(
            analyzer,
            img_dir,
            label_dir,
            variant=variant,
            glob_pattern=glob_pattern,
            export_errors_dir=errors_dir,
            export_partial=export_partial,
            clear_export_dir=clear_export_dir and idx == 0,
            results_csv_path=csv_path,
            aug_copies=aug_copies,
            aug_config=aug_config,
            seed=seed,
            preset_name=preset_name,
            log=log,
        )
        if variant == "raw":
            compare.raw = result
        else:
            compare.augmented = result

    return compare
