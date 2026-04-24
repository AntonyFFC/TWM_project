"""Exploratory Data Analysis: class distribution, example crops, bbox stats.

Generates plots in ``results/plots/`` so the team can inspect dataset quality
before training anything.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CLASS_NAMES, PLOTS_DIR, ensure_dirs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402 - config sets MPLBACKEND=Agg
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.dataset_loader import iter_samples, parse_label_file  # noqa: E402
from config import RAW_LABELS_DIR  # noqa: E402


def plot_class_distribution() -> dict[str, int]:
    """Plot (and return) number of bboxes per class across the full dataset."""
    dist: Counter[str] = Counter()
    for label_path in RAW_LABELS_DIR.glob("*.txt"):
        for cls, *_ in parse_label_file(label_path):
            dist[CLASS_NAMES[cls]] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [c for c in CLASS_NAMES if c in dist]
    values = [dist[c] for c in labels]
    bars = ax.bar(labels, values)
    ax.set_title("Class distribution (bboxes per class)")
    ax.set_ylabel("count")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = PLOTS_DIR / "eda_class_distribution.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")
    return dict(dist)


def plot_bbox_size_stats() -> None:
    """Plot histogram of bbox widths and heights (normalized 0-1)."""
    widths = []
    heights = []
    for label_path in RAW_LABELS_DIR.glob("*.txt"):
        for _cls, _cx, _cy, w, h in parse_label_file(label_path):
            widths.append(w)
            heights.append(h)
    if not widths:
        print("  no bboxes found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(widths, bins=30)
    axes[0].set_title("bbox width (normalized)")
    axes[1].hist(heights, bins=30)
    axes[1].set_title("bbox height (normalized)")
    plt.tight_layout()
    out = PLOTS_DIR / "eda_bbox_sizes.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_sample_grid(n_per_class: int = 3) -> None:
    """Plot a grid of example crops: one row per class."""
    by_class: dict[str, list[np.ndarray]] = {c: [] for c in CLASS_NAMES}
    for sample in tqdm(iter_samples(), desc="Collecting examples"):
        bucket = by_class[sample.class_name]
        if len(bucket) < n_per_class:
            bucket.append(sample.crop)
        if all(len(v) >= n_per_class for v in by_class.values()):
            break

    classes_with_data = [c for c, v in by_class.items() if v]
    if not classes_with_data:
        print("  no images yet -- did you download the dataset?")
        return

    n_rows = len(classes_with_data)
    n_cols = n_per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row, class_name in enumerate(classes_with_data):
        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(by_class[class_name]):
                bgr = by_class[class_name][col]
                ax.imshow(bgr[..., ::-1])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(class_name, fontsize=9)

    plt.suptitle("Example crops per class")
    plt.tight_layout()
    out = PLOTS_DIR / "eda_examples.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ensure_dirs()
    print("EDA: class distribution...")
    plot_class_distribution()
    print("EDA: bbox size statistics...")
    plot_bbox_size_stats()
    print("EDA: example crops per class...")
    plot_sample_grid()
    print("Done.")


if __name__ == "__main__":
    main()
