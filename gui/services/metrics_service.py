"""Metrics and comparison helpers for the GUI."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import METRICS_DIR, PLOTS_DIR  # noqa: E402

COMPARISON_PLOTS = [
    "comparison_accuracy.png",
    "comparison_speed.png",
    "augmentation_gain.png",
    "robustness.png",
]

SUMMARY_CSV = METRICS_DIR / "summary.csv"


def list_runs(*, kind: str | None = None) -> list[str]:
    runs: list[str] = []
    for fp in sorted(METRICS_DIR.glob("*.json")):
        if fp.stem.endswith("_robustness") or fp.name == "summary.json":
            continue
        runs.append(fp.stem)
    if kind is None:
        return runs

    filtered: list[str] = []
    for run_name in runs:
        report = load_report(run_name)
        if report and report.get("kind") == kind:
            filtered.append(run_name)
    return filtered


def load_report(run_name: str) -> dict[str, Any] | None:
    path = METRICS_DIR / f"{run_name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def confusion_plot_path(run_name: str) -> Path:
    return PLOTS_DIR / f"confusion_{run_name}.png"


def comparison_plot_path(filename: str) -> Path:
    return PLOTS_DIR / filename


def run_comparison() -> Path:
    from evaluation import compare

    compare.main()
    return SUMMARY_CSV


def load_summary_table() -> list[dict[str, Any]]:
    if not SUMMARY_CSV.exists():
        return []
    import pandas as pd

    df = pd.read_csv(SUMMARY_CSV)
    return df.to_dict(orient="records")


def format_report_summary(report: dict[str, Any]) -> str:
    return (
        f"Method: {report.get('method')}\n"
        f"Kind: {report.get('kind')}\n"
        f"Trained on: {report.get('trained_on')}\n"
        f"Accuracy: {report.get('accuracy', 0):.4f}\n"
        f"F1 macro: {report.get('f1_macro', 0):.4f}\n"
        f"Precision macro: {report.get('precision_macro', 0):.4f}\n"
        f"Recall macro: {report.get('recall_macro', 0):.4f}\n"
        f"Train time: {report.get('train_time_s', 0):.1f} s\n"
        f"Inference: {report.get('inference_ms_per_image', 0):.2f} ms/img\n"
        f"Train samples: {report.get('n_train', 0)}\n"
        f"Test samples: {report.get('n_test', 0)}\n"
    )
