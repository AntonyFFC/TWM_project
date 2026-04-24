"""Metric computation: accuracy, F1, per-class metrics, inference timing."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvalReport:
    method: str  # run name, e.g. "hog_svm_raw" / "hog_svm_aug"
    base_method: str  # method name without trained_on suffix
    kind: str  # "classical" or "ml"
    trained_on: str  # "raw" or "aug"
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    inference_ms_per_image: float
    confusion_matrix: list[list[int]] = field(default_factory=list)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    train_time_s: float = 0.0
    n_train: int = 0
    n_test: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_row(self) -> dict[str, Any]:
        """A flat dict suitable for a CSV summary table."""
        return {
            "method": self.method,
            "base_method": self.base_method,
            "kind": self.kind,
            "trained_on": self.trained_on,
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "inference_ms": round(self.inference_ms_per_image, 3),
            "train_time_s": round(self.train_time_s, 2),
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def measure_inference_time(
    predict_fn,
    X: np.ndarray,
    warmup_batches: int = 1,
    batch_size: int = 32,
) -> float:
    """Return mean inference time per image in milliseconds.

    Runs a short warmup to exclude one-off overheads (lazy imports, CUDA warmup).
    """
    if len(X) == 0:
        return 0.0

    for _ in range(warmup_batches):
        _ = predict_fn(X[: min(batch_size, len(X))])

    t0 = time.perf_counter()
    _ = predict_fn(X)
    elapsed = time.perf_counter() - t0
    return (elapsed / len(X)) * 1000.0


def compute_report(
    method_name: str,
    base_method: str,
    kind: str,
    trained_on: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    inference_ms_per_image: float,
    train_time_s: float,
    n_train: int = 0,
) -> EvalReport:
    """Build a fully-populated :class:`EvalReport`."""
    labels = list(range(len(class_names)))
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    prec = float(
        precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    rec = float(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    per_class_raw = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class: dict[str, dict[str, float]] = {}
    for cname in class_names:
        entry = per_class_raw.get(cname, {})
        per_class[cname] = {
            "precision": float(entry.get("precision", 0.0)),
            "recall": float(entry.get("recall", 0.0)),
            "f1": float(entry.get("f1-score", 0.0)),
            "support": float(entry.get("support", 0.0)),
        }

    return EvalReport(
        method=method_name,
        base_method=base_method,
        kind=kind,
        trained_on=trained_on,
        accuracy=acc,
        f1_macro=f1,
        precision_macro=prec,
        recall_macro=rec,
        inference_ms_per_image=inference_ms_per_image,
        confusion_matrix=cm,
        per_class=per_class,
        train_time_s=train_time_s,
        n_train=n_train,
        n_test=int(len(y_true)),
    )
