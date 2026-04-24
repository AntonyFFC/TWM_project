"""Robustness benchmark: evaluate accuracy under controlled image corruptions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import NUM_CLASSES  # noqa: E402
from data.augmentation import apply_corruption, robustness_corruptions  # noqa: E402


def evaluate_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Run the same test set through each corruption and return metrics.

    Returns ``{corruption_name: {"accuracy": ..., "f1_macro": ...}}``.
    A ``"clean"`` key holds the uncorrupted reference.
    """
    results: dict[str, dict[str, float]] = {}
    labels = list(range(NUM_CLASSES))

    y_pred_clean = model.predict(X_test)
    results["clean"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_clean)),
        "f1_macro": float(
            f1_score(y_test, y_pred_clean, labels=labels, average="macro", zero_division=0)
        ),
    }

    for name, fn in robustness_corruptions().items():
        Xc = apply_corruption(X_test, fn)
        y_pred = model.predict(Xc)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(
                f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
            ),
        }
    return results
