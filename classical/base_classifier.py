"""Common interface every classical (and ML) classifier must implement.

The evaluator only touches these 5 methods, so any class that conforms to
this interface plugs straight into ``evaluation/evaluator.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """Abstract base class for every classical classifier in this project."""

    #: Human-readable method name used in plots / CSVs / logged lines.
    name: str = "unnamed_classifier"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Train on BGR uint8 images (N, H, W, 3)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class ids, shape (N,) int64."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability-like scores, shape (N, NUM_CLASSES).

        If the underlying method does not produce probabilities, return a
        one-hot encoding of ``predict(X)``.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore a previously saved model from disk."""
