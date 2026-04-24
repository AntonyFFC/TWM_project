"""Common interface every ML model must implement.

Kept identical in shape to ``classical.base_classifier.BaseClassifier`` so that
``evaluation/evaluator.py`` treats both kinds uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Abstract base class for every ML model in this project."""

    name: str = "unnamed_model"

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
        """Return per-class probability scores, shape (N, NUM_CLASSES)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore a previously saved model from disk."""
