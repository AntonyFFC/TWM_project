"""Edge + contour-based classifier.

STATUS: STUB for Kolega A.

Idea:
    Apply Canny edge detector, find contours, extract geometric/shape
    features from the largest contour (area, perimeter, circularity,
    aspect ratio, number of contour points, Hu moments, etc.), then
    train a classical classifier (e.g. RandomForest) on those features.

The current implementation uses a *very* simplistic feature set just so the
framework pipeline runs end-to-end. Replace ``_extract_features`` below with
your own, more expressive feature extractor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))
from classical.base_classifier import BaseClassifier  # noqa: E402
from config import NUM_CLASSES  # noqa: E402
from data.preprocessing import to_gray  # noqa: E402


def _extract_features(img_gray: np.ndarray) -> np.ndarray:
    """Return a 1-D feature vector for a single grayscale image.

    TODO(Kolega A): expand this with your own features!
    Suggested additions:
        - Hu moments: cv2.HuMoments(cv2.moments(...))
        - Bounding rect aspect ratio
        - Solidity = area / convex_hull_area
        - Eccentricity from ellipse fit
    """
    edges = cv2.Canny(img_gray, 80, 160)
    edge_density = float(edges.mean()) / 255.0

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.array([edge_density, 0, 0, 0, 0, 0], dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perim = float(cv2.arcLength(largest, closed=True))
    circularity = 4 * np.pi * area / (perim * perim + 1e-6)
    x, y, w, h = cv2.boundingRect(largest)
    aspect = float(w) / (h + 1e-6)
    n_points = float(len(largest))
    return np.array(
        [edge_density, area, perim, circularity, aspect, n_points],
        dtype=np.float32,
    )


def _batch_features(X_bgr: np.ndarray) -> np.ndarray:
    gray = to_gray(X_bgr)
    return np.stack([_extract_features(img) for img in gray], axis=0)


class EdgeContourClassifier(BaseClassifier):
    """STUB classifier: Canny edges + contour features + RandomForest.

    TODO(Kolega A): extend _extract_features and possibly swap RF for SVM
    once your feature set is richer.
    """

    name = "edge_contour_rf"

    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.clf: RandomForestClassifier | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        feats = _batch_features(X_train)
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.clf.fit(feats, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.clf is not None, "Call fit() before predict()."
        return self.clf.predict(_batch_features(X)).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.clf is not None
        proba = self.clf.predict_proba(_batch_features(X))
        if proba.shape[1] < NUM_CLASSES:
            padded = np.zeros((proba.shape[0], NUM_CLASSES), dtype=proba.dtype)
            for i, c in enumerate(self.clf.classes_):
                padded[:, int(c)] = proba[:, i]
            return padded
        return proba

    def save(self, path: str) -> None:
        assert self.clf is not None
        joblib.dump(self.clf, path)

    def load(self, path: str) -> None:
        self.clf = joblib.load(path)
