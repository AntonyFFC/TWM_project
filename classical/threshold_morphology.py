"""Thresholding + morphology-based classifier.

STATUS: STUB for Kolega A.

Idea:
    Binarize the image (Otsu or adaptive), clean with morphology
    (opening/closing), then extract shape descriptors from the resulting
    blob (area, solidity, Hu moments, orientation, eccentricity, ...).
    Feed them into a classical classifier (RandomForest / SVM).

The feature extractor here is intentionally minimal so that the pipeline
runs end-to-end. Kolega A should enrich ``_extract_features``.
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
    """Feature vector built from Otsu binarization + morphology.

    TODO(Kolega A): add Hu moments, solidity, eccentricity, angle of
    min-area rect (see main.py for an example), etc.
    """
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    total_pixels = clean.size
    white_ratio = float(clean.sum()) / (255.0 * total_pixels)

    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.array([white_ratio, 0, 0, 0, 0], dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perim = float(cv2.arcLength(largest, closed=True))
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    solidity = area / hull_area

    if len(largest) >= 5:
        (_cx, _cy), (ma, MA), _angle = cv2.fitEllipse(largest)
        eccentricity = float(np.sqrt(1 - (ma / (MA + 1e-6)) ** 2))
    else:
        eccentricity = 0.0

    return np.array([white_ratio, area, perim, solidity, eccentricity], dtype=np.float32)


def _batch_features(X_bgr: np.ndarray) -> np.ndarray:
    gray = to_gray(X_bgr)
    return np.stack([_extract_features(img) for img in gray], axis=0)


class ThresholdMorphologyClassifier(BaseClassifier):
    """STUB classifier: Otsu + morphology + shape features + RandomForest."""

    name = "threshold_morphology_rf"

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
        assert self.clf is not None
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
