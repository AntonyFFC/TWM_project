"""HOG (Histogram of Oriented Gradients) + Linear SVM classifier.

This is the reference implementation of ``BaseClassifier`` --
use it as a template when implementing ``edge_contour.py`` and
``threshold_morphology.py``.

Pipeline:
    BGR image -> grayscale -> HOG descriptor -> StandardScaler -> LinearSVC

Why HOG+SVM:
    Classic baseline from the pre-deep-learning era; captures edge/gradient
    structure well, which is relevant for cap rim/thread features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

sys.path.append(str(Path(__file__).resolve().parents[1]))
from classical.base_classifier import BaseClassifier  # noqa: E402
from config import IMAGE_SIZE, NUM_CLASSES  # noqa: E402
from data.preprocessing import to_gray  # noqa: E402


def _hog_features(img_gray: np.ndarray) -> np.ndarray:
    return hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def _batch_hog(X_bgr: np.ndarray) -> np.ndarray:
    gray = to_gray(X_bgr)
    feats = [_hog_features(img) for img in gray]
    return np.stack(feats, axis=0)


class HogSvmClassifier(BaseClassifier):
    """HOG features followed by a calibrated linear SVM."""

    name = "hog_svm"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.pipeline: Pipeline | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        feats = _batch_hog(X_train)
        base_svm = LinearSVC(C=self.C, max_iter=50000, dual="auto", tol=1e-3)
        # CalibratedClassifierCV gives us .predict_proba on LinearSVC.
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svm", CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")),
            ]
        )
        self.pipeline.fit(feats, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.pipeline is not None, "Call fit() before predict()."
        feats = _batch_hog(X)
        return self.pipeline.predict(feats).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.pipeline is not None, "Call fit() before predict_proba()."
        feats = _batch_hog(X)
        proba = self.pipeline.predict_proba(feats)
        if proba.shape[1] < NUM_CLASSES:
            padded = np.zeros((proba.shape[0], NUM_CLASSES), dtype=proba.dtype)
            classes = self.pipeline.classes_
            for i, c in enumerate(classes):
                padded[:, int(c)] = proba[:, i]
            return padded
        return proba

    def save(self, path: str) -> None:
        assert self.pipeline is not None
        joblib.dump({"pipeline": self.pipeline, "C": self.C}, path)

    def load(self, path: str) -> None:
        blob = joblib.load(path)
        self.pipeline = blob["pipeline"]
        self.C = blob.get("C", 1.0)


__all__ = ["HogSvmClassifier"]

if __name__ == "__main__":
    # Quick smoke test on random data.
    rng = np.random.default_rng(0)
    Xd = rng.integers(0, 256, (40, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    yd = rng.integers(0, NUM_CLASSES, 40)
    clf = HogSvmClassifier()
    clf.fit(Xd, yd)
    print("predict shape:", clf.predict(Xd).shape)
    print("proba shape:", clf.predict_proba(Xd).shape)
