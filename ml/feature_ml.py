"""CNN-as-feature-extractor + classical tabular ML classifier.

Pipeline:
    frozen ResNet18 (ImageNet) -> pooled feature vector (512-dim)
        -> XGBoost / RandomForest classifier

This is the "hybrid" approach: we take advantage of pretrained CNN features
without fine-tuning, and let a classical tabular model do the classification.
Much faster than fine-tuning and often surprisingly strong on small datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    BATCH_SIZE,
    CNN_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    SEED,
)
from ml.base_model import BaseModel  # noqa: E402


class _FeatureExtractor:
    """Frozen ResNet18 backbone up to the global average pool."""

    def __init__(self, input_size: int = CNN_INPUT_SIZE, device: str | None = None) -> None:
        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.model = backbone.to(self.device)
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def _prepare_batch(self, X_bgr: np.ndarray) -> torch.Tensor:
        resized = np.stack(
            [
                cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
                if (img.shape[0] != self.input_size or img.shape[1] != self.input_size)
                else img
                for img in X_bgr
            ],
            axis=0,
        )
        rgb = resized[..., ::-1].astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
        return tensor.to(self.device)

    @torch.no_grad()
    def extract(self, X_bgr: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
        self.model.eval()
        feats: list[np.ndarray] = []
        for i in range(0, len(X_bgr), batch_size):
            batch = self._prepare_batch(X_bgr[i : i + batch_size])
            out = self.model(batch).cpu().numpy()
            feats.append(out)
        return np.concatenate(feats, axis=0)


def _build_classifier(kind: str):
    kind = kind.lower()
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        )
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=400,
            n_jobs=-1,
            random_state=SEED,
        )
    raise ValueError(f"Unknown classifier kind: {kind}")


class CNNFeatureClassifier(BaseModel):
    """ImageNet features + tabular ML (XGBoost or RandomForest)."""

    def __init__(self, classifier: str = "xgboost") -> None:
        self.classifier_kind = classifier
        self.extractor: _FeatureExtractor | None = None
        self.clf = None

    @property
    def name(self) -> str:
        return f"cnnfeat_{self.classifier_kind}"

    def _ensure_extractor(self) -> None:
        if self.extractor is None:
            self.extractor = _FeatureExtractor()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        self._ensure_extractor()
        feats_train = self.extractor.extract(X_train)
        self.clf = _build_classifier(self.classifier_kind)
        self.clf.fit(feats_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.clf is not None, "Call fit() before predict()."
        self._ensure_extractor()
        feats = self.extractor.extract(X)
        return self.clf.predict(feats).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.clf is not None
        self._ensure_extractor()
        feats = self.extractor.extract(X)
        proba = self.clf.predict_proba(feats)
        if proba.shape[1] < NUM_CLASSES:
            padded = np.zeros((proba.shape[0], NUM_CLASSES), dtype=proba.dtype)
            classes = getattr(self.clf, "classes_", range(proba.shape[1]))
            for i, c in enumerate(classes):
                padded[:, int(c)] = proba[:, i]
            return padded
        return proba

    def save(self, path: str) -> None:
        assert self.clf is not None
        joblib.dump({"kind": self.classifier_kind, "clf": self.clf}, path)

    def load(self, path: str) -> None:
        blob = joblib.load(path)
        self.classifier_kind = blob["kind"]
        self.clf = blob["clf"]
        self._ensure_extractor()
