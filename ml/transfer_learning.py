"""ResNet18 fine-tuning na klasyfikacji nakretek.

Strategia:
    - bierzemy ResNet18 wytrenowany na ImageNet,
    - zamrazamy wszystkie warstwy oprocz ostatniego bloku (layer4) + glowy klasyfikatora,
    - trenujemy Adam + CrossEntropy przez kilka epok.

Wagi zapisujemy jako ``.pt`` (state_dict + meta).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    BATCH_SIZE,
    CNN_EPOCHS,
    CNN_INPUT_SIZE,
    CNN_LR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    NUM_WORKERS,
    SEED,
)
from ml.base_model import BaseModel  # noqa: E402


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_acc: list[float]


class _ArrayDataset(Dataset):
    """Owijamy BGR uint8 (N, H, W, 3) + etykiety w torchowy Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray, input_size: int) -> None:
        self.X = X
        self.y = y
        self.input_size = input_size
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        img = self.X[idx]
        if img.shape[0] != self.input_size or img.shape[1] != self.input_size:
            img = cv2.resize(
                img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA
            )
        rgb = img[..., ::-1].astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        label = int(self.y[idx]) if self.y is not None else 0
        return tensor, label


def _build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    # Odmrazamy ostatni blok rezydualny - to wystarczy na maly dataset.
    for p in model.layer4.parameters():
        p.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class TransferLearningModel(BaseModel):
    """Fine-tuning ResNet18 na cropach nakretek."""

    name = "resnet18"

    def __init__(
        self,
        epochs: int = CNN_EPOCHS,
        lr: float = CNN_LR,
        batch_size: int = BATCH_SIZE,
        input_size: int = CNN_INPUT_SIZE,
        device: str | None = None,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.history: TrainHistory | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.model = _build_resnet18(NUM_CLASSES).to(self.device)

        train_ds = _ArrayDataset(X_train, y_train, self.input_size)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = _ArrayDataset(X_val, y_val, self.input_size)
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, num_workers=NUM_WORKERS
            )

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        history = TrainHistory(train_loss=[], val_acc=[])
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_batches = 0
            for imgs, labels in train_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1
            mean_loss = running_loss / max(1, n_batches)
            history.train_loss.append(mean_loss)

            val_acc_str = ""
            if val_loader is not None:
                val_acc = self._accuracy(val_loader)
                history.val_acc.append(val_acc)
                val_acc_str = f"  val_acc={val_acc:.3f}"
            print(
                f"[{self.name}] epoch {epoch}/{self.epochs}"
                f"  loss={mean_loss:.4f}{val_acc_str}"
            )

        self.history = history

    @torch.no_grad()
    def _accuracy(self, loader: DataLoader) -> float:
        assert self.model is not None
        self.model.eval()
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(imgs).argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
        return correct / max(1, total)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Wywolaj fit() przed predict_proba()."
        self.model.eval()
        ds = _ArrayDataset(X, np.zeros(len(X), dtype=np.int64), self.input_size)
        loader = DataLoader(ds, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        probs = []
        for imgs, _ in loader:
            imgs = imgs.to(self.device)
            logits = self.model(imgs)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def save(self, path: str) -> None:
        assert self.model is not None
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_size": self.input_size,
            },
            path,
        )

    def load(self, path: str) -> None:
        blob = torch.load(path, map_location=self.device)
        self.input_size = blob.get("input_size", CNN_INPUT_SIZE)
        self.model = _build_resnet18(NUM_CLASSES)
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device)
