"""Globalna konfiguracja: sciezki, seed, klasy, hiperparametry treningu.

Edytuj ten plik, jesli chcesz zmienic lokalizacje datasetu, rozmiar obrazow,
batch size itd. Kazdy modul czyta stad sciezki, wiec sa spojne.
"""
from __future__ import annotations

import os
from pathlib import Path

# Wymuszamy backend Agg na matplotlib zanim cokolwiek innego sciagnie pyplot.
# Bez tego domyslny TkAgg na Windowsie potrafi crashnac z "Tcl_AsyncDelete..."
# gdy sklearn/torch spawnuja watki, ktore garbage-collectuja figury.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib  # noqa: E402

    matplotlib.use("Agg", force=True)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATASET_DIR = PROJECT_ROOT / "bottle-cap.yolov8"
RAW_IMAGES_DIR = RAW_DATASET_DIR / "train" / "images"
RAW_LABELS_DIR = RAW_DATASET_DIR / "train" / "labels"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CROPS_DIR = PROCESSED_DIR / "crops"
SPLITS_DIR = PROCESSED_DIR / "splits"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

CLASS_NAMES: list[str] = [
    "Broken Cap",
    "Broken Ring",
    "Good Cap",
    "Loose Cap",
    "No Cap",
]
NUM_CLASSES: int = len(CLASS_NAMES)

SEED: int = 42

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Wspolny rozmiar crop-a wejsciowego (wycinka GT bbox).
IMAGE_SIZE: int = 128

# Ile kontekstu (gwint, pierscien) zostawic wokol GT bbox.
BBOX_PADDING: float = 0.05

# Hiperparametry treningu CNN.
BATCH_SIZE: int = 32
NUM_WORKERS: int = 0
CNN_EPOCHS: int = 10
CNN_LR: float = 1e-3
CNN_INPUT_SIZE: int = 224

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def ensure_dirs() -> None:
    """Stworz wszystkie katalogi pochodne, jesli jeszcze nie istnieja."""
    for p in (
        PROCESSED_DIR,
        CROPS_DIR,
        SPLITS_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        MODELS_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)
