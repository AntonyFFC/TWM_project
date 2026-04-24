"""Global configuration: paths, seed, class names, training hyperparameters.

Edit this file to change dataset location, image size, batch size, etc.
Every module reads from here so paths are consistent across the project.
"""
from __future__ import annotations

import os
from pathlib import Path

# Force matplotlib to use the non-interactive Agg backend *before* anything
# else in the project imports pyplot. Without this, the default TkAgg backend
# on Windows crashes with "Tcl_AsyncDelete: async handler deleted by the wrong
# thread" when sklearn or torch spawn worker threads that garbage-collect
# figures. Setting MPLBACKEND is honoured by every subsequent matplotlib import.
os.environ.setdefault("MPLBACKEND", "Agg")
try:  # defensive: if matplotlib is already imported, switch the backend now
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

# Uniform crop size (square) fed to every classifier. Chosen as a
# compromise between HOG expressiveness and CNN input resolution.
IMAGE_SIZE: int = 128

# Expand GT bbox by this fraction on each side before cropping to keep a bit
# of context (thread, ring) around the cap.
BBOX_PADDING: float = 0.05

BATCH_SIZE: int = 32
NUM_WORKERS: int = 0
CNN_EPOCHS: int = 10
CNN_LR: float = 1e-3
CNN_INPUT_SIZE: int = 224

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def ensure_dirs() -> None:
    """Create all derived directories if they don't yet exist."""
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
