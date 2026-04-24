"""Load a saved model (by run name) and run inference on a single image or folder.

Examples:

    # single image, automatically picks the model file by run name
    python -m evaluation.infer --model transfer_resnet18_aug --image path/to/cap.jpg

    # run every image in a folder, print top-1 predictions
    python -m evaluation.infer --model hog_svm_raw --folder path/to/folder

    # list every trained model available on disk
    python -m evaluation.infer --list

The utility uses the ``<run_name>.meta.json`` sidecar to know which class to
instantiate (``base_method`` + ``kind``). If you trained a custom method, add
its registry entry at the bottom of this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    BBOX_PADDING,  # noqa: F401  (re-exported for completeness)
    CLASS_NAMES,
    IMAGE_SIZE,
    MODELS_DIR,
)

ModelFactory = Callable[[], object]

# Map base_method -> factory producing a fresh instance whose .load() will
# then pull in the saved state. New methods can be registered here.
_REGISTRY: dict[str, ModelFactory] = {}


def _register_defaults() -> None:
    from classical.edge_contour import EdgeContourClassifier
    from classical.hog_svm import HogSvmClassifier
    from classical.threshold_morphology import ThresholdMorphologyClassifier
    from ml.feature_ml import CNNFeatureClassifier
    from ml.transfer_learning import TransferLearningModel

    _REGISTRY.update(
        {
            HogSvmClassifier.name: HogSvmClassifier,
            EdgeContourClassifier.name: EdgeContourClassifier,
            ThresholdMorphologyClassifier.name: ThresholdMorphologyClassifier,
            "cnnfeat_xgboost": lambda: CNNFeatureClassifier(classifier="xgboost"),
            "cnnfeat_random_forest": lambda: CNNFeatureClassifier(
                classifier="random_forest"
            ),
            "transfer_resnet18": lambda: TransferLearningModel(backbone="resnet18"),
            "transfer_mobilenet_v2": lambda: TransferLearningModel(
                backbone="mobilenet_v2"
            ),
        }
    )


def _meta_path(run_name: str) -> Path:
    return MODELS_DIR / f"{run_name}.meta.json"


def _load_meta(run_name: str) -> dict:
    path = _meta_path(run_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No metadata file at {path}. Has this model been trained?"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_trained_method(run_name: str):
    """Load a method by its run name (e.g. ``transfer_resnet18_aug``)."""
    if not _REGISTRY:
        _register_defaults()

    meta = _load_meta(run_name)
    base_method = meta["base_method"]
    factory = _REGISTRY.get(base_method)
    if factory is None:
        raise KeyError(
            f"No registered factory for base_method={base_method!r}. "
            "Register it at the bottom of evaluation/infer.py."
        )
    instance = factory()
    model_file = MODELS_DIR / meta["model_file"]
    if not model_file.exists():
        raise FileNotFoundError(f"Model file missing: {model_file}")
    instance.load(str(model_file))
    return instance, meta


def _load_image_as_crop(image_path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)


def predict_image(
    method, image_path: Path, image_size: int = IMAGE_SIZE
) -> tuple[int, str, np.ndarray]:
    """Predict one image. Assumes the image is already a (roughly) square crop of a cap.

    Returns ``(class_id, class_name, proba_vector)``.
    """
    crop = _load_image_as_crop(image_path, image_size=image_size)
    batch = crop[None, ...]
    proba = method.predict_proba(batch)[0]
    class_id = int(np.argmax(proba))
    class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
    return class_id, class_name, proba


def list_available_models() -> list[str]:
    return sorted(p.stem.removesuffix(".meta") for p in MODELS_DIR.glob("*.meta.json"))


def _format_proba(proba: np.ndarray) -> str:
    return " | ".join(f"{n}:{p:.2f}" for n, p in zip(CLASS_NAMES, proba))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="List trained models and exit.")
    parser.add_argument("--model", type=str, help="Run name, e.g. transfer_resnet18_aug")
    parser.add_argument("--image", type=str, help="Path to one image.")
    parser.add_argument("--folder", type=str, help="Path to a folder of images.")
    args = parser.parse_args()

    if args.list or (not args.model and not args.image and not args.folder):
        models = list_available_models()
        if not models:
            print(
                f"No trained models found in {MODELS_DIR}. "
                "Run `python run_all.py` first."
            )
            return
        print("Available trained models:")
        for name in models:
            print(f"  {name}")
        return

    if not args.model:
        parser.error("--model is required when doing inference.")

    method, meta = load_trained_method(args.model)
    print(f"Loaded {args.model} (trained on {meta.get('trained_on', '?')})")
    print(
        f"Final metrics from training: "
        f"acc={meta['final_metrics']['accuracy']:.3f}, "
        f"f1={meta['final_metrics']['f1_macro']:.3f}"
    )

    targets: list[Path] = []
    if args.image:
        targets.append(Path(args.image))
    if args.folder:
        folder = Path(args.folder)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            targets.extend(sorted(folder.glob(ext)))

    if not targets:
        print("No images specified. Use --image or --folder.")
        return

    print()
    for path in targets:
        cls_id, cls_name, proba = predict_image(method, path)
        print(f"{path.name:50s}  -> {cls_name:12s}  [{_format_proba(proba)}]")


if __name__ == "__main__":
    main()
