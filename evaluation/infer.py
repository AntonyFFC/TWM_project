"""Inferencja jednym z wytrenowanych modeli (po nazwie run-a).

Przyklady:

    # lista wytrenowanych modeli
    python -m evaluation.infer --list

    # predykcja na jednym obrazie
    python -m evaluation.infer --model resnet18_aug --image path/to/cap.jpg

    # predykcja dla calego folderu
    python -m evaluation.infer --model hog_svm_raw --folder path/to/folder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CLASS_NAMES, IMAGE_SIZE, MODELS_DIR  # noqa: E402

# base_method -> klasa, ktorej instancja .load(path) wczyta wagi.
_REGISTRY = {}


def _register_defaults() -> None:
    from classical.hog_svm import HogSvmClassifier
    from ml.transfer_learning import TransferLearningModel

    _REGISTRY.update(
        {
            HogSvmClassifier.name: HogSvmClassifier,
            TransferLearningModel.name: TransferLearningModel,
        }
    )


def _load_meta(run_name: str) -> dict:
    path = MODELS_DIR / f"{run_name}.meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Brak pliku metadanych {path}. Czy ten model byl trenowany?"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_trained_method(run_name: str):
    """Wczytaj wytrenowany model po jego ``run_name`` (np. ``resnet18_aug``)."""
    if not _REGISTRY:
        _register_defaults()

    meta = _load_meta(run_name)
    base_method = meta["base_method"]
    factory = _REGISTRY.get(base_method)
    if factory is None:
        raise KeyError(
            f"Brak zarejestrowanej fabryki dla base_method={base_method!r}."
        )
    instance = factory()
    model_file = MODELS_DIR / meta["model_file"]
    if not model_file.exists():
        raise FileNotFoundError(f"Brak pliku modelu: {model_file}")
    instance.load(str(model_file))
    return instance, meta


def _load_image_as_crop(image_path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Nie mozna wczytac obrazu: {image_path}")
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)


def predict_image(
    method, image_path: Path, image_size: int = IMAGE_SIZE
) -> tuple[int, str, np.ndarray]:
    """Predykcja na jednym obrazie. Zwraca ``(class_id, class_name, proba)``."""
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
    parser.add_argument("--list", action="store_true", help="Wypisz dostepne modele.")
    parser.add_argument("--model", type=str, help="Nazwa run-a, np. resnet18_aug")
    parser.add_argument("--image", type=str, help="Sciezka do jednego obrazu.")
    parser.add_argument("--folder", type=str, help="Sciezka do folderu z obrazami.")
    args = parser.parse_args()

    if args.list or (not args.model and not args.image and not args.folder):
        models = list_available_models()
        if not models:
            print(
                f"Brak wytrenowanych modeli w {MODELS_DIR}. "
                "Odpal najpierw `python run_all.py`."
            )
            return
        print("Dostepne wytrenowane modele:")
        for name in models:
            print(f"  {name}")
        return

    if not args.model:
        parser.error("--model jest wymagane do inferencji.")

    method, meta = load_trained_method(args.model)
    print(f"Wczytano {args.model} (trenowany na: {meta.get('trained_on', '?')})")
    print(
        f"Metryki z treningu: "
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
        print("Nie podano obrazow. Uzyj --image lub --folder.")
        return

    print()
    for path in targets:
        cls_id, cls_name, proba = predict_image(method, path)
        print(f"{path.name:50s}  -> {cls_name:12s}  [{_format_proba(proba)}]")


if __name__ == "__main__":
    main()
