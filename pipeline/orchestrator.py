"""Callable end-to-end pipeline (EDA -> split -> train -> compare -> demo)."""
from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CROPS_DIR, ensure_dirs  # noqa: E402
from evaluation.evaluator import TRAINED_ON_CHOICES  # noqa: E402

AugmentationChoice = Literal["raw", "aug", "both"]


def has_splits() -> bool:
    return (
        CROPS_DIR.exists()
        and any((CROPS_DIR / split).exists() for split in ("train", "val", "test"))
        and any(CROPS_DIR.rglob("*.png"))
    )


def check_dataset(log: Callable[[str], None] = print) -> dict[str, int | bool]:
    """Return dataset status: labels, images, has_splits."""
    from data.download_dataset import count_label_image_pairs

    labels, images, _missing = count_label_image_pairs()
    status = {
        "labels": labels,
        "images": images,
        "has_splits": has_splits(),
        "ok": images > 0,
    }
    if images == 0:
        log("Brak obrazow w bottle-cap.yolov8/train/images/.")
        log("Odpal `python data/download_dataset.py` lub wrzuc je recznie.")
    elif images < labels:
        log(
            f"UWAGA: mniej obrazow ({images}) niz etykiet ({labels}). "
            "Czesc probek bedzie pominieta."
        )
    return status


def ensure_images_present(log: Callable[[str], None] = print) -> bool:
    return bool(check_dataset(log)["ok"])


def run_eda(log: Callable[[str], None] = print) -> None:
    log("\n### EDA ###")
    from data import eda

    eda.main()


def run_split(log: Callable[[str], None] = print) -> None:
    if has_splits():
        log("\n### Splity juz istnieja -- pomijam splitter.")
        return
    log("\n### SPLIT ###")
    from data import splitter

    splitter.run_split()


def run_pipeline(
    *,
    augmentation: AugmentationChoice = "both",
    skip_eda: bool = False,
    skip_split: bool = False,
    skip_classical: bool = False,
    skip_ml: bool = False,
    skip_compare: bool = False,
    skip_demo: bool = False,
    log: Callable[[str], None] = print,
) -> bool:
    """Run full pipeline. Returns False if dataset check fails."""
    ensure_dirs()

    if not ensure_images_present(log):
        return False

    if not skip_eda:
        run_eda(log)
    else:
        log("\n### EDA (pominiety) ###")

    if not skip_split:
        run_split(log)
    else:
        log("\n### SPLIT (pominiety) ###")

    variants = (
        list(TRAINED_ON_CHOICES) if augmentation == "both" else [augmentation]
    )
    log(f"\n### Warianty treningowe: {variants} ###")

    if not skip_classical:
        log("\n### METODA KLASYCZNA: HOG + SVM ###")
        from classical import run_classical

        run_classical.run_all(variants)

    if not skip_ml:
        log("\n### MODEL ML: ResNet18 (transfer learning) ###")
        from ml import run_ml

        run_ml.run_all(variants)

    if not skip_compare:
        log("\n### POROWNANIE WYNIKOW ###")
        from evaluation import compare

        compare.main()
    else:
        log("\n### POROWNANIE (pominiety) ###")

    if not skip_demo:
        log("\n### DEMO PREDYKCJI ###")
        try:
            import demo

            demo.main()
        except Exception as exc:  # noqa: BLE001
            log(f"Demo nie wystartowalo: {exc}")
    else:
        log("\n### DEMO (pominiety) ###")

    log("\nGotowe. Zobacz folder 'results/' po wykresy i metryki.")
    return True


@contextlib.contextmanager
def redirect_stdout(log: Callable[[str], None]):
    """Capture print() from backend modules and forward to log."""
    old_stdout = sys.stdout
    buffer = io.StringIO()

    class _Writer(io.TextIOBase):
        def write(self, s: str) -> int:
            if s:
                buffer.write(s)
                if "\n" in s:
                    for line in buffer.getvalue().splitlines():
                        if line.strip():
                            log(line)
                    buffer.seek(0)
                    buffer.truncate(0)
            return len(s)

        def flush(self) -> None:
            rest = buffer.getvalue()
            if rest.strip():
                log(rest.rstrip())
            buffer.seek(0)
            buffer.truncate(0)

    try:
        sys.stdout = _Writer()  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = old_stdout
        rest = buffer.getvalue()
        if rest.strip():
            log(rest.rstrip())
