"""Run a single method end-to-end: train -> test -> save report & confusion plot.

Entry point used by ``classical/run_classical.py`` and ``ml/run_ml.py``:

    run_method(method, method_kind="classical" | "ml", trained_on="raw" | "aug")

Each method can be run twice -- once on raw training data (``trained_on="raw"``)
and once on augmented training data (``trained_on="aug"``) -- to measure how
much offline augmentation helps.

Writes to ``results/`` for every run (``<run_name> = <method.name>_<trained_on>``):
    - results/metrics/<run_name>.json             full report
    - results/metrics/<run_name>_robustness.json  robustness-under-corruption test
    - results/plots/confusion_<run_name>.png      confusion matrix
    - results/models/<run_name>.{pkl,pt}          saved model
    - results/models/<run_name>.meta.json         metadata (time, hyperparams, metrics)
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CLASS_NAMES,
    METRICS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    SEED,
    ensure_dirs,
)

import matplotlib.pyplot as plt  # noqa: E402 - config sets MPLBACKEND=Agg
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from data.augmentation import apply_train_augmentation_batch  # noqa: E402
from data.preprocessing import load_split_as_arrays  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    EvalReport,
    compute_report,
    measure_inference_time,
)
from evaluation.robustness import evaluate_robustness  # noqa: E402


TRAINED_ON_CHOICES = ("raw", "aug")
AUG_COPIES = 2  # how many augmented copies per original image when trained_on='aug'


def _model_extension(method_kind: str) -> str:
    return ".pt" if method_kind == "ml" else ".pkl"


def _run_name(method_name: str, trained_on: str) -> str:
    return f"{method_name}_{trained_on}"


def _public_hyperparams(method) -> dict:
    """Extract JSON-serializable public hyperparameters from a method instance."""
    out: dict = {}
    for k, v in vars(method).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, str, bool, list, tuple)) or v is None:
            out[k] = v
    return out


def plot_confusion_matrix(cm: list[list[int]], run_name: str) -> Path:
    mat = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix: {run_name}")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = PLOTS_DIR / f"confusion_{run_name}.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return out


def save_report(report: EvalReport) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"{report.method}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    return out


def save_metadata(
    method,
    method_kind: str,
    run_name: str,
    trained_on: str,
    model_path: Path,
    report: EvalReport,
    n_train_original: int,
    n_train_used: int,
) -> Path:
    """Save a small sidecar JSON next to the model with all provenance info."""
    meta = {
        "run_name": run_name,
        "base_method": method.name,
        "kind": method_kind,
        "trained_on": trained_on,
        "model_file": model_path.name,
        "trained_at_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "hyperparameters": _public_hyperparams(method),
        "dataset": {
            "class_names": CLASS_NAMES,
            "n_train_original": n_train_original,
            "n_train_used": n_train_used,
            "aug_copies_per_image": AUG_COPIES if trained_on == "aug" else 0,
            "n_test": report.n_test,
        },
        "final_metrics": {
            "accuracy": report.accuracy,
            "f1_macro": report.f1_macro,
            "precision_macro": report.precision_macro,
            "recall_macro": report.recall_macro,
            "inference_ms_per_image": report.inference_ms_per_image,
            "train_time_s": report.train_time_s,
        },
    }
    out = MODELS_DIR / f"{run_name}.meta.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out


def _prepare_training_data(
    X_train: np.ndarray, y_train: np.ndarray, trained_on: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) either as-is (raw) or expanded with augmented copies (aug)."""
    if trained_on == "raw":
        return X_train, y_train
    if trained_on == "aug":
        X_aug, y_aug = apply_train_augmentation_batch(
            X_train, y_train, n_copies=AUG_COPIES, seed=SEED
        )
        return X_aug, y_aug
    raise ValueError(f"Unknown trained_on: {trained_on}")


def run_method(
    method,
    method_kind: str,
    trained_on: str = "raw",
) -> EvalReport:
    """Train + evaluate + persist results for a single (method, trained_on) pair."""
    if trained_on not in TRAINED_ON_CHOICES:
        raise ValueError(
            f"trained_on must be one of {TRAINED_ON_CHOICES}, got {trained_on!r}"
        )
    ensure_dirs()

    run_name = _run_name(method.name, trained_on)
    print(f"\n=== {run_name}  ({method_kind}) ===")
    print("Loading splits...")
    X_train_raw, y_train_raw, _ = load_split_as_arrays("train")
    X_val, y_val, _ = load_split_as_arrays("val")
    X_test, y_test, _ = load_split_as_arrays("test")
    print(
        f"  train_raw={len(X_train_raw)}, val={len(X_val)}, test={len(X_test)}"
    )

    X_train, y_train = _prepare_training_data(X_train_raw, y_train_raw, trained_on)
    if trained_on == "aug":
        print(f"  training samples after augmentation: {len(X_train)}")

    t0 = time.perf_counter()
    method.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    train_time = time.perf_counter() - t0
    print(f"  trained in {train_time:.1f}s")

    model_path = MODELS_DIR / f"{run_name}{_model_extension(method_kind)}"
    try:
        method.save(str(model_path))
        print(f"  saved model to {model_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not save model: {exc}")

    inf_ms = measure_inference_time(method.predict, X_test)
    y_pred = method.predict(X_test)

    report = compute_report(
        method_name=run_name,
        base_method=method.name,
        kind=method_kind,
        trained_on=trained_on,
        y_true=y_test,
        y_pred=y_pred,
        class_names=CLASS_NAMES,
        inference_ms_per_image=inf_ms,
        train_time_s=train_time,
        n_train=int(len(X_train)),
    )

    json_path = save_report(report)
    cm_path = plot_confusion_matrix(report.confusion_matrix, run_name)
    robustness = evaluate_robustness(method, X_test, y_test)
    rob_path = METRICS_DIR / f"{run_name}_robustness.json"
    with open(rob_path, "w", encoding="utf-8") as fh:
        json.dump(robustness, fh, indent=2)

    meta_path = save_metadata(
        method=method,
        method_kind=method_kind,
        run_name=run_name,
        trained_on=trained_on,
        model_path=model_path,
        report=report,
        n_train_original=int(len(X_train_raw)),
        n_train_used=int(len(X_train)),
    )

    print(
        f"  acc={report.accuracy:.3f}  f1={report.f1_macro:.3f}  "
        f"inf={report.inference_ms_per_image:.2f} ms/img"
    )
    print(f"  wrote {json_path}")
    print(f"  wrote {cm_path}")
    print(f"  wrote {rob_path}")
    print(f"  wrote {meta_path}")
    return report
