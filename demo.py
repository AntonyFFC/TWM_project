"""Wizualne demo: HOG+SVM vs ResNet18 obok siebie na probkach z testu.

Generuje JEDEN duzy obraz (``results/plots/demo_predictions.png``)
z gridem: po jednym przykladzie z kazdej klasy + losowe dodatkowe.
Dla kazdego obrazu pokazujemy:

    - oryginalny crop
    - GT (prawdziwa klasa)
    - predykcje HOG+SVM (klasa + pewnosc)
    - predykcje ResNet18 (klasa + pewnosc)

Idealny material na slajd / prezentacje obronna.

Uzycie:
    python demo.py                            # domyslnie modele *_aug
    python demo.py --classical hog_svm_raw    # konkretny run klasyczny
    python demo.py --ml resnet18_raw          # konkretny run ML
    python demo.py --n-per-class 3            # wiecej probek
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import CLASS_NAMES, MODELS_DIR, PLOTS_DIR, SEED, ensure_dirs  # noqa: E402
from data.preprocessing import load_split_as_arrays  # noqa: E402
from evaluation.infer import list_available_models, load_trained_method  # noqa: E402


def _pick_run(prefix: str, fallback_suffixes: tuple[str, ...] = ("_aug", "_raw")) -> str | None:
    """Wybierz pierwszy istniejacy run z preferowanym suffix-em."""
    available = set(list_available_models())
    for suffix in fallback_suffixes:
        cand = f"{prefix}{suffix}"
        if cand in available:
            return cand
    for name in available:
        if name.startswith(prefix):
            return name
    return None


def _green_red(ok: bool) -> str:
    return "#2a9d3f" if ok else "#c8412a"


def _short(name: str, n: int = 12) -> str:
    return name if len(name) <= n else name[: n - 1] + "."


def make_grid(
    classical_run: str,
    ml_run: str,
    n_per_class: int,
    seed: int = SEED,
) -> Path:
    ensure_dirs()
    print(f"Wczytuje modele: {classical_run} vs {ml_run}")
    classical, classical_meta = load_trained_method(classical_run)
    ml, ml_meta = load_trained_method(ml_run)

    print("Wczytuje split testowy...")
    X_test, y_test, paths = load_split_as_arrays("test")
    print(f"  test={len(X_test)} probek")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cid in range(len(CLASS_NAMES)):
        idxs = np.where(y_test == cid)[0]
        if len(idxs) == 0:
            continue
        take = min(n_per_class, len(idxs))
        selected.extend(rng.choice(idxs, size=take, replace=False).tolist())

    if not selected:
        raise RuntimeError("Test set jest pusty -- czy odpaliles splitter?")

    selected = sorted(selected)
    Xs = X_test[selected]
    ys = y_test[selected]

    print(f"Wybralem {len(Xs)} probek, biegne predykcje...")
    proba_c = classical.predict_proba(Xs)
    proba_m = ml.predict_proba(Xs)
    pred_c = proba_c.argmax(axis=1)
    pred_m = proba_m.argmax(axis=1)
    conf_c = proba_c.max(axis=1)
    conf_m = proba_m.max(axis=1)

    n = len(Xs)
    n_cols = min(5, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.6 * n_rows))
    axes = np.atleast_2d(axes)

    fig.suptitle(
        f"Demo: {classical_run}  vs  {ml_run}\n"
        f"acc(classical)={classical_meta['final_metrics']['accuracy']:.2f}  |  "
        f"acc(ml)={ml_meta['final_metrics']['accuracy']:.2f}",
        fontsize=13,
        fontweight="bold",
    )

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        bgr = Xs[i]
        rgb = bgr[..., ::-1]
        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])

        gt_name = CLASS_NAMES[int(ys[i])]
        c_name = CLASS_NAMES[int(pred_c[i])]
        m_name = CLASS_NAMES[int(pred_m[i])]
        c_ok = pred_c[i] == ys[i]
        m_ok = pred_m[i] == ys[i]

        title = f"GT: {gt_name}"
        ax.set_title(title, fontsize=10, fontweight="bold")

        # Etykiety pod obrazem ze stylem tick/cross.
        ax.text(
            0.02,
            -0.02,
            f"HOG+SVM: {_short(c_name)}  ({conf_c[i]:.2f}) {'✓' if c_ok else '✗'}",
            transform=ax.transAxes,
            fontsize=9,
            color=_green_red(c_ok),
            verticalalignment="top",
        )
        ax.text(
            0.02,
            -0.14,
            f"ResNet18: {_short(m_name)}  ({conf_m[i]:.2f}) {'✓' if m_ok else '✗'}",
            transform=ax.transAxes,
            fontsize=9,
            color=_green_red(m_ok),
            verticalalignment="top",
        )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = PLOTS_DIR / "demo_predictions.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    n_correct_c = int((pred_c == ys).sum())
    n_correct_m = int((pred_m == ys).sum())
    print(
        f"Trafione na demo set: HOG+SVM {n_correct_c}/{n}, "
        f"ResNet18 {n_correct_m}/{n}"
    )
    print(f"Zapisano {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--classical",
        type=str,
        default=None,
        help="Konkretny run klasyczny (np. hog_svm_aug). Default: auto.",
    )
    parser.add_argument(
        "--ml",
        type=str,
        default=None,
        help="Konkretny run ML (np. resnet18_aug). Default: auto.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=2,
        help="Ile probek na klase pokazac (default: 2).",
    )
    args = parser.parse_args()

    classical_run = args.classical or _pick_run("hog_svm")
    ml_run = args.ml or _pick_run("resnet18")
    if classical_run is None or ml_run is None:
        print(
            "Brakuje wytrenowanych modeli.\n"
            "  Klasyczny: " + str(classical_run) + "\n"
            "  ML       : " + str(ml_run) + "\n"
            "Odpal `python run_all.py` zeby je wytrenowac."
        )
        print("\nDostepne modele:", list_available_models() or "(brak)")
        sys.exit(1)

    make_grid(classical_run, ml_run, n_per_class=args.n_per_class)


if __name__ == "__main__":
    main()
