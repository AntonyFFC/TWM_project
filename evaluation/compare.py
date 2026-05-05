"""Zbiera wyniki wszystkich runow w tabele i wykresy porownawcze.

Czyta wszystko z ``results/metrics/*.json`` (pomijajac ``*_robustness.json``)
i produkuje:

- ``results/metrics/summary.csv``
- ``results/plots/comparison_accuracy.png`` (paired raw vs aug)
- ``results/plots/comparison_speed.png``
- ``results/plots/robustness.png``
- ``results/plots/augmentation_gain.png``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import METRICS_DIR, PLOTS_DIR, ensure_dirs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _load_reports() -> tuple[pd.DataFrame, dict[str, dict]]:
    ensure_dirs()
    rows = []
    robustness_map: dict[str, dict] = {}

    for fp in sorted(METRICS_DIR.glob("*.json")):
        if fp.name == "summary.json":
            continue
        if fp.stem.endswith("_robustness"):
            run_name = fp.stem[: -len("_robustness")]
            with open(fp, "r", encoding="utf-8") as fh:
                robustness_map[run_name] = json.load(fh)
            continue
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        rows.append(
            {
                "method": data.get("method"),
                "base_method": data.get("base_method", data.get("method")),
                "kind": data.get("kind"),
                "trained_on": data.get("trained_on", "raw"),
                "accuracy": data.get("accuracy"),
                "f1_macro": data.get("f1_macro"),
                "precision_macro": data.get("precision_macro"),
                "recall_macro": data.get("recall_macro"),
                "inference_ms": data.get("inference_ms_per_image"),
                "train_time_s": data.get("train_time_s"),
                "n_train": data.get("n_train", 0),
                "n_test": data.get("n_test"),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["kind", "base_method", "trained_on"], ascending=[True, True, True]
    )
    return df, robustness_map


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    """Klasyczne najpierw, potem ML, w kazdej grupie alfabetycznie."""
    order = []
    for kind in ("classical", "ml"):
        subset = df.loc[df["kind"] == kind, "base_method"].drop_duplicates()
        order.extend(sorted(subset.tolist()))
    return order


def plot_accuracy_comparison(df: pd.DataFrame) -> Path:
    methods = _ordered_methods(df)
    x = np.arange(len(methods))
    fig, (ax_acc, ax_f1) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    width = 0.35
    for ax, metric, title in (
        (ax_acc, "accuracy", "Accuracy"),
        (ax_f1, "f1_macro", "F1 macro"),
    ):
        raw_vals = []
        aug_vals = []
        for bm in methods:
            row_raw = df[(df["base_method"] == bm) & (df["trained_on"] == "raw")]
            row_aug = df[(df["base_method"] == bm) & (df["trained_on"] == "aug")]
            raw_vals.append(row_raw[metric].mean() if not row_raw.empty else np.nan)
            aug_vals.append(row_aug[metric].mean() if not row_aug.empty else np.nan)

        ax.bar(x - width / 2, raw_vals, width, label="raw", color="#5c8ec9")
        ax.bar(x + width / 2, aug_vals, width, label="aug", color="#e08a3c")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        for xi, val in zip(x - width / 2, raw_vals):
            if not np.isnan(val):
                ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
        for xi, val in zip(x + width / 2, aug_vals):
            if not np.isnan(val):
                ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)

    fig.suptitle("Porownanie metod: raw vs augmented training")
    plt.tight_layout()
    out = PLOTS_DIR / "comparison_accuracy.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_speed_comparison(df: pd.DataFrame) -> Path:
    speed = (
        df.groupby("base_method")["inference_ms"].mean().sort_values().reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(speed))
    ax.bar(x, speed["inference_ms"], color="#4e9a73")
    ax.set_xticks(x)
    ax.set_xticklabels(speed["base_method"], rotation=0)
    ax.set_ylabel("ms / obraz")
    ax.set_title("Czas inferencji (srednia raw + aug)")
    for xi, val in zip(x, speed["inference_ms"]):
        ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = PLOTS_DIR / "comparison_speed.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_augmentation_gain(df: pd.DataFrame) -> Path | None:
    methods = _ordered_methods(df)
    deltas = {"accuracy": [], "f1_macro": []}
    kept = []
    for bm in methods:
        r_raw = df[(df["base_method"] == bm) & (df["trained_on"] == "raw")]
        r_aug = df[(df["base_method"] == bm) & (df["trained_on"] == "aug")]
        if r_raw.empty or r_aug.empty:
            continue
        kept.append(bm)
        deltas["accuracy"].append(r_aug["accuracy"].mean() - r_raw["accuracy"].mean())
        deltas["f1_macro"].append(r_aug["f1_macro"].mean() - r_raw["f1_macro"].mean())

    if not kept:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(kept))
    width = 0.35
    ax.bar(x - width / 2, deltas["accuracy"], width, label="Δ Accuracy", color="#5c8ec9")
    ax.bar(x + width / 2, deltas["f1_macro"], width, label="Δ F1 macro", color="#e08a3c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(kept, rotation=0)
    ax.set_ylabel("aug - raw")
    ax.set_title("Efekt augmentacji per metoda")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "augmentation_gain.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_robustness(robustness_map: dict[str, dict]) -> Path | None:
    if not robustness_map:
        return None
    all_corruptions: list[str] = []
    for rmap in robustness_map.values():
        for k in rmap.keys():
            if k not in all_corruptions:
                all_corruptions.append(k)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(all_corruptions))
    n_methods = len(robustness_map)
    bar_w = 0.8 / max(1, n_methods)

    for i, (method, rmap) in enumerate(sorted(robustness_map.items())):
        values = [rmap.get(c, {}).get("accuracy", np.nan) for c in all_corruptions]
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(x + offset, values, bar_w, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(all_corruptions, rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Odpornosc: accuracy pod zaburzeniami obrazu")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "robustness.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    df, robustness_map = _load_reports()
    if df.empty:
        print("Brak raportow w results/metrics/. Odpal najpierw `python run_all.py`.")
        return

    csv_path = METRICS_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Zapisano {csv_path}")

    print("\n=== Tabela podsumowujaca ===")
    print(df.to_string(index=False))

    acc_plot = plot_accuracy_comparison(df)
    print(f"Zapisano {acc_plot}")
    speed_plot = plot_speed_comparison(df)
    print(f"Zapisano {speed_plot}")
    gain_plot = plot_augmentation_gain(df)
    if gain_plot is not None:
        print(f"Zapisano {gain_plot}")
    rob_plot = plot_robustness(robustness_map)
    if rob_plot is not None:
        print(f"Zapisano {rob_plot}")


if __name__ == "__main__":
    main()
