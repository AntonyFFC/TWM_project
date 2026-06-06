"""Training operations for HOG+SVM and neural network methods."""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

sys.path.append(str(Path(__file__).resolve().parents[2]))

from evaluation.evaluator import TRAINED_ON_CHOICES  # noqa: E402
from pipeline.orchestrator import redirect_stdout  # noqa: E402

TrainedOn = Literal["raw", "aug", "both"]


def resolve_variants(trained_on: TrainedOn) -> list[str]:
    if trained_on == "both":
        return list(TRAINED_ON_CHOICES)
    return [trained_on]


def run_classical(trained_on: TrainedOn, log: Callable[[str], None]) -> None:
    from classical import run_classical as rc

    variants = resolve_variants(trained_on)
    log(f"HOG+SVM: {len(variants)} training variant(s)")
    with redirect_stdout(log):
        rc.run_all(variants)


def run_ml(trained_on: TrainedOn, log: Callable[[str], None]) -> None:
    from ml import run_ml as rm

    variants = resolve_variants(trained_on)
    log(f"Neural network (ResNet18): {len(variants)} training variant(s)")
    with redirect_stdout(log):
        rm.run_all(variants)
