"""Pipeline operations for the GUI."""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pipeline.orchestrator import (  # noqa: E402
    check_dataset,
    redirect_stdout,
    run_eda,
    run_pipeline,
    run_split,
)
from pipeline.orchestrator import AugmentationChoice  # noqa: E402


def get_dataset_status(log: Callable[[str], None] | None = None) -> dict[str, Any]:
    log_fn = log or (lambda _m: None)
    return check_dataset(log_fn)


def execute_eda(log: Callable[[str], None]) -> None:
    with redirect_stdout(log):
        run_eda(log)


def execute_split(log: Callable[[str], None]) -> None:
    with redirect_stdout(log):
        run_split(log)


def execute_full_pipeline(
    *,
    augmentation: AugmentationChoice,
    skip_eda: bool,
    skip_split: bool,
    skip_classical: bool,
    skip_ml: bool,
    skip_demo: bool,
    log: Callable[[str], None],
) -> bool:
    with redirect_stdout(log):
        return run_pipeline(
            augmentation=augmentation,
            skip_eda=skip_eda,
            skip_split=skip_split,
            skip_classical=skip_classical,
            skip_ml=skip_ml,
            skip_demo=skip_demo,
            log=log,
        )
