"""Load/save GUI-editable hyperparameters in config.py."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.py"

BEGIN_MARKER = "# BEGIN_GUI_EDITABLE"
END_MARKER = "# END_GUI_EDITABLE"

EDITABLE_FIELDS: dict[str, type] = {
    "SEED": int,
    "TRAIN_RATIO": float,
    "VAL_RATIO": float,
    "TEST_RATIO": float,
    "IMAGE_SIZE": int,
    "BBOX_PADDING": float,
    "BATCH_SIZE": int,
    "NUM_WORKERS": int,
    "CNN_EPOCHS": int,
    "CNN_LR": float,
    "CNN_INPUT_SIZE": int,
}

def load_values() -> dict[str, Any]:
    import config

    return {name: getattr(config, name) for name in EDITABLE_FIELDS}


def validate(values: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name, expected_type in EDITABLE_FIELDS.items():
        if name not in values:
            errors.append(f"Missing field: {name}")
            continue
        val = values[name]
        if not isinstance(val, expected_type):
            try:
                values[name] = expected_type(val)
            except (TypeError, ValueError):
                errors.append(f"{name} must be {expected_type.__name__}")
                continue

    seed = int(values.get("SEED", 0))
    if seed < 0:
        errors.append("SEED must be >= 0")

    ratios = [
        float(values.get("TRAIN_RATIO", 0)),
        float(values.get("VAL_RATIO", 0)),
        float(values.get("TEST_RATIO", 0)),
    ]
    if any(r <= 0 or r >= 1 for r in ratios):
        errors.append("Split ratios must be between 0 and 1")
    elif abs(sum(ratios) - 1.0) > 0.01:
        errors.append("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")

    if int(values.get("IMAGE_SIZE", 0)) < 16:
        errors.append("IMAGE_SIZE must be >= 16")
    if int(values.get("BATCH_SIZE", 0)) < 1:
        errors.append("BATCH_SIZE must be >= 1")
    if int(values.get("CNN_EPOCHS", 0)) < 1:
        errors.append("CNN_EPOCHS must be >= 1")
    if float(values.get("CNN_LR", 0)) <= 0:
        errors.append("CNN_LR must be > 0")
    if int(values.get("CNN_INPUT_SIZE", 0)) < 32:
        errors.append("CNN_INPUT_SIZE must be >= 32")

    return errors


def _format_value(val: Any, py_type: type) -> str:
    if py_type is float:
        if val == int(val):
            return f"{float(val):.6g}"
        return repr(float(val))
    if py_type is int:
        return str(int(val))
    return repr(val)


def save_values(values: dict[str, Any]) -> None:
    errors = validate(values)
    if errors:
        raise ValueError("\n".join(errors))

    text = CONFIG_PATH.read_text(encoding="utf-8")
    begin = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if begin < 0 or end < 0 or end <= begin:
        raise RuntimeError("config.py missing GUI editable markers")

    before = text[: begin + len(BEGIN_MARKER)]
    after = text[end:]
    block_lines = [""]
    for name, py_type in EDITABLE_FIELDS.items():
        val = py_type(values[name])
        block_lines.append(f"{name}: {py_type.__name__} = {_format_value(val, py_type)}")
    block_lines.append("")
    new_block = "\n".join(block_lines)

    new_text = before + "\n" + new_block + after
    tmp = CONFIG_PATH.with_suffix(".py.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(CONFIG_PATH)

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])


def field_labels() -> dict[str, str]:
    return {
        "SEED": "Random seed",
        "TRAIN_RATIO": "Train ratio",
        "VAL_RATIO": "Validation ratio",
        "TEST_RATIO": "Test ratio",
        "IMAGE_SIZE": "Crop size (px)",
        "BBOX_PADDING": "BBox padding",
        "BATCH_SIZE": "Batch size",
        "NUM_WORKERS": "DataLoader workers",
        "CNN_EPOCHS": "CNN epochs",
        "CNN_LR": "CNN learning rate",
        "CNN_INPUT_SIZE": "CNN input size",
    }
