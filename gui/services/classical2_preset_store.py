"""Load/save Classical2 JSON parameter presets."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = PROJECT_ROOT / "classical" / "presets"

sys.path.append(str(PROJECT_ROOT))
from classical.classical_2 import Classical2  # noqa: E402

RANGE_KEYS = Classical2._RANGE_KEYS
INT_KEYS = {
    "v_normalize_target",
    "erode_iter",
    "dilate_iter",
    "kernel_size",
    "min_bottle_contour_area",
    "canny_low",
    "canny_high",
    "hough_threshold",
    "hough_max_line_gap",
    "cap_roi_pad",
}
FLOAT_KEYS = {
    "angle_thresh_deg",
    "distance_prop_thresh",
    "line_length_prop",
    "cap_rectangularity_thresh",
    "cap_relative_area_thresh",
    "second_cap_area_ratio",
    "contact_band_prop",
    "upper_half_prop",
    "loose_contact_prop_thresh",
    "cap_area_missing_thresh",
    "cap_area_broken_thresh",
    "cap_hole_area_prop_thresh",
    "ring_edge_angle_thresh",
    "straight_edge_threshold_ratio",
    "straight_edge_angle_tol",
}

PARAM_GROUPS: dict[str, list[str]] = {
    "Segmentation": [
        "bg_v_range",
        "bottle_v_range",
        "cap_v_range",
        "v_normalize_target",
    ],
    "Morphology": ["erode_iter", "dilate_iter", "kernel_size"],
    "Cap detection": [
        "cap_rectangularity_thresh",
        "cap_relative_area_thresh",
        "second_cap_area_ratio",
        "min_bottle_contour_area",
    ],
    "Bottle geometry": ["contact_band_prop", "upper_half_prop"],
    "Loose cap": ["loose_contact_prop_thresh", "angle_thresh_deg"],
    "Broken cap": [
        "cap_area_missing_thresh",
        "cap_area_broken_thresh",
        "cap_hole_area_prop_thresh",
    ],
    "Broken ring": ["ring_edge_angle_thresh"],
    "Line detection": [
        "line_length_prop",
        "canny_low",
        "canny_high",
        "hough_threshold",
        "hough_max_line_gap",
        "cap_roi_pad",
        "straight_edge_threshold_ratio",
        "straight_edge_angle_tol",
    ],
}

PARAM_LABELS: dict[str, str] = {
    "bg_v_range": "Background V range",
    "bottle_v_range": "Bottle V range",
    "cap_v_range": "Cap V range",
    "v_normalize_target": "V normalize target",
    "erode_iter": "Erode iterations",
    "dilate_iter": "Dilate iterations",
    "kernel_size": "Morphology kernel size",
    "angle_thresh_deg": "Crooked angle threshold (deg)",
    "distance_prop_thresh": "Distance prop (unused legacy)",
    "line_length_prop": "Hough min line length ratio",
    "cap_rectangularity_thresh": "Cap rectangularity min",
    "cap_relative_area_thresh": "Min cap area (image fraction)",
    "second_cap_area_ratio": "Second cap region ratio",
    "min_bottle_contour_area": "Min bottle contour area",
    "contact_band_prop": "Contact band height ratio",
    "upper_half_prop": "Upper half height ratio",
    "loose_contact_prop_thresh": "Loose contact width ratio",
    "cap_area_missing_thresh": "Cap missing hull ratio",
    "cap_area_broken_thresh": "Cap broken hull ratio",
    "cap_hole_area_prop_thresh": "Cap hole area ratio",
    "ring_edge_angle_thresh": "Ring edge angle diff (deg)",
    "canny_low": "Canny low threshold",
    "canny_high": "Canny high threshold",
    "hough_threshold": "Hough vote threshold",
    "hough_max_line_gap": "Hough max line gap",
    "cap_roi_pad": "Cap ROI padding (px)",
    "straight_edge_threshold_ratio": "Straight edge length ratio",
    "straight_edge_angle_tol": "Straight edge angle tol (deg)",
}


def _preset_path(name: str) -> Path:
    safe = name.strip().replace(" ", "_")
    return PRESETS_DIR / f"{safe}.json"


def list_presets() -> list[str]:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in PRESETS_DIR.glob("*.json"))
    if "default" not in names:
        save_preset("default", Classical2.get_default_params(), "Built-in defaults")
        names = sorted(p.stem for p in PRESETS_DIR.glob("*.json"))
    return names


def _params_for_json(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in params.items():
        if isinstance(val, tuple):
            out[key] = list(val)
        else:
            out[key] = val
    return out


def validate_params(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    defaults = Classical2.get_default_params()
    for key in defaults:
        if key not in params:
            errors.append(f"Missing parameter: {key}")
    for key in params:
        if key not in defaults:
            errors.append(f"Unknown parameter: {key}")
    for key in RANGE_KEYS:
        if key not in params:
            continue
        val = params[key]
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            errors.append(f"{key} must be [low, high]")
            continue
        lo, hi = int(val[0]), int(val[1])
        if not 0 <= lo <= hi <= 255:
            errors.append(f"{key} values must be 0-255 with low <= high")
    for key in INT_KEYS:
        if key in params and int(params[key]) < 0:
            errors.append(f"{key} must be >= 0")
    ratio_keys = (
        "cap_rectangularity_thresh",
        "cap_relative_area_thresh",
        "second_cap_area_ratio",
        "contact_band_prop",
        "upper_half_prop",
        "loose_contact_prop_thresh",
        "cap_area_missing_thresh",
        "cap_area_broken_thresh",
        "cap_hole_area_prop_thresh",
        "straight_edge_threshold_ratio",
        "line_length_prop",
        "distance_prop_thresh",
    )
    for key in ratio_keys:
        if key in params and not 0 <= float(params[key]) <= 1:
            errors.append(f"{key} must be between 0 and 1")
    return errors


def load_preset(name: str) -> dict[str, Any]:
    path = _preset_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {name}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    params = Classical2.normalize_params(data.get("params", data))
    errors = validate_params(params)
    if errors:
        raise ValueError(f"Invalid preset {name}: " + "; ".join(errors))
    return params


def load_preset_meta(name: str) -> dict[str, Any]:
    path = _preset_path(name)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_preset(name: str, params: dict[str, Any], description: str = "") -> Path:
    errors = validate_params(params)
    if errors:
        raise ValueError("\n".join(errors))
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "description": description,
        "params": _params_for_json(params),
    }
    path = _preset_path(name)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def get_default_params() -> dict[str, Any]:
    try:
        return load_preset("default")
    except (FileNotFoundError, ValueError):
        return Classical2.normalize_params(Classical2.get_default_params())
