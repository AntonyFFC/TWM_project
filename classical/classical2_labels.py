"""Human-readable labels for Classical2 status codes."""
from __future__ import annotations

CLASSICAL2_LABELS: dict[int, str] = {
    0: "Broken Cap",
    1: "Broken Ring",
    2: "Good Cap",
    3: "Loose Cap",
    4: "No Cap",
}

CLASSICAL2_SLUGS: dict[int, str] = {
    0: "broken_cap",
    1: "broken_ring",
    2: "good_cap",
    3: "loose_cap",
    4: "no_cap",
}

STATUS_FLAG_LABELS: dict[str, str] = {
    "ok": "Good Cap — no defect flags",
    "cap_missing": "No Cap — cap not detected",
    "cap_loose": "Loose Cap — neck contact too narrow",
    "cap_broken": "Broken Cap — irregular shape or hole in cap",
    "ring_broken": "Broken Ring — split cap or edge angle mismatch",
    "cap_crooked": "Broken Cap — cap top edge tilt from horizontal exceeds threshold",
}

MEASUREMENT_LABELS: dict[str, str] = {
    "bottle_contact_prop": "Contact/upper width ratio (loose if below threshold)",
    "bottle_contact_width": "Contact width at top of bottle (px)",
    "bottle_upper_width": "Upper bottle span (px)",
    "angle_diff_deg": "Cap top-edge tilt from horizontal (degrees)",
    "cap_area_ratio": "Cap contour / convex hull (broken if low)",
    "cap_area_ratio_status": "Cap shape check result",
    "cap_hole_area_prop": "Hole area inside cap mask (broken if above threshold)",
    "cap_edge_angle_diff": "Top vs bottom cap edge angle (ring broken if high)",
    "cap_top_edge_straight": "Top cap edge looks straight (Hough lines)",
    "cap_regions": "Number of cap-like regions detected",
}


def label_for_code(code: int) -> str:
    return CLASSICAL2_LABELS.get(code, str(code))


def format_code(code: int) -> str:
    return f"{code} ({label_for_code(code)})"


def format_codes(codes: list[int]) -> str:
    if not codes:
        return "—"
    return ", ".join(format_code(c) for c in codes)


def format_status_list(flags: list[str]) -> str:
    if not flags:
        return "—"
    parts = []
    for flag in flags:
        meaning = STATUS_FLAG_LABELS.get(flag, flag)
        parts.append(f"{flag} — {meaning}")
    return "; ".join(parts)


def score_output(output: int, expected: list[int]) -> float:
    if not expected:
        return 0.0
    matches = sum(1 for label in expected if label == output)
    if matches == 0:
        return 0.0
    if matches == len(expected):
        return 1.0
    return 0.5


def describe_match(expected: list[int] | None, predicted: int) -> str:
    if expected is None:
        return "unknown (no label file found)"
    score = score_output(predicted, expected)
    if score == 1.0:
        return "correct"
    if score == 0.5:
        return f"partial (predicted {format_code(predicted)}, expected one of {format_codes(expected)})"
    return f"incorrect (predicted {format_code(predicted)}, expected {format_codes(expected)})"
