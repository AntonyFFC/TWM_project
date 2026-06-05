import math
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


class Classical2:
    """Analyzer for bottle + cap checks.

    Usage: create an instance with optional thresholds, then call `analyze(image_path)`.
    The returned dict contains status, measurements and an annotated image.
    """

    _RANGE_KEYS = ("bg_v_range", "bottle_v_range", "cap_v_range")

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "bg_v_range": (0, 45),
            "bottle_v_range": (46, 129),
            "cap_v_range": (130, 255),
            "v_normalize_target": 210,
            "erode_iter": 2,
            "dilate_iter": 1,
            "kernel_size": 2,
            "angle_thresh_deg": 10.0,
            "distance_prop_thresh": 0.12,
            "line_length_prop": 0.6,
            "cap_rectangularity_thresh": 0.6,
            "cap_relative_area_thresh": 0.05,
            "second_cap_area_ratio": 0.1,
            "min_bottle_contour_area": 100,
            "contact_band_prop": 0.15,
            "upper_half_prop": 0.5,
            "loose_contact_prop_thresh": 0.85,
            "cap_area_missing_thresh": 0.8,
            "cap_area_broken_thresh": 0.95,
            "cap_hole_area_prop_thresh": 0.04,
            "ring_edge_angle_thresh": 5.0,
            "canny_low": 50,
            "canny_high": 150,
            "hough_threshold": 20,
            "hough_max_line_gap": 10,
            "cap_roi_pad": 2,
            "straight_edge_threshold_ratio": 0.7,
            "straight_edge_angle_tol": 20.0,
        }

    @classmethod
    def normalize_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(params)
        for key in cls._RANGE_KEYS:
            if key in out and isinstance(out[key], list):
                out[key] = tuple(out[key])
        return out

    def __init__(self, params: Dict[str, Any] | None = None):
        self.p = self.get_default_params()
        if params:
            self.p.update(self.normalize_params(params))

    def _morph_clean(self, mask: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.p["kernel_size"],) * 2)
        mask = cv2.erode(mask, k, iterations=self.p["erode_iter"])
        mask = cv2.dilate(mask, k, iterations=self.p["dilate_iter"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _morph_clean2(self, mask: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.p["kernel_size"]*2,) * 2)
        #mask = cv2.erode(mask, k, iterations=self.p["erode_iter"])
        mask = cv2.erode(mask, k, iterations=self.p["erode_iter"])
        mask = cv2.dilate(mask, k, iterations=self.p["dilate_iter"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _morph_clean3(self, mask: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.p["kernel_size"],) * 2)
        #mask = cv2.erode(mask, k, iterations=self.p["erode_iter"])
        mask = cv2.erode(mask, k, iterations=self.p["erode_iter"])
        #mask = cv2.dilate(mask, k, iterations=self.p["dilate_iter"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _find_largest_contour(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours[0]

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs(a - b) % 180
        return diff if diff <= 90 else 180 - diff

    @staticmethod
    def _cap_corner_points(contour) -> dict[str, tuple[int, int]] | None:
        pts = contour.reshape(-1, 2)
        if len(pts) < 4:
            return None
        x = pts[:, 0].astype(float)
        y = pts[:, 1].astype(float)
        sum_xy = x + y
        diff_xy = x - y
        return {
            "top_left": tuple(pts[int(np.argmin(sum_xy))]),
            "top_right": tuple(pts[int(np.argmax(diff_xy))]),
            "bottom_left": tuple(pts[int(np.argmin(diff_xy))]),
            "bottom_right": tuple(pts[int(np.argmax(sum_xy))]),
        }

    def _edge_tilt_from_horizontal(
        self, p1: tuple[int, int], p2: tuple[int, int]
    ) -> float:
        """Degrees the edge deviates from the image horizontal (0 = level)."""
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        return self._angle_diff(angle, 0.0)

    @staticmethod
    def _line_length(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _is_straight_edge(
        self,
        lines: np.ndarray,
        edge_p1: tuple[int, int],
        edge_p2: tuple[int, int],
        offset_x: int,
        offset_y: int,
        threshold_ratio: float = 0.7,
        angle_tol: float = 10.0,
    ) -> bool:
        edge_vec = np.array((edge_p2[0] - edge_p1[0], edge_p2[1] - edge_p1[1]), dtype=float)
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1.0:
            return False
        edge_angle = math.degrees(math.atan2(edge_vec[1], edge_vec[0]))
        for l in lines:
            x1, y1, x2, y2 = l[0]
            pt1 = (x1 + offset_x, y1 + offset_y)
            pt2 = (x2 + offset_x, y2 + offset_y)
            line_vec = np.array((pt2[0] - pt1[0], pt2[1] - pt1[1]), dtype=float)
            line_len = np.linalg.norm(line_vec)
            if line_len < 1.0:
                continue
            line_angle = math.degrees(math.atan2(line_vec[1], line_vec[0]))
            if self._angle_diff(edge_angle, line_angle) > angle_tol:
                continue
            proj = abs(np.dot(line_vec, edge_vec) / edge_len)
            if proj >= threshold_ratio * edge_len:
                return True
        return False

    def analyze(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        out = img.copy()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        if int(v.max()) < self.p["v_normalize_target"]:
            v_norm = cv2.normalize(
                v, None, alpha=0, beta=self.p["v_normalize_target"], norm_type=cv2.NORM_MINMAX
            )
            hsv[:, :, 2] = v_norm
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            v = hsv[:, :, 2]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Masks using the V channel ranges: background, cap; bottle is derived later.
        bg_low, bg_high = self.p["bg_v_range"]
        bottle_low, bottle_high = self.p["bottle_v_range"]
        cap_low, cap_high = self.p["cap_v_range"]

        raw_bg_mask = cv2.inRange(v, bg_low, bg_high)
        cap_mask = cv2.inRange(v, cap_low, cap_high)

        raw_bg_mask = self._morph_clean(raw_bg_mask)
        cap_mask = self._morph_clean3(cap_mask)

        bg_mask = np.zeros_like(raw_bg_mask)
        bg_contours, _ = cv2.findContours(raw_bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if bg_contours:
            bg_contours = sorted(bg_contours, key=cv2.contourArea, reverse=True)
            cv2.drawContours(bg_mask, [bg_contours[0]], -1, 255, thickness=cv2.FILLED)

        bottle_mask = cv2.bitwise_not(cv2.bitwise_or(bg_mask, cap_mask))
        bottle_mask = self._morph_clean2(bottle_mask)

        # Find bottle contour (largest)
        bottle_cnt = self._find_largest_contour(bottle_mask)
        bottle_box = None
        bottle_angle = 0.0
        if bottle_cnt is not None and cv2.contourArea(bottle_cnt) > self.p["min_bottle_contour_area"]:
            rect_b = cv2.minAreaRect(bottle_cnt)
            bottle_box_pts = cv2.boxPoints(rect_b).astype(int)
            x, y, w, h = cv2.boundingRect(bottle_cnt)
            bottle_box = (x, y, w, h)
            # keep angle for measurements but do not draw here
            bottle_angle = rect_b[2]
        else:
            bottle_box_pts = None

        # Find cap contours (largest and second largest if large enough)
        contours, _ = cv2.findContours(cap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cap_regions = []
        cap_box = None
        cap_angle = 0.0
        cap_center = None
        cap_cnt = None
        img_area = img.shape[0] * img.shape[1]
        min_cap_area = self.p["cap_relative_area_thresh"] * img_area
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_area = cv2.contourArea(contours[0])
            if largest_area > min_cap_area:
                candidates = [contours[0]]
                if len(contours) > 1 and cv2.contourArea(contours[1]) > self.p["second_cap_area_ratio"] * largest_area:
                    candidates.append(contours[1])

                for idx, cnt in enumerate(candidates):
                    area = cv2.contourArea(cnt)
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect).astype(int)
                    bbox = cv2.boundingRect(cnt)
                    center = (int(rect[0][0]), int(rect[0][1]))
                    angle = rect[2]

                    rect_width, rect_height = rect[1]
                    rect_area = max(rect_width * rect_height, 1.0)
                    rectangularity = float(area) / rect_area
                    if rectangularity < self.p["cap_rectangularity_thresh"]:
                        continue

                    angle_diff = None
                    corners = self._cap_corner_points(cnt)
                    if corners and corners["top_left"] != corners["top_right"]:
                        angle_diff = self._edge_tilt_from_horizontal(
                            corners["top_left"], corners["top_right"]
                        )

                    cap_regions.append({
                        "rect": rect,
                        "box": box,
                        "bbox": bbox,
                        "center": center,
                        "angle": angle,
                        "area": area,
                        "rectangularity": rectangularity,
                        "angle_diff_deg": angle_diff,
                        "contour": cnt,
                    })

                    if idx == 0:
                        cap_cnt = cnt
                        cap_box = bbox
                        cap_center = center
                        cap_angle = angle

        # Measurements
        measurements = {
            "bottle_box": bottle_box,
            "bottle_box_pts": bottle_box_pts,
            "cap_box": cap_box,
            "cap_regions": cap_regions,
            "bottle_angle": bottle_angle,
            "cap_angle": cap_angle,
            "cap_center": cap_center,
        }

        # Assess
        status = []
        if cap_cnt is None:
            status.append("cap_missing")
        else:
            corners = self._cap_corner_points(cap_cnt)
            if corners and corners["top_left"] != corners["top_right"]:
                diff = self._edge_tilt_from_horizontal(
                    corners["top_left"], corners["top_right"]
                )
            else:
                diff = 0.0
            measurements["angle_diff_deg"] = diff
            if diff > self.p["angle_thresh_deg"]:
                status.append("cap_crooked")

            # loose cap if the upper bottle edge contacting the cap is too narrow
            measurements["bottle_contact_width"] = None
            measurements["bottle_width"] = None
            measurements["bottle_upper_width"] = None
            measurements["bottle_upper_width_line"] = None
            measurements["bottle_contact_prop"] = None
            measurements["bottle_contact_line"] = None
            if bottle_box is not None:
                bx, by, bw, bh = bottle_box
                measurements["bottle_width"] = bw
                band_h = max(1, int(bh * self.p["contact_band_prop"]))
                band = bottle_mask[by:by + band_h, bx:bx + bw]
                max_contact = 0.0
                contact_line = None
                # Test: widest span in top 15% (same idea as upper_width, not per-row segments)
                ys_band, xs_band = np.where(band > 0)
                if xs_band.size > 0:
                    left_idx = int(np.argmin(xs_band))
                    right_idx = int(np.argmax(xs_band))
                    left_pt = (bx + int(xs_band[left_idx]), by + int(ys_band[left_idx]))
                    right_pt = (bx + int(xs_band[right_idx]), by + int(ys_band[right_idx]))
                    max_contact = float(
                        np.linalg.norm(
                            np.array(right_pt, dtype=float) - np.array(left_pt, dtype=float)
                        )
                    )
                    contact_line = (left_pt, right_pt)
                measurements["bottle_contact_width"] = max_contact
                measurements["bottle_contact_line"] = contact_line
                upper_width = bw
                upper_width_line = None
                half_h = max(1, int(bh * self.p["upper_half_prop"]))
                upper_mask = bottle_mask[by:by + half_h, bx:bx + bw]
                ys, xs = np.where(upper_mask > 0)
                if xs.size > 0:
                    left_idx = int(np.argmin(xs))
                    right_idx = int(np.argmax(xs))
                    left_pt = (bx + int(xs[left_idx]), by + int(ys[left_idx]))
                    right_pt = (bx + int(xs[right_idx]), by + int(ys[right_idx]))
                    upper_width = float(
                        np.linalg.norm(np.array(right_pt, dtype=float) - np.array(left_pt, dtype=float))
                    )
                    upper_width_line = (left_pt, right_pt)
                measurements["bottle_upper_width"] = upper_width
                measurements["bottle_upper_width_line"] = upper_width_line
                if upper_width > 0:
                    prop = max_contact / float(upper_width)
                else:
                    prop = 0.0
                measurements["bottle_contact_prop"] = prop
                if prop < self.p["loose_contact_prop_thresh"]:
                    status.append("cap_loose")

            # Top edge straightness helper: detect lines inside the cap bbox for cap edge validation.
            cap_broken = False
            ring_broken = False
            measurements["cap_edge_angle_diff"] = None
            measurements["cap_top_edge_straight"] = None
            lines = None
            if cap_box is not None:
                x, y, w, h = cap_box
                pad = self.p["cap_roi_pad"]
                rx1, ry1 = max(x - pad, 0), max(y - pad, 0)
                rx2, ry2 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
                roi = gray[ry1:ry2, rx1:rx2]
                edges = cv2.Canny(roi, self.p["canny_low"], self.p["canny_high"])
                min_len = int(max(w, h) * self.p["line_length_prop"])
                lines = cv2.HoughLinesP(
                    edges,
                    1,
                    np.pi / 180,
                    threshold=self.p["hough_threshold"],
                    minLineLength=min_len,
                    maxLineGap=self.p["hough_max_line_gap"],
                )
                if lines is None or len(lines) == 0:
                    lines = None

            if len(cap_regions) > 1:
                ring_broken = True
            elif len(cap_regions) == 1:
                # Check convex hull area ratio for single cap zone
                region = cap_regions[0]
                cap_cnt = region.get("contour")
                if cap_cnt is not None:
                    cap_area = cv2.contourArea(cap_cnt)
                    hull = cv2.convexHull(cap_cnt)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        area_ratio = cap_area / hull_area
                        measurements["cap_area_ratio"] = area_ratio
                        if area_ratio < self.p["cap_area_missing_thresh"]:
                            measurements["cap_area_ratio_status"] = "cap_missing"
                            status.append("cap_missing")
                            cap_broken = False
                            ring_broken = False
                        elif area_ratio < self.p["cap_area_broken_thresh"]:
                            measurements["cap_area_ratio_status"] = "cap_broken"
                            cap_broken = True
                        else:
                            measurements["cap_area_ratio_status"] = "ok"

                measurements["cap_top_edge_line"] = None
                measurements["cap_bottom_edge_line"] = None
                if "cap_missing" not in status:
                    cap_cnt = region.get("contour")
                    if cap_cnt is not None:
                        corners = self._cap_corner_points(cap_cnt)
                        if corners is not None:
                            top_left = corners["top_left"]
                            top_right = corners["top_right"]
                            bottom_left = corners["bottom_left"]
                            bottom_right = corners["bottom_right"]
                        else:
                            top_left = top_right = bottom_left = bottom_right = None
                        if (
                            top_left is not None
                            and top_right is not None
                            and bottom_left is not None
                            and bottom_right is not None
                            and top_left != top_right
                            and bottom_left != bottom_right
                        ):
                            top_line = (top_left, top_right)
                            bottom_line = (bottom_left, bottom_right)
                            measurements["cap_top_edge_line"] = top_line
                            measurements["cap_bottom_edge_line"] = bottom_line
                            top_angle = math.degrees(
                                math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0])
                            )
                            bottom_angle = math.degrees(
                                math.atan2(
                                    bottom_right[1] - bottom_left[1],
                                    bottom_right[0] - bottom_left[0],
                                )
                            )
                            edge_diff = self._angle_diff(top_angle, bottom_angle)
                            measurements["cap_edge_angle_diff"] = edge_diff
                            if edge_diff > self.p["ring_edge_angle_thresh"]:
                                ring_broken = True
                            if lines is not None:
                                top_edge_straight = self._is_straight_edge(
                                    lines,
                                    top_line[0],
                                    top_line[1],
                                    rx1,
                                    ry1,
                                    threshold_ratio=self.p["straight_edge_threshold_ratio"],
                                    angle_tol=self.p["straight_edge_angle_tol"],
                                )
                                measurements["cap_top_edge_straight"] = top_edge_straight

            # Cap broken if a single cap zone contains a fully surrounded hole above threshold.
            hole_contours = []
            if "cap_missing" not in status and len(cap_regions) == 1 and cap_mask is not None:
                contours_holes, hierarchy_holes = cv2.findContours(cap_mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchy_holes is not None and contours_holes:
                    hierarchy = hierarchy_holes[0]
                    external_indices = [idx for idx, h in enumerate(hierarchy) if h[3] == -1]
                    if external_indices:
                        largest_ext = max(external_indices, key=lambda idx: cv2.contourArea(contours_holes[idx]))
                        hole_indices = [idx for idx, h in enumerate(hierarchy) if h[3] == largest_ext]
                        hole_contours = [contours_holes[idx] for idx in hole_indices]
                        hole_area = sum(cv2.contourArea(cnt) for cnt in hole_contours)
                        cap_area = cv2.contourArea(cap_cnt)
                        if hole_area >= self.p["cap_hole_area_prop_thresh"] * max(cap_area, 1.0):
                            cap_broken = True
                            measurements["cap_hole_area"] = hole_area
                            measurements["cap_hole_area_prop"] = hole_area / max(cap_area, 1.0)
                        else:
                            measurements["cap_hole_area"] = hole_area
                            measurements["cap_hole_area_prop"] = hole_area / max(cap_area, 1.0)
            else:
                hole_contours = []

            if cap_broken:
                status.append("cap_broken")
            if ring_broken:
                status.append("ring_broken")

        if not status:
            status.append("ok")

        if not status:
            status.append("ok")

        # Determine numeric status code (fixed priority; first match wins)
        status_code = 2
        if "cap_missing" in status:
            status_code = 4
        elif cap_cnt is None:
            status_code = 4
        elif "ring_broken" in status:
            status_code = 1
        elif "cap_broken" in status or "cap_crooked" in status:
            status_code = 0
        elif "cap_loose" in status:
            status_code = 3
        else:
            status_code = 2

        measurements["status_code"] = status_code
        measurements["status_label"] = status[0] if status else "ok"

        # No visualization/drawing is performed here; annotated image is the original image copy
        txt = ", ".join(status)
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        result = {
            "status_list": status,
            "status_code": status_code,
            "measurements": measurements,
            "annotated": out,
            "bg_mask": bg_mask,
            "bottle_mask": bottle_mask,
            "cap_mask": cap_mask,
            "hole_contours": hole_contours if 'hole_contours' in locals() else [],
        }

        return result


def build_analysis_visualizations(result: dict[str, Any]) -> dict[str, Any]:
    """Return BGR preview images: annotated, background, bottle, cap."""
    import cv2

    bg_img = cv2.cvtColor(result["bg_mask"], cv2.COLOR_GRAY2BGR)
    bottle_img = cv2.cvtColor(result["bottle_mask"], cv2.COLOR_GRAY2BGR)
    cap_img = cv2.cvtColor(result["cap_mask"], cv2.COLOR_GRAY2BGR)
    measurements = result["measurements"]

    bottle_box_pts = measurements.get("bottle_box_pts")
    if bottle_box_pts is not None:
        cv2.polylines(bottle_img, [bottle_box_pts], True, (0, 255, 0), 2)
        bx, by = bottle_box_pts[0]
        cv2.putText(
            bottle_img, "B", (bx + 4, by - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

    for idx, region in enumerate(measurements.get("cap_regions", []), start=1):
        box = region["box"]
        cv2.polylines(cap_img, [box], True, (0, 255, 0), 2)
        bx, by, _bw, _bh = region["bbox"]
        label = str(idx)
        text_pos = (bx + 6, by + 20)
        cv2.putText(cap_img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(cap_img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    hole_contours = result.get("hole_contours", [])
    if hole_contours:
        hole_overlay = cap_img.copy()
        for cnt in hole_contours:
            cv2.drawContours(hole_overlay, [cnt], -1, (0, 0, 255), cv2.FILLED)
        cv2.addWeighted(hole_overlay, 0.5, cap_img, 0.5, 0, cap_img)

    cap_top_edge_line = measurements.get("cap_top_edge_line")
    if cap_top_edge_line is not None:
        edge_overlay = cap_img.copy()
        cv2.line(edge_overlay, cap_top_edge_line[0], cap_top_edge_line[1], (0, 255, 255), 3)
        cv2.addWeighted(edge_overlay, 0.5, cap_img, 0.5, 0, cap_img)
    cap_bottom_edge_line = measurements.get("cap_bottom_edge_line")
    if cap_bottom_edge_line is not None:
        edge_overlay = cap_img.copy()
        cv2.line(edge_overlay, cap_bottom_edge_line[0], cap_bottom_edge_line[1], (255, 0, 0), 3)
        cv2.addWeighted(edge_overlay, 0.5, cap_img, 0.5, 0, cap_img)

    contact_line = measurements.get("bottle_contact_line")
    if contact_line is not None:
        line_overlay = bottle_img.copy()
        cv2.line(line_overlay, contact_line[0], contact_line[1], (255, 0, 0), 4)
        cv2.addWeighted(line_overlay, 0.5, bottle_img, 0.5, 0, bottle_img)

    upper_width_line = measurements.get("bottle_upper_width_line")
    if upper_width_line is not None:
        upper_overlay = bottle_img.copy()
        cv2.line(upper_overlay, upper_width_line[0], upper_width_line[1], (0, 255, 255), 4)
        cv2.addWeighted(upper_overlay, 0.5, bottle_img, 0.5, 0, bottle_img)

    return {
        "annotated": result["annotated"],
        "background": bg_img,
        "bottle": bottle_img,
        "cap": cap_img,
    }


def save_analysis_visualizations(
    result: dict[str, Any],
    out_dir: Path,
    *,
    annotated_name: str = "annotated.jpg",
    original_path: Path | None = None,
) -> None:
    """Save annotated image and debug mask overlays to ``out_dir``."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)

    if original_path is not None and original_path.exists():
        import shutil

        shutil.copy2(original_path, out_dir / "original.jpg")

    images = build_analysis_visualizations(result)
    cv2.imwrite(str(out_dir / annotated_name), images["annotated"])
    cv2.imwrite(str(out_dir / "background.jpg"), images["background"])
    cv2.imwrite(str(out_dir / "bottle.jpg"), images["bottle"])
    cv2.imwrite(str(out_dir / "cap.jpg"), images["cap"])


def format_analysis_report(
    result: dict[str, Any],
    *,
    expected: list[int] | None = None,
    score: float | None = None,
) -> str:
    """Return report text without writing to disk."""
    from classical.classical2_labels import (
        MEASUREMENT_LABELS,
        describe_match,
        format_code,
        format_codes,
        format_status_list,
    )

    m = result["measurements"]
    code = result.get("status_code")
    status_list = result.get("status_list", [])

    lines = [
        "=== Classification ===",
        f"Predicted: {format_code(code)}",
        f"Status flags: {format_status_list(status_list)}",
    ]
    if expected is not None:
        lines.append(f"Expected (dataset): {format_codes(expected)}")
        lines.append(f"Match: {describe_match(expected, code)}")
    elif score is None:
        lines.append("Expected (dataset): not loaded — set labels folder or use dataset images")
    if score is not None:
        lines.append(f"Score: {score} (1.0=full, 0.5=partial, 0.0=wrong)")
    lines.append("")
    lines.append("=== Measurements ===")
    keys = (
        "bottle_contact_prop",
        "bottle_contact_width",
        "bottle_upper_width",
        "angle_diff_deg",
        "cap_area_ratio",
        "cap_area_ratio_status",
        "cap_hole_area_prop",
        "cap_edge_angle_diff",
        "cap_top_edge_straight",
    )
    for key in keys:
        if key in m and m[key] is not None:
            val = m[key]
            desc = MEASUREMENT_LABELS.get(key, key)
            if isinstance(val, float):
                lines.append(f"{desc}")
                lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"{desc}")
                lines.append(f"  {key}: {val}")
    n_regions = len(m.get("cap_regions", []))
    lines.append(MEASUREMENT_LABELS.get("cap_regions", "cap_regions"))
    lines.append(f"  cap_regions: {n_regions}")
    return "\n".join(lines) + "\n"


def write_analysis_report(
    result: dict[str, Any],
    out_path: Path,
    *,
    expected: list[int] | None = None,
    score: float | None = None,
) -> None:
    """Write a text summary of analysis results for manual review."""
    out_path.write_text(
        format_analysis_report(result, expected=expected, score=score),
        encoding="utf-8",
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run classical_2 cap analyzer on an image")
    parser.add_argument("image", help="Path to image to analyze")
    parser.add_argument("--out", help="Path to save annotated image", default=None)
    args = parser.parse_args()

    c = Classical2()
    r = c.analyze(args.image)
    status_code = r["status_code"]
    print(status_code, end="")
    if status_code == 0:
        reasons = ", ".join(r.get("status_list", []))
        if reasons:
            print(" ", reasons)
        else:
            print()
    else:
        print()

    result_dir = Path(__file__).resolve().parent / "result"
    save_analysis_visualizations(r, result_dir, original_path=Path(args.image))

    if args.out:
        import cv2

        cv2.imwrite(args.out, r["annotated"])


if __name__ == "__main__":
    main()
