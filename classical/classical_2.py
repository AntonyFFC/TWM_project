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

    def __init__(self, params: Dict[str, Any] = None):
        # default parameters (tune these for your dataset)
        defaults = {
            # V-channel ranges for zone detection
            "bg_v_range": (0, 45),
            "bottle_v_range": (46, 134),
            "cap_v_range": (135, 255),
            # morphology
            "erode_iter": 2,
            "dilate_iter": 1,
            "kernel_size": 2,
            # heuristics
            "angle_thresh_deg": 10.0,  # degrees -> crooked
            "distance_prop_thresh": 0.12,  # proportion of bottle height -> loose
            "line_length_prop": 0.6,  # proportion of cap diameter for line detection
            "cap_rectangularity_thresh": 0.6,  # contour area / minAreaRect area
            "cap_relative_area_thresh": 0.05,  # relative to whole image area
        }
        self.p = defaults
        if params:
            self.p.update(params)

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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v = hsv[:, :, 2]

        # Masks using the V channel ranges: background, cap; bottle is derived later.
        bg_low, bg_high = self.p["bg_v_range"]
        bottle_low, bottle_high = self.p["bottle_v_range"]
        cap_low, cap_high = self.p["cap_v_range"]

        raw_bg_mask = cv2.inRange(v, bg_low, bg_high)
        cap_mask = cv2.inRange(v, cap_low, cap_high)

        raw_bg_mask = self._morph_clean(raw_bg_mask)
        cap_mask = self._morph_clean(cap_mask)

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
        if bottle_cnt is not None and cv2.contourArea(bottle_cnt) > 100:
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
                if len(contours) > 1 and cv2.contourArea(contours[1]) > 0.1 * largest_area:
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
                    if bottle_angle is not None:
                        try:
                            diff = abs(angle - bottle_angle)
                            diff = diff % 180
                            if diff > 90:
                                diff = 180 - diff
                            angle_diff = diff
                        except Exception:
                            angle_diff = None

                    cap_regions.append({
                        "rect": rect,
                        "box": box,
                        "bbox": bbox,
                        "center": center,
                        "angle": angle,
                        "area": area,
                        "rectangularity": rectangularity,
                        "angle_diff_deg": angle_diff,
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
            # angle difference
            try:
                # normalize angles to [-90,90]
                diff = abs(cap_angle - bottle_angle)
                diff = diff % 180
                if diff > 90:
                    diff = 180 - diff
            except Exception:
                diff = 0.0
            measurements["angle_diff_deg"] = diff

            # loose cap if the upper bottle edge contacting the cap is too narrow
            measurements["bottle_contact_width"] = None
            measurements["bottle_width"] = None
            measurements["bottle_contact_prop"] = None
            measurements["bottle_contact_line"] = None
            if bottle_box is not None:
                _, _, bw, bh = bottle_box
                measurements["bottle_width"] = bw
                top_y = bottle_box[1]
                band_h = max(1, int(bh * 0.10))
                band = bottle_mask[top_y:top_y + band_h, bottle_box[0]:bottle_box[0] + bw]
                max_contact = 0
                contact_line = None
                for row_idx, row in enumerate(band):
                    cols = np.where(row > 0)[0]
                    if cols.size > 0:
                        segments = np.split(cols, np.where(np.diff(cols) != 1)[0] + 1)
                        for seg in segments:
                            length = len(seg)
                            if length > max_contact:
                                max_contact = length
                                contact_line = (
                                    (bottle_box[0] + int(seg[0]), top_y + row_idx),
                                    (bottle_box[0] + int(seg[-1]), top_y + row_idx),
                                )
                measurements["bottle_contact_width"] = max_contact
                measurements["bottle_contact_line"] = contact_line
                if bw > 0:
                    prop = max_contact / float(bw)
                else:
                    prop = 0.0
                measurements["bottle_contact_prop"] = prop
                if prop < 0.8:
                    status.append("cap_loose")
                if diff > self.p["angle_thresh_deg"]:
                    status.append("cap_crooked")

            # Top edge straightness helper: detect lines inside the cap bbox for cap edge validation.
            cap_broken = False
            ring_broken = False
            measurements["cap_edge_angle_diff"] = None
            measurements["cap_top_edge_straight"] = None
            lines = None
            if cap_box is not None:
                x, y, w, h = cap_box
                pad = 2
                rx1, ry1 = max(x - pad, 0), max(y - pad, 0)
                rx2, ry2 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
                roi = gray[ry1:ry2, rx1:rx2]
                edges = cv2.Canny(roi, 50, 150)
                min_len = int(max(w, h) * self.p["line_length_prop"])
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=min_len, maxLineGap=10)
                if lines is None or len(lines) == 0:
                    lines = None

            if len(cap_regions) > 1:
                ring_broken = True
            elif len(cap_regions) == 1:
                box = cap_regions[0]["box"]
                sorted_pts = sorted(box.tolist(), key=lambda p: (p[1], p[0]))
                top_pts = sorted_pts[:2]
                bottom_pts = sorted_pts[2:]
                if len(top_pts) == 2 and len(bottom_pts) == 2:
                    top_angle = math.degrees(math.atan2(top_pts[1][1] - top_pts[0][1], top_pts[1][0] - top_pts[0][0]))
                    bottom_angle = math.degrees(math.atan2(bottom_pts[1][1] - bottom_pts[0][1], bottom_pts[1][0] - bottom_pts[0][0]))
                    edge_diff = self._angle_diff(top_angle, bottom_angle)
                    measurements["cap_edge_angle_diff"] = edge_diff
                    if edge_diff > 10.0:
                        ring_broken = True
                    if lines is not None:
                        top_edge_straight = self._is_straight_edge(
                            lines,
                            tuple(top_pts[0]),
                            tuple(top_pts[1]),
                            rx1,
                            ry1,
                            angle_tol=20.0,
                        )
                        measurements["cap_top_edge_straight"] = top_edge_straight

            # Cap broken if a single cap zone contains a fully surrounded hole above threshold.
            hole_contours = []
            if len(cap_regions) == 1 and cap_mask is not None:
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
                        if hole_area >= 0.05 * max(cap_area, 1.0):
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

        # Determine numeric status code
        status_code = 2
        if cap_cnt is None:
            status_code = 4
        elif "cap_crooked" in status or "cap_broken" in status:
            status_code = 0
        elif "ring_broken" in status:
            status_code = 1
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
    result_dir.mkdir(parents=True, exist_ok=True)

    bg_img = cv2.cvtColor(r["bg_mask"], cv2.COLOR_GRAY2BGR)
    bottle_img = cv2.cvtColor(r["bottle_mask"], cv2.COLOR_GRAY2BGR)
    cap_img = cv2.cvtColor(r["cap_mask"], cv2.COLOR_GRAY2BGR)

    bottle_box_pts = r["measurements"].get("bottle_box_pts")
    if bottle_box_pts is not None:
        cv2.polylines(bottle_img, [bottle_box_pts], True, (0, 255, 0), 2)
        bx, by = bottle_box_pts[0]
        cv2.putText(bottle_img, "B", (bx + 4, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw detected cap regions on saved cap.jpg
    for idx, region in enumerate(r["measurements"].get("cap_regions", []), start=1):
        box = region["box"]
        cv2.polylines(cap_img, [box], True, (0, 255, 0), 2)
        bx, by, bw, bh = region["bbox"]
        label = str(idx)
        text_pos = (bx + 6, by + 20)
        cv2.putText(cap_img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(cap_img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Mark hole region on cap.jpg if detected
    hole_contours = r.get("hole_contours", [])
    if hole_contours:
        hole_overlay = cap_img.copy()
        for cnt in hole_contours:
            cv2.drawContours(hole_overlay, [cnt], -1, (0, 0, 255), cv2.FILLED)
        cv2.addWeighted(hole_overlay, 0.5, cap_img, 0.5, 0, cap_img)

    # Mark bottle contact line on bottle.jpg if detected
    contact_line = r["measurements"].get("bottle_contact_line")
    if contact_line is not None:
        line_overlay = bottle_img.copy()
        cv2.line(line_overlay, contact_line[0], contact_line[1], (255, 0, 0), 4)
        cv2.addWeighted(line_overlay, 0.5, bottle_img, 0.5, 0, bottle_img)

    cv2.imwrite(str(result_dir / "background.jpg"), bg_img)
    cv2.imwrite(str(result_dir / "bottle.jpg"), bottle_img)
    cv2.imwrite(str(result_dir / "cap.jpg"), cap_img)

    if args.out:
        cv2.imwrite(args.out, r["annotated"])
    


if __name__ == "__main__":
    main()
