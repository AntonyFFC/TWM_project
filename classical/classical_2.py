import math
import cv2
import numpy as np
from typing import Tuple, Dict, Any


class Classical2:
    """Analyzer for bottle + cap checks.

    Usage: create an instance with optional thresholds, then call `analyze(image_path)`.
    The returned dict contains status, measurements and an annotated image.
    """

    def __init__(self, params: Dict[str, Any] = None, visualize: bool = True):
        # default parameters (tune these for your dataset)
        defaults = {
            # HSV ranges as (low_h, low_s, low_v), (high_h, high_s, high_v)
            "cap_hsv": ((0, 0, 150), (180, 80, 255)),
            "bottle_hsv": ((0, 0, 0), (180, 255, 120)),
            "bg_hsv": ((0, 0, 0), (180, 255, 255)),
            # morphology
            "erode_iter": 1,
            "dilate_iter": 2,
            "kernel_size": 5,
            # heuristics
            "angle_thresh_deg": 15.0,  # degrees -> crooked
            "distance_prop_thresh": 0.12,  # proportion of bottle height -> loose
            "crack_line_prop": 0.6,  # proportion of cap diameter for detectable crack
        }
        self.p = defaults
        self.visualize = visualize
        if params:
            self.p.update(params)

    def _morph_clean(self, mask: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.p["kernel_size"],) * 2)
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

    def analyze(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        out = img.copy()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Masks (initial placeholders - tune HSV ranges)
        cap_low, cap_high = self.p["cap_hsv"]
        bottle_low, bottle_high = self.p["bottle_hsv"]

        cap_mask = cv2.inRange(hsv, np.array(cap_low), np.array(cap_high))
        bottle_mask = cv2.inRange(hsv, np.array(bottle_low), np.array(bottle_high))

        cap_mask = self._morph_clean(cap_mask)
        bottle_mask = self._morph_clean(bottle_mask)

        # Find bottle contour (largest)
        bottle_cnt = self._find_largest_contour(bottle_mask)
        bottle_box = None
        bottle_angle = 0.0
        bottle_top_center = None
        if bottle_cnt is not None and cv2.contourArea(bottle_cnt) > 100:
            rect_b = cv2.minAreaRect(bottle_cnt)
            box_b = cv2.boxPoints(rect_b).astype(int)
            x, y, w, h = cv2.boundingRect(bottle_cnt)
            bottle_box = (x, y, w, h)
            # choose orientation (minAreaRect angle semantics vary)
            bottle_angle = rect_b[2]
            bottle_top_center = (int(x + w / 2), int(y))
            cv2.drawContours(out, [box_b], 0, (255, 0, 0), 2)

        # Find cap contour (largest in cap_mask)
        cap_cnt = self._find_largest_contour(cap_mask)
        cap_box = None
        cap_angle = 0.0
        cap_center = None
        if cap_cnt is not None and cv2.contourArea(cap_cnt) > 50:
            rect_c = cv2.minAreaRect(cap_cnt)
            box_c = cv2.boxPoints(rect_c).astype(int)
            x, y, w, h = cv2.boundingRect(cap_cnt)
            cap_box = (x, y, w, h)
            cap_angle = rect_c[2]
            M = cv2.moments(cap_cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cap_center = (cx, cy)
            cv2.drawContours(out, [box_c], 0, (0, 255, 0), 2)
            # rectangle around cap
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Measurements
        measurements = {
            "bottle_box": bottle_box,
            "cap_box": cap_box,
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

            # distance: between cap center and bottle top center (proportional threshold)
            if bottle_top_center and cap_center and bottle_box:
                dist = math.hypot(cap_center[0] - bottle_top_center[0], cap_center[1] - bottle_top_center[1])
                measurements["cap_distance_px"] = dist
                _, _, bw, bh = bottle_box
                prop = dist / float(max(bh, 1))
                measurements["cap_distance_prop"] = prop
                # check loose
                if prop > self.p["distance_prop_thresh"]:
                    status.append("cap_loose")
                # check crooked
                if diff > self.p["angle_thresh_deg"]:
                    status.append("cap_crooked")
            else:
                measurements["cap_distance_px"] = None

            # Crack detection heuristic: look for long linear edges inside cap bbox
            cracked = False
            if cap_box is not None:
                x, y, w, h = cap_box
                pad = 2
                rx1, ry1 = max(x - pad, 0), max(y - pad, 0)
                rx2, ry2 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
                roi = gray[ry1:ry2, rx1:rx2]
                edges = cv2.Canny(roi, 50, 150)
                min_len = int(max(w, h) * self.p["crack_line_prop"])
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=min_len, maxLineGap=10)
                if lines is not None and len(lines) > 0:
                    cracked = True
                    # draw detected crack lines on out
                    for l in lines:
                        x1, y1, x2, y2 = l[0]
                        cv2.line(out, (rx1 + x1, ry1 + y1), (rx1 + x2, ry1 + y2), (0, 0, 255), 2)
            if cracked:
                status.append("cap_cracked")

        if not status:
            status.append("ok")

        # Visualization
        if self.visualize:
            # Draw cap rectangle and info
            if cap_box is not None:
                x, y, w, h = cap_box
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display angle
                angle_text = f"Angle: {cap_angle:.1f}°"
                cv2.putText(out, angle_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display distance if available
                if measurements.get("cap_distance_px") is not None:
                    dist_text = f"Distance: {measurements['cap_distance_px']:.1f}px"
                    cv2.putText(out, dist_text, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if measurements.get("cap_distance_prop") is not None:
                        prop_text = f"Prop: {measurements['cap_distance_prop']:.3f}"
                        cv2.putText(out, prop_text, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw center point
                if cap_center is not None:
                    cv2.circle(out, cap_center, 5, (0, 255, 0), -1)
                
                # Draw angle line (direction indicator)
                if cap_center is not None:
                    angle_rad = np.radians(cap_angle)
                    line_len = max(w, h) // 2
                    end_x = int(cap_center[0] + line_len * np.cos(angle_rad))
                    end_y = int(cap_center[1] + line_len * np.sin(angle_rad))
                    cv2.line(out, cap_center, (end_x, end_y), (0, 255, 0), 2)
        
        # Annotate status text
        txt = ", ".join(status)
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        result = {
            "status_list": status,
            "measurements": measurements,
            "annotated": out,
        }

        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run classical_2 cap analyzer on an image")
    parser.add_argument("image", help="Path to image to analyze")
    parser.add_argument("--out", help="Path to save annotated image", default=None)
    parser.add_argument("--display", action="store_true", help="Display result in a new window")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization annotations")
    args = parser.parse_args()

    visualize = not args.no_viz
    c = Classical2(visualize=visualize)
    r = c.analyze(args.image)
    print("Status:", r["status_list"]) 
    if visualize:
        print("Measurements:", r["measurements"])
    if args.out:
        cv2.imwrite(args.out, r["annotated"]) 
        print(f"Annotated image saved to {args.out}")
    
    if args.display:
        cv2.imshow("Classical2 - Cap Analysis", r["annotated"])
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()
