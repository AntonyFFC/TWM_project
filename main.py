import cv2
import numpy as np

def analyze_roboflow_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find '{image_path}'")
        return

    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No cap found!")
        return

    cap_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(cap_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    (x, y), (width, height), angle = rect
    
    if width < height:
        angle = 90 - angle
    else:
        angle = -angle

    angle = (angle + 90) % 180 - 90

    tolerance = 2.0
    is_crooked = abs(angle) > tolerance

    color = (0, 0, 255) if is_crooked else (0, 255, 0)
    cv2.drawContours(output_img, [box], 0, color, 2)
    
    status_text = "FAIL: Crooked" if is_crooked else "PASS: Straight"
    cv2.putText(output_img, f"Angle: {angle:.1f} deg", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(output_img, status_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("1. Original", img)
    cv2.imshow("2. Threshold Blob", thresh)
    cv2.imshow("3. Final Result", output_img)

    print("Press any key on the image windows to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_name = "photos/WIN_20220701_14_52_15_Pro_jpg.rf.Wen5wNtFOOaoIjhY41Dv.jpg"
    analyze_roboflow_image(image_name)