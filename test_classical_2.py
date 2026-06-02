import csv
import re
from pathlib import Path
from classical.classical_2 import Classical2


def load_labels(label_path: Path) -> list[int] | None:
    if not label_path.exists():
        return None
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    labels = []
    for line in lines:
        match = re.match(r"\s*(\d)", line)
        if match:
            labels.append(int(match.group(1)))
    return labels if labels else None


def score_output(output: int, expected: list[int]) -> float:
    if not expected:
        return 0.0
    matches = sum(1 for label in expected if label == output)
    if matches == 0:
        return 0.0
    if matches == len(expected):
        return 1.0
    return 0.5


def main() -> None:
    root = Path(__file__).resolve().parent
    img_dir = root / "bottle-cap.yolov8" / "train" / "images"
    label_dir = root / "bottle-cap.yolov8" / "train" / "labels"

    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Required folders bottle-cap.yolov8/train/images or bottle-cap.yolov8/train/labels not found")

    model = Classical2()
    files = sorted(img_dir.glob("WIN_*.jpg"))
    total = 0
    full_correct = 0
    partial_correct = 0
    missing_labels = []
    classes = [0, 1, 2, 3, 4]
    error_matrix = {exp: {pred: 0 for pred in classes} for exp in classes}

    results_path = root / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["photo", "correct_classification", "classical_2"])

        for image_path in files:
            label_path = label_dir / (image_path.stem + ".txt")
            expected = load_labels(label_path)
            if expected is None:
                missing_labels.append(image_path.name)
                continue

            result = model.analyze(str(image_path))
            status_code = result.get("status_code")
            if status_code is None:
                raise ValueError(f"No status_code returned for {image_path}")

            total += 1
            score = score_output(status_code, expected)
            if score != 1.0:
                writer.writerow([image_path.name, ";".join(str(x) for x in expected), status_code])
            if score == 1.0:
                full_correct += 1
            elif score == 0.5:
                partial_correct += 1

            for exp_label in set(expected):
                if exp_label not in error_matrix:
                    error_matrix[exp_label] = {pred: 0 for pred in classes}
                if status_code not in error_matrix[exp_label]:
                    error_matrix[exp_label][status_code] = 0
                error_matrix[exp_label][status_code] += 1

    if total == 0:
        print("No tested images found or no valid labels.")
        return

    accuracy = ((full_correct + 0.5 * partial_correct) / total) * 100.0
    print(f"Tested images: {total}")
    print(f"Fully correct: {full_correct}")
    print(f"Partially correct: {partial_correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    print("Error matrix (expected rows, predicted columns):")
    header = "    " + " ".join(f"{pred:>5}" for pred in classes)
    print(header)
    for exp_label in classes:
        row = f"{exp_label:>2} " + " ".join(f"{error_matrix[exp_label].get(pred, 0):>5}" for pred in classes)
        print(row)

    if missing_labels:
        print(f"Skipped images with missing or invalid labels: {len(missing_labels)}")


if __name__ == "__main__":
    main()
