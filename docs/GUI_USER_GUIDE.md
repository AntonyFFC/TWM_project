# TWM Project — GUI User Guide

Graphical interface for the **Bottle Cap Classification** project. The application supports data inspection, augmentation tuning, machine-learning training (HOG+SVM and ResNet18), rule-based computer vision (Classical2), cross-method comparison, and export.

**Launch the GUI:**

```bash
python run_gui.py
```

---

## Table of contents

1. [Overview](#overview)
2. [Recommended workflow](#recommended-workflow)
3. [Menu bar](#menu-bar)
4. [Tabs](#tabs)
   - [Data Loader](#1-data-loader)
   - [Augmentation Config](#2-augmentation-config)
   - [Augmentation Viewer](#3-augmentation-viewer)
   - [Training Pipeline](#4-training-pipeline)
   - [Machine Learning](#5-machine-learning)
   - [Rule-Based Config & Analyze](#6-rule-based-config--analyze)
   - [Rule-Based Evaluation](#7-rule-based-evaluation)
   - [Results Comparison](#8-results-comparison)
   - [Export](#9-export)
5. [Classification classes](#classification-classes)
6. [Output folders](#output-folders)

---

## Overview

The GUI is organized as a **linear pipeline** with nine tabs:

| # | Tab | Purpose |
|---|-----|---------|
| 1 | Data Loader | Browse and preview dataset images |
| 2 | Augmentation Config | Tune augmentation probabilities and limits |
| 3 | Augmentation Viewer | Preview augmentations on the selected image |
| 4 | Training Pipeline | Run EDA, split, and full ML training |
| 5 | Machine Learning | Train or review HOG+SVM and ResNet18 runs |
| 6 | Rule-Based Config & Analyze | Tune Classical2 thresholds; analyze one image |
| 7 | Rule-Based Evaluation | Batch-evaluate Classical2 on the dataset |
| 8 | Results Comparison | Compare all methods in one table and plots |
| 9 | Export | Inference, demo PNG, export results bundle |

Three method families are compared in the project:

| Family | Type | Description |
|--------|------|-------------|
| **HOG + SVM** | Feature-based ML | Hand-crafted features + linear classifier on 128×128 crops |
| **Neural Network (ResNet18)** | Deep learning | Transfer learning on cropped bottle caps |
| **Rule-based CV (Classical2)** | Classical CV | Segmentation + geometry heuristics on full images |

---

## Recommended workflow

### First-time setup

1. Ensure the dataset exists under `bottle-cap.yolov8/train/` (images + labels).
2. Open **Training Pipeline** → verify **Prerequisites** → click **Run Split** if splits are missing.
3. Optionally run **Run EDA** for exploratory plots in `results/plots/`.

### Machine learning path

1. **Data Loader** — confirm images load correctly.
2. **Augmentation Config / Viewer** — adjust augmentation if needed.
3. **Training Pipeline** — set hyperparameters, choose **Augmentation: both**, click **Run Full Pipeline**.
4. **Machine Learning** — inspect individual runs, confusion matrices, and metrics.
5. **Results Comparison** — refresh table and comparison plots.
6. **Export** — generate demo PNG or run inference on a single image.

### Rule-based CV path

1. **Rule-Based Config & Analyze** — load a preset, pick test images, tune parameters, use **Analyze**.
2. **Rule-Based Evaluation** — run on the full labeled dataset (raw / aug / both).
3. **Results Comparison** — Classical2 metrics appear alongside ML methods after evaluation.

---

## Menu bar

| Menu | Item | Action |
|------|------|--------|
| **File** | Open results folder | Opens `results/` in the file explorer |
| **File** | Exit | Closes the application |
| **View** | Open rule-based errors folder | Opens `classical/result/errors/` |
| **View** | Reset Layout | Resets window size to 1280×860 |
| **Help** | About | Short application description |

The **status bar** at the bottom shows the current operation and the last loaded image name.

---

## Tabs

### 1. Data Loader

![Data Loader](../gui/screenshots/01-data-loader.png)

**Purpose:** Central place to select images used by other tabs (Augmentation Viewer, Rule-Based Analyze, Export inference).

**How to use:**

1. Click **Browse Raw Images** to pick a file from `bottle-cap.yolov8/train/images/`, or **Browse Processed Crops** for split crops under `data/processed/crops/`.
2. Read metadata in **Selected Image Info** (path, resolution, file size).
3. Confirm the image in **Image Preview** on the right.

**Tip:** Keep one representative image loaded before opening Augmentation Viewer or Rule-Based Analyze — those tabs read the current selection via **Use Data Loader**.

---

### 2. Augmentation Config

![Augmentation Config](../gui/screenshots/02-augmentation-config.png)

**Purpose:** Configure the stochastic augmentation pipeline used during ML training and Classical2 augmented evaluation.

**Controls (each augmentation has Enable, parameters, and Probability):**

| Augmentation | Typical use |
|--------------|-------------|
| Rotation | Simulates tilted bottles/caps |
| Horizontal flip | Mirror variation |
| Brightness / Contrast | Lighting changes |
| Gaussian / Motion blur | Focus and motion artifacts |
| Gaussian noise | Sensor noise |

**Visualization settings:**

- **Number of copies** — how many pipeline samples to show in the Viewer.
- **View mode** — `individual`, `copies`, or `both`.
- **Reset to Defaults** — restore factory settings.

Changes here affect **Training Pipeline** (aug training) and **Rule-Based Evaluation** when **Use Augmentation Config** is checked.

---

### 3. Augmentation Viewer

![Augmentation Viewer](../gui/screenshots/03-augmentation-viewer.png)

**Purpose:** Visual proof of augmentation settings before training or evaluation.

**How to use:**

1. Load an image on **Data Loader**.
2. Adjust settings on **Augmentation Config**.
3. Click **Generate Visualization**.
4. Optionally **Choose Save Location** and save the grid as PNG.

**Reading the preview:**

- **Top row** — each effect applied **alone** (always on) to the original.
- **Bottom row (Pipeline Copy #N)** — each copy is an **independent** random pass through the full pipeline from the **original** image. Copies are not chained; a rotated Copy #1 does not affect Copy #2.

Check the **Status** log for errors (e.g. no image selected).

---

### 4. Training Pipeline

![Training Pipeline](../gui/screenshots/04-training-pipeline.png)

**Purpose:** End-to-end orchestration — EDA, dataset split, HOG+SVM training, ResNet18 training, comparison charts, and demo.

**Prerequisites** — shows raw image/label counts and whether train/val/test splits exist. Click **Refresh status** after external changes.

**Hyperparameters** — edit values then **Save to config.py** (or **Reload from config**). Key fields:

| Parameter | Meaning |
|-----------|---------|
| Train / val / test ratio | Stratified split proportions |
| Crop size | Output crop dimension (default 128 px) |
| BBox padding | Extra context around label box |
| Batch size, CNN epochs, LR | ResNet18 training |

**Pipeline options:**

| Option | Description |
|--------|-------------|
| **Augmentation: raw / aug / both** | Train on originals only, augmented only, or both variants |
| **Skip EDA / split / HOG+SVM / neural network / demo** | Run only the steps you need |

**Buttons:**

- **Run EDA** — dataset statistics and plots.
- **Run Split** — create crops and splits (skipped if splits already exist).
- **Run Full Pipeline** — runs all non-skipped steps; output streams to the **Log** panel.

Long runs (especially ResNet18) block the buttons until finished; watch the log and status bar.

---

### 5. Machine Learning

![Machine Learning](../gui/screenshots/05-machine-learning.png)

**Purpose:** Train or inspect **HOG + SVM** (left) and **Neural Network / ResNet18** (right) side by side.

**Per column:**

1. **Train on** — `raw`, `aug`, or `both` (runs one or two training jobs).
2. **Run training** — starts training in a background thread.
3. **Run** dropdown — pick a saved run from `results/metrics/`.
4. **Refresh** — reload the run list after training.

The text panel shows accuracy, F1, precision, recall, train time, inference ms/image, and sample counts. Below that, the **confusion matrix** plot for the selected run.

**Note:** Hyperparameters for training are edited on the **Training Pipeline** tab (`config.py` block).

---

### 6. Rule-Based Config & Analyze

![Rule-Based Config & Analyze](../gui/screenshots/06-rule-based-config-analyze.png)

**Purpose:** Configure and debug **Classical2** — rule-based analysis using V-channel segmentation and geometric checks on **full-resolution** images.

**Left panel — parameters:**

- **Preset** — load/save JSON presets from `classical/presets/`.
- Grouped thresholds: segmentation (V ranges), morphology, cap detection, loose/broken cap, ring, line detection (Hough/Canny).

**Right panel — single-image analysis:**

1. Set **Labels dir** (default: dataset labels folder).
2. **Use Data Loader** or **Browse** for an image.
3. Click **Analyze**.

**Classification result** shows:

- **Expected (dataset)** — ground truth from the label file (with class names).
- **Predicted** — Classical2 status code and name.
- **Match** — correct / partial / incorrect (color-coded).
- **Flags** — triggered heuristics (e.g. `cap_loose`, `cap_broken`).

**Preview tabs:** Original, Annotated, Bottle mask, Cap mask, Background. **Measurements** lists numeric values with short descriptions.

Parameter changes on this tab sync to **Rule-Based Evaluation** when you run batch eval.

---

### 7. Rule-Based Evaluation

![Rule-Based Evaluation](../gui/screenshots/07-rule-based-evaluation.png)

**Purpose:** Batch accuracy test of Classical2 on the labeled dataset, with error browser and metrics export.

**Dataset & run:**

| Control | Description |
|---------|-------------|
| Images / Labels / Errors | Paths to dataset and error export folder |
| Preset | JSON parameter preset |
| Evaluate: raw / aug / both | Test on originals, augmented copies, or compare both |
| Copies | Augmented copies per image (aug mode) |
| Use Augmentation Config | Use GUI augmentation settings |
| Export errors / Include partial | Save misclassified folders; include score 0.5 cases |
| View | Switch results between raw and aug when **both** was run |
| **Run evaluation** | Start batch job |

**Layout:**

- **Left — Results:** metrics summary, confusion matrix plot, log. Inference speed (ms/image) is recorded.
- **Right — Misclassified images:** filterable table; select a row to preview. **Re-analyze** applies current parameters to that case.

After a successful run, metrics JSON is written to `results/metrics/classical2_{preset}_{raw|aug}.json` and the comparison table is updated automatically.

**Accuracy note:** Uses partial credit (full match = 1.0, partial multi-label match = 0.5) consistent with dataset labeling.

---

### 8. Results Comparison

![Results Comparison](../gui/screenshots/08-results-comparison.png)

**Purpose:** Unified view of **all** methods — HOG+SVM, ResNet18, and Classical2.

**How to use:**

1. **Refresh table** — reload `results/metrics/summary.csv`.
2. **Run comparison** — rebuild summary CSV and comparison plots from all JSON reports in `results/metrics/`.

**Summary table columns:**

| Column | Meaning |
|--------|---------|
| method | Full run name (e.g. `hog_svm_raw`, `classical2_default_raw`) |
| type | HOG + SVM, Neural network, or Rule-based CV |
| trained_on | `raw` or `aug` |
| accuracy / f1_macro | Performance metrics |
| inference_ms | Mean inference time per image |
| train_time_s | Training time (0 for rule-based CV) |

**Comparison plot** dropdown — `comparison_accuracy.png`, `comparison_speed.png`, `augmentation_gain.png`, `robustness.png`. Click **Show** to display.

Run **Rule-Based Evaluation** and **Training Pipeline** first so all methods appear in the table.

---

### 9. Export

![Export](../gui/screenshots/09-export.png)

**Purpose:** Export artifacts, run single-image inference, and generate presentation-ready demo grids.

**Export files:**

- **Export summary.csv** — copy metrics summary to a chosen path.
- **Export results bundle (zip)** — zip metrics, plots, and models.
- **Open results folder** — quick access to `results/`.

**Inference:**

1. **Refresh models** — list saved models from `results/models/`.
2. Select model, choose image (**Use Data Loader** or **Browse**).
3. **Predict** — shows class probabilities in the text area.

**Demo visualization:**

1. Set **HOG+SVM run** and **Neural network run** names (or **Auto-fill**).
2. **Generate demo PNG** — creates a side-by-side grid (ground truth vs both models) saved to `results/plots/demo_predictions.png`.

---

## Classification classes

| Code | Name |
|------|------|
| 0 | Broken Cap |
| 1 | Broken Ring |
| 2 | Good Cap |
| 3 | Loose Cap |
| 4 | No Cap |

ML methods classify **crops** from bounding boxes. Classical2 analyzes **full images** and maps heuristics to the same five classes.

---

## Output folders

| Path | Contents |
|------|----------|
| `results/metrics/` | JSON reports per run, `summary.csv` |
| `results/plots/` | Confusion matrices, comparison charts, EDA, demo PNG |
| `results/models/` | Saved HOG+SVM (`.pkl`) and ResNet18 (`.pt`) weights |
| `classical/result/errors/` | Exported Classical2 misclassification cases |
| `classical/presets/` | Classical2 JSON parameter presets |
| `data/processed/crops/` | Train/val/test crop splits |

---

## Screenshots

All tab screenshots are stored in [`gui/screenshots/`](../gui/screenshots/) with numbered filenames (`01-data-loader.png` … `09-export.png`) matching the tab order above.

For project setup and CLI usage, see the main [`README.md`](../README.md).
