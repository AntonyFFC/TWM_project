# Augmentation Visualization Guide

## Overview
The `data/visualize_augmentation.py` script allows you to see how the augmentation pipeline transforms images step-by-step. This is useful for understanding what transformations are being applied during training.

## Features
- **Individual augmentation view**: Shows the original image + each augmentation applied separately
- **Multiple copies view**: Shows the original image + N augmented versions created by the full pipeline
- **Save to file**: Optionally save visualizations as PNG images
- **Flexible input**: Use any image from your dataset or provide a specific image path

## Usage

### Basic Usage (interactive display)
Display augmentations for the first available dataset image:
```bash
python data/visualize_augmentation.py
```

### Use a specific image
```bash
python data/visualize_augmentation.py --image path/to/my_image.png
```

### Generate more augmented copies
By default shows 5 augmented copies. To show 10:
```bash
python data/visualize_augmentation.py --n-copies 10
```

### View only individual augmentations
Show each augmentation separately (not the full pipeline):
```bash
python data/visualize_augmentation.py --mode individual
```

### View only multiple copies
Show only the full pipeline applied multiple times:
```bash
python data/visualize_augmentation.py --mode copies
```

### Save to file instead of displaying
```bash
python data/visualize_augmentation.py --save output.png
```
This generates two files:
- `output_individual.png` - Each augmentation separately
- `output_copies.png` - Multiple augmented copies

## What Augmentations Are Applied?

The script visualizes the augmentation pipeline defined in `data/augmentation.py`:

1. **Rotate** ± 20 degrees (70% probability)
2. **Horizontal Flip** (50% probability)
3. **Brightness/Contrast** adjustments ± 20% (60% probability)
4. **Blur** - either Gaussian or Motion blur (30% probability)
5. **Gaussian Noise** (30% probability)
6. **Resize** - all images resized to `IMAGE_SIZE` (256x256)

## Example Workflows

### 1. Check if augmentations are appropriate for your dataset
```bash
python data/visualize_augmentation.py --mode both --save aug_check.png
```
Then open `aug_check_individual.png` and `aug_check_copies.png` to verify transformations look reasonable.

### 2. See how many copies are generated
```bash
python data/visualize_augmentation.py --n-copies 20 --mode copies --save many_copies.png
```

### 3. Visualize with a specific problematic image
```bash
python data/visualize_augmentation.py --image data/processed/crops/train/broken_cap_001.png
```

## Tips

- The **individual** view is useful for understanding what each transform does
- The **copies** view shows how much variation is created in training data
- If augmentations look too aggressive or too weak, you can adjust parameters in `data/augmentation.py`
- Each time you run with the same settings, you may get different results due to randomness (except when viewing individual transforms)
