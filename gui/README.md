# GUI Application Structure

## Overview
This folder contains a tkinter-based GUI application for the TWM project. It provides an interactive interface for:
- Loading and selecting image data
- Configuring augmentation parameters
- Visualizing the augmentation process
- Saving visualization results

## Folder Structure

```
gui/
├── __init__.py                 # GUI package initialization
├── main_window.py              # Main window with tabbed interface
├── components/                 # Individual GUI components
│   ├── __init__.py
│   ├── data_loader.py         # Image selection and preview
│   ├── augmentation_config.py  # Parameter configuration
│   └── augmentation_viewer.py  # Visualization display and saving
└── utils/                      # Utility functions
    └── __init__.py
```

## Main Components

### 1. **main_window.py** - Main Application Window
- Root window with menu bar and tabbed interface
- Coordinates all components
- Status bar for user feedback
- Menu options: File, View, Help

### 2. **components/data_loader.py** - Data Selection
- Browse raw images from dataset
- Browse processed crops from training splits
- Image preview with metadata display
- Select images for augmentation testing

### 3. **components/augmentation_config.py** - Parameter Configuration
- Enable/disable individual augmentations:
  - Rotation (±20° by default)
  - Horizontal flip
  - Brightness/Contrast
  - Blur (Gaussian or Motion)
  - Gaussian Noise
- Adjust probability and parameters for each transform
- Configure visualization mode (individual, copies, both)
- Set number of augmented copies to generate

### 4. **components/augmentation_viewer.py** - Visualization & Export
- Generate augmentation visualizations
- Display preview of results
- Choose output file location
- Save visualizations to disk
- Status messages for user feedback

## Usage

### Launch the GUI
```bash
python run_gui.py
```

### Current State
The GUI is in **outline/skeleton phase**:
- ✅ Window structure and layout
- ✅ Tab organization
- ✅ Component placeholders
- ❌ Functional implementations (in progress)
- ❌ Data binding and callbacks (to be added)

## Development Roadmap

### Phase 1: Core Functionality (Current)
- [x] Create folder structure
- [x] Create main window outline
- [x] Create component outlines
- [ ] Implement data loader functionality
- [ ] Implement augmentation config controls
- [ ] Implement visualization generation

### Phase 2: Integration
- [ ] Connect components to existing `data/visualize_augmentation.py`
- [ ] Implement config → visualization pipeline
- [ ] Add real-time preview
- [ ] Add parameter validation

### Phase 3: Polish & Features
- [ ] Add drag-and-drop image loading
- [ ] Add preset configurations
- [ ] Add favorites/bookmarks
- [ ] Add batch processing
- [ ] Add advanced augmentation options

### Phase 4: Full Application Integration
- [ ] Training pipeline control
- [ ] Classical methods interface
- [ ] ML training interface
- [ ] Results comparison view
- [ ] Export reports

## Notes

- Uses **tkinter** (built-in, no extra dependencies)
- Component-based architecture for easy maintenance
- Status bar for all user operations
- Designed for future expansion to full application control

## Implementation Notes

### TODO Markers
Search for `# TODO:` comments in the code for specific implementation tasks.

### Key Functions to Implement
1. `DataLoaderComponent._browse_raw_images()` - Browse file dialog
2. `DataLoaderComponent._update_info_display()` - Display image metadata
3. `AugmentationConfigComponent.get_config()` - Extract UI values
4. `AugmentationViewerComponent._generate_visualization()` - Call visualization script
5. `AugmentationViewerComponent._update_status()` - Status updates

### Data Flow (When Implemented)
```
Data Loader → Augmentation Config → Generate → Augmentation Viewer → Save
   (select)      (set params)       (run)      (display)            (export)
```

## Testing

To test the GUI without implementations:
```bash
python run_gui.py
```

You should see:
- Main window with title "TWM Project - Bottle Cap Augmentation Viewer"
- Three tabs: "Data Loader", "Augmentation Config", "Augmentation Viewer"
- Menu bar with File, View, and Help menus
- Status bar at the bottom
