# GUI Architecture & Data Flow

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TWM Project GUI Application                  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                       Menu Bar                              ││
│  │   File   │   View   │   Help                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      Tabbed Interface                       ││
│  ├──────────────┬──────────────────┬─────────────────────────┤│
│  │ Data Loader  │ Augmentation Cfg │ Augmentation Viewer   ││
│  │              │                  │                        ││
│  │ ┌──────────┐ │ ┌──────────────┐ │ ┌────────────────────┐││
│  │ │ Browse   │ │ │ Rotate       │ │ │ Generate Button   │││
│  │ │ Images   │ │ │ Flip         │ │ │ Preview Canvas    │││
│  │ │ Preview  │ │ │ Brightness   │ │ │ Save Location     │││
│  │ │ Info     │ │ │ Blur         │ │ │ Status Log        │││
│  │ │          │ │ │ Noise        │ │ │                   │││
│  │ │          │ │ │ Settings     │ │ │                   │││
│  │ └──────────┘ │ └──────────────┘ │ └────────────────────┘││
│  └──────────────┴──────────────────┴─────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Status Bar: Ready                              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Component Interactions

```
Data Loader Component
├─ Browse Raw Images
├─ Browse Processed Crops
├─ Image Preview
└─ Metadata Display
   │
   └──────────────────────────────┐
                                  │
                                  ▼
              Augmentation Config Component
              ├─ Rotation Parameters
              ├─ Flip Parameters
              ├─ Brightness/Contrast Parameters
              ├─ Blur Parameters
              ├─ Noise Parameters
              └─ Visualization Settings
                 │
                 └──────────────────────────────┐
                                                │
                                                ▼
                        Augmentation Viewer Component
                        ├─ Generate Visualization
                        ├─ Display Preview
                        ├─ Save Location Selection
                        └─ Status Updates
```

## Data Flow

```
USER SELECTS IMAGE          USER CONFIGURES AUGMENTATION
       │                              │
       ▼                              ▼
┌─────────────┐          ┌────────────────────────┐
│ Data Loader │          │ Augmentation Config    │
│             │          │                        │
│ - Image Path│          │ - Rotation: ±20°       │
│ - Size      │          │ - Flip: 50%            │
│ - Metadata  │          │ - Brightness: ±20%     │
└─────┬───────┘          │ - Blur: 30%            │
      │                  │ - Noise: 30%           │
      │                  │ - Mode: both/copies/   │
      │                  │   individual           │
      │                  │ - N Copies: 5          │
      │                  └────────┬───────────────┘
      │                           │
      └───────────┬───────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Generate Button │
         └────────┬────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │ Call Visualization Script│
    │ (data/visualize_augment) │
    │ with config parameters   │
    └────────────┬─────────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │ Augmentation Viewer      │
    │                          │
    │ - Display Image Grid     │
    │ - Show Transformations   │
    │ - Save Results           │
    └──────────────────────────┘
```

## File Organization

```
gui/
├── __init__.py
│   └─ Package marker and documentation
│
├── main_window.py
│   └─ MainWindow class
│       ├─ __init__() - Initialize window
│       ├─ _create_menu_bar() - Build menu bar
│       ├─ _create_notebook() - Create tabs
│       ├─ _create_status_bar() - Status display
│       └─ update_status() - Update status messages
│
├── components/
│   ├── __init__.py
│   │
│   ├── data_loader.py
│   │   └─ DataLoaderComponent class
│   │       ├─ _create_widgets() - Build UI
│   │       ├─ _browse_raw_images() - File dialog
│   │       ├─ _browse_crops() - File dialog
│   │       └─ _update_info_display() - Show metadata
│   │
│   ├── augmentation_config.py
│   │   └─ AugmentationConfigComponent class
│   │       ├─ _create_widgets() - Build UI
│   │       ├─ _create_augmentation_controls() - Add controls
│   │       ├─ get_config() - Extract values
│   │       └─ reset_to_defaults() - Reset parameters
│   │
│   └── augmentation_viewer.py
│       └─ AugmentationViewerComponent class
│           ├─ _create_widgets() - Build UI
│           ├─ _generate_visualization() - Call script
│           ├─ _choose_save_location() - Save dialog
│           └─ _update_status() - Log messages
│
├── utils/
│   └── __init__.py
│       └─ (Future utility functions)
│
└── README.md
    └─ This documentation
```

## State Management

### Current Implementation
- Configuration stored in component state variables (tk.IntVar, tk.DoubleVar, etc.)
- Each component manages its own state
- Status updates passed to main window

### Future Enhancements
- Central state manager for cross-component communication
- Configuration presets/profiles
- Undo/redo functionality
- Session save/load

## Integration Points with Existing Code

### Connects To:
1. **data/visualize_augmentation.py**
   - Used by AugmentationViewerComponent
   - Passes parameters from config
   - Receives generated images

2. **data/dataset_loader.py**
   - Used by DataLoaderComponent
   - Loads image samples
   - Displays metadata

3. **data/augmentation.py**
   - Parameters configured in GUI
   - Applied in visualization script

4. **config.py**
   - RAW_IMAGES_DIR
   - CROPS_DIR
   - IMAGE_SIZE

## Future Extensions

```
┌─────────────────────────────────────────┐
│     Main Application Controller         │
├─────────────────────────────────────────┤
│ ├─ Augmentation Viewer (Current)       │
│ ├─ Data Processing Pipeline            │
│ ├─ Training Control                    │
│ │  ├─ Classical Methods                │
│ │  └─ ML Methods                       │
│ ├─ Evaluation & Comparison             │
│ └─ Results Dashboard                   │
└─────────────────────────────────────────┘
```
