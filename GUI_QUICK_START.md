# Quick Start Guide for GUI

## 🚀 Launch the GUI

```powershell
cd c:\Users\antek\Documents\programowanie\studia\TWM_project
python run_gui.py
```

## 📋 What You'll See

```
┌────────────────────────────────────────────────────────────────────┐
│  TWM Project - Bottle Cap Augmentation Viewer              [_□✕]   │
├────────────────────────────────────────────────────────────────────┤
│ File    View    Help                                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─ Data Loader ─┬─ Augmentation Config ─┬─ Augmentation Viewer ┐ │
│  │               │                       │                      │ │
│  │ [Data Loader  │ [Augmentation Config  │ [Augmentation        │ │
│  │  Tab Content] │  Tab Content]         │  Viewer Tab Content] │ │
│  │               │                       │                      │ │
│  └───────────────┴───────────────────────┴──────────────────────┘ │
│                                                                     │
│ Ready                                                               │
└────────────────────────────────────────────────────────────────────┘
```

## 📚 Documentation Files (In Root Directory)

1. **GUI_SETUP_SUMMARY.md** ← Start here! Overview of what was created
2. **gui/README.md** - What each component does
3. **gui/ARCHITECTURE.md** - How components interact
4. **gui/IMPLEMENTATION.md** - How to implement functionality

## 🎯 Current State

| Aspect | Status |
|--------|--------|
| Folder structure | ✅ Complete |
| File organization | ✅ Complete |
| Window layout | ✅ Complete |
| Menu bar | ✅ Complete |
| Tabs & panels | ✅ Complete |
| Component classes | ✅ Complete |
| UI placeholders | ✅ Complete |
| Functionality | ⏳ Ready to implement |

## 🛠️ Implementing Features

Once you're ready to implement, follow **gui/IMPLEMENTATION.md**

The order is:
1. **Data Loader** - Load and preview images
2. **Augmentation Config** - Configure parameters
3. **Augmentation Viewer** - Generate and save visualizations

Each is independent, so you can work on them separately.

## 📁 File Locations

```
Root/
├── run_gui.py                    ← Launch the app
├── GUI_SETUP_SUMMARY.md          ← Overview
└── gui/
    ├── main_window.py            ← Main window class
    ├── README.md                 ← Component docs
    ├── ARCHITECTURE.md           ← System design
    ├── IMPLEMENTATION.md         ← Implementation guide
    ├── components/
    │   ├── data_loader.py
    │   ├── augmentation_config.py
    │   ├── augmentation_viewer.py
    │   └── __init__.py
    ├── utils/
    │   └── __init__.py
    └── __init__.py
```

## 🔍 Code Structure

### Main Window (`gui/main_window.py`)
```python
MainWindow
├── _create_menu_bar()        # File, View, Help menus
├── _create_notebook()        # Tabbed interface
├── _create_status_bar()      # Bottom status bar
└── update_status(message)    # Update status
```

### Data Loader (`gui/components/data_loader.py`)
```python
DataLoaderComponent(ttk.Frame)
├── _browse_raw_images()      # TODO: Browse dialog
├── _browse_crops()           # TODO: Browse dialog
└── _update_info_display()    # TODO: Show metadata
```

### Augmentation Config (`gui/components/augmentation_config.py`)
```python
AugmentationConfigComponent(ttk.Frame)
├── _create_augmentation_controls()  # TODO: Add sliders
├── get_config()                     # TODO: Extract values
└── reset_to_defaults()              # TODO: Reset to defaults
```

### Augmentation Viewer (`gui/components/augmentation_viewer.py`)
```python
AugmentationViewerComponent(ttk.Frame)
├── _generate_visualization()   # TODO: Call script
├── _choose_save_location()     # TODO: File dialog
└── _update_status()            # TODO: Log messages
```

## 🔗 Connections

When implemented, the flow will be:

```
User selects image       User adjusts parameters
    ↓                           ↓
Data Loader          Augmentation Config
    │                          │
    └──────────────┬───────────┘
                   │
                   ↓
           User clicks "Generate"
                   │
                   ↓
        Augmentation Viewer
         calls visualization
              script
                   │
                   ↓
          Displays result
          & saves to disk
```

## 💡 Tips

- **Don't modify the outline** - Stub implementations are safe
- **Check IMPLEMENTATION.md first** - Has all the details
- **Test each tab separately** - No dependencies between them
- **Keep functions focused** - One thing per function
- **Use TODO comments** - Mark what still needs implementation

## ✅ Success Criteria

The GUI outline is successful when:
- [x] Window launches without errors
- [x] All three tabs appear
- [x] Menu bar is functional (About dialog works)
- [x] Status bar updates when you click things
- [x] No circular imports or missing dependencies

## 🎓 Learning Path

1. **Understand the structure** - Read GUI_SETUP_SUMMARY.md
2. **See the design** - Read gui/ARCHITECTURE.md
3. **Learn how to extend** - Read gui/IMPLEMENTATION.md
4. **Implement one component** - Start with data_loader.py
5. **Test & iterate** - Run GUI, add features, repeat

---

**Happy coding! Your GUI foundation is ready. 🚀**
