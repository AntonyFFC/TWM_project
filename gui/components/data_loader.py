"""Component for loading and selecting image data."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


# Note: RAW_IMAGES_DIR and CROPS_DIR will be used when
# implementing the browsing functionality


class DataLoaderComponent(ttk.Frame):
    """Frame for selecting image data sources.

    Features:
      - Browse and select from raw images
      - Browse and select from processed crops
      - Show image preview
      - Display image metadata (size, path, etc.)
    """

    def __init__(self, parent: ttk.Frame) -> None:
        """Initialize the data loader component.

        Args:
            parent: The parent frame.
        """
        super().__init__(parent)
        self.selected_image_path: Path | None = None

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the component UI widgets."""
        # Title
        title = tk.Label(
            self, text="Data Loader", font=("Arial", 14, "bold")
        )
        title.pack(pady=10)

        # Source selection frame
        source_frame = ttk.LabelFrame(self, text="Select Image Source")
        source_frame.pack(fill=tk.X, padx=10, pady=5)

        # Buttons for different sources
        ttk.Button(
            source_frame,
            text="Browse Raw Images",
            command=self._browse_raw_images,
        ).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(
            source_frame,
            text="Browse Processed Crops",
            command=self._browse_crops,
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Selected image info frame
        info_frame = ttk.LabelFrame(self, text="Selected Image Info")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.info_text = tk.Text(
            info_frame, height=10, width=60, state=tk.DISABLED
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Preview frame (placeholder)
        preview_frame = ttk.LabelFrame(self, text="Image Preview")
        preview_frame.pack(fill=tk.BOTH, padx=10, pady=5)

        tk.Label(
            preview_frame,
            text="[Image preview will be displayed here]",
            bg="gray",
            height=5,
        ).pack(fill=tk.BOTH, expand=True)

    def _browse_raw_images(self) -> None:
        """Browse and select a raw image."""
        # TODO: Implement image browsing and selection
        pass

    def _browse_crops(self) -> None:
        """Browse and select a processed crop."""
        # TODO: Implement crop browsing and selection
        pass

    def _update_info_display(self) -> None:
        """Update the info text display with selected image details."""
        # TODO: Implement info display update
        pass
