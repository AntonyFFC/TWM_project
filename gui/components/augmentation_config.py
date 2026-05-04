"""Component for configuring augmentation parameters."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


class AugmentationConfigComponent(ttk.Frame):
    """Frame for configuring augmentation pipeline parameters.

    Features:
      - Enable/disable individual augmentations (rotate, flip, blur, etc.)
      - Adjust parameter ranges for each augmentation
      - Set probability for each transformation
      - Choose between individual vs. pipeline visualization
      - Set number of augmented copies to generate
    """

    def __init__(self, parent: ttk.Frame) -> None:
        """Initialize the augmentation config component.

        Args:
            parent: The parent frame.
        """
        super().__init__(parent)

        # Store current configuration
        self.config: dict[str, bool | float | int] = {
            "rotate_enabled": tk.BooleanVar(value=True),
            "rotate_limit": tk.IntVar(value=20),
            "rotate_prob": tk.DoubleVar(value=0.7),
            "h_flip_enabled": tk.BooleanVar(value=True),
            "h_flip_prob": tk.DoubleVar(value=0.5),
            "brightness_enabled": tk.BooleanVar(value=True),
            "brightness_limit": tk.DoubleVar(value=0.2),
            "contrast_limit": tk.DoubleVar(value=0.2),
            "brightness_prob": tk.DoubleVar(value=0.6),
            "blur_enabled": tk.BooleanVar(value=True),
            "blur_prob": tk.DoubleVar(value=0.3),
            "noise_enabled": tk.BooleanVar(value=True),
            "noise_prob": tk.DoubleVar(value=0.3),
            "n_copies": tk.IntVar(value=5),
            "view_mode": tk.StringVar(value="both"),
        }

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the component UI widgets."""
        # Title
        title = tk.Label(
            self, text="Augmentation Configuration", font=("Arial", 14, "bold")
        )
        title.pack(pady=10)

        # Create canvas with scrollbar for augmentation parameters
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=canvas.yview
        )
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Individual augmentation enable/disable checkboxes and parameters
        self._create_augmentation_controls(scrollable_frame)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom controls frame
        controls_frame = ttk.LabelFrame(self, text="Visualization Settings")
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        # Number of copies
        ttk.Label(controls_frame, text="Number of copies:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Spinbox(
            controls_frame,
            from_=1,
            to=20,
            textvariable=self.config["n_copies"],
            width=5,
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # View mode selection
        ttk.Label(controls_frame, text="View mode:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Combobox(
            controls_frame,
            textvariable=self.config["view_mode"],
            values=["individual", "copies", "both"],
            state="readonly",
            width=15,
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

    def _create_augmentation_controls(self, parent: ttk.Frame) -> None:
        """Create enable/disable and parameter controls for each augmentation.

        Args:
            parent: The parent frame for augmentation controls.
        """
        # Placeholder structure - will be filled with actual controls
        augmentations = [
            ("Rotation", "rotate"),
            ("Horizontal Flip", "h_flip"),
            ("Brightness/Contrast", "brightness"),
            ("Blur", "blur"),
            ("Gaussian Noise", "noise"),
        ]

        for aug_name, aug_key in augmentations:
            aug_frame = ttk.LabelFrame(parent, text=aug_name)
            aug_frame.pack(fill=tk.X, padx=5, pady=5)

            # Placeholder for controls
            tk.Label(
                aug_frame,
                text=f"[Parameters for {aug_name} - Coming Soon]",
                foreground="gray",
            ).pack(padx=5, pady=5)

    def get_config(self) -> dict:
        """Get current configuration values.

        Returns:
            Dictionary with current configuration.
        """
        # TODO: Extract actual values from widgets
        return {}

    def reset_to_defaults(self) -> None:
        """Reset all parameters to default values."""
        # TODO: Reset all config variables to defaults
        pass
