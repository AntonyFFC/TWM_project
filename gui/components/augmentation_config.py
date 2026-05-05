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

        # Reset button
        ttk.Button(
            controls_frame,
            text="Reset to Defaults",
            command=self.reset_to_defaults,
        ).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)

    def _create_augmentation_controls(self, parent: ttk.Frame) -> None:
        """Create enable/disable and parameter controls for each augmentation.

        Args:
            parent: The parent frame for augmentation controls.
        """
        # Rotation
        rotate_frame = ttk.LabelFrame(parent, text="Rotation")
        rotate_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(
            rotate_frame,
            text="Enable Rotation",
            variable=self.config["rotate_enabled"],
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(rotate_frame, text="Limit (degrees):").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Spinbox(
            rotate_frame,
            from_=0,
            to=90,
            textvariable=self.config["rotate_limit"],
            width=5,
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(rotate_frame, text="Probability:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            rotate_frame,
            from_=0.0,
            to=1.0,
            variable=self.config["rotate_prob"],
            orient=tk.HORIZONTAL,
        ).grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self._add_prob_label(rotate_frame, self.config["rotate_prob"], 2)

        # Horizontal Flip
        flip_frame = ttk.LabelFrame(parent, text="Horizontal Flip")
        flip_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(
            flip_frame,
            text="Enable Horizontal Flip",
            variable=self.config["h_flip_enabled"],
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(flip_frame, text="Probability:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            flip_frame,
            from_=0.0,
            to=1.0,
            variable=self.config["h_flip_prob"],
            orient=tk.HORIZONTAL,
        ).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self._add_prob_label(flip_frame, self.config["h_flip_prob"], 1)

        # Brightness/Contrast
        bright_frame = ttk.LabelFrame(parent, text="Brightness / Contrast")
        bright_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(
            bright_frame,
            text="Enable Brightness/Contrast",
            variable=self.config["brightness_enabled"],
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(bright_frame, text="Brightness Limit:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            bright_frame,
            from_=0.0,
            to=1.0,
            variable=self.config["brightness_limit"],
            orient=tk.HORIZONTAL,
        ).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self._add_pct_label(bright_frame, self.config["brightness_limit"], 1)

        ttk.Label(bright_frame, text="Contrast Limit:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            bright_frame,
            from_=0.0,
            to=1.0,
            variable=self.config["contrast_limit"],
            orient=tk.HORIZONTAL,
        ).grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self._add_pct_label(bright_frame, self.config["contrast_limit"], 2)

        ttk.Label(bright_frame, text="Probability:").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            bright_frame,
            from_=0.0,
            to=1.0,
            variable=self.config["brightness_prob"],
            orient=tk.HORIZONTAL,
        ).grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self._add_prob_label(bright_frame, self.config["brightness_prob"], 3)

        # Blur
        blur_frame = ttk.LabelFrame(parent, text="Blur")
        blur_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(
            blur_frame,
            text="Enable Blur",
            variable=self.config["blur_enabled"],
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(blur_frame, text="Probability:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            blur_frame,
            from_=0.0,
            to=1.0,
            variable=self.config["blur_prob"],
            orient=tk.HORIZONTAL,
        ).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self._add_prob_label(blur_frame, self.config["blur_prob"], 1)

        # Gaussian Noise
        noise_frame = ttk.LabelFrame(parent, text="Gaussian Noise")
        noise_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(
            noise_frame,
            text="Enable Gaussian Noise",
            variable=self.config["noise_enabled"],
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(noise_frame, text="Probability:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            noise_frame,
            from_=0.0,
            to=1.0,
            variable=self.config["noise_prob"],
            orient=tk.HORIZONTAL,
        ).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self._add_prob_label(noise_frame, self.config["noise_prob"], 1)

    def _add_prob_label(
        self, parent: tk.Widget, var: tk.DoubleVar, row: int
    ) -> None:
        """Add a label showing probability percentage.

        Args:
            parent: Parent widget.
            var: The DoubleVar to display.
            row: Grid row.
        """
        label = tk.Label(parent, text="", width=5, anchor=tk.W)
        label.grid(row=row, column=2, padx=5, pady=5)

        def update_label(*_: object) -> None:
            label.config(text=f"{var.get() * 100:.0f}%")

        var.trace_add("write", update_label)
        update_label()

    def _add_pct_label(
        self, parent: tk.Widget, var: tk.DoubleVar, row: int
    ) -> None:
        """Add a label showing percentage value.

        Args:
            parent: Parent widget.
            var: The DoubleVar to display.
            row: Grid row.
        """
        label = tk.Label(parent, text="", width=5, anchor=tk.W)
        label.grid(row=row, column=2, padx=5, pady=5)

        def update_label(*_: object) -> None:
            label.config(text=f"{var.get() * 100:.0f}%")

        var.trace_add("write", update_label)
        update_label()

    def get_config(self) -> dict:
        """Get current configuration values.

        Returns:
            Dictionary with current configuration.
        """
        return {
            "rotate_enabled": self.config["rotate_enabled"].get(),
            "rotate_limit": self.config["rotate_limit"].get(),
            "rotate_prob": self.config["rotate_prob"].get(),
            "h_flip_enabled": self.config["h_flip_enabled"].get(),
            "h_flip_prob": self.config["h_flip_prob"].get(),
            "brightness_enabled": self.config["brightness_enabled"].get(),
            "brightness_limit": self.config["brightness_limit"].get(),
            "contrast_limit": self.config["contrast_limit"].get(),
            "brightness_prob": self.config["brightness_prob"].get(),
            "blur_enabled": self.config["blur_enabled"].get(),
            "blur_prob": self.config["blur_prob"].get(),
            "noise_enabled": self.config["noise_enabled"].get(),
            "noise_prob": self.config["noise_prob"].get(),
            "n_copies": self.config["n_copies"].get(),
            "view_mode": self.config["view_mode"].get(),
        }

    def reset_to_defaults(self) -> None:
        """Reset all parameters to default values."""
        defaults = {
            "rotate_enabled": True,
            "rotate_limit": 20,
            "rotate_prob": 0.7,
            "h_flip_enabled": True,
            "h_flip_prob": 0.5,
            "brightness_enabled": True,
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "brightness_prob": 0.6,
            "blur_enabled": True,
            "blur_prob": 0.3,
            "noise_enabled": True,
            "noise_prob": 0.3,
            "n_copies": 5,
            "view_mode": "both",
        }
        for key, value in defaults.items():
            self.config[key].set(value)
