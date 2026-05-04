"""Component for viewing augmentation results."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


class AugmentationViewerComponent(ttk.Frame):
    """Frame for displaying and saving augmentation visualizations.

    Features:
      - Display generated augmentation visualizations
      - Select output format and location
      - Generate visualizations using configured parameters
      - Save results to disk
    """

    def __init__(self, parent: ttk.Frame) -> None:
        """Initialize the augmentation viewer component.

        Args:
            parent: The parent frame.
        """
        super().__init__(parent)

        self.output_path: Path | None = None
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the component UI widgets."""
        # Title
        title = tk.Label(
            self, text="Augmentation Viewer", font=("Arial", 14, "bold")
        )
        title.pack(pady=10)

        # Control frame
        control_frame = ttk.LabelFrame(
            self, text="Visualization Controls"
        )
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Generate button
        ttk.Button(
            control_frame,
            text="Generate Visualization",
            command=self._generate_visualization,
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Save button
        ttk.Button(
            control_frame,
            text="Choose Save Location",
            command=self._choose_save_location,
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Output path display
        path_frame = ttk.Frame(control_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(path_frame, text="Output path:").pack(
            side=tk.LEFT, padx=5
        )
        self.path_label = tk.Label(
            path_frame, text="(Not selected)", foreground="gray"
        )
        self.path_label.pack(side=tk.LEFT, padx=5)

        # Visualization display area
        display_frame = ttk.LabelFrame(self, text="Visualization Preview")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Canvas for image display (placeholder)
        self.canvas = tk.Canvas(
            display_frame, bg="gray", height=400
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas.create_text(
            self.canvas.winfo_reqwidth() // 2,
            self.canvas.winfo_reqheight() // 2,
            text="[Visualization will be displayed here]",
            fill="white",
            font=("Arial", 12),
        )

        # Status frame
        status_frame = ttk.LabelFrame(self, text="Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_text = tk.Text(
            status_frame, height=4, width=60, state=tk.DISABLED
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _generate_visualization(self) -> None:
        """Generate the augmentation visualization."""
        # TODO: Call visualization function with current config
        self._update_status("Generating visualization...")

    def _choose_save_location(self) -> None:
        """Choose where to save the visualization."""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if path:
            self.output_path = Path(path)
            self.path_label.config(
                text=str(self.output_path), foreground="black"
            )
            self._update_status(f"Output path set to: {path}")

    def _update_status(self, message: str) -> None:
        """Update the status text.

        Args:
            message: The message to display.
        """
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
