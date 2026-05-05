from __future__ import annotations

import tkinter as tk
from typing import Callable
from pathlib import Path
from tkinter import filedialog, ttk
import sys

from PIL import Image, ImageTk

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import CROPS_DIR, RAW_IMAGES_DIR  # noqa: E402


class DataLoaderComponent(ttk.Frame):

    def __init__(
        self,
        parent: ttk.Frame,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.selected_image_path: Path | None = None
        self.selected_source: str = "Unknown"
        self.on_status = on_status
        self.preview_photo: ImageTk.PhotoImage | None = None

        self._create_widgets()

    def _create_widgets(self) -> None:
        title = tk.Label(
            self, text="Data Loader", font=("Arial", 14, "bold")
        )
        title.pack(pady=10)

        source_frame = ttk.LabelFrame(self, text="Select Image Source")
        source_frame.pack(fill=tk.X, padx=10, pady=5)

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

        content_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        content_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        info_frame = ttk.LabelFrame(content_pane, text="Selected Image Info")
        info_container = ttk.Frame(info_frame)
        info_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        info_scroll_y = ttk.Scrollbar(info_container, orient=tk.VERTICAL)
        self.info_text = tk.Text(
            info_container,
            height=12,
            width=50,
            state=tk.DISABLED,
            yscrollcommand=info_scroll_y.set,
            wrap=tk.WORD,
        )
        info_scroll_y.config(command=self.info_text.yview)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        preview_frame = ttk.LabelFrame(content_pane, text="Image Preview")

        self.preview_label = tk.Label(
            preview_frame,
            text="[Image preview will be displayed here]",
            bg="#2f2f2f",
            fg="white",
            anchor=tk.CENTER,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        content_pane.add(info_frame, weight=1)
        content_pane.add(preview_frame, weight=2)

    def _browse_raw_images(self) -> None:
        self._browse_image(
            initial_dir=RAW_IMAGES_DIR,
            source_label="Raw Images",
        )

    def _browse_crops(self) -> None:
        self._browse_image(
            initial_dir=CROPS_DIR,
            source_label="Processed Crops",
        )

    def _browse_image(self, initial_dir: Path, source_label: str) -> None:
        if not initial_dir.exists():
            self._notify_status(
                f"Directory not found: {initial_dir}. Falling back to project root."
            )
            initial_dir = Path(__file__).resolve().parents[2]

        selected = filedialog.askopenfilename(
            title=f"Select image from {source_label}",
            initialdir=str(initial_dir),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            self._notify_status("Image selection cancelled.")
            return

        self.selected_image_path = Path(selected)
        self.selected_source = source_label
        self._update_info_display()
        self._update_preview_display()
        self._notify_status(f"Loaded image: {self.selected_image_path.name}")

    def _update_info_display(self) -> None:
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)

        if self.selected_image_path is None:
            self.info_text.insert(tk.END, "No image selected.\n")
            self.info_text.config(state=tk.DISABLED)
            return

        try:
            with Image.open(self.selected_image_path) as img:
                width, height = img.size
                mode = img.mode
            file_size_kb = self.selected_image_path.stat().st_size / 1024.0
        except Exception as exc:
            self.info_text.insert(
                tk.END,
                f"Failed to read image metadata:\n{exc}\n",
            )
            self.info_text.config(state=tk.DISABLED)
            return

        metadata = (
            f"Source: {self.selected_source}\n"
            f"File name: {self.selected_image_path.name}\n"
            f"Full path: {self.selected_image_path}\n"
            f"Resolution: {width} x {height}\n"
            f"Color mode: {mode}\n"
            f"File size: {file_size_kb:.2f} KB\n"
        )
        self.info_text.insert(tk.END, metadata)
        self.info_text.config(state=tk.DISABLED)

    def _update_preview_display(self) -> None:
        if self.selected_image_path is None:
            return

        try:
            with Image.open(self.selected_image_path) as img:
                preview = img.convert("RGB")
                preview.thumbnail((520, 320), Image.Resampling.LANCZOS)
                self.preview_photo = ImageTk.PhotoImage(preview)
        except Exception as exc:
            self.preview_label.config(
                text=f"Failed to load preview:\n{exc}",
                image="",
                bg="#5a5a5a",
                fg="white",
            )
            return

        self.preview_label.config(
            image=self.preview_photo,
            text="",
            bg="#1f1f1f",
        )

    def _notify_status(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)

    def get_selected_image_path(self) -> Path | None:
        return self.selected_image_path
