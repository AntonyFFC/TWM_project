from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
import sys
from typing import Any
import io
import cv2
import numpy as np
import albumentations as A
import matplotlib
import matplotlib.pyplot as plt

try:
    from PIL import Image as PILImage, ImageTk
except ImportError:
    raise ImportError("Please install Pillow to display images: pip install Pillow")

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).resolve().parents[2]))


class AugmentationViewerComponent(ttk.Frame):

    def __init__(self, parent: ttk.Frame, data_loader: Any, config_component: Any) -> None:
        super().__init__(parent)

        self.data_loader = data_loader
        self.config_component = config_component

        self.output_path: Path | None = None
        self.photo_image: tk.PhotoImage | None = None
        self._create_widgets()

    def _create_widgets(self) -> None:
        title = tk.Label(
            self, text="Augmentation Viewer", font=("Arial", 14, "bold")
        )
        title.pack(pady=10)

        control_frame = ttk.LabelFrame(
            self, text="Visualization Controls"
        )
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            control_frame,
            text="Generate Visualization",
            command=self._generate_visualization,
        ).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(
            control_frame,
            text="Choose Save Location",
            command=self._choose_save_location,
        ).pack(side=tk.LEFT, padx=5, pady=5)

        path_frame = ttk.Frame(control_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(path_frame, text="Output path:").pack(
            side=tk.LEFT, padx=5
        )
        self.path_label = tk.Label(
            path_frame, text="(Not selected)", foreground="gray"
        )
        self.path_label.pack(side=tk.LEFT, padx=5)

        display_frame = ttk.LabelFrame(self, text="Visualization Preview")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

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

        status_frame = ttk.LabelFrame(self, text="Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_text = tk.Text(
            status_frame, height=4, width=60, state=tk.DISABLED
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _generate_visualization(self) -> None:
        if not self.data_loader or not self.config_component:
            self._update_status("Error: Data loader or config component not available.")
            return
        
        image_path = self.data_loader.get_selected_image_path()
        if not image_path:
            self._update_status("Error: No image selected in data loader.")
            return
        
        config = self.config_component.get_config()
        if not config:
            self._update_status("Error: No augmentation configuration available.")
            return
        self._update_status("Generating visualization...")

        try:
            generated_pil_image = self._create_visualization(image_path, config)

            if self.output_path:
                generated_pil_image.save(self.output_path)
                self._update_status(f"Result saved automatically to: {self.output_path}")

            self._display_image(generated_pil_image)
            self._update_status("Visualization generated successfully.")
        
        except Exception as e:
            self._update_status(f"Error during visualization: {e}")
            messagebox.showerror("Visualization Error", f"An error occurred while generating the visualization:\n{e}")

    def _display_image(self, pil_image: PILImage.Image) -> None:
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width, canvas_height = 800, 600

        display_img = pil_image.copy()
        
        display_img.thumbnail((canvas_width, canvas_height), PILImage.Resampling.LANCZOS)
        
        self.photo_image = ImageTk.PhotoImage(display_img)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor=tk.CENTER,
            image=self.photo_image
        )

    def _on_canvas_resize(self, event: tk.Event) -> None:
        if self.photo_image:
            self.canvas.coords(1, event.width // 2, event.height // 2)
        elif self.canvas.find_withtag("placeholder"):
            self.canvas.coords("placeholder", event.width // 2, event.height // 2)
    
    def _create_visualization(self, image_path: str, config: dict) -> PILImage.Image:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image at {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mode = config.get("view_mode", "both")
        n_copies = config.get("n_copies", 5)

        transforms_info = []

        if config.get("rotate_enabled", True):
            p = config.get("rotate_prob", 0.7)
            limit = config.get("rotate_limit", 20)
            transforms_info.append((
                f"Rotate (±{limit}°)", 
                lambda prob, l=limit: A.Rotate(limit=l, border_mode=cv2.BORDER_REFLECT_101, p=prob),
                p
            ))

        if config.get("h_flip_enabled", True):
            p = config.get("h_flip_prob", 0.5)
            transforms_info.append((
                "Horizontal Flip", 
                lambda prob: A.HorizontalFlip(p=prob),
                p
            ))

        if config.get("brightness_enabled", True):
            p = config.get("brightness_prob", 0.6)
            b_limit = config.get("brightness_limit", 0.2)
            c_limit = config.get("contrast_limit", 0.2)
            transforms_info.append((
                "Bright / Contrast", 
                lambda prob, bl=b_limit, cl=c_limit: A.RandomBrightnessContrast(brightness_limit=bl, contrast_limit=cl, p=prob),
                p
            ))

        if config.get("blur_enabled", True):
            p = config.get("blur_prob", 0.3)
            b_limit = config.get("blur_limit", 5)
            b_limit = max(3, b_limit if b_limit % 2 != 0 else b_limit + 1)
            transforms_info.append((
                f"Gauss Blur (k={b_limit})", 
                lambda prob, bl=b_limit: A.GaussianBlur(blur_limit=(3, bl), p=prob),
                p
            ))

        if config.get("m_blur_enabled", True):
            p = config.get("m_blur_prob", 0.3)
            mb_limit = config.get("m_blur_limit", 5)
            mb_limit = max(3, mb_limit if mb_limit % 2 != 0 else mb_limit + 1)
            transforms_info.append((
                f"Motion Blur (k={mb_limit})", 
                lambda prob, mbl=mb_limit: A.MotionBlur(blur_limit=(3, mbl), p=prob),
                p
            ))

        if config.get("noise_enabled", True):
            p = config.get("noise_prob", 0.3)
            n_std = config.get("noise_std", 0.1)
            transforms_info.append((
                f"Gauss Noise (s={n_std:.2f})", 
                lambda prob, std=n_std: A.GaussNoise(std_range=(0.0, std), p=prob), 
                p
            ))

        rows = 2 if mode == "both" else 1
        fig = plt.figure(figsize=(15, 4 * rows))
        
        gs = fig.add_gridspec(rows, 1)
        current_row_idx = 0
        
        def format_ax(ax, image, title):
            ax.imshow(image)
            ax.set_title(title, fontsize=10, fontweight="bold" if "Original" in title else "normal")
            ax.axis("off")

        if mode in ("individual", "both"):
            cols = len(transforms_info) + 1
            gs_row = gs[current_row_idx].subgridspec(1, cols)
            
            ax = fig.add_subplot(gs_row[0, 0])
            format_ax(ax, img_rgb, "Original")
            
            for i, (name, transform_factory, _) in enumerate(transforms_info, 1):
                ax = fig.add_subplot(gs_row[0, i])
                try:
                    transform = transform_factory(1.0)
                    augmented = transform(image=img_rgb)["image"]
                    format_ax(ax, augmented, name)
                except Exception as e:
                    print(f"Transform {name} failed: {e}")
                    format_ax(ax, img_rgb, f"{name}\n(Failed)")
            
            current_row_idx += 1

        if mode in ("copies", "both"):
            pipeline_transforms = [factory(prob) for _, factory, prob in transforms_info]
            pipeline = A.Compose(pipeline_transforms)
            
            cols = n_copies + 1
            gs_row = gs[current_row_idx].subgridspec(1, cols)
            
            ax = fig.add_subplot(gs_row[0, 0])
            format_ax(ax, img_rgb, "Original")
            
            for i in range(1, cols):
                ax = fig.add_subplot(gs_row[0, i])
                try:
                    augmented = pipeline(image=img_rgb)["image"]
                    format_ax(ax, augmented, f"Pipeline Copy #{i}")
                except Exception as e:
                    print(f"Pipeline Copy {i} failed: {e}")
                    format_ax(ax, img_rgb, "Failed")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor='#2b2b2b')
        plt.close(fig)
        buf.seek(0)

        return PILImage.open(buf)

    def _choose_save_location(self) -> None:
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
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
