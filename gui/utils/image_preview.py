"""Shared helpers for displaying PIL images on tk labels."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


def show_image_on_label(
    label: tk.Label,
    image_path: Path | None,
    *,
    max_size: tuple[int, int] = (480, 320),
    placeholder: str = "No image",
) -> ImageTk.PhotoImage | None:
    if image_path is None or not image_path.exists():
        label.config(image="", text=placeholder, bg="#2f2f2f", fg="white")
        return None

    try:
        with Image.open(image_path) as img:
            preview = img.convert("RGB")
            preview.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
    except Exception as exc:
        label.config(image="", text=f"Failed to load:\n{exc}", bg="#5a5a5a", fg="white")
        return None

    label.config(image=photo, text="", bg="#1f1f1f")
    return photo


def show_bgr_on_label(
    label: tk.Label,
    bgr_image,
    *,
    max_size: tuple[int, int] = (480, 320),
    placeholder: str = "No image",
):
    """Display an OpenCV BGR ndarray on a tk label."""
    import cv2
    from PIL import Image

    if bgr_image is None:
        label.config(image="", text=placeholder, bg="#2f2f2f", fg="white")
        return None

    try:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(rgb)
        preview.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(preview)
    except Exception as exc:
        label.config(image="", text=f"Failed to render:\n{exc}", bg="#5a5a5a", fg="white")
        return None

    label.config(image=photo, text="", bg="#1f1f1f")
    return photo
