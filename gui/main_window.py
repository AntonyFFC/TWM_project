from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from gui.components.data_loader import DataLoaderComponent  # noqa: E402
from gui.components.augmentation_config import AugmentationConfigComponent  # noqa: E402
from gui.components.augmentation_viewer import AugmentationViewerComponent  # noqa: E402


class MainWindow:

    def __init__(self, root: tk.Tk) -> None:

        self.root = root
        self.root.title("TWM Project - Bottle Cap Augmentation Viewer")
        self.root.geometry("1200x800")

        self._create_menu_bar()

        self._create_notebook()

        self._create_status_bar()

    def _create_menu_bar(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(
            label="Reset Layout", command=self._reset_layout
        )

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.data_loader_tab = DataLoaderComponent(
            self.notebook,
            on_status=self.update_status,
        )
        self.notebook.add(self.data_loader_tab, text="Data Loader")
        
        self.augmentation_config_tab = AugmentationConfigComponent(
            self.notebook
        )
        self.notebook.add(self.augmentation_config_tab, text="Augmentation Config")

        self.augmentation_viewer_tab = AugmentationViewerComponent(
            self.notebook,
            data_loader=self.data_loader_tab,
            config_component=self.augmentation_config_tab
        )
        self.notebook.add(self.augmentation_viewer_tab, text="Augmentation Viewer")

    def _create_status_bar(self) -> None:
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 9),
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _reset_layout(self) -> None:
        self.root.geometry("1200x800")

    def _show_about(self) -> None:
        from tkinter import messagebox

        messagebox.showinfo(
            "About",
            "TWM Project - Bottle Cap Augmentation Viewer\n\n"
            "A GUI application for visualizing and testing "
            "image augmentation pipelines.",
        )

    def update_status(self, message: str) -> None:
        self.status_bar.config(text=message)
        self.root.update_idletasks()


def main() -> None:
    root = tk.Tk()
    _ = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
