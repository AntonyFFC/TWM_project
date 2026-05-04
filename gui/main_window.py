"""Main GUI window for the TWM project application.

This module contains the root window structure and layout,
coordinating all GUI components.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


class MainWindow:
    """Main application window with tabs for different features."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the main window.

        Args:
            root: The root tkinter window.
        """
        self.root = root
        self.root.title("TWM Project - Bottle Cap Augmentation Viewer")
        self.root.geometry("1200x800")

        # Create menu bar
        self._create_menu_bar()

        # Create main notebook (tabbed interface)
        self._create_notebook()

        # Create status bar
        self._create_status_bar()

    def _create_menu_bar(self) -> None:
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(
            label="Reset Layout", command=self._reset_layout
        )

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_notebook(self) -> None:
        """Create the main tabbed interface."""
        # Note: Components will be imported and added here
        # to avoid circular imports during development
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Placeholder tabs - will be replaced with actual components
        placeholder_frame_1 = ttk.Frame(self.notebook)
        self.notebook.add(placeholder_frame_1, text="Data Loader")
        tk.Label(
            placeholder_frame_1,
            text="[Data Loader Tab - Coming Soon]",
            font=("Arial", 14),
        ).pack(pady=50)

        placeholder_frame_2 = ttk.Frame(self.notebook)
        self.notebook.add(placeholder_frame_2, text="Augmentation Config")
        tk.Label(
            placeholder_frame_2,
            text="[Augmentation Config Tab - Coming Soon]",
            font=("Arial", 14),
        ).pack(pady=50)

        placeholder_frame_3 = ttk.Frame(self.notebook)
        self.notebook.add(placeholder_frame_3, text="Augmentation Viewer")
        tk.Label(
            placeholder_frame_3,
            text="[Augmentation Viewer Tab - Coming Soon]",
            font=("Arial", 14),
        ).pack(pady=50)

    def _create_status_bar(self) -> None:
        """Create the status bar at the bottom of the window."""
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
        """Reset the window layout to default."""
        self.root.geometry("1200x800")

    def _show_about(self) -> None:
        """Show the about dialog."""
        from tkinter import messagebox

        messagebox.showinfo(
            "About",
            "TWM Project - Bottle Cap Augmentation Viewer\n\n"
            "A GUI application for visualizing and testing "
            "image augmentation pipelines.",
        )

    def update_status(self, message: str) -> None:
        """Update the status bar message.

        Args:
            message: The message to display.
        """
        self.status_bar.config(text=message)
        self.root.update_idletasks()


def main() -> None:
    """Run the main GUI application."""
    root = tk.Tk()
    _ = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
