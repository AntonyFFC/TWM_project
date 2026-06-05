from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import RESULTS_DIR  # noqa: E402
from gui.services.classical2_params_state import Classical2ParamsState  # noqa: E402
from gui.components.augmentation_config import AugmentationConfigComponent  # noqa: E402
from gui.components.augmentation_viewer import AugmentationViewerComponent  # noqa: E402
from gui.components.data_loader import DataLoaderComponent  # noqa: E402
from gui.components.classical2_config_analyze import Classical2ConfigAnalyzeComponent  # noqa: E402
from gui.components.classical2_evaluation import Classical2EvaluationComponent  # noqa: E402
from gui.components.export_results import ExportResultsComponent  # noqa: E402
from gui.components.methods_panel import MethodsPanelComponent  # noqa: E402
from gui.components.results_comparison import ResultsComparisonComponent  # noqa: E402
from gui.components.training_pipeline import TrainingPipelineComponent  # noqa: E402


class MainWindow:

    def __init__(self, root: tk.Tk) -> None:

        self.root = root
        self.root.title("TWM Project - Bottle Cap Classification")
        self.root.geometry("1280x860")

        self._create_menu_bar()

        self._create_status_bar()

        self._create_notebook()

    def _create_menu_bar(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open results folder", command=self._open_results_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(
            label="Open classical errors folder", command=self._open_classical_errors_folder
        )
        view_menu.add_separator()
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
            config_component=self.augmentation_config_tab,
        )
        self.notebook.add(self.augmentation_viewer_tab, text="Augmentation Viewer")

        self.pipeline_tab = TrainingPipelineComponent(
            self.notebook,
            on_status=self.update_status,
        )
        self.notebook.add(self.pipeline_tab, text="Training Pipeline")

        self.methods_tab = MethodsPanelComponent(
            self.notebook,
            on_status=self.update_status,
        )
        self.notebook.add(self.methods_tab, text="Classical & ML")

        self.compare_tab = ResultsComparisonComponent(
            self.notebook,
            on_status=self.update_status,
        )
        self.notebook.add(self.compare_tab, text="Results Comparison")

        self.export_tab = ExportResultsComponent(
            self.notebook,
            data_loader=self.data_loader_tab,
            on_status=self.update_status,
        )
        self.notebook.add(self.export_tab, text="Export")

        self.classical2_params = Classical2ParamsState()

        self.classical2_config_tab = Classical2ConfigAnalyzeComponent(
            self.notebook,
            data_loader=self.data_loader_tab,
            params_state=self.classical2_params,
            on_status=self.update_status,
        )
        self.notebook.add(self.classical2_config_tab, text="Classical Config & Analyze")

        self.classical2_eval_tab = Classical2EvaluationComponent(
            self.notebook,
            params_state=self.classical2_params,
            config_tab=self.classical2_config_tab,
            on_status=self.update_status,
        )
        self.notebook.add(self.classical2_eval_tab, text="Classical Evaluation")

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
        self.root.geometry("1280x860")

    def _open_classical_errors_folder(self) -> None:
        import os
        import subprocess

        path = Path(__file__).resolve().parents[1] / "classical" / "result" / "errors"
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        self.update_status("Opened classical errors folder")

    def _open_results_folder(self) -> None:
        import os
        import subprocess

        path = str(RESULTS_DIR)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        self.update_status("Opened results folder")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "TWM Project - Bottle Cap Classification\n\n"
            "GUI for data loading, augmentation, training pipeline,\n"
            "classical/ML methods, Classical2 rule-based CV, and export.",
        )

    def update_status(self, message: str) -> None:
        if hasattr(self, "status_bar"):
            self.status_bar.config(text=message)
            self.root.update_idletasks()


def main() -> None:
    root = tk.Tk()
    _ = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
