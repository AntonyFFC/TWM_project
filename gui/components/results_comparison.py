"""Results comparison tab: summary table and comparison plots."""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from gui.services import metrics_service  # noqa: E402
from gui.utils.image_preview import show_image_on_label  # noqa: E402
from gui.utils.task_runner import TaskRunner  # noqa: E402


class ResultsComparisonComponent(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Frame,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_status = on_status
        self._task_runner = TaskRunner(self.winfo_toplevel())
        self._plot_var = tk.StringVar()
        self._photo = None
        self._tree: ttk.Treeview
        self._create_widgets()
        self.refresh_table()

    def _create_widgets(self) -> None:
        title = tk.Label(self, text="Results Comparison", font=("Arial", 14, "bold"))
        title.pack(pady=8)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_row, text="Refresh table", command=self.refresh_table).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_row, text="Run comparison", command=self._run_comparison).pack(
            side=tk.LEFT, padx=4
        )

        table_frame = ttk.LabelFrame(self, text="Summary (summary.csv)")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        cols = (
            "method",
            "kind",
            "trained_on",
            "accuracy",
            "f1_macro",
            "inference_ms",
            "train_time_s",
        )
        tree_container = ttk.Frame(table_frame)
        tree_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll_y = ttk.Scrollbar(tree_container, orient=tk.VERTICAL)
        scroll_x = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL)
        self._tree = ttk.Treeview(
            tree_container,
            columns=cols,
            show="headings",
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
        )
        scroll_y.config(command=self._tree.yview)
        scroll_x.config(command=self._tree.xview)
        for col in cols:
            self._tree.heading(col, text=col)
            self._tree.column(col, width=100, anchor=tk.CENTER)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        plot_frame = ttk.LabelFrame(self, text="Comparison plot")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        pick_row = ttk.Frame(plot_frame)
        pick_row.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(pick_row, text="Plot:").pack(side=tk.LEFT)
        self._plot_combo = ttk.Combobox(
            pick_row,
            textvariable=self._plot_var,
            values=metrics_service.COMPARISON_PLOTS,
            state="readonly",
            width=32,
        )
        self._plot_combo.pack(side=tk.LEFT, padx=4)
        self._plot_combo.bind("<<ComboboxSelected>>", lambda _e: self._show_plot())
        ttk.Button(pick_row, text="Show", command=self._show_plot).pack(side=tk.LEFT, padx=4)

        self._plot_label = tk.Label(
            plot_frame,
            text="Select a plot",
            bg="#2f2f2f",
            fg="white",
        )
        self._plot_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        if metrics_service.COMPARISON_PLOTS:
            self._plot_var.set(metrics_service.COMPARISON_PLOTS[0])

    def refresh_table(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

        rows = metrics_service.load_summary_table()
        if not rows:
            self._notify("No summary.csv — run comparison after training.")
            self._show_plot()
            return

        for row in rows:
            self._tree.insert(
                "",
                tk.END,
                values=(
                    row.get("method", ""),
                    row.get("kind", ""),
                    row.get("trained_on", ""),
                    f"{row.get('accuracy', 0):.4f}",
                    f"{row.get('f1_macro', 0):.4f}",
                    f"{row.get('inference_ms', 0):.2f}",
                    f"{row.get('train_time_s', 0):.1f}",
                ),
            )
        self._notify(f"Loaded {len(rows)} rows from summary")
        self._show_plot()

    def _show_plot(self) -> None:
        name = self._plot_var.get()
        if not name:
            return
        path = metrics_service.comparison_plot_path(name)
        self._photo = show_image_on_label(
            self._plot_label,
            path if path.exists() else None,
            max_size=(700, 400),
            placeholder=f"Plot not found: {name}\nRun comparison first.",
        )

    def _run_comparison(self) -> None:
        if self._task_runner.is_running:
            self._notify("A task is already running.")
            return
        self._notify("Running comparison...")

        def _work():
            return metrics_service.run_comparison()

        self._task_runner.run(
            _work,
            on_success=lambda _p: self.after(
                0, lambda: (self.refresh_table(), self._notify("Comparison complete"))
            ),
            on_error=lambda err: self.after(0, lambda: _show_error(err)),
        )

    def _notify(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)


def _show_error(err: str) -> None:
    from tkinter import messagebox

    messagebox.showerror("Comparison failed", err[:2000])
