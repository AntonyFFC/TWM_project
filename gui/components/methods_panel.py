"""Classical and ML training panels with metrics preview."""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from gui.services import metrics_service, training_service  # noqa: E402
from gui.utils.image_preview import show_image_on_label  # noqa: E402
from gui.utils.task_runner import TaskRunner  # noqa: E402


class _MethodColumn(ttk.LabelFrame):
    def __init__(
        self,
        parent: ttk.Frame,
        *,
        title: str,
        kind: str,
        train_fn,
        on_status: Callable[[str], None] | None,
        task_runner: TaskRunner,
        on_complete: Callable[[], None],
    ) -> None:
        super().__init__(parent, text=title)
        self.kind = kind
        self._train_fn = train_fn
        self.on_status = on_status
        self._task_runner = task_runner
        self._on_complete = on_complete
        self._trained_on = tk.StringVar(value="both")
        self._run_var = tk.StringVar()
        self._photo = None
        self._run_btn: ttk.Button
        self._create_widgets()

    def _create_widgets(self) -> None:
        aug_row = ttk.Frame(self)
        aug_row.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(aug_row, text="Train on:").pack(side=tk.LEFT)
        for val in ("raw", "aug", "both"):
            ttk.Radiobutton(aug_row, text=val, value=val, variable=self._trained_on).pack(
                side=tk.LEFT, padx=4
            )

        self._run_btn = ttk.Button(self, text="Run training", command=self._run_training)
        self._run_btn.pack(anchor=tk.W, padx=5, pady=4)

        pick_row = ttk.Frame(self)
        pick_row.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(pick_row, text="Run:").pack(side=tk.LEFT)
        self._run_combo = ttk.Combobox(
            pick_row, textvariable=self._run_var, state="readonly", width=28
        )
        self._run_combo.pack(side=tk.LEFT, padx=4)
        self._run_combo.bind("<<ComboboxSelected>>", lambda _e: self._show_selected_run())
        ttk.Button(pick_row, text="Refresh", command=self.refresh_runs).pack(side=tk.LEFT, padx=4)

        self._metrics_text = tk.Text(self, height=10, width=36, state=tk.DISABLED, wrap=tk.WORD)
        self._metrics_text.pack(fill=tk.X, padx=5, pady=4)

        self._plot_label = tk.Label(
            self,
            text="Confusion matrix",
            bg="#2f2f2f",
            fg="white",
            height=10,
        )
        self._plot_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def refresh_runs(self) -> None:
        runs = metrics_service.list_runs(kind=self.kind)
        self._run_combo["values"] = runs
        if runs:
            self._run_var.set(runs[-1])
            self._show_selected_run()
        else:
            self._run_var.set("")
            self._set_metrics_text("No training runs yet.")
            show_image_on_label(self._plot_label, None, placeholder="No confusion plot")

    def _set_metrics_text(self, text: str) -> None:
        self._metrics_text.config(state=tk.NORMAL)
        self._metrics_text.delete("1.0", tk.END)
        self._metrics_text.insert(tk.END, text)
        self._metrics_text.config(state=tk.DISABLED)

    def _show_selected_run(self) -> None:
        run_name = self._run_var.get()
        if not run_name:
            return
        report = metrics_service.load_report(run_name)
        if report is None:
            self._set_metrics_text(f"Report not found: {run_name}")
            return
        self._set_metrics_text(metrics_service.format_report_summary(report))
        plot_path = metrics_service.confusion_plot_path(run_name)
        self._photo = show_image_on_label(
            self._plot_label,
            plot_path if plot_path.exists() else None,
            max_size=(360, 260),
            placeholder="Confusion plot not found",
        )

    def _run_training(self) -> None:
        if self._task_runner.is_running:
            self._notify("Another task is running.")
            return
        trained_on = self._trained_on.get()  # type: ignore[arg-type]
        label = f"{self.kind} training ({trained_on})"
        self._run_btn.config(state=tk.DISABLED)
        self._notify(f"Starting {label}")

        def _work():
            def log(msg: str) -> None:
                if self.on_status and msg.strip():
                    self.after(0, lambda m=msg.strip(): self.on_status(m[:200]))

            self._train_fn(trained_on, log)
            return True

        def _finished() -> None:
            self._run_btn.config(state=tk.NORMAL)
            self.refresh_runs()
            self._on_complete()

        self._task_runner.run(
            _work,
            on_success=lambda _r: self.after(0, lambda: self._notify(f"Finished: {label}")),
            on_error=lambda err: self.after(
                0,
                lambda: self._notify(f"Failed: {label}") or _show_error(err),
            ),
            on_finished=lambda: self.after(0, _finished),
        )

    def _notify(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)


def _show_error(err: str) -> None:
    from tkinter import messagebox

    messagebox.showerror("Training failed", err[:2000])


class MethodsPanelComponent(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Frame,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_status = on_status
        self._task_runner = TaskRunner(self.winfo_toplevel())
        self._create_widgets()

    def _create_widgets(self) -> None:
        title = tk.Label(self, text="Classical & ML Methods", font=("Arial", 14, "bold"))
        title.pack(pady=8)

        hp_note = ttk.Label(
            self,
            text="Edit hyperparameters on the Training Pipeline tab, then Save to config.py.",
        )
        hp_note.pack(padx=10, anchor=tk.W)

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._classical = _MethodColumn(
            paned,
            title="Classical (HOG + SVM)",
            kind="classical",
            train_fn=training_service.run_classical,
            on_status=self.on_status,
            task_runner=self._task_runner,
            on_complete=self._on_training_complete,
        )
        self._ml = _MethodColumn(
            paned,
            title="ML (ResNet18)",
            kind="ml",
            train_fn=training_service.run_ml,
            on_status=self.on_status,
            task_runner=self._task_runner,
            on_complete=self._on_training_complete,
        )
        paned.add(self._classical, weight=1)
        paned.add(self._ml, weight=1)

        self._classical.refresh_runs()
        self._ml.refresh_runs()

    def _on_training_complete(self) -> None:
        self._classical.refresh_runs()
        self._ml.refresh_runs()

    def refresh_all(self) -> None:
        self._classical.refresh_runs()
        self._ml.refresh_runs()
