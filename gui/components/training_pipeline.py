"""Training pipeline tab: data prep, full pipeline, logging."""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from gui.components.hyperparams_editor import HyperparamsEditor  # noqa: E402
from gui.services import pipeline_service  # noqa: E402
from gui.utils.task_runner import TaskRunner  # noqa: E402


class TrainingPipelineComponent(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Frame,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_status = on_status
        self.hyperparams = HyperparamsEditor(self, on_status=on_status)
        self._task_runner = TaskRunner(self.winfo_toplevel())
        self._augmentation = tk.StringVar(value="both")
        self._skip_eda = tk.BooleanVar(value=False)
        self._skip_split = tk.BooleanVar(value=False)
        self._skip_classical = tk.BooleanVar(value=False)
        self._skip_ml = tk.BooleanVar(value=False)
        self._skip_demo = tk.BooleanVar(value=False)
        self._prereq_label: tk.Label
        self._buttons: list[ttk.Button] = []
        self._create_widgets()
        self.refresh_prerequisites()

    def _create_widgets(self) -> None:
        title = tk.Label(self, text="Training Pipeline", font=("Arial", 14, "bold"))
        title.pack(pady=8)

        prereq_frame = ttk.LabelFrame(self, text="Prerequisites")
        prereq_frame.pack(fill=tk.X, padx=10, pady=5)
        self._prereq_label = tk.Label(prereq_frame, text="Checking...", justify=tk.LEFT)
        self._prereq_label.pack(anchor=tk.W, padx=8, pady=5)
        ttk.Button(prereq_frame, text="Refresh status", command=self.refresh_prerequisites).pack(
            anchor=tk.W, padx=8, pady=4
        )

        self.hyperparams.pack(fill=tk.X, padx=10, pady=5)

        opts = ttk.LabelFrame(self, text="Pipeline options")
        opts.pack(fill=tk.X, padx=10, pady=5)

        aug_row = ttk.Frame(opts)
        aug_row.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(aug_row, text="Augmentation:").pack(side=tk.LEFT)
        for val, label in (("raw", "raw"), ("aug", "aug"), ("both", "both")):
            ttk.Radiobutton(
                aug_row, text=label, value=val, variable=self._augmentation
            ).pack(side=tk.LEFT, padx=6)

        skip_row = ttk.Frame(opts)
        skip_row.pack(fill=tk.X, padx=5, pady=4)
        for var, text in (
            (self._skip_eda, "Skip EDA"),
            (self._skip_split, "Skip split"),
            (self._skip_classical, "Skip classical"),
            (self._skip_ml, "Skip ML"),
            (self._skip_demo, "Skip demo"),
        ):
            ttk.Checkbutton(skip_row, text=text, variable=var).pack(side=tk.LEFT, padx=6)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=10, pady=8)
        for text, cmd in (
            ("Run EDA", self._run_eda),
            ("Run Split", self._run_split),
            ("Run Full Pipeline", self._run_full_pipeline),
        ):
            btn = ttk.Button(btn_row, text=text, command=cmd)
            btn.pack(side=tk.LEFT, padx=5)
            self._buttons.append(btn)

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._log = scrolledtext.ScrolledText(log_frame, height=12, state=tk.DISABLED)
        self._log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def refresh_prerequisites(self) -> None:
        status = pipeline_service.get_dataset_status()
        text = (
            f"Raw images: {status['images']}\n"
            f"Labels: {status['labels']}\n"
            f"Train/val/test splits: {'yes' if status['has_splits'] else 'no'}\n"
            f"Dataset OK: {'yes' if status['ok'] else 'no'}"
        )
        self._prereq_label.config(text=text)
        self._notify("Prerequisites refreshed")

    def _append_log(self, message: str) -> None:
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, message + "\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in self._buttons:
            btn.config(state=state)

    def _run_task(self, label: str, fn) -> None:
        self._append_log(f"--- {label} ---")
        self._set_busy(True)
        self._notify(f"Running: {label}")

        def _log(msg: str) -> None:
            self.after(0, lambda m=msg: self._append_log(m))

        def _work():
            return fn(_log)

        self._task_runner.run(
            _work,
            on_success=lambda _r: self.after(
                0, lambda: (self.refresh_prerequisites(), self._notify(f"Finished: {label}"))
            ),
            on_error=lambda err: self.after(
                0,
                lambda: (
                    self._append_log(err),
                    messagebox.showerror("Task failed", err[:2000]),
                ),
            ),
            on_finished=lambda: self.after(0, lambda: self._set_busy(False)),
        )

    def _run_eda(self) -> None:
        self._run_task("EDA", lambda log: pipeline_service.execute_eda(log))

    def _run_split(self) -> None:
        self._run_task("Split", lambda log: pipeline_service.execute_split(log))

    def _run_full_pipeline(self) -> None:
        aug = self._augmentation.get()
        self._run_task(
            "Full pipeline",
            lambda log: pipeline_service.execute_full_pipeline(
                augmentation=aug,  # type: ignore[arg-type]
                skip_eda=self._skip_eda.get(),
                skip_split=self._skip_split.get(),
                skip_classical=self._skip_classical.get(),
                skip_ml=self._skip_ml.get(),
                skip_demo=self._skip_demo.get(),
                log=log,
            ),
        )

    def _notify(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)
