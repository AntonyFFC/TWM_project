"""Classical2 batch evaluation and error browser tab."""
from __future__ import annotations

import os
import subprocess
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from classical.classical2_labels import CLASSICAL2_LABELS, label_for_code
from classical.classical_2 import (
    Classical2,
    build_analysis_visualizations,
    format_analysis_report,
)
from classical.evaluate_classical2 import ErrorCase, EvalResult, evaluate_dataset
from gui.services.classical2_params_state import Classical2ParamsState
from gui.utils.image_preview import show_bgr_on_label, show_image_on_label
from gui.utils.task_runner import TaskRunner


class Classical2EvaluationComponent(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Frame,
        *,
        params_state: Classical2ParamsState,
        config_tab: Classical2ConfigAnalyzeComponent | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.params_state = params_state
        self.config_tab = config_tab
        self.on_status = on_status
        self._task_runner = TaskRunner(self.winfo_toplevel())
        self._eval_result: EvalResult | None = None
        self._filtered_errors: list[ErrorCase] = []
        self._photos: dict[str, tk.PhotoImage] = {}
        self._selected_error: ErrorCase | None = None

        root = Path(__file__).resolve().parents[2]
        self._img_dir = tk.StringVar(value=str(root / "bottle-cap.yolov8" / "train" / "images"))
        self._label_dir = tk.StringVar(value=str(root / "bottle-cap.yolov8" / "train" / "labels"))
        self._errors_dir = tk.StringVar(value=str(root / "classical" / "result" / "errors"))
        self._preset_var = tk.StringVar(value=params_state.get_preset_name())
        self._export_errors = tk.BooleanVar(value=True)
        self._export_partial = tk.BooleanVar(value=False)
        self._filter_exp = tk.StringVar(value="all")
        self._filter_pred = tk.StringVar(value="all")
        self._filter_search = tk.StringVar(value="")

        self._create_widgets()
        self._refresh_preset_names()

    def _create_widgets(self) -> None:
        title = tk.Label(self, text="Classical Evaluation", font=("Arial", 14, "bold"))
        title.pack(pady=6)

        top = ttk.LabelFrame(self, text="Dataset & run")
        top.pack(fill=tk.X, padx=8, pady=4)
        self._add_dir_row(top, "Images:", self._img_dir)
        self._add_dir_row(top, "Labels:", self._label_dir)
        self._add_dir_row(top, "Errors out:", self._errors_dir)

        opts = ttk.Frame(top)
        opts.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(opts, text="Preset:").pack(side=tk.LEFT)
        self._preset_combo = ttk.Combobox(
            opts, textvariable=self._preset_var, state="readonly", width=16
        )
        self._preset_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(opts, text="Reload preset", command=self._reload_preset).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(opts, text="Export error folders", variable=self._export_errors).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Checkbutton(opts, text="Include partial", variable=self._export_partial).pack(
            side=tk.LEFT, padx=6
        )
        self._run_btn = ttk.Button(opts, text="Run evaluation", command=self._run_evaluation)
        self._run_btn.pack(side=tk.LEFT, padx=8)

        self._summary_label = ttk.Label(top, text="No evaluation run yet.")
        self._summary_label.pack(anchor=tk.W, padx=4, pady=4)

        self._log = scrolledtext.ScrolledText(top, height=4, state=tk.DISABLED)
        self._log.pack(fill=tk.X, padx=4, pady=4)

        mid = ttk.LabelFrame(self, text="Confusion matrix (expected rows → predicted cols)")
        mid.pack(fill=tk.X, padx=8, pady=4)
        cols = ("exp", "p0", "p1", "p2", "p3", "p4")
        self._matrix_tree = ttk.Treeview(mid, columns=cols, show="headings", height=5)
        for c in cols:
            self._matrix_tree.heading(c, text=c)
            self._matrix_tree.column(c, width=60, anchor=tk.CENTER)
        self._matrix_tree.pack(fill=tk.X, padx=4, pady=4)

        bottom = ttk.Panedwindow(self, orient=tk.VERTICAL)
        bottom.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        err_frame = ttk.LabelFrame(bottom, text="Misclassified images")
        preview_frame = ttk.LabelFrame(bottom, text="Preview")
        bottom.add(err_frame, weight=2)
        bottom.add(preview_frame, weight=3)

        filt = ttk.Frame(err_frame)
        filt.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(filt, text="Expected:").pack(side=tk.LEFT)
        exp_values = ["all"] + [f"{k} {v}" for k, v in CLASSICAL2_LABELS.items()]
        ttk.Combobox(
            filt, textvariable=self._filter_exp, values=exp_values, state="readonly", width=14
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(filt, text="Predicted:").pack(side=tk.LEFT)
        ttk.Combobox(
            filt, textvariable=self._filter_pred, values=exp_values, state="readonly", width=14
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(filt, text="Search:").pack(side=tk.LEFT)
        ent = ttk.Entry(filt, textvariable=self._filter_search, width=20)
        ent.pack(side=tk.LEFT, padx=4)
        ent.bind("<KeyRelease>", lambda _e: self._apply_error_filter())
        ttk.Button(filt, text="Apply filter", command=self._apply_error_filter).pack(
            side=tk.LEFT, padx=4
        )

        tree_wrap = ttk.Frame(err_frame)
        tree_wrap.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        scroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL)
        err_cols = ("photo", "expected", "predicted", "score", "status")
        self._error_tree = ttk.Treeview(
            tree_wrap,
            columns=err_cols,
            show="headings",
            yscrollcommand=scroll.set,
        )
        scroll.config(command=self._error_tree.yview)
        for c, w in zip(err_cols, (220, 60, 60, 50, 160)):
            self._error_tree.heading(c, text=c)
            self._error_tree.column(c, width=w)
        self._error_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._error_tree.bind("<<TreeviewSelect>>", self._on_error_select)

        btn_row = ttk.Frame(err_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="Open error folder", command=self._open_error_folder).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_row, text="Re-analyze with current params", command=self._reanalyze).pack(
            side=tk.LEFT, padx=4
        )

        preview_nb = ttk.Notebook(preview_frame)
        preview_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._preview_labels: dict[str, tk.Label] = {}
        for name in ("Original", "Annotated", "Bottle", "Cap"):
            tab = ttk.Frame(preview_nb)
            preview_nb.add(tab, text=name)
            lbl = tk.Label(tab, text=name, bg="#2f2f2f", fg="white")
            lbl.pack(fill=tk.BOTH, expand=True)
            self._preview_labels[name.lower()] = lbl

        self._report_text = scrolledtext.ScrolledText(preview_frame, height=8, state=tk.DISABLED)
        self._report_text.pack(fill=tk.X, padx=4, pady=4)

    def _add_dir_row(self, parent: ttk.Frame, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row, text="...", width=3, command=lambda: self._browse_dir(var)).pack(
            side=tk.LEFT
        )

    def _browse_dir(self, var: tk.StringVar) -> None:
        selected = filedialog.askdirectory()
        if selected:
            var.set(selected)

    def _refresh_preset_names(self) -> None:
        names = self.params_state.refresh_preset_list()
        self._preset_combo["values"] = names

    def _reload_preset(self) -> None:
        from gui.services import classical2_preset_store as preset_store

        name = self._preset_var.get()
        try:
            params = preset_store.load_preset(name)
            self.params_state.set_params(name, params)
            self._notify(f"Loaded preset: {name}")
        except Exception as exc:
            messagebox.showerror("Preset error", str(exc))

    def _append_log(self, msg: str) -> None:
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, msg + "\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _run_evaluation(self) -> None:
        if self._task_runner.is_running:
            return
        if self.config_tab is not None:
            try:
                self.config_tab.sync_params_to_state()
            except ValueError as exc:
                messagebox.showerror("Invalid parameters", str(exc))
                return
        else:
            self._reload_preset()

        params = self.params_state.get_params()
        img_dir = Path(self._img_dir.get())
        label_dir = Path(self._label_dir.get())
        errors_dir = Path(self._errors_dir.get()) if self._export_errors.get() else None

        self._run_btn.config(state=tk.DISABLED)
        self._append_log("Starting evaluation...")
        self._notify("Running Classical2 evaluation...")

        def _work():
            analyzer = Classical2(params)
            return evaluate_dataset(
                analyzer,
                img_dir,
                label_dir,
                export_errors_dir=errors_dir,
                export_partial=self._export_partial.get(),
                clear_export_dir=bool(errors_dir),
                results_csv_path=Path(__file__).resolve().parents[2] / "results.csv",
                log=lambda m: self.after(0, lambda msg=m: self._append_log(msg)),
            )

        self._task_runner.run(
            _work,
            on_success=lambda r: self.after(0, lambda: self._on_eval_done(r)),
            on_error=lambda err: self.after(
                0, lambda: messagebox.showerror("Evaluation failed", err[:2000])
            ),
            on_finished=lambda: self.after(0, lambda: self._run_btn.config(state=tk.NORMAL)),
        )

    def _on_eval_done(self, result: EvalResult) -> None:
        self._eval_result = result
        self._summary_label.config(
            text=(
                f"Tested: {result.total} | Full correct: {result.full_correct} | "
                f"Partial: {result.partial_correct} | Accuracy: {result.accuracy:.2f}% | "
                f"Errors: {len(result.errors)}"
            )
        )
        self._populate_matrix(result)
        self._apply_error_filter()
        self._append_log("Evaluation complete.")
        self._notify(f"Evaluation done: {result.accuracy:.2f}% accuracy")

    def _populate_matrix(self, result: EvalResult) -> None:
        for item in self._matrix_tree.get_children():
            self._matrix_tree.delete(item)
        for exp in range(5):
            row = {"exp": str(exp)}
            for pred in range(5):
                row[f"p{pred}"] = str(result.confusion.get(exp, {}).get(pred, 0))
            self._matrix_tree.insert("", tk.END, values=tuple(row[c] for c in self._matrix_tree["columns"]))

    def _parse_filter_class(self, value: str) -> int | None:
        if value == "all" or not value:
            return None
        return int(value.split()[0])

    def _apply_error_filter(self) -> None:
        if self._eval_result is None:
            return
        exp_f = self._parse_filter_class(self._filter_exp.get())
        pred_f = self._parse_filter_class(self._filter_pred.get())
        search = self._filter_search.get().strip().lower()

        filtered: list[ErrorCase] = []
        for err in self._eval_result.errors:
            if exp_f is not None and exp_f not in err.expected:
                continue
            if pred_f is not None and err.predicted != pred_f:
                continue
            if search and search not in err.photo.lower():
                continue
            filtered.append(err)

        self._filtered_errors = filtered
        for item in self._error_tree.get_children():
            self._error_tree.delete(item)
        for err in filtered:
            self._error_tree.insert(
                "",
                tk.END,
                iid=err.photo,
                values=(
                    err.photo,
                    ";".join(str(x) for x in err.expected),
                    err.predicted,
                    err.score,
                    ", ".join(err.status_list),
                ),
            )

    def _on_error_select(self, _event=None) -> None:
        sel = self._error_tree.selection()
        if not sel or self._eval_result is None:
            return
        photo = sel[0]
        err = next((e for e in self._filtered_errors if e.photo == photo), None)
        if err is None:
            return
        self._selected_error = err
        self._show_error_preview(err)

    def _show_error_preview(self, err: ErrorCase) -> None:
        result = err.result
        images = build_analysis_visualizations(result)
        self._photos["original"] = show_image_on_label(
            self._preview_labels["original"], err.image_path, max_size=(480, 280)
        )
        for key in ("annotated", "bottle", "cap"):
            self._photos[key] = show_bgr_on_label(
                self._preview_labels[key], images[key], max_size=(480, 280)
            )
        text = format_analysis_report(result, expected=err.expected, score=err.score)
        self._report_text.config(state=tk.NORMAL)
        self._report_text.delete("1.0", tk.END)
        self._report_text.insert(tk.END, text)
        self._report_text.config(state=tk.DISABLED)

    def _open_error_folder(self) -> None:
        if self._selected_error is None or self._selected_error.export_folder is None:
            messagebox.showinfo("No folder", "Re-run evaluation with export enabled.")
            return
        path = str(self._selected_error.export_folder)
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)

    def _reanalyze(self) -> None:
        if self._selected_error is None:
            return
        if self.config_tab is not None:
            try:
                self.config_tab.sync_params_to_state()
            except ValueError as exc:
                messagebox.showerror("Invalid parameters", str(exc))
                return
        params = self.params_state.get_params()
        try:
            result = Classical2(params).analyze(str(self._selected_error.image_path))
        except Exception as exc:
            messagebox.showerror("Re-analyze failed", str(exc))
            return
        code = result.get("status_code")
        self._notify(f"Re-analyzed: {label_for_code(code)} ({code})")
        err = self._selected_error
        err.result = result
        err.predicted = code
        err.status_list = list(result.get("status_list", []))
        self._show_error_preview(err)

    def _notify(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)
