"""Classical2 batch evaluation and error browser tab."""
from __future__ import annotations

import os
import subprocess
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import TYPE_CHECKING
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2

from classical.classical2_labels import CLASSICAL2_LABELS, label_for_code
from classical.classical_2 import (
    Classical2,
    build_analysis_visualizations,
    format_analysis_report,
)
from classical.evaluate_classical2 import (
    CompareEvalResult,
    ErrorCase,
    EvalResult,
    format_eval_summary,
    stable_eval_seed,
    evaluate_dataset,
)
from gui.services import metrics_service
from data.augmentation import augment_bgr_image, train_augmentations_full_image
from gui.services.classical2_params_state import Classical2ParamsState
from gui.utils.image_preview import show_bgr_on_label, show_image_on_label
from gui.utils.task_runner import TaskRunner

if TYPE_CHECKING:
    from gui.components.augmentation_config import AugmentationConfigComponent
    from gui.components.classical2_config_analyze import Classical2ConfigAnalyzeComponent


class Classical2EvaluationComponent(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Frame,
        *,
        params_state: Classical2ParamsState,
        config_tab: Classical2ConfigAnalyzeComponent | None = None,
        augmentation_config: AugmentationConfigComponent | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.params_state = params_state
        self.config_tab = config_tab
        self.augmentation_config = augmentation_config
        self.on_status = on_status
        self._task_runner = TaskRunner(self.winfo_toplevel())
        self._compare_result: CompareEvalResult | None = None
        self._active_result: EvalResult | None = None
        self._filtered_errors: list[ErrorCase] = []
        self._photos: dict[str, tk.PhotoImage] = {}
        self._cm_photo = None
        self._selected_error: ErrorCase | None = None
        self._eval_seed = 42

        root = Path(__file__).resolve().parents[2]
        self._img_dir = tk.StringVar(value=str(root / "bottle-cap.yolov8" / "train" / "images"))
        self._label_dir = tk.StringVar(value=str(root / "bottle-cap.yolov8" / "train" / "labels"))
        self._errors_dir = tk.StringVar(value=str(root / "classical" / "result" / "errors"))
        self._preset_var = tk.StringVar(value=params_state.get_preset_name())
        self._eval_on = tk.StringVar(value="raw")
        self._aug_copies = tk.IntVar(value=2)
        self._use_gui_aug = tk.BooleanVar(value=True)
        self._result_view = tk.StringVar(value="raw")
        self._export_errors = tk.BooleanVar(value=True)
        self._export_partial = tk.BooleanVar(value=False)
        self._filter_exp = tk.StringVar(value="all")
        self._filter_pred = tk.StringVar(value="all")
        self._filter_search = tk.StringVar(value="")

        self._create_widgets()
        self._refresh_preset_names()

    def _create_widgets(self) -> None:
        title = tk.Label(self, text="Rule-Based Evaluation", font=("Arial", 14, "bold"))
        title.pack(pady=4)

        top = ttk.LabelFrame(self, text="Dataset & run")
        top.pack(fill=tk.X, padx=8, pady=2)

        paths = ttk.Frame(top)
        paths.pack(fill=tk.X, padx=4, pady=2)
        self._add_dir_row(paths, "Images:", self._img_dir)
        self._add_dir_row(paths, "Labels:", self._label_dir)
        self._add_dir_row(paths, "Errors:", self._errors_dir)

        row1 = ttk.Frame(top)
        row1.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(row1, text="Preset:").pack(side=tk.LEFT)
        self._preset_combo = ttk.Combobox(
            row1, textvariable=self._preset_var, state="readonly", width=14
        )
        self._preset_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Reload", command=self._reload_preset).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="Evaluate:").pack(side=tk.LEFT, padx=(8, 2))
        for val, label in (("raw", "raw"), ("aug", "aug"), ("both", "both")):
            ttk.Radiobutton(row1, text=label, value=val, variable=self._eval_on).pack(
                side=tk.LEFT, padx=2
            )
        ttk.Label(row1, text="Copies:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Spinbox(row1, from_=1, to=10, textvariable=self._aug_copies, width=3).pack(
            side=tk.LEFT
        )
        self._run_btn = ttk.Button(row1, text="Run evaluation", command=self._run_evaluation)
        self._run_btn.pack(side=tk.RIGHT, padx=4)

        row2 = ttk.Frame(top)
        row2.pack(fill=tk.X, padx=4, pady=2)
        ttk.Checkbutton(
            row2, text="Use Augmentation Config", variable=self._use_gui_aug
        ).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(row2, text="Export errors", variable=self._export_errors).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Checkbutton(row2, text="Include partial", variable=self._export_partial).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Label(row2, text="View:").pack(side=tk.LEFT, padx=(12, 2))
        self._result_view_combo = ttk.Combobox(
            row2,
            textvariable=self._result_view,
            values=["raw", "aug"],
            state="disabled",
            width=8,
        )
        self._result_view_combo.pack(side=tk.LEFT, padx=2)
        self._result_view_combo.bind("<<ComboboxSelected>>", lambda _e: self._switch_result_view())
        self._summary_label = ttk.Label(row2, text="No evaluation run yet.")
        self._summary_label.pack(side=tk.LEFT, padx=8)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.LabelFrame(main, text="Results")
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=1)

        self._metrics_text = tk.Text(left, height=9, width=34, state=tk.DISABLED, wrap=tk.WORD)
        self._metrics_text.pack(fill=tk.X, padx=5, pady=4)

        self._plot_label = tk.Label(
            left,
            text="Confusion matrix",
            bg="#2f2f2f",
            fg="white",
        )
        self._plot_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=4)

        self._log = scrolledtext.ScrolledText(left, height=3, state=tk.DISABLED)
        self._log.pack(fill=tk.X, padx=5, pady=(0, 4))

        right_paned = ttk.Panedwindow(right, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)

        err_frame = ttk.LabelFrame(right_paned, text="Misclassified images")
        preview_frame = ttk.LabelFrame(right_paned, text="Preview")
        right_paned.add(err_frame, weight=2)
        right_paned.add(preview_frame, weight=3)

        filt = ttk.Frame(err_frame)
        filt.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(filt, text="Expected:").pack(side=tk.LEFT)
        exp_values = ["all"] + [f"{k} {v}" for k, v in CLASSICAL2_LABELS.items()]
        ttk.Combobox(
            filt, textvariable=self._filter_exp, values=exp_values, state="readonly", width=12
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(filt, text="Predicted:").pack(side=tk.LEFT)
        ttk.Combobox(
            filt, textvariable=self._filter_pred, values=exp_values, state="readonly", width=12
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(filt, text="Search:").pack(side=tk.LEFT)
        ent = ttk.Entry(filt, textvariable=self._filter_search, width=14)
        ent.pack(side=tk.LEFT, padx=2)
        ent.bind("<KeyRelease>", lambda _e: self._apply_error_filter())
        ttk.Button(filt, text="Filter", command=self._apply_error_filter).pack(side=tk.LEFT, padx=2)

        tree_wrap = ttk.Frame(err_frame)
        tree_wrap.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        scroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL)
        err_cols = ("sample", "expected", "predicted", "score", "status")
        self._error_tree = ttk.Treeview(
            tree_wrap,
            columns=err_cols,
            show="headings",
            yscrollcommand=scroll.set,
        )
        scroll.config(command=self._error_tree.yview)
        for c, w in zip(err_cols, (200, 52, 52, 44, 120)):
            self._error_tree.heading(c, text=c)
            self._error_tree.column(c, width=w)
        self._error_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._error_tree.bind("<<TreeviewSelect>>", self._on_error_select)

        btn_row = ttk.Frame(err_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(btn_row, text="Open error folder", command=self._open_error_folder).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_row, text="Re-analyze", command=self._reanalyze).pack(side=tk.LEFT, padx=2)

        preview_nb = ttk.Notebook(preview_frame)
        preview_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self._preview_labels: dict[str, tk.Label] = {}
        for name in ("Original", "Annotated", "Bottle", "Cap"):
            tab = ttk.Frame(preview_nb)
            preview_nb.add(tab, text=name)
            lbl = tk.Label(tab, text=name, bg="#2f2f2f", fg="white")
            lbl.pack(fill=tk.BOTH, expand=True)
            self._preview_labels[name.lower()] = lbl

        self._report_text = scrolledtext.ScrolledText(preview_frame, height=6, state=tk.DISABLED)
        self._report_text.pack(fill=tk.X, padx=4, pady=2)

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

    def _aug_config(self) -> dict | None:
        if self._use_gui_aug.get() and self.augmentation_config is not None:
            return self.augmentation_config.get_config()
        return None

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
        eval_on = self._eval_on.get()
        aug_copies = max(1, int(self._aug_copies.get()))
        aug_config = self._aug_config()

        self._run_btn.config(state=tk.DISABLED)
        self._append_log(f"Starting evaluation (eval_on={eval_on}, aug_copies={aug_copies})...")
        self._notify("Running rule-based evaluation...")

        def _work():
            analyzer = Classical2(params)
            return evaluate_dataset(
                analyzer,
                img_dir,
                label_dir,
                eval_on=eval_on,  # type: ignore[arg-type]
                export_errors_dir=errors_dir,
                export_partial=self._export_partial.get(),
                clear_export_dir=bool(errors_dir),
                results_csv_path=Path(__file__).resolve().parents[2] / "results.csv",
                aug_copies=aug_copies,
                aug_config=aug_config,
                seed=self._eval_seed,
                preset_name=self._preset_var.get(),
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

    def _on_eval_done(self, compare: CompareEvalResult) -> None:
        self._compare_result = compare
        if compare.eval_on == "both":
            self._result_view_combo.config(state="readonly")
            self._result_view.set("raw")
        else:
            self._result_view_combo.config(state="disabled")
            self._result_view.set(compare.eval_on)

        lines = []
        if compare.raw is not None:
            lines.append(
                f"Raw: {compare.raw.total} samples, {compare.raw.accuracy:.2f}% accuracy, "
                f"{compare.raw.inference_ms_per_image:.1f} ms/img, {len(compare.raw.errors)} errors"
            )
        if compare.augmented is not None:
            lines.append(
                f"Aug: {compare.augmented.total} samples, {compare.augmented.accuracy:.2f}% accuracy, "
                f"{compare.augmented.inference_ms_per_image:.1f} ms/img, "
                f"{len(compare.augmented.errors)} errors"
            )
        if compare.raw and compare.augmented:
            delta = compare.augmented.accuracy - compare.raw.accuracy
            lines.append(f"Delta (aug - raw): {delta:+.2f}%")
        self._summary_label.config(text=" | ".join(lines) if lines else "No results.")

        self._active_result = compare.primary
        if compare.eval_on == "both" and compare.raw is not None:
            self._active_result = compare.raw
        self._update_results_panel()
        self._apply_error_filter()
        self._append_log("Evaluation complete.")
        saved = [
            str(r.metrics_json_path)
            for r in (compare.raw, compare.augmented)
            if r and r.metrics_json_path
        ]
        if saved:
            self._append_log("Metrics saved: " + ", ".join(saved))
            try:
                metrics_service.run_comparison()
                self._append_log("Results comparison table/plots updated.")
            except Exception as exc:
                self._append_log(f"Comparison update skipped: {exc}")
        primary = compare.primary
        if primary:
            self._notify(f"Evaluation done: {primary.accuracy:.2f}% accuracy ({compare.eval_on})")

    def _set_metrics_text(self, text: str) -> None:
        self._metrics_text.config(state=tk.NORMAL)
        self._metrics_text.delete("1.0", tk.END)
        self._metrics_text.insert(tk.END, text)
        self._metrics_text.config(state=tk.DISABLED)

    def _update_results_panel(self) -> None:
        result = self._active_result
        compare = self._compare_result
        preset = self._preset_var.get()

        if result is None:
            self._set_metrics_text("No evaluation run yet.")
            self._cm_photo = show_image_on_label(
                self._plot_label, None, placeholder="No confusion plot"
            )
            return

        summary = format_eval_summary(result, preset_name=preset)
        if compare and compare.eval_on == "both" and compare.raw and compare.augmented:
            delta = compare.augmented.accuracy - compare.raw.accuracy
            summary += (
                f"\n\n--- Comparison ---\n"
                f"Raw accuracy: {compare.raw.accuracy:.2f}%\n"
                f"Aug accuracy: {compare.augmented.accuracy:.2f}%\n"
                f"Delta (aug - raw): {delta:+.2f}%"
            )
        self._set_metrics_text(summary)

        plot_path = result.confusion_plot_path
        self._cm_photo = show_image_on_label(
            self._plot_label,
            plot_path if plot_path and plot_path.exists() else None,
            max_size=(420, 320),
            placeholder="Confusion plot not found",
        )

    def _switch_result_view(self) -> None:
        if self._compare_result is None:
            return
        view = self._result_view.get()
        if view == "aug" and self._compare_result.augmented is not None:
            self._active_result = self._compare_result.augmented
        elif self._compare_result.raw is not None:
            self._active_result = self._compare_result.raw
        else:
            self._active_result = self._compare_result.primary
        self._update_results_panel()
        self._apply_error_filter()

    def _parse_filter_class(self, value: str) -> int | None:
        if value == "all" or not value:
            return None
        return int(value.split()[0])

    def _apply_error_filter(self) -> None:
        if self._active_result is None:
            return
        exp_f = self._parse_filter_class(self._filter_exp.get())
        pred_f = self._parse_filter_class(self._filter_pred.get())
        search = self._filter_search.get().strip().lower()

        filtered: list[ErrorCase] = []
        for err in self._active_result.errors:
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
            row_id = err.sample_label or err.photo
            self._error_tree.insert(
                "",
                tk.END,
                iid=row_id,
                values=(
                    err.sample_label or err.photo,
                    ";".join(str(x) for x in err.expected),
                    err.predicted,
                    err.score,
                    ", ".join(err.status_list),
                ),
            )

    def _on_error_select(self, _event=None) -> None:
        sel = self._error_tree.selection()
        if not sel or self._active_result is None:
            return
        row_id = sel[0]
        err = next(
            (
                e
                for e in self._filtered_errors
                if (e.sample_label or e.photo) == row_id
            ),
            None,
        )
        if err is None:
            return
        self._selected_error = err
        self._show_error_preview(err)

    def _show_error_preview(self, err: ErrorCase) -> None:
        result = err.result
        images = build_analysis_visualizations(result)
        if err.variant == "aug" and err.aug_copy_index is not None:
            img_bgr = cv2.imread(str(err.image_path))
            if img_bgr is not None:
                compose = train_augmentations_full_image(self._aug_config())
                aug_img = augment_bgr_image(
                    img_bgr,
                    compose,
                    seed=stable_eval_seed(self._eval_seed, err.image_path.stem, err.aug_copy_index),
                )
                self._photos["original"] = show_bgr_on_label(
                    self._preview_labels["original"], aug_img, max_size=(480, 280)
                )
            else:
                self._photos["original"] = show_image_on_label(
                    self._preview_labels["original"], err.image_path, max_size=(480, 280)
                )
        else:
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
        err = self._selected_error
        try:
            if err.variant == "aug" and err.aug_copy_index is not None:
                img_bgr = cv2.imread(str(err.image_path))
                if img_bgr is None:
                    raise ValueError(f"Could not load {err.image_path}")
                compose = train_augmentations_full_image(self._aug_config())
                aug_img = augment_bgr_image(
                    img_bgr,
                    compose,
                    seed=stable_eval_seed(self._eval_seed, err.image_path.stem, err.aug_copy_index),
                )
                result = Classical2(params).analyze(aug_img)
            else:
                result = Classical2(params).analyze(str(err.image_path))
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
