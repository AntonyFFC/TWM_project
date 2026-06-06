"""Classical2 parameter editor and single-image analysis tab."""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import RAW_LABELS_DIR  # noqa: E402
from classical.classical2_labels import (
    describe_match,
    format_code,
    format_codes,
    format_status_list,
    score_output,
)
from classical.classical_2 import Classical2, build_analysis_visualizations, format_analysis_report
from classical.evaluate_classical2 import find_expected_labels
from gui.components.data_loader import DataLoaderComponent
from gui.services import classical2_preset_store as preset_store
from gui.services.classical2_params_state import Classical2ParamsState
from gui.utils.image_preview import show_bgr_on_label, show_image_on_label
from gui.utils.task_runner import TaskRunner


class Classical2ConfigAnalyzeComponent(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Frame,
        *,
        data_loader: DataLoaderComponent,
        params_state: Classical2ParamsState,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.data_loader = data_loader
        self.params_state = params_state
        self.on_status = on_status
        self._task_runner = TaskRunner(self.winfo_toplevel())
        self._vars: dict[str, tk.Variable] = {}
        self._image_path: Path | None = None
        self._photos: dict[str, tk.PhotoImage] = {}
        self._last_result: dict | None = None
        self._preset_var = tk.StringVar(value=params_state.get_preset_name())
        self._labels_dir = tk.StringVar(value=str(RAW_LABELS_DIR))
        self._create_widgets()
        self._load_preset_into_form(params_state.get_preset_name())

    def _create_widgets(self) -> None:
        title = tk.Label(self, text="Rule-Based Config & Analyze", font=("Arial", 14, "bold"))
        title.pack(pady=6)

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=2)
        paned.add(right, weight=3)

        self._build_preset_bar(left)
        self._build_param_form(left)
        self._build_analyze_panel(right)

    def _build_preset_bar(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(row, text="Preset:").pack(side=tk.LEFT)
        self._preset_combo = ttk.Combobox(
            row, textvariable=self._preset_var, state="readonly", width=18
        )
        self._preset_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Load", command=self._on_load_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Save as...", command=self._on_save_preset).pack(side=tk.LEFT, padx=2)
        self._refresh_preset_names()

    def _build_param_form(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        canvas = tk.Canvas(container, highlightthickness=0)
        scroll_y = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        canvas.configure(yscrollcommand=scroll_y.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        defaults = preset_store.get_default_params()
        for group, keys in preset_store.PARAM_GROUPS.items():
            frame = ttk.LabelFrame(inner, text=group)
            frame.pack(fill=tk.X, padx=2, pady=4)
            for key in keys:
                if key not in defaults:
                    continue
                self._add_param_row(frame, key, defaults[key])

    def _add_param_row(self, parent: ttk.Frame, key: str, default_val) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=4, pady=2)
        label = preset_store.PARAM_LABELS.get(key, key)
        ttk.Label(row, text=label, width=28).pack(side=tk.LEFT)

        if key in preset_store.RANGE_KEYS:
            lo_var = tk.IntVar(value=default_val[0])
            hi_var = tk.IntVar(value=default_val[1])
            self._vars[f"{key}_lo"] = lo_var
            self._vars[f"{key}_hi"] = hi_var
            ttk.Entry(row, textvariable=lo_var, width=5).pack(side=tk.LEFT, padx=2)
            ttk.Label(row, text="-").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=hi_var, width=5).pack(side=tk.LEFT, padx=2)
        elif key in preset_store.INT_KEYS:
            var = tk.IntVar(value=int(default_val))
            self._vars[key] = var
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=2)
        else:
            var = tk.DoubleVar(value=float(default_val))
            self._vars[key] = var
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=2)

    def _build_analyze_panel(self, parent: ttk.Frame) -> None:
        src = ttk.LabelFrame(parent, text="Image")
        src.pack(fill=tk.X, padx=4, pady=4)
        btn_row = ttk.Frame(src)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="Use Data Loader", command=self._use_data_loader).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_row, text="Browse...", command=self._browse_image).pack(side=tk.LEFT, padx=2)
        self._analyze_btn = ttk.Button(btn_row, text="Analyze", command=self._run_analyze)
        self._analyze_btn.pack(side=tk.LEFT, padx=8)
        self._path_label = ttk.Label(src, text="No image selected", wraplength=420)
        self._path_label.pack(anchor=tk.W, padx=4, pady=2)

        labels_row = ttk.Frame(src)
        labels_row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(labels_row, text="Labels dir:").pack(side=tk.LEFT)
        ttk.Entry(labels_row, textvariable=self._labels_dir).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(labels_row, text="...", width=3, command=self._browse_labels_dir).pack(
            side=tk.LEFT
        )

        result_frame = ttk.LabelFrame(parent, text="Classification result")
        result_frame.pack(fill=tk.X, padx=4, pady=4)
        self._expected_label = tk.Label(
            result_frame, text="Expected: —", anchor=tk.W, justify=tk.LEFT, font=("Arial", 10)
        )
        self._expected_label.pack(fill=tk.X, padx=8, pady=2)
        self._predicted_label = tk.Label(
            result_frame,
            text="Predicted: —",
            anchor=tk.W,
            justify=tk.LEFT,
            font=("Arial", 10, "bold"),
        )
        self._predicted_label.pack(fill=tk.X, padx=8, pady=2)
        self._match_label = tk.Label(
            result_frame, text="Match: —", anchor=tk.W, justify=tk.LEFT, font=("Arial", 10)
        )
        self._match_label.pack(fill=tk.X, padx=8, pady=2)
        self._flags_label = tk.Label(
            result_frame,
            text="Flags: —",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=480,
            font=("Arial", 9),
            fg="#444444",
        )
        self._flags_label.pack(fill=tk.X, padx=8, pady=(2, 6))

        preview_nb = ttk.Notebook(parent)
        preview_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._preview_labels: dict[str, tk.Label] = {}
        for name in ("Original", "Annotated", "Bottle", "Cap", "Background"):
            tab = ttk.Frame(preview_nb)
            preview_nb.add(tab, text=name)
            lbl = tk.Label(tab, text=name, bg="#2f2f2f", fg="white")
            lbl.pack(fill=tk.BOTH, expand=True)
            self._preview_labels[name.lower()] = lbl

        meas_frame = ttk.LabelFrame(parent, text="Measurements")
        meas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._meas_text = scrolledtext.ScrolledText(meas_frame, height=8, state=tk.DISABLED)
        self._meas_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _browse_labels_dir(self) -> None:
        selected = filedialog.askdirectory(title="Select labels folder")
        if selected:
            self._labels_dir.set(selected)

    def _lookup_expected(self, image_path: Path) -> list[int] | None:
        return find_expected_labels(image_path, Path(self._labels_dir.get()))

    def _update_result_display(
        self, result: dict, expected: list[int] | None, image_path: Path
    ) -> None:
        code = result.get("status_code", -1)
        status_list = result.get("status_list", [])

        if expected is not None:
            self._expected_label.config(
                text=f"Expected (dataset): {format_codes(expected)}",
                fg="#1a1a1a",
            )
        else:
            self._expected_label.config(
                text=(
                    "Expected (dataset): not found — check labels folder "
                    f"({self._labels_dir.get()})"
                ),
                fg="#666666",
            )

        self._predicted_label.config(text=f"Predicted: {format_code(code)}")

        if expected is not None:
            match_text = describe_match(expected, code)
            sc = score_output(code, expected)
            if sc == 1.0:
                match_color = "#2a9d3f"
            elif sc == 0.5:
                match_color = "#e9a319"
            else:
                match_color = "#c8412a"
            self._match_label.config(text=f"Match: {match_text}", fg=match_color)
        else:
            self._match_label.config(text="Match: —", fg="#666666")

        self._flags_label.config(text=f"Flags: {format_status_list(status_list)}")

        score = score_output(code, expected) if expected else None
        text = format_analysis_report(result, expected=expected, score=score)
        self._meas_text.config(state=tk.NORMAL)
        self._meas_text.delete("1.0", tk.END)
        self._meas_text.insert(tk.END, text)
        self._meas_text.config(state=tk.DISABLED)

    def _refresh_preset_names(self) -> None:
        names = self.params_state.refresh_preset_list()
        self._preset_combo["values"] = names

    def _collect_params(self) -> dict:
        params = preset_store.get_default_params()
        for key in list(params.keys()):
            if key in preset_store.RANGE_KEYS:
                lo = int(self._vars[f"{key}_lo"].get())
                hi = int(self._vars[f"{key}_hi"].get())
                params[key] = (lo, hi)
            elif key in self._vars:
                if key in preset_store.INT_KEYS:
                    params[key] = int(self._vars[key].get())
                else:
                    params[key] = float(self._vars[key].get())
        errors = preset_store.validate_params(params)
        if errors:
            raise ValueError("\n".join(errors))
        return Classical2.normalize_params(params)

    def _apply_vars_from_params(self, params: dict) -> None:
        for key, val in params.items():
            if key in preset_store.RANGE_KEYS:
                self._vars[f"{key}_lo"].set(int(val[0]))
                self._vars[f"{key}_hi"].set(int(val[1]))
            elif key in self._vars:
                self._vars[key].set(val)

    def _load_preset_into_form(self, name: str) -> None:
        try:
            params = preset_store.load_preset(name)
        except Exception as exc:
            messagebox.showerror("Preset error", str(exc))
            return
        self.params_state.set_params(name, params)
        self._preset_var.set(name)
        self._apply_vars_from_params(params)
        self._notify(f"Loaded preset: {name}")

    def _on_load_preset(self) -> None:
        self._load_preset_into_form(self._preset_var.get())

    def _on_save_preset(self) -> None:
        try:
            params = self._collect_params()
        except ValueError as exc:
            messagebox.showerror("Validation error", str(exc))
            return
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self)
        if not name:
            return
        desc = simpledialog.askstring("Description", "Optional description:", parent=self) or ""
        try:
            preset_store.save_preset(name.strip(), params, desc)
        except ValueError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._refresh_preset_names()
        self._preset_var.set(name.strip())
        self.params_state.set_params(name.strip(), params)
        messagebox.showinfo("Saved", f"Preset saved: {name.strip()}")
        self._notify(f"Saved preset: {name.strip()}")

    def sync_params_to_state(self) -> None:
        """Push current form values into shared state (for evaluation tab)."""
        params = self._collect_params()
        self.params_state.set_params(self._preset_var.get(), params)

    def _use_data_loader(self) -> None:
        path = self.data_loader.get_selected_image_path()
        if path is None:
            messagebox.showwarning("No image", "Select an image on the Data Loader tab first.")
            return
        self._image_path = path
        self._path_label.config(text=str(path))
        self._notify(f"Using image: {path.name}")

    def _browse_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if selected:
            self._image_path = Path(selected)
            self._path_label.config(text=str(self._image_path))

    def _run_analyze(self) -> None:
        if self._image_path is None or not self._image_path.exists():
            messagebox.showwarning("No image", "Choose an image to analyze.")
            return
        if self._task_runner.is_running:
            self._notify("Analysis already running.")
            return
        try:
            params = self._collect_params()
            self.params_state.set_params(self._preset_var.get(), params)
        except ValueError as exc:
            messagebox.showerror("Validation error", str(exc))
            return

        path = self._image_path
        self._analyze_btn.config(state=tk.DISABLED)
        self._notify("Analyzing...")

        def _work():
            analyzer = Classical2(params)
            return analyzer.analyze(str(path))

        def _success(result: dict) -> None:
            self._last_result = result
            expected = self._lookup_expected(path)
            code = result.get("status_code", -1)
            self._update_result_display(result, expected, path)
            self._show_previews(result, path)
            if expected is not None:
                self._notify(
                    f"Analysis: predicted {format_code(code)}, expected {format_codes(expected)}"
                )
            else:
                self._notify(f"Analysis complete: {format_code(code)} (no label file)")

        def _finished() -> None:
            self._analyze_btn.config(state=tk.NORMAL)

        self._task_runner.run(
            _work,
            on_success=lambda r: self.after(0, lambda: _success(r)),
            on_error=lambda err: self.after(
                0, lambda: messagebox.showerror("Analysis failed", err[:2000])
            ),
            on_finished=lambda: self.after(0, _finished),
        )

    def _show_previews(self, result: dict, original_path: Path) -> None:
        images = build_analysis_visualizations(result)
        self._photos["original"] = show_image_on_label(
            self._preview_labels["original"], original_path, max_size=(520, 360)
        )
        for key in ("annotated", "bottle", "cap", "background"):
            self._photos[key] = show_bgr_on_label(
                self._preview_labels[key], images[key], max_size=(520, 360)
            )

    def _notify(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)
