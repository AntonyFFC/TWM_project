"""Export tab: copy results, inference, demo."""
from __future__ import annotations

import shutil
import tkinter as tk
import zipfile
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import CLASS_NAMES, METRICS_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR  # noqa: E402
from gui.components.data_loader import DataLoaderComponent  # noqa: E402
from gui.services import metrics_service  # noqa: E402
from gui.utils.image_preview import show_image_on_label  # noqa: E402
from gui.utils.task_runner import TaskRunner  # noqa: E402


class ExportResultsComponent(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Frame,
        *,
        data_loader: DataLoaderComponent,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.data_loader = data_loader
        self.on_status = on_status
        self._task_runner = TaskRunner(self.winfo_toplevel())
        self._model_var = tk.StringVar()
        self._infer_image_var = tk.StringVar()
        self._classical_demo_var = tk.StringVar()
        self._ml_demo_var = tk.StringVar()
        self._photo = None
        self._create_widgets()
        self.refresh_models()

    def _create_widgets(self) -> None:
        title = tk.Label(self, text="Export & Inference", font=("Arial", 14, "bold"))
        title.pack(pady=8)

        files_frame = ttk.LabelFrame(self, text="Export files")
        files_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(
            files_frame, text="Export summary.csv...", command=self._export_summary
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(
            files_frame, text="Export results bundle (zip)...", command=self._export_zip
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(
            files_frame, text="Open results folder", command=self._open_results_folder
        ).pack(side=tk.LEFT, padx=5, pady=5)

        infer_frame = ttk.LabelFrame(self, text="Inference")
        infer_frame.pack(fill=tk.X, padx=10, pady=5)

        row1 = ttk.Frame(infer_frame)
        row1.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(row1, text="Model:").pack(side=tk.LEFT)
        self._model_combo = ttk.Combobox(
            row1, textvariable=self._model_var, state="readonly", width=36
        )
        self._model_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Refresh models", command=self.refresh_models).pack(
            side=tk.LEFT, padx=4
        )

        row2 = ttk.Frame(infer_frame)
        row2.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(row2, text="Image:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self._infer_image_var, width=50).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(row2, text="Use Data Loader", command=self._use_data_loader_image).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(row2, text="Browse...", command=self._browse_infer_image).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(row2, text="Predict", command=self._run_inference).pack(
            side=tk.LEFT, padx=4
        )

        self._infer_output = scrolledtext.ScrolledText(infer_frame, height=6, state=tk.DISABLED)
        self._infer_output.pack(fill=tk.X, padx=5, pady=5)

        demo_frame = ttk.LabelFrame(self, text="Demo visualization")
        demo_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        demo_opts = ttk.Frame(demo_frame)
        demo_opts.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(demo_opts, text="Classical run:").pack(side=tk.LEFT)
        ttk.Entry(demo_opts, textvariable=self._classical_demo_var, width=18).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(demo_opts, text="ML run:").pack(side=tk.LEFT)
        ttk.Entry(demo_opts, textvariable=self._ml_demo_var, width=18).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(demo_opts, text="Auto-fill", command=self._autofill_demo_runs).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(demo_opts, text="Generate demo PNG", command=self._run_demo).pack(
            side=tk.LEFT, padx=4
        )

        self._demo_label = tk.Label(
            demo_frame,
            text="Demo preview",
            bg="#2f2f2f",
            fg="white",
        )
        self._demo_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def refresh_models(self) -> None:
        from evaluation.infer import list_available_models

        models = list_available_models()
        self._model_combo["values"] = models
        if models and not self._model_var.get():
            self._model_var.set(models[0])
        self._autofill_demo_runs()
        self._notify(f"Found {len(models)} trained model(s)")

    def _autofill_demo_runs(self) -> None:
        from demo import _pick_run

        if not self._classical_demo_var.get():
            c = _pick_run("hog_svm")
            if c:
                self._classical_demo_var.set(c)
        if not self._ml_demo_var.get():
            m = _pick_run("resnet18")
            if m:
                self._ml_demo_var.set(m)

    def _use_data_loader_image(self) -> None:
        path = self.data_loader.get_selected_image_path()
        if path is None:
            messagebox.showwarning("No image", "Select an image on the Data Loader tab first.")
            return
        self._infer_image_var.set(str(path))

    def _browse_infer_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select image for inference",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self._infer_image_var.set(selected)

    def _set_infer_output(self, text: str) -> None:
        self._infer_output.config(state=tk.NORMAL)
        self._infer_output.delete("1.0", tk.END)
        self._infer_output.insert(tk.END, text)
        self._infer_output.config(state=tk.DISABLED)

    def _run_inference(self) -> None:
        model_name = self._model_var.get()
        image_path = self._infer_image_var.get().strip()
        if not model_name:
            messagebox.showwarning("No model", "Train a model or refresh the model list.")
            return
        if not image_path:
            messagebox.showwarning("No image", "Choose an image for inference.")
            return

        from evaluation.infer import load_trained_method, predict_image

        try:
            method, meta = load_trained_method(model_name)
            _cls_id, cls_name, proba = predict_image(method, Path(image_path))
            lines = [
                f"Model: {model_name}",
                f"Trained on: {meta.get('trained_on', '?')}",
                f"Image: {image_path}",
                f"Prediction: {cls_name}",
                "",
                "Class probabilities:",
            ]
            for name, p in zip(CLASS_NAMES, proba):
                lines.append(f"  {name}: {p:.4f}")
            self._set_infer_output("\n".join(lines))
            self._notify(f"Prediction: {cls_name}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Inference failed", str(exc))

    def _export_summary(self) -> None:
        src = metrics_service.SUMMARY_CSV
        if not src.exists():
            messagebox.showwarning(
                "No summary",
                "summary.csv not found. Run comparison on the Results Comparison tab.",
            )
            return
        dest = filedialog.asksaveasfilename(
            title="Save summary as",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="summary.csv",
        )
        if not dest:
            return
        shutil.copy2(src, dest)
        self._notify(f"Exported summary to {dest}")

    def _export_zip(self) -> None:
        dest = filedialog.asksaveasfilename(
            title="Save results bundle",
            defaultextension=".zip",
            filetypes=[("ZIP", "*.zip"), ("All files", "*.*")],
            initialfile="twm_results.zip",
        )
        if not dest:
            return
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in (METRICS_DIR, PLOTS_DIR, MODELS_DIR):
                if not folder.exists():
                    continue
                for fp in folder.rglob("*"):
                    if fp.is_file():
                        zf.write(fp, fp.relative_to(RESULTS_DIR.parent))
        self._notify(f"Exported bundle to {dest}")

    def _open_results_folder(self) -> None:
        import os
        import subprocess

        path = str(RESULTS_DIR)
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        self._notify("Opened results folder")

    def _run_demo(self) -> None:
        if self._task_runner.is_running:
            self._notify("A task is already running.")
            return

        classical = self._classical_demo_var.get().strip() or None
        ml = self._ml_demo_var.get().strip() or None

        def _work():
            import demo

            return demo.make_grid(classical, ml, n_per_class=2)

        self._notify("Generating demo...")
        self._task_runner.run(
            _work,
            on_success=lambda out_path: self.after(
                0,
                lambda: (
                    self._show_demo(out_path),
                    self._notify(f"Demo saved: {out_path}"),
                ),
            ),
            on_error=lambda err: self.after(
                0, lambda: messagebox.showerror("Demo failed", err[:2000])
            ),
        )

    def _show_demo(self, path: Path) -> None:
        self._photo = show_image_on_label(
            self._demo_label,
            path,
            max_size=(700, 420),
            placeholder="Demo image not found",
        )

    def _notify(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)
