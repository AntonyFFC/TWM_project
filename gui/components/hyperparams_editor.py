"""Shared hyperparameter editor backed by config.py."""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from gui.services import config_store  # noqa: E402


class HyperparamsEditor(ttk.LabelFrame):
    def __init__(
        self,
        parent: ttk.Frame,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent, text="Hyperparameters (saved to config.py)")
        self.on_status = on_status
        self._vars: dict[str, tk.Variable] = {}
        self._create_widgets()
        self.reload_from_config()

    def _create_widgets(self) -> None:
        grid = ttk.Frame(self)
        grid.pack(fill=tk.X, padx=5, pady=5)

        labels = config_store.field_labels()
        row = 0
        col = 0
        for name, py_type in config_store.EDITABLE_FIELDS.items():
            ttk.Label(grid, text=labels.get(name, name)).grid(
                row=row, column=col * 2, sticky=tk.W, padx=4, pady=2
            )
            if py_type is int:
                var: tk.Variable = tk.IntVar()
            elif py_type is float:
                var = tk.DoubleVar()
            else:
                var = tk.StringVar()
            self._vars[name] = var
            entry = ttk.Entry(grid, textvariable=var, width=12)
            entry.grid(row=row, column=col * 2 + 1, sticky=tk.W, padx=4, pady=2)
            col += 1
            if col >= 3:
                col = 0
                row += 1

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_row, text="Reload from config", command=self.reload_from_config).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_row, text="Save to config.py", command=self.save_to_config).pack(
            side=tk.LEFT, padx=4
        )

    def reload_from_config(self) -> None:
        values = config_store.load_values()
        for name, var in self._vars.items():
            var.set(values[name])
        self._notify("Hyperparameters loaded from config.py")

    def get_values(self) -> dict:
        out = {}
        for name, py_type in config_store.EDITABLE_FIELDS.items():
            out[name] = py_type(self._vars[name].get())
        return out

    def save_to_config(self) -> None:
        try:
            values = self.get_values()
            config_store.save_values(values)
        except ValueError as exc:
            messagebox.showerror("Validation error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save failed", str(exc))
            return
        messagebox.showinfo("Saved", "Hyperparameters written to config.py")
        self._notify("Hyperparameters saved to config.py")

    def _notify(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)
