"""Run blocking work on a background thread; update UI on the main thread."""
from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

import tkinter as tk


class TaskRunner:
    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def run(
        self,
        fn: Callable[[], Any],
        *,
        on_log: Callable[[str], None] | None = None,
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        if self._running:
            if on_log:
                on_log("A task is already running.")
            return

        self._running = True

        def _done_callback(result: Any = None, error: str | None = None) -> None:
            self._running = False
            if error:
                if on_error:
                    on_error(error)
            elif on_success:
                on_success(result)
            if on_finished:
                on_finished()

        def _worker() -> None:
            try:
                result = fn()
                self.root.after(0, lambda: _done_callback(result=result))
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                self.root.after(0, lambda: _done_callback(error=f"{exc}\n{tb}"))

        import threading

        threading.Thread(target=_worker, daemon=True).start()
