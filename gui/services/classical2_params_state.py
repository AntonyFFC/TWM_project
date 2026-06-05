"""Shared Classical2 preset/params state for GUI tabs."""
from __future__ import annotations

from typing import Any

from gui.services.classical2_preset_store import get_default_params, list_presets, load_preset


class Classical2ParamsState:
    def __init__(self) -> None:
        self.preset_name: str = "default"
        self.params: dict[str, Any] = get_default_params()

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def get_preset_name(self) -> str:
        return self.preset_name

    def set_from_preset(self, name: str) -> None:
        self.preset_name = name
        self.params = load_preset(name)

    def set_params(self, name: str, params: dict[str, Any]) -> None:
        self.preset_name = name
        self.params = dict(params)

    def refresh_preset_list(self) -> list[str]:
        return list_presets()
