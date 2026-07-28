from __future__ import annotations

import importlib


def test_app_import_does_not_start_streamlit() -> None:
    module = importlib.import_module("app")

    assert hasattr(module, "main")
