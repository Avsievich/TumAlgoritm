"""Окно открывается и предлагает роль выхода для столбца y."""

from pathlib import Path

import pandas as pd
import pytest

from tumalgoritm.data_io import ROLE_Y


def test_gui_suggests_output_role(tmp_path: Path):
    """После загрузки файла столбец y получает роль выхода."""
    customtkinter = pytest.importorskip("customtkinter")
    del customtkinter
    from arm.gui import App

    path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "y": [0.0, 1.0, 0.0]}).to_csv(
        path, index=False
    )
    app = App()
    try:
        app.load_file(path)
        app.update()
        assert app._role_boxes["y"].get().startswith(ROLE_Y)
        assert "a" in app._role_boxes
    finally:
        app.destroy()
