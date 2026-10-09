"""Перебор выходов Y: рейтинг, таблицы и исключённые столбцы."""

from pathlib import Path

import pandas as pd
import pytest

from arm.y_scan import (
    LINK_PDO_SHARE,
    SHEET_LINKS,
    SHEET_PAIRS,
    SHEET_PDO,
    SHEET_RANK,
    build_scan_frames,
    candidate_y_names,
    linked_factor_indices,
    scan_outputs,
    successful_y_names,
    summary_line,
    write_y_scan_report,
)


def _toy_frame() -> pd.DataFrame:
    """Маленькая таблица: два сильных сигнала и шум."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "a": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "b": [0.0, 0.1, 0.0, 0.9, 1.0, 0.95],
            "noise": [0.2, 0.8, 0.3, 0.7, 0.1, 0.6],
            "y_strong": [0.0, 0.05, 0.1, 0.9, 0.95, 1.0],
            "y_weak": [0.4, 0.6, 0.5, 0.55, 0.45, 0.5],
        }
    )


def test_excluded_column_is_not_a_candidate():
    """Столбец с ролью «исключить» не становится выходом."""
    frame = _toy_frame()
    names = candidate_y_names(frame, exclude_names=["id", "noise"])
    assert "id" not in names
    assert "noise" not in names
    assert "y_strong" in names


def test_scan_ranks_by_mdo_and_builds_frames(tmp_path: Path):
    """Рейтинг отсортирован по MDO, листы отчёта на месте."""
    frame = _toy_frame()
    scan = scan_outputs(
        frame,
        exclude_names=["id"],
        params={
            "nadc_x": 1,
            "nadc_y": 2,
            "effect_order": 1,
            "min_dominant": 1,
            "max_dominant": 3,
            "eta0": 0.0,
        },
    )
    assert scan.n_candidates == len(candidate_y_names(frame, ["id"]))
    assert len(scan.rows) == scan.n_candidates
    ok_rows = [row for row in scan.rows if row.ok]
    assert ok_rows
    mdo_values = [row.mdo for row in ok_rows]
    assert mdo_values == sorted(mdo_values, reverse=True)
    frames = build_scan_frames(scan)
    assert SHEET_RANK in frames
    assert SHEET_LINKS in frames
    assert SHEET_PDO in frames
    assert SHEET_PAIRS in frames
    assert list(frames[SHEET_RANK]["Y"]) == [
        row.y_name for row in scan.rows
    ]
    assert "Связанные X" in frames[SHEET_RANK].columns
    best = scan.best
    assert best is not None
    assert best.linked_names
    assert best.result is not None
    assert linked_factor_indices(best.result) == best.linked_idx
    peak = max(float(item.pdo_raw) for item in best.result.factors)
    floor = peak * LINK_PDO_SHARE
    for index in best.linked_idx:
        assert best.result.factors[index].pdo_raw >= floor
    assert best.linked_names[0] in str(
        frames[SHEET_RANK].iloc[0]["Связанные X"]
    )
    links = frames[SHEET_LINKS]
    assert {"Y", "X", "ПДО", "Порог"}.issubset(links.columns)
    assert best.y_name in set(links["Y"])
    assert set(best.linked_names) == set(
        links.loc[links["Y"] == best.y_name, "X"]
    )
    for _, row in links.loc[links["Y"] == best.y_name].iterrows():
        assert row["ПДО"] >= row["Порог"] - 1e-12
    filtered = build_scan_frames(scan, best.y_name)
    assert set(filtered[SHEET_LINKS]["Y"]) == {best.y_name}
    assert set(filtered[SHEET_PDO]["Y"]) == {best.y_name}
    path = write_y_scan_report(scan, tmp_path / "y_scan_report.xlsx")
    assert path.is_file()
    assert path.stat().st_size > 0
    assert best.y_name in summary_line(scan)


def test_scan_summary_visible_after_gui_fill():
    """После скана на экране результатов видна строка лучшего Y."""
    pytest.importorskip("customtkinter")
    from arm.gui import App

    frame = _toy_frame()
    scan = scan_outputs(
        frame,
        exclude_names=["id"],
        params={
            "nadc_x": 1,
            "nadc_y": 2,
            "effect_order": 1,
            "min_dominant": 1,
            "max_dominant": 3,
            "eta0": 0.0,
        },
    )
    app = App()
    try:
        app._frame = frame
        app._finish_scan(scan, frame, ["готово"])
        app.update()
        text = app._scan_summary.cget("text")
        assert scan.best is not None
        assert scan.best.y_name in text
        assert "MDO" in text
        shown = app._result_tree.get_children()
        assert shown
        first = app._result_tree.item(shown[0], "values")
        assert scan.best.y_name in first or any(
            name in first for name in scan.best.linked_names
        )
        assert app._table_pick.get() == SHEET_LINKS
        assert app._chart_paths
        other = [
            name for name in successful_y_names(scan)
            if name != scan.best.y_name
        ]
        if other:
            app._on_scan_y_filter(other[0])
            app.update()
            assert any(
                other[0].replace(":", "_") in str(path)
                or "scan" in str(path)
                for path in app._chart_paths.values()
            )
    finally:
        app.destroy()
