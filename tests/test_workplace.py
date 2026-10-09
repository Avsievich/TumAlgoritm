"""Экраны рабочего места: таблица, правки, ход, числа, графики, выгрузка."""

import time
from pathlib import Path

import pandas as pd
import pytest

from arm.charts import FILE_HEATMAP, FILE_SURFACE
from arm.export_run import export_run
from arm.gui import (
    CHART_ZOOM_STEP,
    HEATMAP_VIEW,
    OTHER_CHART_VIEW,
    factor_count,
    image_to_dib,
)
from arm.help_text import help_for
from arm.session import drop_columns, load_working_table, save_working_csv
from tumalgoritm.data_io import ROLE_Y, build_dataset
from tumalgoritm.pipeline import run_frame

ROOT = Path(__file__).resolve().parents[1]
PARKINSON_PATH = ROOT / "reference" / "parkinsons.csv"


def _two_factor_frame() -> pd.DataFrame:
    """Короткий пример, в котором есть и тепловая карта, и поверхность."""
    return pd.DataFrame(
        {
            "x1": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "x2": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
        }
    )


def test_progress_reaches_100_in_order():
    """Проценты идут по шагам и заканчиваются сотней."""
    seen: list[int] = []
    run_frame(
        _two_factor_frame(),
        "y",
        min_dominant=1,
        max_dominant=2,
        effect_order=1,
        on_progress=lambda _name, percent: seen.append(percent),
    )
    assert seen[0] == 10
    assert seen[-1] == 100
    assert seen == sorted(seen)


def test_edited_table_is_what_gets_calculated():
    """Удаление столбца name оставляет 22 фактора, файл не обязателен."""
    if not PARKINSON_PATH.is_file():
        pytest.skip("Нет reference/parkinsons.csv")
    before = PARKINSON_PATH.read_bytes()
    original = load_working_table(PARKINSON_PATH)
    assert original.shape == (195, 24)
    edited = drop_columns(original, ["name"])
    dataset = build_dataset(edited, "status")
    assert dataset.x.shape[1] == 22
    assert "name" not in dataset.x_names
    assert PARKINSON_PATH.read_bytes() == before


def test_save_refuses_to_overwrite_source(tmp_path: Path):
    """Рабочий CSV нельзя записать поверх открытого файла."""
    source = tmp_path / "data.csv"
    pd.DataFrame({"y": [1.0, 2.0]}).to_csv(source, index=False)
    frame = load_working_table(source)
    with pytest.raises(ValueError):
        save_working_csv(frame, source, source=source)
    saved = save_working_csv(
        frame, tmp_path / "copy.csv", source=source
    )
    assert saved.is_file()
    assert source.read_text(encoding="utf-8") == saved.read_text(
        encoding="utf-8"
    )


def test_export_folder_keeps_source(tmp_path: Path):
    """Папка прогона содержит четыре вида файлов и не трогает исходник."""
    source = tmp_path / "source.csv"
    frame = _two_factor_frame()
    frame.to_csv(source, index=False)
    before = source.read_bytes()
    result = run_frame(
        frame,
        "y",
        min_dominant=2,
        max_dominant=2,
        effect_order=1,
    )
    first = export_run(frame, result, ["100%  Готово"], tmp_path / "out")
    second = export_run(frame, result, ["100%  Готово"], tmp_path / "out")
    assert first != second
    for folder in (first, second):
        assert (folder / "working.csv").is_file()
        assert (folder / "report.xlsx").is_file()
        assert (folder / "journal.txt").is_file()
        assert (folder / FILE_HEATMAP).is_file()
        assert (folder / FILE_SURFACE).is_file()
    assert source.read_bytes() == before
    summary = pd.read_excel(first / "report.xlsx", sheet_name="Summary")
    values = dict(zip(summary["Показатель"], summary["Значение"]))
    assert float(values["MDO"]) + float(values["ODO"]) == pytest.approx(1.0)


def test_help_explains_mdo():
    """Карточка MDO и список справки берут один и тот же текст."""
    from arm.help_text import format_help

    text = help_for("MDO").meaning
    assert "объяснённая доминантными факторами" in text
    shown = format_help(help_for("MDO"))
    assert "Как считается" in shown
    assert "объяснённая доминантными факторами" in shown


def test_help_covers_parameters():
    """В справке есть параметры, которых не было в коротких карточках."""
    from arm.help_guide import guide_names
    from arm.help_text import BY_NAME, format_help, help_for

    needed = (
        "отброшенные строки",
        "минимум доминантных",
        "максимум доминантных",
        "нормированные ряды",
        "коды квантования",
        "A1 и A2",
        "предупреждения",
        "Дисперсия квантования",
    )
    for name in needed:
        card = help_for(name)
        text = format_help(card)
        assert "Как считается" in text
        assert "Как читать" in text
        assert card.short
    assert set(guide_names()).isdisjoint(BY_NAME)


def test_chart_copy_uses_windows_dib():
    """В буфер уходит DIB, а не файл BMP с 14-байтовым заголовком."""
    from PIL import Image

    image = Image.new("RGB", (2, 2), "red")
    dib = image_to_dib(image)
    assert int.from_bytes(dib[:4], "little") == 40


def test_guide_covers_screens_and_export():
    """Инструкция описывает порядок работы и не перекрывает показатели."""
    from arm.help_guide import GUIDE_TITLE, full_guide, guide_for, guide_names
    from arm.help_text import BY_NAME, HOW_TO

    whole = full_guide()
    assert whole == HOW_TO
    assert "Открыть файл" in whole
    assert "22 фактора" in whole
    assert "results" in whole
    assert "100" in guide_for("Экран «Расчёт»")
    charts = guide_for("Графики")
    assert "Тепловая карта" in charts
    square = guide_for("График: радарная диаграмма")
    assert "square_radar.png" in square
    assert "ДОЭВД" in square
    assert "ДОЭВС < 0" in square
    scan_help = guide_for("Сканирование выходов")
    assert "Сканировать выходы" in scan_help
    assert "Связи" in scan_help
    assert "Связанные X" in scan_help
    surface = guide_for("График: условное среднее")
    assert "Серый крест" in surface
    names = set(guide_names())
    assert GUIDE_TITLE in names
    assert names.isdisjoint(BY_NAME)


def test_window_previews_edits_and_results(tmp_path: Path):
    """Окно показывает правки, числа, график и пояснение MDO."""
    pytest.importorskip("customtkinter")
    from arm.gui import App

    if not PARKINSON_PATH.is_file():
        pytest.skip("Нет reference/parkinsons.csv")

    sample = tmp_path / "sample.csv"
    _two_factor_frame().to_csv(sample, index=False)
    app = App()
    try:
        app.load_file(PARKINSON_PATH)
        app.update()
        size_text = app._size_label.cget("text")
        assert "195" in size_text and "24" in size_text
        assert len(app._data_tree.get_children()) == 195
        assert len(app._data_tree["columns"]) == 24

        app._column_pick.set("name")
        app._delete_column()
        app.update()
        assert "name" not in list(app._frame.columns)
        assert app._role_boxes["status"].get().startswith(ROLE_Y)
        assert factor_count(app._frame, "status") == 22

        app.load_file(sample)
        app.update()
        first_row = app._data_tree.get_children()[0]
        app._data_tree.selection_set(first_row)
        app._delete_rows()
        assert len(app._frame) == 5
        assert len(pd.read_csv(sample)) == 6

        app._column_pick.set("x1")
        app._filter_low.delete(0, "end")
        app._filter_low.insert(0, "1")
        app._filter_high.delete(0, "end")
        app._apply_filter()
        assert set(app._frame["x1"]) == {1.0}

        app.show_indicator("MDO")
        app.update()
        shown = app._help_view.get("1.0", "end")
        assert "объяснённая доминантными факторами" in shown
        app._on_help_pick("Как работать")
        guide = app._help_view.get("1.0", "end")
        assert "Открыть файл" in guide
        assert "Выгрузить" in guide

        result = run_frame(
            _two_factor_frame(),
            "y",
            nadc_x=2,
            nadc_y=3,
            min_dominant=2,
            max_dominant=2,
            effect_order=1,
        )
        app._finish_ok(result, _two_factor_frame(), ["100%  Готово"])
        app.update()
        mdo = float(app._card_values["MDO"].cget("text"))
        odo = float(app._card_values["ODO"].cget("text"))
        assert mdo + odo == pytest.approx(1.0)
        assert app._chart_box == HEATMAP_VIEW
        assert app._chart_zoom == 1.0
        assert app._copy_chart_button.cget("state") == "normal"
        base_width = app._chart_size[0]
        app._zoom_chart(CHART_ZOOM_STEP)
        assert app._chart_zoom == pytest.approx(1.0 + CHART_ZOOM_STEP)
        assert app._chart_size[0] > base_width
        assert (app._preview_dir / FILE_HEATMAP).is_file()
        assert (app._preview_dir / FILE_SURFACE).is_file()
        app._show_chart("Условное среднее")
        assert app._chart_box == OTHER_CHART_VIEW
        assert app._chart_zoom == 1.0
        assert app._run_button.cget("state") == "normal"
    finally:
        app.destroy()


def test_run_button_waits_and_keeps_warnings(tmp_path: Path):
    """Кнопка занята до конца шага, предупреждение пишется в журнал."""
    pytest.importorskip("customtkinter")
    from arm.gui import App

    path = tmp_path / "thin.csv"
    pd.DataFrame(
        {"x": [0.1, 0.2, 0.8, 0.9], "y": [0.0, 1.0, 0.0, 1.0]}
    ).to_csv(path, index=False)
    app = App()
    try:
        app.load_file(path)
        app.update()
        app._nadc_x.delete(0, "end")
        app._nadc_x.insert(0, "4")
        app._min_dom.delete(0, "end")
        app._min_dom.insert(0, "1")
        app._max_dom.delete(0, "end")
        app._max_dom.insert(0, "1")
        app._order.delete(0, "end")
        app._order.insert(0, "1")
        app._start()
        assert app._run_button.cget("state") == "disabled"
        deadline = time.time() + 30
        while time.time() < deadline:
            app.update()
            if app._run_button.cget("state") == "normal":
                break
            time.sleep(0.05)
        log = app._log.get("1.0", "end")
        assert app._run_button.cget("state") == "normal"
        assert "100%" in log
        assert "Предупреждение" in log
        assert app.winfo_exists()
    finally:
        app.destroy()


def _wait_run(app) -> None:
    """Ждёт, пока кнопка «Считать» снова станет доступна."""
    deadline = time.time() + 30
    while time.time() < deadline:
        app.update()
        if app._run_button.cget("state") == "normal" and not app._running:
            return
        time.sleep(0.05)
    raise AssertionError(app._log.get("1.0", "end"))


def test_second_run_after_parameter_change(tmp_path: Path):
    """После смены параметров повторное «Считать» берёт новые значения."""
    pytest.importorskip("customtkinter")
    from arm.gui import App

    path = tmp_path / "pair.csv"
    pd.DataFrame(
        {
            "x1": [0.0, 0.0, 1.0, 1.0],
            "x2": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.2, 0.8, 1.0],
        }
    ).to_csv(path, index=False)
    app = App()
    try:
        app.load_file(path)
        app.update()
        app.show_screen("Параметры")
        app._min_dom.delete(0, "end")
        app._min_dom.insert(0, "1")
        app._max_dom.delete(0, "end")
        app._max_dom.insert(0, "2")
        app._order.delete(0, "end")
        app._order.insert(0, "1")
        app._nadc_x.delete(0, "end")
        app._nadc_x.insert(0, "2")
        app.show_screen("Расчёт")
        app.update()
        assert app._run_button._mouse_inside is True
        app._run_button._on_release()
        _wait_run(app)
        assert app._result.prepared.nadc_x == 2

        app.show_screen("Параметры")
        app._nadc_x.delete(0, "end")
        app._nadc_x.insert(0, "3")
        app.show_screen("Расчёт")
        app.update()
        assert app._run_button._mouse_inside is True
        app._run_button._on_release()
        _wait_run(app)
        assert app._result.prepared.nadc_x == 3
        assert app._run_button.cget("state") == "normal"
        assert "100%" in app._log.get("1.0", "end")
    finally:
        app.destroy()
