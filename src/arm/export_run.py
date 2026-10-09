"""Папка одного прогона: таблица, Excel, рисунки и журнал.

Повторный запуск создаёт новую папку и не затирает предыдущую.
Исходный файл данных эта запись не открывает на изменение.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from arm.charts import save_charts
from arm.report import write_report
from tumalgoritm.pipeline import AnalysisResult


def export_run(
    frame: pd.DataFrame,
    result: AnalysisResult,
    journal: list[str],
    root: str | Path,
) -> Path:
    """Складывает все файлы прогона в папку с датой и временем.

    Args:
        frame: рабочая таблица, которая участвовала в расчёте.
        result: показатели.
        journal: строки хода расчёта и предупреждения.
        root: родительская папка. Внутри неё создаётся подпапка.

    Returns:
        Путь к папке прогона.
    """
    folder = _unique_folder(Path(root))
    frame.to_csv(folder / "working.csv", index=False)
    write_report(result, folder / "report.xlsx")
    save_charts(result, folder)
    lines = list(journal)
    for text in result.warnings:
        warning = f"Предупреждение: {text}"
        if warning not in lines:
            lines.append(warning)
    (folder / "journal.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return folder


def _unique_folder(root: Path) -> Path:
    """Создаёт папку ГГГГММДД_ЧЧММСС, при совпадении добавляет номер."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / stamp
    number = 2
    while candidate.exists():
        candidate = root / f"{stamp}_{number}"
        number += 1
    candidate.mkdir()
    return candidate
