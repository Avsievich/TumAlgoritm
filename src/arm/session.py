"""Правка загруженной таблицы до расчёта.

Исходный файл здесь не записывается. Рабочая копия живёт в памяти,
пока пользователь сам не сохранит её отдельным файлом.
"""

from pathlib import Path

import pandas as pd

from tumalgoritm.data_io import ROLE_EXCLUDE, ROLE_Y, ROLE_Z, load_table

# Имена, которые при открытии предлагаются как выход.
OUTPUT_NAMES = {"y", "status"}


def load_working_table(path: str | Path) -> pd.DataFrame:
    """Читает CSV или Excel в рабочую таблицу.

    Args:
        path: путь к файлу данных.

    Returns:
        Копия таблицы. Запись в этот файл не выполняется.
    """
    return load_table(path).copy()


def column_overview(frame: pd.DataFrame) -> pd.DataFrame:
    """Краткая сводка столбцов: тип и число пустых ячеек.

    Args:
        frame: рабочая таблица.

    Returns:
        Таблица со столбцами «Столбец», «Тип», «Пустых».
    """
    rows = []
    for name in frame.columns:
        series = frame[name]
        rows.append(
            {
                "Столбец": str(name),
                "Тип": str(series.dtype),
                "Пустых": int(series.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def suggest_role(name: str, series: pd.Series) -> str:
    """Предлагает роль: status и y — выход, текст — исключить, иначе Z.

    Args:
        name: имя столбца.
        series: значения столбца.

    Returns:
        Код роли Z, Y или «исключить».
    """
    numeric = bool(pd.api.types.is_numeric_dtype(series))
    if str(name).strip().lower() in OUTPUT_NAMES and numeric:
        return ROLE_Y
    if not numeric:
        return ROLE_EXCLUDE
    return ROLE_Z


def drop_columns(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Удаляет столбцы из рабочей копии.

    Args:
        frame: текущая таблица.
        names: имена столбцов, которые нужно убрать.

    Returns:
        Новая таблица без этих столбцов.
    """
    existing = [name for name in names if name in frame.columns]
    return frame.drop(columns=existing).reset_index(drop=True)


def drop_rows(frame: pd.DataFrame, indexes: list[int]) -> pd.DataFrame:
    """Удаляет строки по номерам текущей таблицы.

    Args:
        frame: текущая таблица.
        indexes: номера строк, как они видны в таблице.

    Returns:
        Таблица без выбранных строк, номера начинаются с нуля.
    """
    return frame.drop(index=indexes).reset_index(drop=True)


def apply_range_filter(
    frame: pd.DataFrame,
    column: str,
    low: float | None,
    high: float | None,
) -> pd.DataFrame:
    """Оставляет строки, где числовой столбец лежит в границах.

    Пустая граница означает «без ограничения с этой стороны».
    Нечисловые значения столбца в выборку не попадают.

    Args:
        frame: текущая таблица.
        column: имя числового столбца.
        low: нижняя граница или None.
        high: верхняя граница или None.

    Returns:
        Отфильтрованная таблица.

    Raises:
        ValueError: столбца нет или он не числовой.
    """
    if column not in frame.columns:
        raise ValueError(f"Столбца «{column}» нет в таблице")
    if not pd.api.types.is_numeric_dtype(frame[column]):
        raise ValueError(f"Столбец «{column}» не числовой")
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = values.notna()
    if low is not None:
        mask = mask & (values >= low)
    if high is not None:
        mask = mask & (values <= high)
    return frame.loc[mask].reset_index(drop=True)


def save_working_csv(
    frame: pd.DataFrame,
    path: str | Path,
    source: str | Path | None = None,
) -> Path:
    """Сохраняет рабочую таблицу в новый CSV.

    Args:
        frame: таблица, которая пойдёт в файл.
        path: куда записать.
        source: исходный файл. Совпадающий путь отклоняется,
            чтобы не затереть загруженные данные.

    Returns:
        Путь к записанному файлу.

    Raises:
        ValueError: путь совпадает с исходным файлом.
    """
    target = Path(path)
    if source is not None and target.resolve() == Path(source).resolve():
        raise ValueError("Нельзя перезаписать исходный файл")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target
