"""Загрузка таблицы и разметка ролей переменных.

По разделу 1.1 методики переменные делятся на:
    Z — измеряемые неуправляемые входы;
    U — управляемые входы;
    Y — выход;
    Ξ — столбцы, которые в расчёт не входят.

В формулах Z и U обрабатываются одинаково: оба становятся
столбцами матрицы факторов X. Роль сохраняется для отчёта.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Роли. Буква «Ξ» совпадает с обозначением книги.
ROLE_EXCLUDE = "Ξ"
ROLE_Z = "Z"
ROLE_U = "U"
ROLE_Y = "Y"

# Расширения, которые умеет читать загрузчик.
_CSV_SUFFIXES = {".csv", ".txt"}
_EXCEL_SUFFIXES = {".xlsx", ".xls"}


@dataclass
class Dataset:
    """Факторы и выход после разметки ролей.

    Attributes:
        x: матрица факторов, форма (N, n), только конечные числа.
        y: вектор выхода длины N.
        x_names: имена столбцов X в том же порядке, что и столбцы.
        x_roles: роль каждого столбца X, «Z» или «U».
        y_name: имя столбца выхода.
        n_dropped: сколько строк отброшено из-за пропусков.
    """

    x: np.ndarray
    y: np.ndarray
    x_names: list[str]
    x_roles: list[str]
    y_name: str
    n_dropped: int


def load_table(path: str | Path) -> pd.DataFrame:
    """Читает CSV/TXT или Excel с заголовком в первой строке.

    Args:
        path: путь к файлу данных.

    Returns:
        Таблица с именами столбцов в виде строк.

    Raises:
        FileNotFoundError: файла нет.
        ValueError: неизвестное расширение или пустая таблица.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Файл данных не найден: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in _CSV_SUFFIXES:
        frame = pd.read_csv(file_path)
    elif suffix in _EXCEL_SUFFIXES:
        frame = pd.read_excel(file_path)
    else:
        raise ValueError(
            "Поддерживаются файлы CSV, TXT и Excel, "
            f"получено расширение «{suffix}»"
        )

    if frame.empty:
        raise ValueError(f"Файл не содержит строк: {file_path}")

    frame = frame.copy()
    frame.columns = [str(name) for name in frame.columns]
    _require_unique_columns(frame)
    return frame


def build_dataset(
    frame: pd.DataFrame,
    y_name: str,
    u_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> Dataset:
    """Собирает матрицу X и вектор Y по ролям столбцов.

    Числовые столбцы, не названные выходом, управляемыми или
    исключёнными, получают роль Z. Нечисловые столбцы отбрасываются.

    Args:
        frame: исходная таблица.
        y_name: имя столбца выхода.
        u_names: имена управляемых входов (роль U).
        exclude_names: имена столбцов, которые нужно исключить.

    Returns:
        Dataset с выровненными по строкам X и Y.

    Raises:
        ValueError: нет выхода, нет факторов или роли противоречат друг другу.
    """
    table = frame.copy()
    table.columns = [str(name) for name in table.columns]
    _require_unique_columns(table)

    controlled = list(u_names or [])
    excluded = list(exclude_names or [])
    _validate_role_names(table, y_name, controlled, excluded)

    if not pd.api.types.is_numeric_dtype(table[y_name]):
        raise ValueError(f"Столбец выхода «{y_name}» должен быть числовым")

    x_names: list[str] = []
    x_roles: list[str] = []
    for name in table.columns:
        if name == y_name or name in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(table[name]):
            continue
        role = ROLE_U if name in controlled else ROLE_Z
        x_names.append(name)
        x_roles.append(role)

    if not x_names:
        raise ValueError("После разметки ролей не осталось числовых факторов")

    used = x_names + [y_name]
    before = len(table)
    numeric = table[used].apply(pd.to_numeric, errors="coerce")
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    n_dropped = before - len(clean)
    if clean.empty:
        raise ValueError("После удаления пропусков не осталось наблюдений")

    return Dataset(
        x=clean[x_names].to_numpy(dtype=float),
        y=clean[y_name].to_numpy(dtype=float),
        x_names=x_names,
        x_roles=x_roles,
        y_name=y_name,
        n_dropped=n_dropped,
    )


def load_dataset(
    path: str | Path,
    y_name: str,
    u_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> Dataset:
    """Загружает файл и сразу размечает роли.

    Args:
        path: путь к CSV или Excel.
        y_name: имя столбца выхода.
        u_names: управляемые входы.
        exclude_names: столбцы, которые нужно пропустить.

    Returns:
        Готовый Dataset.
    """
    return build_dataset(
        load_table(path),
        y_name,
        u_names=u_names,
        exclude_names=exclude_names,
    )


def _require_unique_columns(frame: pd.DataFrame) -> None:
    """Проверяет, что имена столбцов не повторяются."""
    names = list(frame.columns)
    if len(names) != len(set(names)):
        raise ValueError("В таблице есть повторяющиеся имена столбцов")


def _validate_role_names(
    frame: pd.DataFrame,
    y_name: str,
    controlled: list[str],
    excluded: list[str],
) -> None:
    """Проверяет, что роли ссылаются на существующие разные столбцы."""
    columns = set(frame.columns)
    if y_name not in columns:
        raise ValueError(f"Столбец выхода «{y_name}» не найден")

    unknown = [name for name in controlled + excluded if name not in columns]
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"В файле нет столбцов: {joined}")

    overlap = set(controlled) & set(excluded)
    if y_name in controlled or y_name in excluded:
        overlap.add(y_name)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(f"Столбцы указаны в нескольких ролях сразу: {joined}")
