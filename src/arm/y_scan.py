"""Перебор столбцов как выходов Y и отчёт о взаимосвязях.

Формулы не дублируются: каждый выход считается через run_frame.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from tumalgoritm.pipeline import AnalysisResult, run_frame

# Имена листов совпадают на экране и в Excel.
SHEET_RANK = "Рейтинг"
SHEET_LINKS = "Связи"
SHEET_PDO = "ПДО"
SHEET_PAIRS = "Пары"
SHEET_DOMINANT = "Доминантные"

# X попадает в отчёт о связях, если его ПДО не ниже этой
# доли от максимального ПДО среди факторов данного Y.
LINK_PDO_SHARE = 0.5

ProgressFn = Callable[[str, int], None]


@dataclass
class YScanRow:
    """Одна строка рейтинга выходов.

    Attributes:
        y_name: имя столбца, который был выходом.
        ok: расчёт завершился без ошибки.
        error: текст ошибки, если ok ложно.
        n_obs: число строк после очистки.
        mdo: итоговое МДО.
        odo: ОДО.
        joint_info: I(X_D; Y).
        correlation_r: множественная корреляция.
        n_dominant: размер доминантного набора.
        pdo_sum: сумма сырого ПДО доминантных факторов.
        doevd_sum: сумма ДОЭВД пар среди связанных X.
        dominant_names: имена доминантных факторов (отбор МДО).
        linked_names: X с ПДО не ниже половины от максимума.
        linked_idx: индексы тех же связанных X.
        result: полный итог, если расчёт удался.
    """

    y_name: str
    ok: bool
    error: str = ""
    n_obs: int = 0
    mdo: float = 0.0
    odo: float = 0.0
    joint_info: float = 0.0
    correlation_r: float = 0.0
    n_dominant: int = 0
    pdo_sum: float = 0.0
    doevd_sum: float = 0.0
    dominant_names: list[str] = field(default_factory=list)
    linked_names: list[str] = field(default_factory=list)
    linked_idx: list[int] = field(default_factory=list)
    result: AnalysisResult | None = None


@dataclass
class YScanResult:
    """Итог перебора всех кандидатов в Y.

    Attributes:
        rows: строки рейтинга, уже отсортированные.
        exclude_names: столбцы, которые не участвовали.
        params: параметры расчёта.
        n_candidates: сколько столбцов пробовали как Y.
    """

    rows: list[YScanRow]
    exclude_names: list[str]
    params: dict
    n_candidates: int

    @property
    def best(self) -> YScanRow | None:
        """Первая успешная строка рейтинга или None."""
        for row in self.rows:
            if row.ok:
                return row
        return None


def candidate_y_names(
    frame: pd.DataFrame,
    exclude_names: list[str] | None = None,
) -> list[str]:
    """Числовые столбцы, которые можно поставить выходом.

    Args:
        frame: рабочая таблица.
        exclude_names: столбцы с ролью «исключить».

    Returns:
        Имена в порядке столбцов таблицы.
    """
    skipped = set(exclude_names or [])
    names: list[str] = []
    for name in frame.columns:
        label = str(name)
        if label in skipped:
            continue
        if not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        names.append(label)
    return names


def scan_outputs(
    frame: pd.DataFrame,
    exclude_names: list[str] | None = None,
    params: dict | None = None,
    on_progress: ProgressFn | None = None,
) -> YScanResult:
    """Считает расчёт для каждого кандидата в Y.

    Остальные неисключённые числовые столбцы становятся факторами.
    Роли U на время скана не выделяются: набор X должен быть
    одинаковым для сравнения выходов.

    Args:
        frame: рабочая таблица.
        exclude_names: столбцы вне расчёта.
        params: аргументы run_frame без y_name.
        on_progress: сообщение шага и процент от 0 до 100.

    Returns:
        Рейтинг и полные итоги по успешным Y.

    Raises:
        ValueError: нет ни одного кандидата в Y.
    """
    excluded = list(exclude_names or [])
    options = dict(params or {})
    candidates = candidate_y_names(frame, excluded)
    if not candidates:
        raise ValueError(
            "Нет числовых столбцов для сканирования выходов"
        )
    rows: list[YScanRow] = []
    total = len(candidates)
    for index, y_name in enumerate(candidates):
        percent = int(round(100 * index / total))
        _notify(
            on_progress,
            f"Y {index + 1}/{total}: {y_name}",
            percent,
        )
        rows.append(_run_one(frame, y_name, excluded, options))
    rows.sort(key=_rank_key)
    _notify(on_progress, "Сканирование выходов готово", 100)
    return YScanResult(
        rows=rows,
        exclude_names=excluded,
        params=options,
        n_candidates=total,
    )


def build_scan_frames(
    scan: YScanResult,
    y_filter: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Таблицы отчёта для экрана и Excel.

    Args:
        scan: итог перебора.
        y_filter: если задан, листы ПДО, Пары и Доминантные
            содержат только этот выход. Рейтинг всегда полный.

    Returns:
        Имя листа и таблица.
    """
    return {
        SHEET_RANK: _rank_frame(scan),
        SHEET_LINKS: _links_frame(scan, y_filter),
        SHEET_PDO: _pdo_frame(scan, y_filter),
        SHEET_PAIRS: _pairs_frame(scan, y_filter),
        SHEET_DOMINANT: _dominant_frame(scan, y_filter),
    }


def write_y_scan_report(
    scan: YScanResult,
    path: str | Path,
) -> Path:
    """Пишет полный отчёт скана в Excel.

    Args:
        scan: итог перебора.
        path: файл xlsx.

    Returns:
        Путь к файлу.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = build_scan_frames(scan)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return output


def export_y_scan(
    scan: YScanResult,
    journal: list[str],
    root: str | Path,
) -> Path:
    """Папка прогона скана: Excel и журнал.

    Args:
        scan: итог перебора.
        journal: строки хода.
        root: родительская папка results.

    Returns:
        Путь к созданной папке.
    """
    folder = _unique_folder(Path(root))
    write_y_scan_report(scan, folder / "y_scan_report.xlsx")
    lines = list(journal)
    best = scan.best
    if best is not None:
        lines.append(
            f"Лучший выход: {best.y_name}, MDO = {best.mdo:.6g}"
        )
    (folder / "journal.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return folder


def summary_line(scan: YScanResult) -> str:
    """Короткая строка для верхней панели результатов."""
    best = scan.best
    if best is None:
        return (
            f"Ни один из {scan.n_candidates} выходов "
            "не посчитался"
        )
    return (
        f"Сильнее всего связан выход {best.y_name}, "
        f"MDO = {best.mdo:.4g}; "
        f"просмотрено {scan.n_candidates} Y"
    )


def successful_y_names(scan: YScanResult) -> list[str]:
    """Имена успешных Y в порядке рейтинга."""
    return [row.y_name for row in scan.rows if row.ok]


def linked_factor_indices(result: AnalysisResult) -> list[int]:
    """Индексы X с ПДО не ниже половины от максимума по этому Y.

    Args:
        result: итог расчёта для одного выхода.

    Returns:
        Индексы по убыванию ПДО. При нулевом максимуме список пуст.
    """
    raw = [float(item.pdo_raw) for item in result.factors]
    if not raw:
        return []
    peak = max(raw)
    if peak <= 0.0:
        return []
    floor = peak * LINK_PDO_SHARE
    order = sorted(
        range(len(raw)),
        key=lambda index: raw[index],
        reverse=True,
    )
    return [index for index in order if raw[index] >= floor]


def _run_one(
    frame: pd.DataFrame,
    y_name: str,
    excluded: list[str],
    params: dict,
) -> YScanRow:
    """Один прогон. Ошибка не останавливает весь скан."""
    try:
        result = run_frame(
            frame,
            y_name,
            exclude_names=excluded,
            **params,
        )
    except Exception as exc:
        return YScanRow(
            y_name=y_name,
            ok=False,
            error=str(exc),
        )
    dominant = list(result.dominant_idx)
    names = result.column_names
    linked = linked_factor_indices(result)
    linked_set = set(linked)
    pdo_sum = float(
        sum(result.factors[index].pdo_raw for index in linked)
    )
    doevd_sum = 0.0
    for link in result.pairs:
        if link.i in linked_set and link.j in linked_set:
            doevd_sum += float(link.doevd)
    return YScanRow(
        y_name=y_name,
        ok=True,
        n_obs=int(result.n_obs),
        mdo=float(result.mdo),
        odo=float(result.odo),
        joint_info=float(result.information.joint_dominant),
        correlation_r=float(result.correlation_r),
        n_dominant=len(dominant),
        pdo_sum=pdo_sum,
        doevd_sum=doevd_sum,
        dominant_names=[names[index] for index in dominant],
        linked_names=[names[index] for index in linked],
        linked_idx=linked,
        result=result,
    )


def _rank_key(row: YScanRow) -> tuple:
    """Сортировка: успех выше, затем MDO, информация, сумма ПДО."""
    return (
        0 if row.ok else 1,
        -row.mdo,
        -row.joint_info,
        -row.pdo_sum,
        row.y_name,
    )


def _rank_frame(scan: YScanResult) -> pd.DataFrame:
    """Сводный рейтинг: Y, связанные X и показатели."""
    rows = []
    for item in scan.rows:
        if item.ok:
            linked = ", ".join(item.linked_names)
            count = len(item.linked_names)
        else:
            linked = ""
            count = None
        rows.append(
            {
                "Y": item.y_name,
                "Связанные X": linked,
                "MDO": item.mdo if item.ok else None,
                "ODO": item.odo if item.ok else None,
                "I(X_D; Y)": item.joint_info if item.ok else None,
                "R": item.correlation_r if item.ok else None,
                "Число X": count,
                "Сумма ПДО": item.pdo_sum if item.ok else None,
                "Сумма ДОЭВД": item.doevd_sum if item.ok else None,
                "N": item.n_obs if item.ok else None,
                "Статус": "ok" if item.ok else item.error,
            }
        )
    return pd.DataFrame(rows)


def _links_frame(
    scan: YScanResult,
    y_filter: str | None,
) -> pd.DataFrame:
    """Имя Y и каждый связанный X отдельной строкой.

    Связь: ПДО фактора не ниже половины от максимального
    ПДО среди всех X для этого Y.
    """
    rows = []
    for item in scan.rows:
        if not item.ok or item.result is None:
            continue
        if y_filter is not None and item.y_name != y_filter:
            continue
        names = item.result.column_names
        raw = [
            float(factor.pdo_raw) for factor in item.result.factors
        ]
        peak = max(raw) if raw else 0.0
        floor = peak * LINK_PDO_SHARE
        for index in item.linked_idx:
            factor = item.result.factors[index]
            rows.append(
                {
                    "Y": item.y_name,
                    "X": names[index],
                    "ПДО": factor.pdo_raw,
                    "Порог": floor,
                    "Макс. ПДО": peak,
                    "Тип связи": "ПДО ≥ 1/2 от макс.",
                }
            )
    return pd.DataFrame(rows)


def _pdo_frame(
    scan: YScanResult,
    y_filter: str | None,
) -> pd.DataFrame:
    """Прямые ПДО всех факторов; связанные помечены отдельно."""
    rows = []
    for item in scan.rows:
        if not item.ok or item.result is None:
            continue
        if y_filter is not None and item.y_name != y_filter:
            continue
        linked = set(item.linked_idx)
        for index, factor in enumerate(item.result.factors):
            rows.append(
                {
                    "Y": item.y_name,
                    "X": item.result.column_names[index],
                    "ПДО_raw": factor.pdo_raw,
                    "ПДО_adj": factor.pdo_adj,
                    "Есть связь": (
                        "да" if index in linked else "нет"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pairs_frame(
    scan: YScanResult,
    y_filter: str | None,
) -> pd.DataFrame:
    """Пары связанных X при данном Y и их ДОЭВД."""
    rows = []
    for item in scan.rows:
        if not item.ok or item.result is None:
            continue
        if y_filter is not None and item.y_name != y_filter:
            continue
        linked = set(item.linked_idx)
        names = item.result.column_names
        for link in item.result.pairs:
            if link.i not in linked or link.j not in linked:
                continue
            rows.append(
                {
                    "Y": item.y_name,
                    "X_i": names[link.i],
                    "X_j": names[link.j],
                    "ДОЭВД": link.doevd,
                    "Тип связи": "пара связанных X",
                }
            )
    return pd.DataFrame(rows)


def _dominant_frame(
    scan: YScanResult,
    y_filter: str | None,
) -> pd.DataFrame:
    """Кратко: Y и перечень имён связанных X."""
    rows = []
    for item in scan.rows:
        if not item.ok:
            continue
        if y_filter is not None and item.y_name != y_filter:
            continue
        rows.append(
            {
                "Y": item.y_name,
                "Связанные X": ", ".join(item.linked_names),
                "Число X": len(item.linked_names),
            }
        )
    return pd.DataFrame(rows)


def _notify(
    callback: ProgressFn | None,
    name: str,
    percent: int,
) -> None:
    """Передаёт шаг в окно, если обратный вызов задан."""
    if callback is not None:
        callback(name, percent)


def _unique_folder(root: Path) -> Path:
    """Папка ГГГГММДД_ЧЧММСС_yscan, при совпадении с номером."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{stamp}_yscan"
    number = 2
    while candidate.exists():
        candidate = root / f"{stamp}_yscan_{number}"
        number += 1
    candidate.mkdir()
    return candidate
