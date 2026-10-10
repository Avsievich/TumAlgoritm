"""Готовит таблицу NHANES lab+diagnoses для расчёта в TumAlgoritm.

Исходник — широкий CSV цикла 2017–2018 с лабораторией и самоотчётом
диагнозов. На выходе — отдельный файл без пропусков: выход status
(диабет по DIQ010), демография, бинарные диагнозы и те же лабораторные
факторы, что в nhanes_2017_2018_for_analysis.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Исходная широкая таблица из проекта NHANES.
SRC_CSV = Path(
    r"D:\Projects\NHANES\data\out"
    r"\nhanes_2017_2018_lab_diagnoses.csv"
)

# Уже принятый набор лабораторных столбцов (полный случай).
REF_LABS_CSV = Path(
    r"D:\Projects\TumAlgoritm\reference"
    r"\nhanes_2017_2018_for_analysis.csv"
)

# Куда писать подготовленный файл в текущем проекте.
DST_CSV = Path(
    r"D:\Projects\TumAlgoritm\reference"
    r"\nhanes_2017_2018_lab_diagnoses_for_analysis.csv"
)

# Минимальный возраст: вопросы MCQ160*/BPQ/KIQ — для взрослых.
MIN_AGE_YEARS = 20

# Коды DIQ010: да / нет. Пограничный (3) и «не знает» (9) отбрасываем.
DIQ_YES = 1
DIQ_NO = 2

# Коды самоотчёта «да/нет» в опросниках NHANES.
CODE_YES = 1
CODE_NO = 2

# Демография: пол, возраст, раса (схема с азиатами).
DEMO_COLUMNS = ("RIAGENDR", "RIDAGEYR", "RIDRETH3")

# Бинарные диагнозы с почти полным заполнением у взрослых.
# Не берём возраст диагноза, детали DIQ175* и DIQ050 (инсулин —
# почти однозначно связан с диабетом и дал бы ложную связь).
DIAG_COLUMNS = (
    "MCQ010",   # астма
    "MCQ053",   # лечение анемии за 3 месяца
    "MCQ092",   # переливание крови
    "MCQ203",   # желтуха
    "MCQ160A",  # артрит
    "MCQ160B",  # сердечная недостаточность
    "MCQ160C",  # ИБС
    "MCQ160D",  # стенокардия
    "MCQ160E",  # инфаркт
    "MCQ160F",  # инсульт
    "MCQ160G",  # эмфизема
    "MCQ160K",  # хронический бронхит
    "MCQ160L",  # болезнь печени
    "MCQ160M",  # щитовидная железа
    "MCQ160N",  # подагра
    "MCQ160O",  # ХОБЛ
    "MCQ220",   # рак
    "BPQ020",   # гипертония
    "BPQ080",   # высокий холестерин
    "KIQ022",   # слабые/отказывающие почки
    "KIQ026",   # камни почек
    "HEQ010",   # гепатит B
    "HEQ030",   # гепатит C
)


def lab_factor_names() -> list[str]:
    """Читает список лабораторных факторов из уже готового файла.

    Returns:
        Имена столбцов лаборатории без SEQN и status.
    """
    columns = list(pd.read_csv(REF_LABS_CSV, nrows=0).columns)
    return [name for name in columns if name not in ("SEQN", "status")]


def to_binary_yes_no(series: pd.Series) -> pd.Series:
    """Переводит коды 1/2 в 1/0; прочие значения станут NaN.

    Args:
        series: столбец с кодами CDC «да/нет».

    Returns:
        Серия 1 (да), 0 (нет) или NaN.
    """
    mapped = series.map({CODE_YES: 1, CODE_NO: 0})
    return mapped.astype("float")


def prepare_frame(source: pd.DataFrame, labs: list[str]) -> pd.DataFrame:
    """Собирает полную по строкам и столбцам панель для анализа.

    Args:
        source: широкая таблица lab_diagnoses.
        labs: лабораторные факторы без пропусков.

    Returns:
        Таблица SEQN, status, демография, диагнозы 0/1, лаборатория.
    """
    needed = (
        ["SEQN", "DIQ010"]
        + list(DEMO_COLUMNS)
        + list(DIAG_COLUMNS)
        + labs
    )
    missing = [name for name in needed if name not in source.columns]
    if missing:
        raise KeyError(f"В исходнике нет столбцов: {missing}")

    frame = source.loc[:, needed].copy()

    # Взрослые с однозначным ответом о диабете.
    frame = frame.loc[frame["RIDAGEYR"] >= MIN_AGE_YEARS]
    frame = frame.loc[frame["DIQ010"].isin([DIQ_YES, DIQ_NO])]

    # Полный случай по лаборатории и демографии.
    frame = frame.dropna(subset=list(DEMO_COLUMNS) + labs)

    # Диагнозы: только «да/нет», отказ и «не знает» убираем.
    for name in DIAG_COLUMNS:
        frame = frame.loc[frame[name].isin([CODE_YES, CODE_NO])]

    # Собираем столбцы словарём, чтобы не фрагментировать DataFrame.
    columns: dict[str, pd.Series] = {
        "SEQN": frame["SEQN"].astype(int),
        # status = 1 при диабете (DIQ010 == 1), иначе 0.
        "status": (frame["DIQ010"] == DIQ_YES).astype(int),
    }
    for name in DEMO_COLUMNS:
        columns[name] = frame[name]
    for name in DIAG_COLUMNS:
        columns[name] = to_binary_yes_no(frame[name]).astype(int)
    for name in labs:
        columns[name] = frame[name]

    out = pd.DataFrame(columns)
    if out.isna().any().any():
        raise ValueError("После подготовки остались пропуски")

    return out.reset_index(drop=True)


def main() -> None:
    """Читает исходник, пишет CSV в reference текущего проекта."""
    labs = lab_factor_names()
    source = pd.read_csv(SRC_CSV)
    prepared = prepare_frame(source, labs)

    DST_CSV.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(DST_CSV, index=False)

    n_pos = int(prepared["status"].sum())
    n_neg = int((prepared["status"] == 0).sum())
    print(f"csv -> {DST_CSV}")
    print(
        f"rows={len(prepared)} cols={prepared.shape[1]} "
        f"status_1={n_pos} status_0={n_neg}"
    )


if __name__ == "__main__":
    main()
