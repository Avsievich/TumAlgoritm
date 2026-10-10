"""Запись результата расчёта в книгу Excel.

Лист — способ показать уже полученные показатели.
Вычисление показателей остаётся в пакете tumalgoritm.
"""

from pathlib import Path

import pandas as pd

from tumalgoritm.pipeline import AnalysisResult

# Имена листов зафиксированы, чтобы тесты и внешние скрипты находили их.
SHEET_SUMMARY = "Summary"
SHEET_PDO = "PDO"
SHEET_DOMINANT = "Dominant"
SHEET_RES_M = "res_M"
SHEET_DOEVS = "DOEVS"
SHEET_CHDO = "CHDO"
SHEET_INFORMATION = "Information"
SHEET_SUBSETS = "I_subsets"


def write_report(result: AnalysisResult, path: str | Path) -> Path:
    """Сохраняет все листы отчёта.

    Args:
        result: результат analyze_dataset или run_analysis.
        path: путь к xlsx. Папка создаётся при необходимости.

    Returns:
        Путь к записанному файлу.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = build_frames(result)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            # Матрица resM сохраняет подписи строк.
            write_index = sheet_name == SHEET_RES_M
            frame.to_excel(
                writer, sheet_name=sheet_name, index=write_index
            )
    return output


def build_frames(result: AnalysisResult) -> dict[str, pd.DataFrame]:
    """Собирает те же таблицы, что уходят в Excel.

    Экран результатов берёт их отсюда, чтобы числа на экране
    и в файле не расходились.

    Args:
        result: итог расчёта.

    Returns:
        Имя листа и таблица.
    """
    return {
        SHEET_SUMMARY: _summary_frame(result),
        SHEET_PDO: _pdo_frame(result),
        SHEET_DOMINANT: _dominant_frame(result),
        SHEET_RES_M: _res_m_frame(result),
        SHEET_DOEVS: _pair_frame(result),
        SHEET_CHDO: _chdo_frame(result),
        SHEET_INFORMATION: _information_frame(result),
        SHEET_SUBSETS: _subset_frame(result),
    }


def frame_to_tsv(
    frame: pd.DataFrame,
    *,
    with_index: bool = False,
) -> str:
    """Переводит одну таблицу в текст с табуляцией.

    Такой текст вставляется в Excel, Word и блокнот без потери
    столбцов. Индекс нужен для матрицы res_M: там подписи строк
    — это имена эффектов.

    Args:
        frame: лист отчёта.
        with_index: писать ли подписи строк первым столбцом.

    Returns:
        Строка TSV с заголовком.
    """
    shown = frame.reset_index() if with_index else frame
    return shown.to_csv(index=False, sep="\t", lineterminator="\n")


def frames_to_copy_text(
    frames: dict[str, pd.DataFrame],
    *,
    index_sheets: set[str] | None = None,
) -> str:
    """Собирает все листы отчёта в один текст для буфера обмена.

    Перед каждой таблицей ставится заголовок листа. Между листами
    пустая строка, чтобы блоки было видно в блокноте.

    Args:
        frames: имя листа и таблица, как в build_frames.
        index_sheets: листы, у которых в текст идёт индекс строк.
            По умолчанию — только res_M.

    Returns:
        Многострочный текст со всеми данными исследования.
    """
    need_index = index_sheets if index_sheets is not None else {SHEET_RES_M}
    parts: list[str] = []
    for sheet_name, frame in frames.items():
        parts.append(f"### {sheet_name}")
        body = frame_to_tsv(
            frame,
            with_index=sheet_name in need_index,
        ).rstrip("\n")
        parts.append(body)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _summary_frame(result: AnalysisResult) -> pd.DataFrame:
    """Скалярные итоги: МДО, ОДО, FDO и уникальная информация."""
    names = result.column_names
    rows: list[tuple[str, object]] = [
        ("N", result.n_obs),
        ("Выход", result.y_name),
        ("NADCX", result.prepared.nadc_x),
        ("NADCY", result.prepared.nadc_y),
        ("q", result.effect_order),
        ("eta0", result.eta0),
        ("m*", result.selection.m_star),
        ("S_m", result.selection.s_m),
        ("FDO", result.effects.fdo),
        ("MDO_raw", result.mdo_raw),
        ("MDO_adj", result.mdo_adj),
        ("MDO", result.mdo),
        ("ODO", result.odo),
        ("R", result.correlation_r),
        ("k_joint", result.k_joint),
        ("H(Y)", result.information.h_y),
        ("I(X_D; Y)", result.information.joint_dominant),
        (
            "Дисперсия квантования",
            result.prepared.variance_ratio,
        ),
    ]
    for index in result.dominant_idx:
        rows.append((f"ПДО {names[index]}", result.factors[index].pdo_raw))
    for link in result.pairs:
        if link.i in result.dominant_idx and link.j in result.dominant_idx:
            label = f"VdDO({names[link.i]},{names[link.j]})"
            rows.append((label, link.doevd))
    for index in result.dominant_idx:
        rows.append(
            (
                f"I({names[index]}; Y | остальные)",
                result.information.unique[index],
            )
        )
    for number, text in enumerate(result.warnings, start=1):
        rows.append((f"Предупреждение {number}", text))
    return pd.DataFrame(rows, columns=["Показатель", "Значение"])


def _pdo_frame(result: AnalysisResult) -> pd.DataFrame:
    """Таблица одиночных ПДО и гетероскедастичности."""
    n_obs = result.n_obs
    rows = []
    for name, role, factor in zip(
        result.column_names, result.roles, result.factors
    ):
        per_cell = n_obs / factor.n_groups if factor.n_groups else 0.0
        rows.append(
            {
                "Переменная": name,
                "Роль": role,
                "ПДО_raw": factor.pdo_raw,
                "ПДО_adj": factor.pdo_adj,
                "k": factor.n_groups,
                "N/k": per_cell,
                "omega": factor.omega,
                "theta": factor.theta,
            }
        )
    return pd.DataFrame(rows)


def _dominant_frame(result: AnalysisResult) -> pd.DataFrame:
    """Ранжирование ПДО и критерий Q(m)."""
    names = result.column_names
    m_star = result.selection.m_star
    limited = set(result.dominant_idx)
    rows = []
    for rank, index in enumerate(result.selection.ranked_idx, start=1):
        index_int = int(index)
        if rank <= m_star:
            group = "A1 (доминантная)"
        else:
            group = "A2 (несущественная)"
        q_value = ""
        optimum = ""
        if rank < len(result.selection.ranked_idx):
            q_value = float(result.selection.q_values[rank - 1])
            if rank == m_star:
                optimum = "m*"
        rows.append(
            {
                "Ранг": rank,
                "Переменная": names[index_int],
                "ПДО для отбора": float(
                    result.selection.ranked_scores[rank - 1]
                ),
                "Группа": group,
                "В расчёте эффектов": "да" if index_int in limited else "нет",
                "Q(m)": q_value,
                "Оптимум": optimum,
            }
        )
    return pd.DataFrame(rows)


def _res_m_frame(result: AnalysisResult) -> pd.DataFrame:
    """Матрица перекрёстных вкладов эффектов."""
    labels = result.effects.labels
    return pd.DataFrame(
        result.effects.res_m,
        index=labels,
        columns=labels,
    )


def _pair_frame(result: AnalysisResult) -> pd.DataFrame:
    """Все пары: чистое взаимодействие и две направленные проекции."""
    names = result.column_names
    rows = []
    for link in result.pairs:
        in_dominant = (
            link.i in result.dominant_idx and link.j in result.dominant_idx
        )
        rows.append(
            {
                "xi": names[link.i],
                "xj": names[link.j],
                "VdDO": link.doevd,
                "DOEVS(xi,xj)(xi)": link.doevs_i,
                "DOEVS(xi,xj)(xj)": link.doevs_j,
                "Оба доминантные": "да" if in_dominant else "нет",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "xi",
                "xj",
                "VdDO",
                "DOEVS(xi,xj)(xi)",
                "DOEVS(xi,xj)(xj)",
                "Оба доминантные",
            ]
        )
    return pd.DataFrame(rows)


def _chdo_frame(result: AnalysisResult) -> pd.DataFrame:
    """Частные отношения только для пар доминантных факторов."""
    names = result.column_names
    dominant = set(result.dominant_idx)
    rows = []
    for link in result.pairs:
        if link.i not in dominant or link.j not in dominant:
            continue
        rows.append(
            {
                "xi": names[link.i],
                "xj": names[link.j],
                "ЧДО(xj|xi)": link.chdo_j_given_i,
                "ЧДО(xi|xj)": link.chdo_i_given_j,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["xi", "xj", "ЧДО(xj|xi)", "ЧДО(xi|xj)"]
        )
    return pd.DataFrame(rows)


def _information_frame(result: AnalysisResult) -> pd.DataFrame:
    """Парная информация и уникальный вклад доминантных факторов."""
    rows = []
    for index, name in enumerate(result.column_names):
        unique = result.information.unique[index]
        rows.append(
            {
                "Переменная": name,
                "Роль": result.roles[index],
                "I(X;Y)": result.information.mutual[index],
                "I(X;Y|остальные доминантные)": unique,
            }
        )
    return pd.DataFrame(rows)


def _subset_frame(result: AnalysisResult) -> pd.DataFrame:
    """Взаимная информация подмножеств доминантных факторов до порядка q."""
    rows = [
        {"Подмножество": label, "I(X_S; Y)": value}
        for label, value in result.information.subset_mutual
    ]
    if not rows:
        return pd.DataFrame(columns=["Подмножество", "I(X_S; Y)"])
    return pd.DataFrame(rows)
