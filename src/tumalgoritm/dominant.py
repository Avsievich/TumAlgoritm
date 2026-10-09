"""Выбор доминантных переменных по критерию Q(m).

Алгоритм раздела 2.2:
    1. Ранжировать прямые дисперсионные отношения по убыванию.
    2. Для каждой границы m посчитать Q(m) = mean(A2) / mean(A1).
    3. Взять m* с минимальным Q.
    4. Нижняя оценка множественного отношения S_m — сумма ПДО группы A1.
    5. Набор достаточен, если S_m не меньше порога eta0.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DominantSelection:
    """Результат разбиения факторов на доминантные и несущественные.

    Attributes:
        ranked_idx: исходные индексы факторов от сильного ПДО к слабому.
        ranked_scores: ПДО в том же порядке.
        q_values: Q(m) для m = 1 .. n-1. Пусто, если фактор один.
        m_star: размер группы A1 до инженерного ограничения min/max.
        s_m: сумма ПДО группы A1, нижняя оценка МДО.
        sufficient: выполняется ли S_m >= eta0.
        dominant_idx: индексы A1 в порядке убывания ПДО.
    """

    ranked_idx: np.ndarray
    ranked_scores: np.ndarray
    q_values: np.ndarray
    m_star: int
    s_m: float
    sufficient: bool
    dominant_idx: list[int]


def criterion_q(sorted_scores: np.ndarray, border: int) -> float:
    """Считает Q(m) = среднее хвоста / среднее головы.

    Args:
        sorted_scores: ПДО, уже отсортированные по убыванию.
        border: размер головы A1. Хвост A2 — всё, что правее.

    Returns:
        Q(m). Бесконечность, если разбиение вырождено или голова нулевая.
    """
    n_factors = len(sorted_scores)
    if border <= 0 or border >= n_factors:
        return float(np.inf)

    mean_head = float(np.mean(sorted_scores[:border]))
    mean_tail = float(np.mean(sorted_scores[border:]))
    if mean_head == 0.0:
        return float(np.inf)
    return mean_tail / mean_head


def select_dominant(
    scores: np.ndarray,
    eta0: float = 0.0,
) -> DominantSelection:
    """Находит границу m* и группу доминантных факторов.

    Args:
        scores: ПДО всех факторов, по одному на столбец X.
        eta0: порог достаточности нижней оценки МДО.

    Returns:
        DominantSelection. При одном факторе он и есть доминантный.

    Raises:
        ValueError: передан пустой набор оценок.
    """
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) == 0:
        raise ValueError("Нет факторов для выбора доминантных переменных")

    order = np.argsort(values)[::-1]
    ranked = values[order]
    if len(values) == 1:
        q_values = np.array([], dtype=float)
        m_star = 1
    else:
        q_values = np.array(
            [criterion_q(ranked, border) for border in range(1, len(values))],
            dtype=float,
        )
        m_star = int(np.argmin(q_values)) + 1

    s_m = float(np.sum(ranked[:m_star]))
    dominant_idx = [int(index) for index in order[:m_star]]
    return DominantSelection(
        ranked_idx=order,
        ranked_scores=ranked,
        q_values=q_values,
        m_star=m_star,
        s_m=s_m,
        sufficient=s_m >= eta0,
        dominant_idx=dominant_idx,
    )


def limit_dominant(
    selection: DominantSelection,
    n_factors: int,
    min_dominant: int,
    max_dominant: int,
) -> list[int]:
    """Доводит размер доминантного набора до заданных границ.

    Если группа A1 короче минимума, к ней добавляются следующие
    по силе ПДО факторы. Если группа длиннее максимума, остаются
    самые сильные. Это ограничение расчёта, а не формула книги.

    Args:
        selection: результат select_dominant.
        n_factors: сколько факторов доступно.
        min_dominant: нижняя граница размера набора.
        max_dominant: верхняя граница размера набора.

    Returns:
        Индексы столбцов X в порядке убывания ПДО.

    Raises:
        ValueError: границы размера заданы неверно.
    """
    if min_dominant < 1 or max_dominant < 1:
        raise ValueError("Размер доминантного набора должен быть не меньше 1")
    if min_dominant > max_dominant:
        raise ValueError("min_dominant не может быть больше max_dominant")

    chosen = list(selection.dominant_idx)
    target_min = min(min_dominant, n_factors)
    for index in selection.ranked_idx:
        if len(chosen) >= target_min:
            break
        index_int = int(index)
        if index_int not in chosen:
            chosen.append(index_int)

    target_max = min(max_dominant, n_factors)
    return chosen[:target_max]
