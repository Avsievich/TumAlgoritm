"""Энтропия и взаимная информация по кодам квантования.

Частоты — эмпирические доли занятых сочетаний уровней.
Информация считается в битах. Полный многомерный массив
вероятностей не строится: для каждого нужного подмножества
факторов берётся своя таблица частот.

    I(X; Y) = H(X) + H(Y) - H(X, Y)
    I(Xi; Y | X_T) = H(Y | X_T) - H(Y | Xi, X_T)
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class InformationResult:
    """Информационные критерии по квантованным уровням.

    Attributes:
        h_y: энтропия выхода H(Y).
        mutual: I(Xi; Y) для каждого фактора, в порядке столбцов X.
        unique: I(Xi; Y | остальные доминантные). NaN, если фактор
            не входит в доминантный набор.
        joint_dominant: I(X_D; Y) по всему доминантному набору.
        subset_mutual: I(X_S; Y) для подмножеств доминантных факторов
            размера от 1 до q. Подпись — имена факторов через запятую.
    """

    h_y: float
    mutual: np.ndarray
    unique: np.ndarray
    joint_dominant: float
    subset_mutual: list[tuple[str, float]]


def entropy(codes: np.ndarray) -> float:
    """Считает энтропию дискретного кода в битах.

    Args:
        codes: столбец кодов или матрица совместных кодов (N, k).

    Returns:
        H = -sum p log2(p) по занятым сочетаниям. Пустой вход даёт 0.
    """
    values = np.asarray(codes)
    if values.size == 0:
        return 0.0
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    _, counts = np.unique(values, axis=0, return_counts=True)
    probs = counts.astype(float) / float(counts.sum())
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log2(probs)))


def mutual_information(x_codes: np.ndarray, y_codes: np.ndarray) -> float:
    """Считает взаимную информацию I(X; Y).

    Отрицательный хвост из-за ошибок округления обнуляется.

    Args:
        x_codes: коды одного или нескольких факторов.
        y_codes: коды выхода.

    Returns:
        Неотрицательное значение в битах.
    """
    factors = _as_columns(x_codes)
    target = np.asarray(y_codes).reshape(-1, 1)
    h_x = entropy(factors)
    h_y = entropy(target)
    h_xy = entropy(np.column_stack([factors, target]))
    return float(max(0.0, h_x + h_y - h_xy))


def conditional_mutual_information(
    x_codes: np.ndarray,
    y_codes: np.ndarray,
    condition: np.ndarray | None,
) -> float:
    """Считает I(X; Y | condition).

    При пустом условии совпадает с обычной взаимной информацией.
    Условные энтропии раскрываются как разности совместных:
    H(Y|T) = H(T, Y) - H(T).

    Args:
        x_codes: коды фактора, информацию которого оцениваем.
        y_codes: коды выхода.
        condition: коды условий (N, k) или None.

    Returns:
        Неотрицательное значение в битах.
    """
    if condition is None or np.asarray(condition).size == 0:
        return mutual_information(x_codes, y_codes)

    factor = _as_columns(x_codes)
    target = np.asarray(y_codes).reshape(-1, 1)
    given = _as_columns(condition)

    h_given = entropy(given)
    h_given_y = entropy(np.column_stack([given, target]))
    h_factor_given = entropy(np.column_stack([factor, given]))
    h_all = entropy(np.column_stack([factor, given, target]))

    h_y_given = h_given_y - h_given
    h_y_given_factor = h_all - h_factor_given
    return float(max(0.0, h_y_given - h_y_given_factor))


def information_report(
    x_codes: np.ndarray,
    y_codes: np.ndarray,
    dominant_idx: list[int],
    names: list[str],
    max_order: int,
) -> InformationResult:
    """Считает парную, уникальную, совместную и подмножественную информацию.

    Уникальный вклад фактора — I(Xi; Y | остальные доминантные).
    Для факторов вне доминантного набора уникальный вклад не определён.
    Подмножества доминантных факторов ограничиваются порядком q, чтобы
    не перебирать все 2^n сочетаний.

    Args:
        x_codes: коды всех факторов (N, n).
        y_codes: коды выхода длины N.
        dominant_idx: индексы доминантных факторов.
        names: имена всех факторов.
        max_order: максимальный размер подмножества q.

    Returns:
        InformationResult.
    """
    codes = np.asarray(x_codes)
    target = np.asarray(y_codes).reshape(-1)
    n_factors = codes.shape[1]
    mutual = np.array(
        [mutual_information(codes[:, col], target) for col in range(n_factors)]
    )
    unique = np.full(n_factors, np.nan)
    dominant = [int(index) for index in dominant_idx]

    for index in dominant:
        others = [other for other in dominant if other != index]
        condition = codes[:, others] if others else None
        unique[index] = conditional_mutual_information(
            codes[:, index],
            target,
            condition,
        )

    if dominant:
        joint = mutual_information(codes[:, dominant], target)
    else:
        joint = 0.0

    subset_mutual = _subset_mutual(
        codes, target, dominant, names, max_order
    )
    return InformationResult(
        h_y=entropy(target),
        mutual=mutual,
        unique=unique,
        joint_dominant=joint,
        subset_mutual=subset_mutual,
    )


def _subset_mutual(
    codes: np.ndarray,
    target: np.ndarray,
    dominant: list[int],
    names: list[str],
    max_order: int,
) -> list[tuple[str, float]]:
    """Считает I(X_S; Y) для подмножеств доминантных факторов до порядка q."""
    if max_order < 1 or not dominant:
        return []
    rows: list[tuple[str, float]] = []
    upper = min(max_order, len(dominant))
    for order in range(1, upper + 1):
        for subset in combinations(dominant, order):
            label = ", ".join(names[index] for index in subset)
            value = mutual_information(codes[:, list(subset)], target)
            rows.append((label, value))
    return rows


def _as_columns(codes: np.ndarray) -> np.ndarray:
    """Приводит коды к двумерному массиву «строки — наблюдения»."""
    values = np.asarray(codes)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values
