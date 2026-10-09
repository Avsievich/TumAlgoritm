"""Эффекты взаимодействия произвольного порядка и матрица resM.

Эффект подмножества факторов S задаётся включениями-исключениями:

    f_S = sum_{T subset S} (-1)^(|S| - |T|) * E(Y | X_T).

Пустое T даёт общее среднее E(Y). Для одного фактора это E(Y|Xi) - E(Y),
для пары — чистое взаимодействие VdDO, для тройки — формула прототипа.
Столбцы матрицы V — эти эффекты по наблюдениям,
resM = V^T V / NDy, FDO — сумма всех элементов resM.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from tumalgoritm.dispersion import Moments, aligned_group_mean


@dataclass
class EffectMatrix:
    """Базис эффектов и их перекрёстная матрица.

    Attributes:
        subsets: подмножества индексов факторов, по столбцу V.
        labels: подписи столбцов для отчёта.
        res_m: матрица вкладов K x K.
        fdo: сумма всех элементов res_m, включая перекрёстные.
    """

    subsets: list[tuple[int, ...]]
    labels: list[str]
    res_m: np.ndarray
    fdo: float


class ConditionalMeanCache:
    """Кэш условных средних, чтобы не группировать одни и те же коды дважды."""

    def __init__(self, y: np.ndarray, codes: np.ndarray, overall_mean: float):
        self._y = np.asarray(y, dtype=float).reshape(-1)
        self._codes = np.asarray(codes)
        self._overall_mean = overall_mean
        self._cache: dict[tuple[int, ...], np.ndarray] = {
            (): np.full(len(self._y), overall_mean)
        }

    def mean_of(self, columns: tuple[int, ...]) -> np.ndarray:
        """Возвращает E(Y | столбцы), выровненное по строкам."""
        key = tuple(columns)
        if key not in self._cache:
            selected = self._codes[:, list(key)]
            conditional, _ = aligned_group_mean(self._y, selected)
            self._cache[key] = conditional
        return self._cache[key]

    def effect(self, subset: tuple[int, ...]) -> np.ndarray:
        """Считает вектор эффекта f_S по формуле включений-исключений."""
        acc = np.zeros(len(self._y), dtype=float)
        size = len(subset)
        for mask in range(1 << size):
            members = tuple(
                subset[pos] for pos in range(size) if mask & (1 << pos)
            )
            # (-1)^(|S|-|T|): плюс, когда разность размеров чётная.
            sign = 1.0 if (size - len(members)) % 2 == 0 else -1.0
            acc += sign * self.mean_of(members)
        return acc


def iter_subsets(
    column_idx: list[int],
    max_order: int,
) -> list[tuple[int, ...]]:
    """Перечисляет непустые подмножества факторов до заданного порядка.

    Args:
        column_idx: индексы столбцов, обычно доминантные, по убыванию ПДО.
        max_order: максимальный размер подмножества q.

    Returns:
        Список кортежей индексов: сначала одиночные, затем пары и т.д.
    """
    if max_order < 1:
        raise ValueError("Порядок взаимодействий q должен быть не меньше 1")
    subsets: list[tuple[int, ...]] = []
    upper = min(max_order, len(column_idx))
    for order in range(1, upper + 1):
        subsets.extend(combinations(column_idx, order))
    return subsets


def effect_label(subset: tuple[int, ...], names: list[str]) -> str:
    """Делает подпись эффекта: имя фактора, VdDO(пары) или VdDO3(тройки)."""
    labels = [names[index] for index in subset]
    if len(subset) == 1:
        return labels[0]
    if len(subset) == 2:
        return f"VdDO({labels[0]},{labels[1]})"
    joined = ",".join(labels)
    return f"VdDO{len(subset)}({joined})"


def build_effect_matrix(
    y: np.ndarray,
    codes: np.ndarray,
    subsets: list[tuple[int, ...]],
    names: list[str],
    moments: Moments,
) -> EffectMatrix:
    """Строит resM и FDO по списку подмножеств.

    Args:
        y: нормированный выход.
        codes: коды всех факторов (N, n).
        subsets: эффекты, которые войдут в базис.
        names: имена всех факторов.
        moments: сводки выхода, отсюда берётся масштаб NDy.

    Returns:
        EffectMatrix. Если подмножеств нет, матрица пустая и FDO = 0.
    """
    if not subsets:
        return EffectMatrix(
            subsets=[],
            labels=[],
            res_m=np.zeros((0, 0)),
            fdo=0.0,
        )

    cache = ConditionalMeanCache(y, codes, moments.mean)
    columns = [cache.effect(subset) for subset in subsets]
    effect_table = np.column_stack(columns)
    res_m = (effect_table.T @ effect_table) / moments.scale
    labels = [effect_label(subset, names) for subset in subsets]
    return EffectMatrix(
        subsets=list(subsets),
        labels=labels,
        res_m=res_m,
        fdo=float(np.sum(res_m)),
    )


def expanded_triple_effect(
    y: np.ndarray,
    codes: np.ndarray,
    overall_mean: float,
) -> np.ndarray:
    """Считает тройной эффект развёрнутой формулой прототипа.

    f = E(Y|ijk) - E(Y|ij) - E(Y|jk) - E(Y|ik)
        + E(Y|i) + E(Y|j) + E(Y|k) - E(Y).

    Нужна как независимая проверка формулы включений-исключений
    на первых трёх столбцах codes.

    Args:
        y: выход.
        codes: не меньше трёх столбцов кодов.
        overall_mean: общее среднее выхода.

    Returns:
        Вектор эффекта длины N.
    """
    mean_i, _ = aligned_group_mean(y, codes[:, 0])
    mean_j, _ = aligned_group_mean(y, codes[:, 1])
    mean_k, _ = aligned_group_mean(y, codes[:, 2])
    mean_ij, _ = aligned_group_mean(y, codes[:, [0, 1]])
    mean_ik, _ = aligned_group_mean(y, codes[:, [0, 2]])
    mean_jk, _ = aligned_group_mean(y, codes[:, [1, 2]])
    mean_ijk, _ = aligned_group_mean(y, codes[:, [0, 1, 2]])
    return (
        mean_ijk
        - mean_ij
        - mean_jk
        - mean_ik
        + mean_i
        + mean_j
        + mean_k
        - overall_mean
    )
