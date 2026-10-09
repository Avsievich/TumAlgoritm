"""Прямые, частные и парные дисперсионные отношения.

Группировка идёт по кодам квантования факторов, а сам выход
остаётся непрерывным нормированным рядом. Масштаб NDy равен
(N - 1) * выборочная дисперсия Y, как в прототипе методики.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Moments:
    """Сводки выхода, нужные всем дисперсионным формулам.

    Attributes:
        mean: общее среднее E(Y).
        variance: выборочная дисперсия D(Y), ddof=1.
        variance_of_square: D(Y^2), знаменатель показателей
            гетероскедастичности.
        scale: (N - 1) * D(Y). Если дисперсия нулевая, равна 1,
            чтобы деление в отношениях не падало: числитель тогда тоже 0.
        n_obs: число наблюдений.
    """

    mean: float
    variance: float
    variance_of_square: float
    scale: float
    n_obs: int


@dataclass
class FactorDispersion:
    """Дисперсионные характеристики одного фактора."""

    pdo_raw: float
    pdo_adj: float
    n_groups: int
    omega: float
    theta: float


@dataclass
class PairLink:
    """Парные величины для факторов i и j.

    doevd — чистое взаимодействие VdDO.
    doevs_i / doevs_j — две направленные проекции формулы (2.18).
    chdo_j_given_i — частное отношение Xj при фиксированном Xi.
    chdo_i_given_j — частное отношение Xi при фиксированном Xj.
    """

    i: int
    j: int
    doevd: float
    doevs_i: float
    doevs_j: float
    chdo_j_given_i: float
    chdo_i_given_j: float


def output_moments(y: np.ndarray) -> Moments:
    """Считает среднее, дисперсии и масштаб NDy.

    Args:
        y: нормированный выход.

    Returns:
        Moments для дальнейших отношений.
    """
    values = np.asarray(y, dtype=float).reshape(-1)
    n_obs = len(values)
    mean = float(np.mean(values))
    if n_obs < 2:
        variance = 0.0
        variance_of_square = 0.0
    else:
        variance = float(np.var(values, ddof=1))
        variance_of_square = float(np.var(values**2, ddof=1))

    # Нулевая дисперсия: все условные средние совпадут с общим, числитель 0.
    if variance == 0.0:
        scale = 1.0
    else:
        scale = (n_obs - 1) * variance

    return Moments(
        mean=mean,
        variance=variance,
        variance_of_square=variance_of_square,
        scale=scale,
        n_obs=n_obs,
    )


def bias_correct_eta2(eta2: float, n_obs: int, n_groups: int) -> float:
    """Убирает завышение дисперсионного отношения при малом N/k.

    Формула та же, что у скорректированного R^2:
        eta_adj = 1 - (1 - eta) * (N - 1) / (N - k).
    k — число занятых ячеек. Если наблюдений не больше, чем ячеек,
    оценка вырождена и возвращается NaN.

    Args:
        eta2: сырое отношение.
        n_obs: число наблюдений.
        n_groups: число занятых групп.

    Returns:
        Оценка в отрезке [0, 1] либо NaN.
    """
    if n_obs <= n_groups:
        return float("nan")
    adjusted = 1.0 - (1.0 - eta2) * (n_obs - 1) / (n_obs - n_groups)
    return float(np.clip(adjusted, 0.0, 1.0))


def aligned_group_mean(
    y: np.ndarray,
    codes: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Возвращает E(Y | коды) в порядке исходных строк.

    Args:
        y: выход длины N.
        codes: один столбец кодов или матрица кодов (N, k).

    Returns:
        Пара (условные средние длины N, число занятых групп).
    """
    target = np.asarray(y, dtype=float).reshape(-1)
    grouped_codes = np.asarray(codes)
    if grouped_codes.ndim == 1:
        grouped_codes = grouped_codes.reshape(-1, 1)

    frame = pd.DataFrame(grouped_codes)
    keys = list(frame.columns)
    frame["Y"] = target
    grouped = frame.groupby(keys, sort=False)["Y"]
    means = grouped.transform("mean").to_numpy(dtype=float)
    return means, int(grouped.ngroups)


def dispersion_ratio(
    y: np.ndarray,
    codes: np.ndarray,
    moments: Moments,
) -> tuple[float, int]:
    """Считает отношение sum (E(Y|X) - E(Y))^2 / NDy.

    Это и прямое ПДО одного фактора, и множественное МДО,
    если в codes несколько столбцов.

    Args:
        y: нормированный выход.
        codes: коды группировки.
        moments: сводки выхода.

    Returns:
        Пара (сырое отношение, число занятых ячеек).
    """
    conditional, n_groups = aligned_group_mean(y, codes)
    raw = float(np.sum((conditional - moments.mean) ** 2) / moments.scale)
    return raw, n_groups


def heteroscedasticity(
    y: np.ndarray,
    codes: np.ndarray,
    moments: Moments,
) -> tuple[float, float]:
    """Считает omega и theta — показатели неоднородности по X.

    omega = sqrt( D[D[Y|X]] / D[Y^2] ) используется, когда среднее
    внутри групп почти постоянно. theta = sqrt( D[M[Y|X]^2] / D[Y^2] )
    — когда почти постоянна внутригрупповая дисперсия.
    Знаменатель именно D(Y^2), а не D(Y).

    Args:
        y: нормированный выход.
        codes: коды одного фактора.
        moments: сводки выхода.

    Returns:
        Пара (omega, theta). При одной группе или нулевом D(Y^2) — нули.
    """
    if moments.variance_of_square == 0.0:
        return 0.0, 0.0

    target = np.asarray(y, dtype=float).reshape(-1)
    frame = pd.DataFrame(
        {"X": np.asarray(codes).reshape(-1), "Y": target}
    )
    grouped = frame.groupby("X", sort=False)["Y"]
    counts = grouped.count()
    if len(counts) <= 1:
        return 0.0, 0.0

    # Группа из одной точки не имеет выборочной дисперсии: берём 0.
    group_var = grouped.var(ddof=1).reindex(counts.index).fillna(0.0)
    group_mean = grouped.mean().reindex(counts.index)
    weights = counts.to_numpy(dtype=float)
    variances = group_var.to_numpy(dtype=float)
    means = group_mean.to_numpy(dtype=float)

    mean_of_var = float(np.average(variances, weights=weights))
    var_of_var = float(
        np.average((variances - mean_of_var) ** 2, weights=weights)
    )
    mean_of_sq_mean = float(np.average(means**2, weights=weights))
    var_of_sq_mean = float(
        np.average((means**2 - mean_of_sq_mean) ** 2, weights=weights)
    )
    denom = moments.variance_of_square
    omega = float(np.sqrt(var_of_var / denom))
    theta = float(np.sqrt(var_of_sq_mean / denom))
    return omega, theta


def factor_dispersion(
    y: np.ndarray,
    codes: np.ndarray,
    moments: Moments,
) -> FactorDispersion:
    """Считает сырое и скорректированное ПДО и гетероскедастичность.

    Args:
        y: нормированный выход.
        codes: коды одного фактора.
        moments: сводки выхода.

    Returns:
        FactorDispersion по этому фактору.
    """
    raw, n_groups = dispersion_ratio(y, codes, moments)
    adjusted = bias_correct_eta2(raw, moments.n_obs, n_groups)
    omega, theta = heteroscedasticity(y, codes, moments)
    return FactorDispersion(
        pdo_raw=raw,
        pdo_adj=adjusted,
        n_groups=n_groups,
        omega=omega,
        theta=theta,
    )


def selection_scores(raw: np.ndarray, adjusted: np.ndarray) -> np.ndarray:
    """Готовит ПДО для отбора: при вырожденной поправке остаётся сырое.

    Args:
        raw: сырые ПДО.
        adjusted: скорректированные ПДО, могут содержать NaN.

    Returns:
        Массив той же длины без NaN.
    """
    raw_values = np.asarray(raw, dtype=float)
    adjusted_values = np.asarray(adjusted, dtype=float)
    return np.where(np.isnan(adjusted_values), raw_values, adjusted_values)


def _partial_ratio(joint_eta: float, fixed_eta: float) -> float:
    """ЧДО = (eta_ij - eta_i) / (1 - eta_i). При нулевом остатке — 0."""
    remainder = 1.0 - fixed_eta
    if remainder <= 0.0:
        return 0.0
    return float((joint_eta - fixed_eta) / remainder)


def pair_link(
    y: np.ndarray,
    codes_i: np.ndarray,
    codes_j: np.ndarray,
    moments: Moments,
    eta_i: float,
    eta_j: float,
) -> tuple[float, float, float, float, float]:
    """Считает VdDO, две проекции ДОЭВС и два частных отношения.

    Чистое взаимодействие:
        f = E(Y|Xi,Xj) - E(Y|Xi) - E(Y|Xj) + E(Y).
    VdDO = sum(f^2) / NDy.
    Проекция на Xi: sum(f * (E(Y|Xi) - E(Y))) / NDy.

    Args:
        y: нормированный выход.
        codes_i: коды первого фактора.
        codes_j: коды второго фактора.
        moments: сводки выхода.
        eta_i: сырое ПДО первого фактора.
        eta_j: сырое ПДО второго фактора.

    Returns:
        (VdDO, DOEVS на i, DOEVS на j, ЧДО j|i, ЧДО i|j).
    """
    mean_i, _ = aligned_group_mean(y, codes_i)
    mean_j, _ = aligned_group_mean(y, codes_j)
    pair_codes = np.column_stack([codes_i, codes_j])
    mean_ij, _ = aligned_group_mean(y, pair_codes)
    interaction = mean_ij - mean_i - mean_j + moments.mean
    effect_i = mean_i - moments.mean
    effect_j = mean_j - moments.mean
    scale = moments.scale

    doevd = float(np.sum(interaction**2) / scale)
    doevs_i = float(np.sum(interaction * effect_i) / scale)
    doevs_j = float(np.sum(interaction * effect_j) / scale)
    joint_eta = float(np.sum((mean_ij - moments.mean) ** 2) / scale)
    return (
        doevd,
        doevs_i,
        doevs_j,
        _partial_ratio(joint_eta, eta_i),
        _partial_ratio(joint_eta, eta_j),
    )


def all_factor_stats(
    y: np.ndarray,
    x_codes: np.ndarray,
) -> tuple[Moments, list[FactorDispersion]]:
    """Считает ПДО и гетероскедастичность по каждому столбцу кодов.

    Args:
        y: нормированный выход.
        x_codes: коды факторов (N, n).

    Returns:
        Сводки выхода и список характеристик факторов.
    """
    moments = output_moments(y)
    stats = [
        factor_dispersion(y, x_codes[:, col], moments)
        for col in range(x_codes.shape[1])
    ]
    return moments, stats


def multiple_correlation(x_norm: np.ndarray, y: np.ndarray) -> float:
    """Считает R по определителям корреляционных матриц.

    R = sqrt( |1 - det(corr(X,Y)) / det(corr(X))| ).
    При вырожденной матрице факторов возвращается 0.

    Args:
        x_norm: нормированные доминантные факторы (N, m).
        y: нормированный выход.

    Returns:
        Неотрицательный показатель R.
    """
    factors = np.asarray(x_norm, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1)
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)
    if factors.shape[1] == 0 or len(target) < 2:
        return 0.0

    joint = np.column_stack([factors, target])
    try:
        det_joint = float(np.linalg.det(np.corrcoef(joint, rowvar=False)))
        det_factors = float(
            np.linalg.det(np.atleast_2d(np.corrcoef(factors, rowvar=False)))
        )
    except np.linalg.LinAlgError:
        return 0.0
    if det_factors == 0.0 or not np.isfinite(det_joint):
        return 0.0
    if not np.isfinite(det_factors):
        return 0.0
    return float(np.sqrt(abs(1.0 - abs(det_joint) / abs(det_factors))))
