"""Нормировка факторов и квантование по разрядности АЦП.

Дисперсионные отношения считаются по непрерывному нормированному
выходу, сгруппированному по кодам факторов. Информационные критерии
используют коды и факторов, и выхода.
"""

from dataclasses import dataclass

import numpy as np

from tumalgoritm.constants import (
    MAX_AUTO_ADC_BITS,
    MAX_EXPLICIT_ADC_BITS,
    MIN_OBS_PER_CELL,
)


@dataclass
class PreparedData:
    """Нормированные ряды и целочисленные коды квантования.

    Attributes:
        x_norm: факторы в диапазоне [0, 1], форма (N, n).
        y_norm: выход в диапазоне [0, 1], длина N.
        x_codes: коды факторов, целые от 0 до 2^nadc_x - 1.
        y_codes: коды выхода, целые от 0 до 2^nadc_y - 1.
        nadc_x: фактическая разрядность факторов.
        nadc_y: фактическая разрядность выхода.
        variance_ratio: отношение дисперсии квантованного Y
            к дисперсии шума квантования.
    """

    x_norm: np.ndarray
    y_norm: np.ndarray
    x_codes: np.ndarray
    y_codes: np.ndarray
    nadc_x: int
    nadc_y: int
    variance_ratio: float


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Приводит каждый столбец к отрезку [0, 1].

    Формула столбца: (x - min) / (max - min). Если столбец
    постоянный, деление не выполняется и столбец заполняется нулями.

    Args:
        matrix: матрица наблюдений формы (N, n) или вектор длины N.

    Returns:
        Нормированная матрица той же формы.
    """
    values = np.asarray(matrix, dtype=float)
    squeezed = values.ndim == 1
    if squeezed:
        values = values.reshape(-1, 1)

    normalized = np.zeros(values.shape, dtype=float)
    for col in range(values.shape[1]):
        column = values[:, col]
        low = float(np.min(column))
        high = float(np.max(column))
        span = high - low
        if span == 0.0:
            normalized[:, col] = 0.0
        else:
            normalized[:, col] = (column - low) / span

    if squeezed:
        return normalized.reshape(-1)
    return normalized


def quantize_signal(signal_norm: np.ndarray, n_adc: int) -> np.ndarray:
    """Переводит нормированный сигнал в номера уровней АЦП.

    Число уровней равно 2^n_adc, шаг равен 1 / число_уровней.
    Индекс считается как round(x / шаг) и обрезается в допустимый
    диапазон. Значение 1 попадает на последний уровень.

    Args:
        signal_norm: нормированный сигнал, значения около [0, 1].
        n_adc: число двоичных разрядов, не меньше 1.

    Returns:
        Массив целых кодов той же формы, что и signal_norm.
    """
    if n_adc < 1:
        raise ValueError("Разрядность квантования должна быть не меньше 1")

    levels = 2**n_adc
    step = 1.0 / levels
    # floor(x + 0.5) для неотрицательных чисел совпадает с обычным округлением.
    raw_levels = np.floor(np.asarray(signal_norm, dtype=float) / step + 0.5)
    return np.clip(raw_levels, 0, levels - 1).astype(int)


def reconstruct_signal(codes: np.ndarray, n_adc: int) -> np.ndarray:
    """Восстанавливает уровень сигнала из кода АЦП.

    Args:
        codes: коды квантования.
        n_adc: разрядность, которой эти коды получены.

    Returns:
        Значения кода, умноженные на шаг квантования.
    """
    step = 1.0 / (2**n_adc)
    return np.asarray(codes, dtype=float) * step


def variance_ratio(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Считает Dy_y / (Dy - Dy_y) для оценки шума квантования.

    Dy — дисперсия исходного нормированного выхода, Dy_y — дисперсия
    сигнала, восстановленного из кодов. Если шум нулевой, возвращается 0,
    чтобы не делить на ноль.

    Args:
        original: нормированный выход до квантования.
        reconstructed: выход, восстановленный из кодов.

    Returns:
        Неотрицательное отношение дисперсий.
    """
    signal_var = float(np.var(np.asarray(original, dtype=float)))
    quantized_var = float(np.var(np.asarray(reconstructed, dtype=float)))
    noise_var = signal_var - quantized_var
    if noise_var == 0.0:
        return 0.0
    return quantized_var / noise_var


def optimal_adc_bits(n_obs: int) -> int:
    """Подбирает разрядность так, чтобы парная сетка была заполнена.

    Условие прототипа: в двумерной сетке в среднем не меньше
    MIN_OBS_PER_CELL точек на ячейку, то есть
    2^(2b) <= N / MIN_OBS_PER_CELL.

    Args:
        n_obs: число наблюдений.

    Returns:
        Разрядность от 1 до MAX_AUTO_ADC_BITS.
    """
    if n_obs <= MIN_OBS_PER_CELL * 2:
        return 1
    bits = int(np.floor(0.5 * np.log2(n_obs / float(MIN_OBS_PER_CELL))))
    return max(1, min(bits, MAX_AUTO_ADC_BITS))


def resolve_adc_bits(requested: int, n_obs: int) -> int:
    """Возвращает разрядность: 0 означает автоподбор.

    Args:
        requested: запрошенные биты; 0 — посчитать автоматически.
        n_obs: число наблюдений, нужно только для автоподбора.

    Returns:
        Фактическая разрядность.

    Raises:
        ValueError: разрядность отрицательная или слишком большая.
    """
    if requested < 0:
        raise ValueError("Разрядность АЦП не может быть отрицательной")
    if requested == 0:
        return optimal_adc_bits(n_obs)
    if requested > MAX_EXPLICIT_ADC_BITS:
        raise ValueError(
            f"Разрядность {requested} больше допустимой "
            f"{MAX_EXPLICIT_ADC_BITS}"
        )
    return requested


def prepare_signals(
    x: np.ndarray,
    y: np.ndarray,
    nadc_x: int,
    nadc_y: int,
) -> PreparedData:
    """Нормирует X и Y и квантует их заданной или авто-разрядностью.

    Args:
        x: матрица факторов (N, n).
        y: вектор выхода длины N.
        nadc_x: разрядность факторов, 0 — подобрать.
        nadc_y: разрядность выхода, 0 — подобрать.

    Returns:
        PreparedData с нормированными рядами и кодами.
    """
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if x_values.ndim != 2:
        raise ValueError("Матрица факторов должна быть двумерной")
    if len(y_values) != x_values.shape[0]:
        raise ValueError("Число строк X и длина Y не совпадают")

    n_obs = len(y_values)
    bits_x = resolve_adc_bits(nadc_x, n_obs)
    bits_y = resolve_adc_bits(nadc_y, n_obs)

    x_norm = normalize_columns(x_values)
    y_norm = normalize_columns(y_values)
    x_codes = quantize_signal(x_norm, bits_x)
    y_codes = quantize_signal(y_norm, bits_y)
    restored = reconstruct_signal(y_codes, bits_y)
    ratio = variance_ratio(y_norm, restored)

    return PreparedData(
        x_norm=x_norm,
        y_norm=y_norm,
        x_codes=x_codes,
        y_codes=y_codes,
        nadc_x=bits_x,
        nadc_y=bits_y,
        variance_ratio=ratio,
    )
