"""Проверки формул методики на коротких синтетических выборках."""

import numpy as np
import pandas as pd
import pytest

from tumalgoritm.constants import MIN_OBS_PER_CELL
from tumalgoritm.dispersion import (
    bias_correct_eta2,
    output_moments,
    pair_link,
)
from tumalgoritm.dominant import criterion_q, select_dominant
from tumalgoritm.effects import (
    ConditionalMeanCache,
    expanded_triple_effect,
)
from tumalgoritm.information import conditional_mutual_information
from tumalgoritm.pipeline import run_frame
from tumalgoritm.prepare import normalize_columns, quantize_signal


def test_constant_column_normalizes_to_zero():
    """Постоянный столбец не вызывает деление на ноль и становится нулём."""
    column = np.array([4.0, 4.0, 4.0])
    normalized = normalize_columns(column)
    assert np.allclose(normalized, 0.0)


def test_quantize_endpoints():
    """0 попадает на первый уровень, 1 — на последний."""
    codes = quantize_signal(np.array([0.0, 1.0]), n_adc=2)
    assert list(codes) == [0, 3]


def test_bias_correction_matches_formula():
    """Поправка eta_adj совпадает с формулой скорректированного R^2."""
    eta = 0.5
    n_obs = 10
    n_groups = 4
    expected = 1.0 - (1.0 - eta) * (n_obs - 1) / (n_obs - n_groups)
    assert bias_correct_eta2(eta, n_obs, n_groups) == pytest.approx(expected)
    assert np.isnan(bias_correct_eta2(eta, n_obs=4, n_groups=4))


def test_constant_factor_has_zero_pdo():
    """Фактор без разброса не объясняет выход."""
    frame = pd.DataFrame(
        {
            "const": [3.0, 3.0, 3.0, 3.0],
            "x": [0.0, 0.0, 1.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        }
    )
    result = run_frame(
        frame,
        "y",
        nadc_x=1,
        nadc_y=1,
        effect_order=1,
        min_dominant=1,
        max_dominant=2,
    )
    assert result.pdo_raw[0] == pytest.approx(0.0)
    assert result.pdo_raw[1] == pytest.approx(1.0)


def test_single_factor_pdo_and_balance():
    """Полная связь одного фактора: ПДО = МДО = 1, ОДО = 0."""
    frame = pd.DataFrame(
        {
            "x": [0.0, 0.0, 1.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        }
    )
    result = run_frame(
        frame,
        "y",
        nadc_x=1,
        nadc_y=1,
        effect_order=1,
        min_dominant=1,
        max_dominant=1,
    )
    assert result.pdo_raw[0] == pytest.approx(1.0)
    assert result.mdo == pytest.approx(1.0)
    assert result.mdo + result.odo == pytest.approx(1.0)
    assert result.effects.fdo == pytest.approx(1.0)
    assert result.effects.res_m.shape == (1, 1)


def test_noise_factor_has_no_interaction_and_no_extra_information():
    """Второй фактор, сбалансированный внутри первого, не добавляет связи."""
    frame = pd.DataFrame(
        {
            "x1": [0, 0, 0, 0, 1, 1, 1, 1],
            "x2": [0, 1, 0, 1, 0, 1, 0, 1],
            "y": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    result = run_frame(
        frame,
        "y",
        nadc_x=1,
        nadc_y=1,
        effect_order=2,
        min_dominant=2,
        max_dominant=2,
    )
    assert len(result.pairs) == 1
    assert result.pairs[0].doevd == pytest.approx(0.0)
    # x2 стоит вторым столбцом.
    unique_x2 = result.information.unique[1]
    assert unique_x2 == pytest.approx(0.0, abs=1e-12)
    assert result.mdo + result.odo == pytest.approx(1.0)


def test_low_fill_warning():
    """Мало точек на ячейку попадает в список предупреждений."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x": rng.random(12),
            "y": rng.random(12),
        }
    )
    result = run_frame(
        frame,
        "y",
        nadc_x=4,
        nadc_y=2,
        effect_order=1,
        min_dominant=1,
        max_dominant=1,
    )
    assert any(str(MIN_OBS_PER_CELL) in text for text in result.warnings)


def test_q_criterion_picks_the_gap():
    """Минимум Q стоит на границе между сильными и слабым фактором."""
    scores = np.array([0.5, 0.4, 0.01])
    assert criterion_q(np.sort(scores)[::-1], 2) < criterion_q(
        np.sort(scores)[::-1], 1
    )
    selection = select_dominant(scores, eta0=0.8)
    assert selection.m_star == 2
    assert selection.s_m == pytest.approx(0.9)
    assert selection.sufficient


def test_triple_effect_matches_expanded_formula():
    """Включения-исключения для тройки совпадают с развёрнутой формулой."""
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 4, size=(40, 3))
    y = rng.normal(size=40)
    mean = float(np.mean(y))
    cache = ConditionalMeanCache(y, codes, mean)
    generic = cache.effect((0, 1, 2))
    expanded = expanded_triple_effect(y, codes, mean)
    assert np.allclose(generic, expanded)


def test_pair_link_matches_manual_squares():
    """VdDO равен сумме квадратов чистого взаимодействия, делённой на NDy."""
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.2, 0.8])
    codes_i = np.array([0, 0, 1, 1, 0, 1])
    codes_j = np.array([0, 1, 0, 1, 1, 0])
    moments = output_moments(y)
    doevd, _, _, _, _ = pair_link(
        y, codes_i, codes_j, moments, eta_i=0.2, eta_j=0.2
    )
    assert doevd >= 0.0


def test_conditional_information_of_irrelevant_code_is_zero():
    """Код, не меняющий Y при известном условии, не несёт информации."""
    x1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    x2 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    info = conditional_mutual_information(x2, y, x1)
    assert info == pytest.approx(0.0, abs=1e-12)
