"""Рисунки: пустые ячейки остаются пустыми, файлы png создаются."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arm.charts import (
    FILE_HEATMAP,
    FILE_PDO,
    FILE_PDO_HEATMAP,
    FILE_SCATTER,
    FILE_SQUARE_RADAR,
    FILE_SURFACE,
    conditional_mean_grid,
    dispersion_ratio_matrix,
    heatmap_inches,
    save_charts,
    sliding_neighbor_effects,
)
from arm.square_radar import (
    build_square_radar,
    radar_fraction,
    square_perimeter,
    square_radar_vertices,
)
from tumalgoritm.pipeline import run_frame


def test_empty_cells_stay_nan():
    """В сетке среднего нет числа там, где пара кодов не встречалась."""
    y = np.array([0.0, 1.0])
    codes_a = np.array([0, 1])
    codes_b = np.array([0, 1])
    grid = conditional_mean_grid(y, codes_a, codes_b, n_levels=3)
    assert grid[0, 0] == 0.0
    assert grid[1, 1] == 1.0
    assert np.isnan(grid[0, 1])
    assert np.isnan(grid[2, 2])


def test_save_charts_writes_pngs(tmp_path: Path):
    """Для двух факторов сохраняются столбцы, тепловая карта и 3D-графики."""
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "x2": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
        }
    )
    result = run_frame(
        frame,
        "y",
        nadc_x=2,
        nadc_y=2,
        effect_order=2,
        min_dominant=2,
        max_dominant=2,
    )
    written = save_charts(result, tmp_path)
    names = {path.name for path in written}
    assert FILE_PDO in names
    assert FILE_PDO_HEATMAP in names
    assert FILE_HEATMAP in names
    assert FILE_SURFACE in names
    assert FILE_SQUARE_RADAR in names
    assert FILE_SCATTER in names
    assert all(path.stat().st_size > 0 for path in written)


def test_square_perimeter_steps_clockwise_from_the_top():
    """Четыре точки — середины сторон, обход по часовой стрелке."""
    points = square_perimeter(4)
    expected = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ]
    )
    assert points == pytest.approx(expected)


def test_square_radar_shares_scale_and_marks_dominant():
    """Большее ПДО дальше от центра, доминантная точка помечена."""
    vertices, low, high = square_radar_vertices(
        ["strong", "weak"],
        [0.9, 0.2],
        [0.1, 0.0],
        [0.4, 0.05],
        [0],
    )
    assert low == 0.0
    assert high == 1.0
    far = vertices[0].pdo_x ** 2 + vertices[0].pdo_y ** 2
    near = vertices[1].pdo_x ** 2 + vertices[1].pdo_y ** 2
    assert far > near
    assert vertices[0].dominant
    assert not vertices[1].dominant
    assert vertices[0].pdo_x == pytest.approx(vertices[0].anchor_x * 0.9)
    assert vertices[0].pdo_y == pytest.approx(vertices[0].anchor_y * 0.9)


def test_negative_doevs_lies_inside_the_zero_level():
    """Отрицательный ДОЭВС ближе к центру, чем уровень нуля."""
    vertices, low, high = square_radar_vertices(
        ["a", "b"],
        [0.5, 0.4],
        [-0.25, 0.2],
        [0.1, 0.1],
        [],
    )
    assert low == pytest.approx(-0.25)
    zero = radar_fraction(0.0, low, high)
    anchor = np.hypot(vertices[0].anchor_x, vertices[0].anchor_y)
    negative = np.hypot(vertices[0].doevs_x, vertices[0].doevs_y)
    assert negative < zero * anchor


def test_sliding_neighbor_effects_follow_column_order():
    """ДОЭВД позиции i — пара с следующим столбцом, по кругу."""
    frame = pd.DataFrame(
        {
            "a": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "b": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "c": [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            "y": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
        }
    )
    result = run_frame(
        frame,
        "y",
        nadc_x=1,
        nadc_y=2,
        effect_order=2,
        min_dominant=2,
        max_dominant=3,
    )
    doevs, doevd = sliding_neighbor_effects(result)
    pairs = {
        (link.i, link.j): link
        for link in result.pairs
    }
    count = len(result.column_names)
    for index in range(count):
        other = (index + 1) % count
        left, right = sorted((index, other))
        link = pairs[(left, right)]
        assert doevd[index] == pytest.approx(link.doevd)
        if index < other:
            assert doevs[index] == pytest.approx(link.doevs_i)
        else:
            assert doevs[index] == pytest.approx(link.doevs_j)


def test_build_square_radar_writes_png_and_vectors(tmp_path: Path):
    """png пишется всегда, svg и pdf — по запросу для статьи."""
    path = tmp_path / "square_radar.png"
    written = build_square_radar(
        ["a", "b", "c"],
        [0.8, 0.2, 0.4],
        [0.1, -0.2, 0.05],
        [0.3, 0.1, 0.6],
        [0],
        path,
        vector=True,
    )
    assert written == path
    assert path.stat().st_size > 0
    assert path.with_suffix(".svg").stat().st_size > 0
    assert path.with_suffix(".pdf").stat().st_size > 0


def test_heatmap_inches_stay_bounded():
    """Сотни эффектов не раздувают рисунок до гигабайтной растеризации."""
    assert heatmap_inches(2) == 6.0
    assert heatmap_inches(1800) == 12.0


def test_pdo_heatmap_matches_joint_ratio():
    """Вне диагонали лежит совместное ПДО пары, на диагонали — прямое."""
    from tumalgoritm.dispersion import dispersion_ratio

    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "x2": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
        }
    )
    result = run_frame(
        frame,
        "y",
        nadc_x=2,
        nadc_y=2,
        effect_order=1,
        min_dominant=1,
        max_dominant=2,
    )
    matrix, names = dispersion_ratio_matrix(result)
    assert names == ["x1", "x2"]
    assert matrix[0, 0] == pytest.approx(result.pdo_raw[0])
    assert matrix[1, 1] == pytest.approx(result.pdo_raw[1])
    joint, _groups = dispersion_ratio(
        result.prepared.y_norm,
        result.prepared.x_codes[:, :2],
        result.moments,
    )
    assert matrix[0, 1] == pytest.approx(joint)
    assert matrix[1, 0] == pytest.approx(joint)


def test_oversized_figure_is_shrunk(tmp_path: Path):
    """Фигура в десятки дюймов записывается в обычный png."""
    import matplotlib.pyplot as plt
    from PIL import Image

    from arm.charts import MAX_FIGURE_INCHES, _write

    fig, ax = plt.subplots(figsize=(80, 80))
    ax.imshow(np.eye(8), interpolation="nearest")
    path = _write(fig, tmp_path / "wide.png")
    with Image.open(path) as image:
        assert max(image.size) <= MAX_FIGURE_INCHES * 120 + 20
