"""Графики автоматизированного рабочего места.

Рисунки строятся по уже посчитанному AnalysisResult.
Пустые ячейки условного среднего не заполняются соседними значениями.
"""

from pathlib import Path

import matplotlib

# Файл рисуется без окна, чтобы расчёт можно было запускать из консоли.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from arm.square_radar import build_square_radar
from tumalgoritm.dispersion import dispersion_ratio
from tumalgoritm.pipeline import AnalysisResult

# Сколько сильнейших пар рисовать столбиками, чтобы график оставался читаемым.
MAX_PAIR_BARS = 15

# Дюймы на одну подпись тепловой карты и пределы стороны рисунка.
# Без потолка сотни эффектов давали фигуру в сотни дюймов, и запись
# png запрашивала десятки гигабайт под растеризацию.
HEATMAP_INCH_PER_LABEL = 0.28
MIN_HEATMAP_INCHES = 6.0
MAX_HEATMAP_INCHES = 12.0

# Больше подписей на осях не читаются, остаются редкие метки.
MAX_HEATMAP_TICKS = 36

# Любой рисунок перед записью ужимается до этой стороны в дюймах.
MAX_FIGURE_INCHES = 14.0

# Имена файлов, которые создаёт save_charts.
FILE_PDO = "pdo_bars.png"
FILE_RADAR = "radar_pdo.png"
FILE_PDO_HEATMAP = "heatmap_pdo.png"
FILE_HEATMAP = "heatmap_resm.png"
FILE_SURFACE = "surface_mean.png"
FILE_PAIRS = "pair_bars.png"
FILE_NETWORK = "network.png"
FILE_SCATTER = "scatter3d.png"
FILE_SQUARE_RADAR = "square_radar.png"


def conditional_mean_grid(
    y: np.ndarray,
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    n_levels: int,
) -> np.ndarray:
    """Сетка среднего Y по паре кодов факторов.

    Ячейка без наблюдений остаётся NaN. Это не пропуск, который нужно
    закрасить интерполяцией, а отсутствие данных.

    Args:
        y: нормированный выход.
        codes_a: коды первого фактора.
        codes_b: коды второго фактора.
        n_levels: число уровней квантования, сторона сетки.

    Returns:
        Массив формы (n_levels, n_levels).
    """
    grid = np.full((n_levels, n_levels), np.nan, dtype=float)
    sums = np.zeros((n_levels, n_levels), dtype=float)
    counts = np.zeros((n_levels, n_levels), dtype=int)
    target = np.asarray(y, dtype=float).reshape(-1)
    left = np.asarray(codes_a, dtype=int).reshape(-1)
    right = np.asarray(codes_b, dtype=int).reshape(-1)
    for code_a, code_b, value in zip(left, right, target):
        if 0 <= code_a < n_levels and 0 <= code_b < n_levels:
            sums[code_a, code_b] += value
            counts[code_a, code_b] += 1
    occupied = counts > 0
    grid[occupied] = sums[occupied] / counts[occupied]
    return grid


def save_charts(result: AnalysisResult, directory: str | Path) -> list[Path]:
    """Сохраняет набор диагностических рисунков.

    Args:
        result: результат полного расчёта.
        directory: папка для png. Создаётся, если её нет.

    Returns:
        Пути к записанным файлам. Часть рисунков пропускается,
        если доминантных факторов меньше двух или трёх.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    written = [
        _save_pdo_bars(result, folder / FILE_PDO),
        _save_pdo_heatmap(result, folder / FILE_PDO_HEATMAP),
        _save_heatmap(result, folder / FILE_HEATMAP),
    ]
    if len(result.column_names) >= 2:
        radar_path = folder / FILE_SQUARE_RADAR
        written.append(_save_square_radar(result, radar_path))
    if len(result.dominant_idx) >= 3:
        written.append(_save_radar(result, folder / FILE_RADAR))
    if len(result.dominant_idx) >= 2:
        written.append(_save_surface(result, folder / FILE_SURFACE))
        written.append(_save_pair_bars(result, folder / FILE_PAIRS))
        written.append(_save_network(result, folder / FILE_NETWORK))
        written.append(_save_scatter(result, folder / FILE_SCATTER))
    return written


def _save_pdo_bars(result: AnalysisResult, path: Path) -> Path:
    """Столбцы сырого ПДО, отсортированные по убыванию."""
    order = np.argsort(result.pdo_raw)[::-1]
    names = [result.column_names[int(index)] for index in order]
    values = result.pdo_raw[order]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
    ax.barh(names[::-1], values[::-1], color="#2E75B6")
    ax.set_xlabel("ПДО")
    ax.set_title("Прямые дисперсионные отношения")
    return _write(fig, path)


def _save_radar(result: AnalysisResult, path: Path) -> Path:
    """Радар ПДО доминантных факторов. Нужно не меньше трёх осей."""
    idx = result.dominant_idx
    labels = [result.column_names[index] for index in idx]
    values = result.pdo_raw[idx]
    angles = np.linspace(0, 2 * np.pi, len(idx), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    closed_values = np.concatenate([values, values[:1]])
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(closed_angles, closed_values, color="#1F4E79")
    ax.fill(closed_angles, closed_values, color="#9DC3E6", alpha=0.4)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title("ПДО доминантных факторов")
    return _write(fig, path)


def heatmap_inches(n_labels: int) -> float:
    """Сторона тепловой карты в дюймах, не больше заданного потолка.

    Args:
        n_labels: число эффектов, сторона матрицы resM.

    Returns:
        Сторона фигуры в дюймах.
    """
    wanted = HEATMAP_INCH_PER_LABEL * max(n_labels, 1)
    return min(MAX_HEATMAP_INCHES, max(MIN_HEATMAP_INCHES, wanted))


def dispersion_ratio_matrix(
    result: AnalysisResult,
) -> tuple[np.ndarray, list[str]]:
    """Матрица дисперсионных отношений факторов с выходом.

    Диагональ — прямое ПДО фактора. Вне диагонали — совместное
    отношение пары, то же, что МДО по двум столбцам кодов.
    Это не resM: там вклады эффектов, здесь связь с Y.

    Args:
        result: итог расчёта с кодами факторов и сводками Y.

    Returns:
        Квадратная матрица и имена факторов в том же порядке.
    """
    names = list(result.column_names)
    count = len(names)
    matrix = np.zeros((count, count), dtype=float)
    raw = np.asarray(result.pdo_raw, dtype=float)
    for index in range(count):
        matrix[index, index] = float(raw[index])
    y = result.prepared.y_norm
    codes = result.prepared.x_codes
    moments = result.moments
    for link in result.pairs:
        pair = np.column_stack(
            [codes[:, link.i], codes[:, link.j]]
        )
        joint, _n_groups = dispersion_ratio(y, pair, moments)
        matrix[link.i, link.j] = joint
        matrix[link.j, link.i] = joint
    return matrix, names


def _save_pdo_heatmap(result: AnalysisResult, path: Path) -> Path:
    """Тепловая матрица прямых и парных дисперсионных отношений."""
    matrix, labels = dispersion_ratio_matrix(result)
    if len(labels) == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_axis_off()
        ax.set_title("Нет факторов для матрицы ПДО")
        return _write(fig, path)
    upper = max(1.0, float(np.max(matrix)))
    return _write(
        _matrix_figure(
            matrix,
            labels,
            "Дисперсионные отношения с Y",
            "viridis",
            0.0,
            upper,
        ),
        path,
    )


def _save_heatmap(result: AnalysisResult, path: Path) -> Path:
    """Тепловая карта матрицы перекрёстных вкладов resM."""
    matrix = np.asarray(result.effects.res_m, dtype=float)
    labels = list(result.effects.labels)
    if len(labels) == 0 or matrix.size == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_axis_off()
        ax.set_title("Матрица resM пуста")
        return _write(fig, path)
    return _write(
        _matrix_figure(
            matrix,
            labels,
            f"Матрица resM, эффектов: {len(labels)}",
            "coolwarm",
            None,
            None,
        ),
        path,
    )


def _matrix_figure(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> plt.Figure:
    """Рисует квадратную тепловую карту фиксированного размера."""
    side = heatmap_inches(len(labels))
    fig, ax = plt.subplots(figsize=(side, side))
    # nearest не раздувает клетки до размера окна при сохранении.
    image = ax.imshow(
        matrix,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    _mark_heatmap_ticks(ax, labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)
    return fig


def _mark_heatmap_ticks(ax: plt.Axes, labels: list[str]) -> None:
    """Ставит подписи осей. При длинном списке оставляет часть имён."""
    count = len(labels)
    if count <= MAX_HEATMAP_TICKS:
        positions = list(range(count))
    else:
        step = max(1, count // MAX_HEATMAP_TICKS)
        positions = list(range(0, count, step))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(
        [labels[index] for index in positions],
        rotation=90,
        fontsize=7,
    )
    ax.set_yticklabels(
        [labels[index] for index in positions],
        fontsize=7,
    )


def _save_surface(result: AnalysisResult, path: Path) -> Path:
    """Поверхность E(Y|X1,X2). Пустые ячейки не интерполируются."""
    first, second = result.dominant_idx[:2]
    n_levels = 2 ** result.prepared.nadc_x
    grid = conditional_mean_grid(
        result.prepared.y_norm,
        result.prepared.x_codes[:, first],
        result.prepared.x_codes[:, second],
        n_levels,
    )
    levels = np.arange(n_levels)
    xx, yy = np.meshgrid(levels, levels, indexing="ij")
    # Маска скрывает грань поверхности там, где наблюдений не было.
    masked = np.ma.masked_invalid(grid)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if np.isfinite(grid).any():
        ax.plot_surface(
            xx, yy, masked, cmap="viridis", linewidth=0, antialiased=False
        )
    empty = ~np.isfinite(grid)
    if empty.any():
        floor = float(np.nanmin(grid)) if np.isfinite(grid).any() else 0.0
        ax.scatter(
            xx[empty],
            yy[empty],
            np.full(int(empty.sum()), floor),
            c="0.6",
            marker="x",
            depthshade=False,
            label="нет наблюдений",
        )
        ax.legend(loc="upper left")
    ax.set_xlabel(result.column_names[first])
    ax.set_ylabel(result.column_names[second])
    ax.set_zlabel(result.y_name)
    ax.set_title("Условное среднее. Пустые ячейки не заполняются")
    return _write(fig, path)


def _save_pair_bars(result: AnalysisResult, path: Path) -> Path:
    """Столбики чистого парного взаимодействия доминантных факторов."""
    dominant = set(result.dominant_idx)
    links = [
        link
        for link in result.pairs
        if link.i in dominant and link.j in dominant
    ]
    links = sorted(links, key=lambda item: item.doevd, reverse=True)
    links = links[:MAX_PAIR_BARS]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    names = result.column_names
    for position, link in enumerate(links):
        height = max(float(link.doevd), 0.0)
        ax.bar3d(position, 0, 0, 0.6, 0.6, height, color="#ED7D31")
        ax.text(
            position,
            0,
            height,
            f"{names[link.i]}-{names[link.j]}",
            fontsize=7,
        )
    ax.set_title("Парные взаимодействия VdDO")
    ax.set_zlabel("VdDO")
    return _write(fig, path)


def _save_network(result: AnalysisResult, path: Path) -> Path:
    """Круговая сеть доминантных факторов. Толщина ребра — VdDO."""
    idx = result.dominant_idx
    place = {factor: position for position, factor in enumerate(idx)}
    angles = np.linspace(0, 2 * np.pi, len(idx), endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)
    fig, ax = plt.subplots(figsize=(7, 7))
    dominant = set(idx)
    for link in result.pairs:
        if link.i not in dominant or link.j not in dominant:
            continue
        if link.doevd <= 0:
            continue
        left = place[link.i]
        right = place[link.j]
        width = 0.4 + 4.0 * float(link.doevd)
        ax.plot(
            [xs[left], xs[right]],
            [ys[left], ys[right]],
            color="#2E75B6",
            linewidth=width,
            alpha=0.7,
        )
    ax.scatter(xs, ys, s=80, color="#1F4E79", zorder=3)
    for position, factor in enumerate(idx):
        ax.text(
            xs[position] * 1.12,
            ys[position] * 1.12,
            result.column_names[factor],
        )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Сеть доминантных факторов")
    return _write(fig, path)


def _save_scatter(result: AnalysisResult, path: Path) -> Path:
    """Облако наблюдений: два сильнейших фактора и выход."""
    first, second = result.dominant_idx[:2]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        result.prepared.x_norm[:, first],
        result.prepared.x_norm[:, second],
        result.prepared.y_norm,
        c=result.prepared.y_norm,
        cmap="viridis",
        s=18,
    )
    ax.set_xlabel(result.column_names[first])
    ax.set_ylabel(result.column_names[second])
    ax.set_zlabel(result.y_name)
    ax.set_title("Наблюдения")
    return _write(fig, path)


def sliding_neighbor_effects(
    result: AnalysisResult,
) -> tuple[np.ndarray, np.ndarray]:
    """ДОЭВС и ДОЭВД скользящих соседних пар.

    Пара позиции i — факторы i и i+1. После последнего идёт первый.
    ДОЭВД одно на пару. ДОЭВС — проекция этой пары на фактор i,
    а не среднее двух направлений: они у одной пары разные.

    Args:
        result: итог расчёта с полем pairs.

    Returns:
        Два массива длины n. Если пары нет, на месте ноль.
    """
    count = len(result.column_names)
    doevs = np.zeros(count, dtype=float)
    doevd = np.zeros(count, dtype=float)
    stored: dict[frozenset[int], tuple[float, float, float]] = {}
    for link in result.pairs:
        stored[frozenset((link.i, link.j))] = (
            float(link.doevd),
            float(link.doevs_i),
            float(link.doevs_j),
        )
    for index in range(count):
        other = (index + 1) % count
        found = stored.get(frozenset((index, other)))
        if found is None or index == other:
            continue
        pair_doevd, on_low, on_high = found
        doevd[index] = pair_doevd
        # В паре i < j: первая проекция относится к меньшему номеру.
        if index < other:
            doevs[index] = on_low
        else:
            doevs[index] = on_high
    return doevs, doevd


def _save_square_radar(result: AnalysisResult, path: Path) -> Path:
    """Квадратный радар ПДО, ДОЭВС и ДОЭВД всех входных переменных."""
    doevs, doevd = sliding_neighbor_effects(result)
    return build_square_radar(
        list(result.column_names),
        np.asarray(result.pdo_raw, dtype=float),
        doevs,
        doevd,
        list(result.dominant_idx),
        path,
    )


def _write(
    fig: plt.Figure,
    path: Path,
    *,
    layout: bool = True,
) -> Path:
    """Сохраняет рисунок и закрывает его, чтобы не копить память.

    Args:
        fig: фигура matplotlib.
        path: файл png.
        layout: вызывать tight_layout перед записью.
    """
    _limit_figure_inches(fig)
    if layout:
        fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def _limit_figure_inches(fig: plt.Figure) -> None:
    """Ужимает слишком большую фигуру до записи файла."""
    width, height = fig.get_size_inches()
    longest = max(float(width), float(height), 1.0)
    if longest <= MAX_FIGURE_INCHES:
        return
    scale = MAX_FIGURE_INCHES / longest
    fig.set_size_inches(width * scale, height * scale)
