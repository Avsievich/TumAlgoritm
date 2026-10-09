"""Квадратная радарная диаграмма дисперсионных отношений.

Переменные стоят на периметре квадрата. По каждому лучу к центру
отложены три величины на одной шкале: ПДО, ДОЭВС и ДОЭВД.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Файл пишется без окна: расчёт запускают и из консоли, и из АРМ.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Внешняя сторона квадрата — уровень 1. Центр — 0, либо ниже,
# если среди ДОЭВС есть отрицательные числа.
RADAR_HIGH = 1.0
RADAR_HALF = 1.0

# Отступ имени за периметром, в долях половины стороны.
RADAR_NAME_GAP = 0.18

# ПДО — основной критерий, поэтому линия толще остальных.
RADAR_PDO_WIDTH = 2.6
RADAR_DOEVS_WIDTH = 1.7
RADAR_DOEVD_WIDTH = 1.3

# Деления общей шкалы: внешний, средний и внутренний квадраты.
RADAR_RINGS = (1.0, 2.0 / 3.0, 1.0 / 3.0)

_RADAR_PDO = "#1F4E79"
_RADAR_DOEVS = "#548235"
_RADAR_DOEVS_NEG = "#7030A0"
_RADAR_DOEVD = "#ED7D31"
_RADAR_DOMINANT = "#C00000"

RADAR_TITLE = "Радарная диаграмма дисперсионных отношений"
RADAR_SUBTITLE = (
    "ПДО — индивидуальная связь; "
    "ДОЭВС — эффект взаимосвязи соседних факторов;\n"
    "ДОЭВД — эффект нелинейного взаимодействия соседних факторов"
)

# Подпись на рисунке не должна раздувать файл, как тепловая карта.
_MAX_INCHES = 14.0


@dataclass(frozen=True)
class RadarVertex:
    """Одна переменная на квадратном радаре.

    Attributes:
        index: номер переменной в переданных массивах.
        name: подпись на периметре.
        dominant: переменная входит в доминантный набор.
        anchor_x: точка переменной на внешней стороне.
        anchor_y: точка переменной на внешней стороне.
        pdo_x: положение ПДО на луче к центру.
        pdo_y: положение ПДО на луче к центру.
        doevs_x: положение ДОЭВС на том же луче.
        doevs_y: положение ДОЭВС на том же луче.
        doevd_x: положение ДОЭВД на том же луче.
        doevd_y: положение ДОЭВД на том же луче.
        pdo: исходное ПДО.
        doevs: исходный ДОЭВС соседней пары.
        doevd: исходный ДОЭВД соседней пары.
    """

    index: int
    name: str
    dominant: bool
    anchor_x: float
    anchor_y: float
    pdo_x: float
    pdo_y: float
    doevs_x: float
    doevs_y: float
    doevd_x: float
    doevd_y: float
    pdo: float
    doevs: float
    doevd: float


def radar_bounds(doevs: np.ndarray) -> tuple[float, float]:
    """Общая шкала трёх контуров.

    Верх всегда 1, чтобы ПДО, ДОЭВС и ДОЭВД читались в одних
    единицах. Если ДОЭВС уходит ниже нуля, низ шкалы опускается
    до этого минимума: ноль перестаёт быть центром.

    Args:
        doevs: значения ДОЭВС длины n.

    Returns:
        Нижняя и верхняя границы шкалы.
    """
    values = np.asarray(doevs, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    low = 0.0
    if len(finite):
        low = min(0.0, float(np.min(finite)))
    return low, RADAR_HIGH


def radar_fraction(value: float, low: float, high: float) -> float:
    """Доля луча от центра до внешней стороны.

    Args:
        value: величина на общей шкале.
        low: нижняя граница, 0 или отрицательная.
        high: верхняя граница, обычно 1.

    Returns:
        Число от 0 до 1. Вне шкалы значение прижимается к краю.
    """
    if not np.isfinite(value):
        return 0.0
    span = high - low
    if span <= 0.0:
        return 0.0
    clipped = min(max(float(value), low), high)
    return (clipped - low) / span


def square_perimeter(count: int, half: float = RADAR_HALF) -> np.ndarray:
    """Точки на квадрате с равным шагом по периметру.

    Обход по часовой стрелке. Первая точка — середина верхней
    стороны, дальше вправо.

    Args:
        count: сколько точек поставить.
        half: половина стороны квадрата.

    Returns:
        Массив формы (count, 2). При count <= 0 массив пустой.
    """
    if count <= 0:
        return np.zeros((0, 2), dtype=float)
    vertices = _square_loop(half)
    lengths = _path_lengths(vertices)
    total = float(lengths[-1])
    points = np.empty((count, 2), dtype=float)
    for index in range(count):
        points[index] = _walk_path(
            vertices,
            lengths,
            total * index / count,
        )
    return points


def square_radar_vertices(
    colnames,
    pdoall,
    doevssliding,
    doevdsliding,
    dominantidx,
) -> tuple[list[RadarVertex], float, float]:
    """Считает точки трёх контуров на общих лучах.

    Args:
        colnames: имена переменных.
        pdoall: ПДО длины n.
        doevssliding: ДОЭВС соседних пар длины n.
        doevdsliding: ДОЭВД соседних пар длины n.
        dominantidx: индексы, которые нужно выделить.

    Returns:
        Вершины, нижняя граница шкалы и верхняя.

    Raises:
        ValueError: длины не совпали или переменных меньше двух.
    """
    names = [str(item) for item in colnames]
    pdo = np.asarray(pdoall, dtype=float).reshape(-1)
    doevs = np.asarray(doevssliding, dtype=float).reshape(-1)
    doevd = np.asarray(doevdsliding, dtype=float).reshape(-1)
    count = len(names)
    same = len(pdo) == count and len(doevs) == count
    if not same or len(doevd) != count:
        raise ValueError(
            "Длины имён, ПДО, ДОЭВС и ДОЭВД должны совпадать."
        )
    if count < 2:
        raise ValueError(
            "Для соседних пар нужны хотя бы две переменные."
        )
    low, high = radar_bounds(doevs)
    dominant = {int(item) for item in dominantidx}
    anchors = square_perimeter(count)
    vertices: list[RadarVertex] = []
    for index, name in enumerate(names):
        anchor = anchors[index]
        vertices.append(
            _vertex(
                index,
                name,
                index in dominant,
                anchor,
                float(pdo[index]),
                float(doevs[index]),
                float(doevd[index]),
                low,
                high,
            )
        )
    return vertices, low, high


def build_square_radar(
    colnames,
    pdoall,
    doevssliding,
    doevdsliding,
    dominantidx,
    output_path,
    title: str = RADAR_TITLE,
    *,
    vector: bool = False,
) -> Path:
    """Рисует квадратный радар и записывает png.

    При vector=True рядом появляются svg и pdf с тем же именем:
    их удобно вставлять в статью.

    Args:
        colnames: имена переменных.
        pdoall: ПДО длины n.
        doevssliding: ДОЭВС соседних пар длины n.
        doevdsliding: ДОЭВД соседних пар длины n.
        dominantidx: индексы доминантных переменных.
        output_path: путь png.
        title: заголовок рисунка.
        vector: записать ещё svg и pdf.

    Returns:
        Путь к png.

    Raises:
        ValueError: длины массивов различаются или n < 2.
    """
    vertices, low, high = square_radar_vertices(
        colnames,
        pdoall,
        doevssliding,
        doevdsliding,
        dominantidx,
    )
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    _draw_rays(ax, vertices)
    _draw_scale_squares(ax, low, high)
    _draw_closed(
        ax,
        [item.doevd_x for item in vertices],
        [item.doevd_y for item in vertices],
        color=_RADAR_DOEVD,
        lw=RADAR_DOEVD_WIDTH,
        linestyle=(0, (4, 1.6)),
        zorder=2,
    )
    _draw_doevs(ax, vertices)
    _draw_closed(
        ax,
        [item.pdo_x for item in vertices],
        [item.pdo_y for item in vertices],
        color=_RADAR_PDO,
        lw=RADAR_PDO_WIDTH,
        zorder=4,
    )
    _draw_markers(ax, vertices)
    _draw_names(ax, vertices)
    limit = RADAR_HALF * float(np.sqrt(2.0)) + RADAR_NAME_GAP
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.text(
        0.5,
        0.93,
        RADAR_SUBTITLE,
        ha="center",
        va="top",
        fontsize=8,
    )
    has_negative = any(item.doevs < 0.0 for item in vertices)
    fig.legend(
        handles=_radar_legend(has_negative),
        loc="lower center",
        ncol=4 if has_negative else 3,
        fontsize=8,
        frameon=False,
    )
    fig.subplots_adjust(left=0.04, right=0.96, top=0.84, bottom=0.08)
    return _save_radar(fig, Path(output_path), vector)


def _vertex(
    index: int,
    name: str,
    dominant: bool,
    anchor: np.ndarray,
    pdo: float,
    doevs: float,
    doevd: float,
    low: float,
    high: float,
) -> RadarVertex:
    """Кладёт три значения на луч от центра к точке периметра."""
    pdo_f = radar_fraction(pdo, low, high)
    doevs_f = radar_fraction(doevs, low, high)
    doevd_f = radar_fraction(doevd, low, high)
    return RadarVertex(
        index=index,
        name=name,
        dominant=dominant,
        anchor_x=float(anchor[0]),
        anchor_y=float(anchor[1]),
        pdo_x=float(anchor[0] * pdo_f),
        pdo_y=float(anchor[1] * pdo_f),
        doevs_x=float(anchor[0] * doevs_f),
        doevs_y=float(anchor[1] * doevs_f),
        doevd_x=float(anchor[0] * doevd_f),
        doevd_y=float(anchor[1] * doevd_f),
        pdo=pdo,
        doevs=doevs,
        doevd=doevd,
    )


def _square_loop(half: float) -> np.ndarray:
    """Вершины обхода: верхняя середина и дальше по часовой."""
    return np.array(
        [
            [0.0, half],
            [half, half],
            [half, -half],
            [-half, -half],
            [-half, half],
            [0.0, half],
        ],
        dtype=float,
    )


def _path_lengths(vertices: np.ndarray) -> np.ndarray:
    """Накопленная длина ломаной, первая точка — ноль."""
    lengths = np.zeros(len(vertices), dtype=float)
    for index in range(1, len(vertices)):
        step = vertices[index] - vertices[index - 1]
        lengths[index] = lengths[index - 1] + float(
            np.hypot(step[0], step[1])
        )
    return lengths


def _walk_path(
    vertices: np.ndarray,
    lengths: np.ndarray,
    distance: float,
) -> np.ndarray:
    """Точка на ломаной на заданном расстоянии от старта."""
    last = len(vertices) - 1
    for index in range(1, len(vertices)):
        at_end = index == last
        if distance <= lengths[index] or at_end:
            span = float(lengths[index] - lengths[index - 1])
            if span <= 0.0:
                return vertices[index]
            done = (distance - lengths[index - 1]) / span
            done = min(max(float(done), 0.0), 1.0)
            return vertices[index - 1] + done * (
                vertices[index] - vertices[index - 1]
            )
    return vertices[0]


def _draw_rays(ax: plt.Axes, vertices: list[RadarVertex]) -> None:
    """Лучи от центра к местам переменных на периметре."""
    for item in vertices:
        ax.plot(
            [0.0, item.anchor_x],
            [0.0, item.anchor_y],
            color="0.82",
            lw=0.7,
            zorder=0,
        )


def _draw_scale_squares(ax: plt.Axes, low: float, high: float) -> None:
    """Три квадрата шкалы и, при отрицательном ДОЭВС, уровень нуля."""
    styles = (
        {"color": "0.15", "linestyle": "-", "lw": 1.15},
        {"color": "0.45", "linestyle": "-", "lw": 0.8},
        {"color": "0.62", "linestyle": "--", "lw": 0.8},
    )
    for level, style in zip(RADAR_RINGS, styles):
        half = RADAR_HALF * radar_fraction(level, low, high)
        _plot_square(ax, half, style)
        _label_level(ax, half, level)
    if low < 0.0:
        half = RADAR_HALF * radar_fraction(0.0, low, high)
        _plot_square(
            ax,
            half,
            {"color": "0.25", "linestyle": ":", "lw": 1.05},
        )
        _label_level(ax, half, 0.0)


def _plot_square(ax: plt.Axes, half: float, style: dict) -> None:
    """Замкнутый квадрат с центром в нуле."""
    if half <= 0.0:
        return
    side = [-half, half, half, -half, -half]
    other = [-half, -half, half, half, -half]
    ax.plot(side, other, zorder=1, **style)


def _label_level(ax: plt.Axes, half: float, level: float) -> None:
    """Число шкалы у верхнего левого угла квадрата."""
    if half <= 0.0:
        return
    if abs(level) < 1e-9 or abs(level - 1.0) < 1e-9:
        caption = f"{level:.0f}"
    else:
        caption = f"{level:.2f}"
    ax.text(
        -half + 0.03,
        half - 0.03,
        caption,
        fontsize=7,
        color="0.35",
        ha="left",
        va="top",
        zorder=2,
    )


def _draw_closed(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    **style,
) -> None:
    """Замкнутая ломаная по всем переменным."""
    closed_x = np.append(np.asarray(xs, dtype=float), xs[0])
    closed_y = np.append(np.asarray(ys, dtype=float), ys[0])
    ax.plot(closed_x, closed_y, **style)


def _draw_doevs(ax: plt.Axes, vertices: list[RadarVertex]) -> None:
    """Контур ДОЭВС. Отрицательные участки — другим цветом."""
    count = len(vertices)
    for index, item in enumerate(vertices):
        other = vertices[(index + 1) % count]
        _draw_signed_segment(
            ax,
            np.array([item.doevs_x, item.doevs_y]),
            item.doevs,
            np.array([other.doevs_x, other.doevs_y]),
            other.doevs,
        )


def _draw_signed_segment(
    ax: plt.Axes,
    start: np.ndarray,
    start_value: float,
    end: np.ndarray,
    end_value: float,
) -> None:
    """Отрезок ДОЭВС, разрезанный в нуле, если знак меняется."""
    crosses = start_value < 0.0 < end_value or end_value < 0.0 < start_value
    if crosses and start_value != end_value:
        share = (0.0 - start_value) / (end_value - start_value)
        middle = start + share * (end - start)
        _plot_segment(ax, start, middle, start_value)
        _plot_segment(ax, middle, end, end_value)
        return
    _plot_segment(ax, start, end, min(start_value, end_value))


def _plot_segment(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    value: float,
) -> None:
    """Кусок контура ДОЭВС одного знака."""
    color = _RADAR_DOEVS_NEG if value < 0.0 else _RADAR_DOEVS
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        lw=RADAR_DOEVS_WIDTH,
        solid_capstyle="round",
        zorder=3,
    )


def _draw_markers(ax: plt.Axes, vertices: list[RadarVertex]) -> None:
    """Точки ПДО. Доминантные — красные и крупнее."""
    for item in vertices:
        if item.dominant:
            color = _RADAR_DOMINANT
            size = 70
        else:
            color = _RADAR_PDO
            size = 22
        ax.scatter(
            [item.pdo_x],
            [item.pdo_y],
            s=size,
            color=color,
            zorder=5,
        )


def _draw_names(ax: plt.Axes, vertices: list[RadarVertex]) -> None:
    """Имена снаружи периметра. Доминантные — красные и жирные."""
    for item in vertices:
        angle = float(np.arctan2(item.anchor_y, item.anchor_x))
        align, valign, rotation = _ray_text_style(angle)
        place = 1.0 + RADAR_NAME_GAP
        if item.dominant:
            color = _RADAR_DOMINANT
            weight = "bold"
        else:
            color = "0.2"
            weight = "normal"
        ax.text(
            item.anchor_x * place,
            item.anchor_y * place,
            item.name,
            ha=align,
            va=valign,
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=8 if item.dominant else 7,
            color=color,
            fontweight=weight,
            clip_on=False,
        )


def _ray_text_style(angle: float) -> tuple[str, str, float]:
    """Имя растёт от квадрата наружу и не переворачивается.

    Returns:
        Выравнивание по горизонтали, по вертикали и поворот.
    """
    degrees = float(np.degrees(angle)) % 360.0
    # Между 90° и 270° прямой поворот ставит буквы вверх ногами.
    if 90.0 < degrees < 270.0:
        return "right", "center", degrees - 180.0
    return "left", "center", degrees


def _radar_legend(has_negative: bool) -> list[Line2D]:
    """Образцы трёх контуров. Отрицательный ДОЭВС — отдельная строка."""
    handles = [
        Line2D(
            [0],
            [0],
            color=_RADAR_PDO,
            lw=RADAR_PDO_WIDTH,
            label="ПДО",
        ),
        Line2D(
            [0],
            [0],
            color=_RADAR_DOEVS,
            lw=RADAR_DOEVS_WIDTH,
            label="ДОЭВС",
        ),
    ]
    if has_negative:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_RADAR_DOEVS_NEG,
                lw=RADAR_DOEVS_WIDTH,
                label="ДОЭВС < 0",
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color=_RADAR_DOEVD,
            lw=RADAR_DOEVD_WIDTH,
            linestyle="--",
            label="ДОЭВД",
        )
    )
    return handles


def _save_radar(fig: plt.Figure, path: Path, vector: bool) -> Path:
    """Пишет png и, если нужно, svg и pdf. Закрывает фигуру."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = fig.get_size_inches()
    longest = max(float(width), float(height), 1.0)
    if longest > _MAX_INCHES:
        scale = _MAX_INCHES / longest
        fig.set_size_inches(width * scale, height * scale)
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.15)
    if vector:
        fig.savefig(
            path.with_suffix(".svg"),
            bbox_inches="tight",
            pad_inches=0.15,
        )
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.15,
        )
    plt.close(fig)
    return path
