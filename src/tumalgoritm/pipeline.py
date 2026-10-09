"""Сквозной расчёт методики без записи файлов.

Порядок шагов:
    1. Нормировка и квантование.
    2. Прямые дисперсионные отношения и гетероскедастичность.
    3. Выбор доминантных факторов по Q(m) и ограничение размера набора.
    4. Множественное и остаточное отношения по совместной группировке.
    5. Базис эффектов до порядка q, матрица resM и FDO.
    6. Парные VdDO, ДОЭВС и частные отношения.
    7. Взаимная и условная информация по кодам.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tumalgoritm.constants import (
    DEFAULT_EFFECT_ORDER,
    DEFAULT_ETA0,
    DEFAULT_MAX_DOMINANT,
    DEFAULT_MIN_DOMINANT,
    DEFAULT_NADC_X,
    DEFAULT_NADC_Y,
    MIN_OBS_PER_CELL,
    MIN_SAMPLE_SIZE,
)
from tumalgoritm.data_io import Dataset, build_dataset, load_dataset
from tumalgoritm.dispersion import (
    FactorDispersion,
    Moments,
    PairLink,
    all_factor_stats,
    bias_correct_eta2,
    dispersion_ratio,
    multiple_correlation,
    pair_link,
    selection_scores,
)
from tumalgoritm.dominant import (
    DominantSelection,
    limit_dominant,
    select_dominant,
)
from tumalgoritm.effects import EffectMatrix, build_effect_matrix, iter_subsets
from tumalgoritm.information import InformationResult, information_report
from tumalgoritm.prepare import PreparedData, prepare_signals

# Доли хода. Это порядок шагов, а не оценка оставшихся секунд.
STEP_PREPARE = 10
STEP_PDO = 25
STEP_DOMINANT = 35
STEP_MDO = 45
STEP_EFFECTS = 60
STEP_PAIRS = 75
STEP_INFORMATION = 90
STEP_DONE = 100

# Сообщение шага: имя и процент от 0 до 100.
ProgressFn = Callable[[str, int], None]


@dataclass
class AnalysisResult:
    """Полный результат одного прогона методики."""

    y_name: str
    column_names: list[str]
    roles: list[str]
    n_obs: int
    n_dropped: int
    prepared: PreparedData
    moments: Moments
    factors: list[FactorDispersion]
    selection: DominantSelection
    dominant_idx: list[int]
    mdo_raw: float
    mdo_adj: float
    mdo: float
    odo: float
    k_joint: int
    effects: EffectMatrix
    pairs: list[PairLink]
    information: InformationResult
    correlation_r: float
    eta0: float
    effect_order: int
    warnings: list[str] = field(default_factory=list)

    @property
    def pdo_raw(self) -> np.ndarray:
        """Сырые ПДО в порядке столбцов X."""
        return np.array([item.pdo_raw for item in self.factors], dtype=float)

    @property
    def pdo_adj(self) -> np.ndarray:
        """Скорректированные ПДО. Могут содержать NaN."""
        return np.array([item.pdo_adj for item in self.factors], dtype=float)


def analyze_dataset(
    dataset: Dataset,
    nadc_x: int = DEFAULT_NADC_X,
    nadc_y: int = DEFAULT_NADC_Y,
    eta0: float = DEFAULT_ETA0,
    effect_order: int = DEFAULT_EFFECT_ORDER,
    min_dominant: int = DEFAULT_MIN_DOMINANT,
    max_dominant: int = DEFAULT_MAX_DOMINANT,
    on_progress: ProgressFn | None = None,
) -> AnalysisResult:
    """Считает дисперсионную и информационную ветви по готовому Dataset.

    Args:
        dataset: факторы X, выход Y и роли.
        nadc_x: разрядность факторов, 0 — подобрать.
        nadc_y: разрядность выхода, 0 — подобрать.
        eta0: порог достаточности нижней оценки МДО.
        effect_order: максимальный порядок взаимодействий q.
        min_dominant: минимум факторов в доминантном наборе.
        max_dominant: максимум факторов в доминантном наборе.
        on_progress: необязательное сообщение о шаге и проценте.
            Окну не нужно, чтобы пакет знал про интерфейс.

    Returns:
        AnalysisResult со всеми показателями и предупреждениями.

    Raises:
        ValueError: слишком мало наблюдений или неверные параметры.
    """
    if dataset.x.shape[0] < MIN_SAMPLE_SIZE:
        raise ValueError(
            f"Нужно не меньше {MIN_SAMPLE_SIZE} наблюдений "
            "для выборочной дисперсии"
        )

    _step(on_progress, "Подготовка данных", STEP_PREPARE)
    prepared = prepare_signals(dataset.x, dataset.y, nadc_x, nadc_y)
    _step(on_progress, "Прямые отношения", STEP_PDO)
    moments, factors = all_factor_stats(prepared.y_norm, prepared.x_codes)
    scores = selection_scores(
        np.array([item.pdo_raw for item in factors]),
        np.array([item.pdo_adj for item in factors]),
    )
    _step(on_progress, "Доминантные факторы", STEP_DOMINANT)
    selection = select_dominant(scores, eta0=eta0)
    dominant_idx = limit_dominant(
        selection,
        n_factors=dataset.x.shape[1],
        min_dominant=min_dominant,
        max_dominant=max_dominant,
    )

    # МДО — по совместным ячейкам доминантных факторов, не по сумме resM.
    _step(on_progress, "Множественное отношение", STEP_MDO)
    joint_codes = prepared.x_codes[:, dominant_idx]
    mdo_raw, k_joint = dispersion_ratio(
        prepared.y_norm, joint_codes, moments
    )
    mdo_adj = bias_correct_eta2(mdo_raw, moments.n_obs, k_joint)
    warnings = _collect_warnings(
        dataset=dataset,
        factors=factors,
        n_obs=moments.n_obs,
        k_joint=k_joint,
        mdo_adj=mdo_adj,
        selection=selection,
        eta0=eta0,
    )
    if np.isnan(mdo_adj):
        mdo_final = float(np.clip(mdo_raw, 0.0, 1.0))
    else:
        mdo_final = float(mdo_adj)
    odo = float(1.0 - mdo_final)

    _step(on_progress, "Эффекты взаимодействия", STEP_EFFECTS)
    subsets = iter_subsets(dominant_idx, effect_order)
    effects = build_effect_matrix(
        prepared.y_norm,
        prepared.x_codes,
        subsets,
        dataset.x_names,
        moments,
    )
    _step(on_progress, "Парные связи", STEP_PAIRS)
    pairs = _all_pairs(prepared, moments, factors)
    _step(on_progress, "Информация", STEP_INFORMATION)
    info = information_report(
        prepared.x_codes,
        prepared.y_codes,
        dominant_idx,
        dataset.x_names,
        effect_order,
    )
    correlation = multiple_correlation(
        prepared.x_norm[:, dominant_idx],
        prepared.y_norm,
    )
    _step(on_progress, "Готово", STEP_DONE)

    return AnalysisResult(
        y_name=dataset.y_name,
        column_names=list(dataset.x_names),
        roles=list(dataset.x_roles),
        n_obs=moments.n_obs,
        n_dropped=dataset.n_dropped,
        prepared=prepared,
        moments=moments,
        factors=factors,
        selection=selection,
        dominant_idx=dominant_idx,
        mdo_raw=mdo_raw,
        mdo_adj=mdo_adj,
        mdo=mdo_final,
        odo=odo,
        k_joint=k_joint,
        effects=effects,
        pairs=pairs,
        information=info,
        correlation_r=correlation,
        eta0=eta0,
        effect_order=effect_order,
        warnings=warnings,
    )


def run_frame(
    frame,
    y_name: str,
    u_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
    **kwargs,
) -> AnalysisResult:
    """Размечает роли в таблице и запускает расчёт.

    Args:
        frame: таблица pandas.
        y_name: имя столбца выхода.
        u_names: управляемые входы.
        exclude_names: исключаемые столбцы.
        **kwargs: параметры analyze_dataset.

    Returns:
        AnalysisResult.
    """
    dataset = build_dataset(
        frame,
        y_name,
        u_names=u_names,
        exclude_names=exclude_names,
    )
    return analyze_dataset(dataset, **kwargs)


def run_analysis(
    path: str | Path,
    y_name: str,
    u_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
    **kwargs,
) -> AnalysisResult:
    """Загружает файл и выполняет полный расчёт.

    Args:
        path: CSV, TXT или Excel.
        y_name: имя столбца выхода.
        u_names: управляемые входы.
        exclude_names: исключаемые столбцы.
        **kwargs: параметры analyze_dataset.

    Returns:
        AnalysisResult.
    """
    dataset = load_dataset(
        path,
        y_name,
        u_names=u_names,
        exclude_names=exclude_names,
    )
    return analyze_dataset(dataset, **kwargs)


def _step(callback: ProgressFn | None, name: str, percent: int) -> None:
    """Передаёт шаг в окно, если обратный вызов задан."""
    if callback is not None:
        callback(name, percent)


def _all_pairs(
    prepared: PreparedData,
    moments: Moments,
    factors: list[FactorDispersion],
) -> list[PairLink]:
    """Считает парные связи для всех сочетаний факторов."""
    n_factors = prepared.x_codes.shape[1]
    links: list[PairLink] = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            doevd, vs_i, vs_j, chdo_j, chdo_i = pair_link(
                prepared.y_norm,
                prepared.x_codes[:, i],
                prepared.x_codes[:, j],
                moments,
                factors[i].pdo_raw,
                factors[j].pdo_raw,
            )
            links.append(
                PairLink(
                    i=i,
                    j=j,
                    doevd=doevd,
                    doevs_i=vs_i,
                    doevs_j=vs_j,
                    chdo_j_given_i=chdo_j,
                    chdo_i_given_j=chdo_i,
                )
            )
    return links


def _collect_warnings(
    dataset: Dataset,
    factors: list[FactorDispersion],
    n_obs: int,
    k_joint: int,
    mdo_adj: float,
    selection: DominantSelection,
    eta0: float,
) -> list[str]:
    """Собирает методические предупреждения, не прерывая расчёт."""
    warnings: list[str] = []
    if dataset.n_dropped:
        warnings.append(
            f"Отброшено строк с пропусками: {dataset.n_dropped}"
        )

    for name, factor in zip(dataset.x_names, factors):
        if factor.n_groups > 0 and n_obs / factor.n_groups < MIN_OBS_PER_CELL:
            warnings.append(
                f"«{name}»: в среднем меньше {MIN_OBS_PER_CELL} "
                "наблюдений на ячейку"
            )

    if k_joint > 0 and n_obs / k_joint < MIN_OBS_PER_CELL:
        warnings.append(
            "Совместное разбиение доминантных факторов: "
            f"в среднем меньше {MIN_OBS_PER_CELL} наблюдений на ячейку"
        )
    if np.isnan(mdo_adj):
        warnings.append(
            "МДО вырождено: наблюдений не больше, чем занятых ячеек. "
            "В отчёт взято сырое МДО, обрезанное к отрезку [0, 1]"
        )
    if eta0 > 0.0 and not selection.sufficient:
        warnings.append(
            f"Нижняя оценка МДО S_m = {selection.s_m:.4f} "
            f"меньше порога eta0 = {eta0}"
        )
    return warnings
