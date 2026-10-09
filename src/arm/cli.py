"""Командная строка автоматизированного рабочего места.

Модуль только принимает параметры и показывает результат.
Сам расчёт выполняет пакет tumalgoritm.

Пример:
    python -m arm data.csv --y status --nadc-x 4 --nadc-y 6 --q 2
"""

import argparse
import sys
from pathlib import Path

from tumalgoritm.constants import (
    DEFAULT_EFFECT_ORDER,
    DEFAULT_ETA0,
    DEFAULT_MAX_DOMINANT,
    DEFAULT_MIN_DOMINANT,
    DEFAULT_NADC_X,
    DEFAULT_NADC_Y,
)
from arm.charts import save_charts
from arm.report import write_report
from tumalgoritm.pipeline import run_analysis


def build_parser() -> argparse.ArgumentParser:
    """Собирает разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="arm",
        description="Дисперсионно-информационный анализ по методике Тюмикова",
    )
    parser.add_argument("filepath", help="CSV, TXT или Excel с заголовком")
    parser.add_argument("--y", required=True, help="Имя столбца выхода")
    parser.add_argument(
        "--u",
        nargs="*",
        default=[],
        help=(
            "Управляемые входы (роль U). "
            "Остальные числовые столбцы — роль Z"
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Столбцы, которые нужно исключить из расчёта",
    )
    parser.add_argument(
        "--nadc-x",
        type=int,
        default=DEFAULT_NADC_X,
        help="Разрядность факторов. 0 — подобрать по объёму выборки",
    )
    parser.add_argument(
        "--nadc-y",
        type=int,
        default=DEFAULT_NADC_Y,
        help="Разрядность выхода. 0 — подобрать по объёму выборки",
    )
    parser.add_argument(
        "--eta0",
        type=float,
        default=DEFAULT_ETA0,
        help="Порог достаточности нижней оценки МДО",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=DEFAULT_EFFECT_ORDER,
        help="Максимальный порядок взаимодействий",
    )
    parser.add_argument(
        "--min-dominant",
        type=int,
        default=DEFAULT_MIN_DOMINANT,
        help="Минимальный размер доминантного набора",
    )
    parser.add_argument(
        "--max-dominant",
        type=int,
        default=DEFAULT_MAX_DOMINANT,
        help="Максимальный размер доминантного набора",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Путь к xlsx. По умолчанию <имя_файла>_tumalgoritm.xlsx",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Запускает расчёт и печатает краткие итоги.

    Без аргументов открывается окно разметки ролей.

    Args:
        argv: аргументы без имени программы. None — взять из командной строки.
    """
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0:
        from arm.gui import launch

        launch()
        return
    args = build_parser().parse_args(argv)
    source = Path(args.filepath)
    output = Path(args.output) if args.output else _default_output(source)
    result = run_analysis(
        source,
        args.y,
        u_names=args.u,
        exclude_names=args.exclude,
        nadc_x=args.nadc_x,
        nadc_y=args.nadc_y,
        eta0=args.eta0,
        effect_order=args.q,
        min_dominant=args.min_dominant,
        max_dominant=args.max_dominant,
    )
    write_report(result, output)
    plot_dir = output.with_name(output.stem + "_plots")
    save_charts(result, plot_dir)
    _print_summary(result, output, plot_dir)


def _default_output(source: Path) -> Path:
    """Кладёт отчёт рядом с запуском, не затирая исходный файл."""
    return Path.cwd() / f"{source.stem}_tumalgoritm.xlsx"


def _print_summary(result, output: Path, plot_dir: Path) -> None:
    """Печатает доминантные факторы и главные скаляры."""
    names = [result.column_names[index] for index in result.dominant_idx]
    print(f"Наблюдений: {result.n_obs}")
    print(f"Доминантные факторы: {', '.join(names)}")
    print(f"МДО = {result.mdo:.6f}    ОДО = {result.odo:.6f}")
    print(f"FDO = {result.effects.fdo:.6f}    R = {result.correlation_r:.6f}")
    print(
        f"I(X_D; Y) = {result.information.joint_dominant:.6f} бит    "
        f"H(Y) = {result.information.h_y:.6f} бит"
    )
    for text in result.warnings:
        print(f"Предупреждение: {text}")
    print(f"Отчёт: {output}")
    print(f"Рисунки: {plot_dir}")


if __name__ == "__main__":
    main()
