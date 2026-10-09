"""Сквозной отчёт, командная строка и контрольный набор Parkinson."""

from pathlib import Path

import pandas as pd
import pytest

from arm.cli import main
from tumalgoritm.pipeline import run_analysis
from arm.report import (
    SHEET_CHDO,
    SHEET_DOEVS,
    SHEET_DOMINANT,
    SHEET_INFORMATION,
    SHEET_PDO,
    SHEET_RES_M,
    SHEET_SUMMARY,
    write_report,
)

# Файл UCI Parkinson кладётся сюда перед прогоном тестов.
PARKINSON_PATH = (
    Path(__file__).resolve().parents[1] / "reference" / "parkinsons.csv"
)

EXPECTED_SHEETS = {
    SHEET_SUMMARY,
    SHEET_PDO,
    SHEET_DOMINANT,
    SHEET_RES_M,
    SHEET_DOEVS,
    SHEET_CHDO,
    SHEET_INFORMATION,
}


def test_report_contains_separated_metrics(tmp_path: Path):
    """Excel хранит ПДО, VdDO, МДО, ОДО, FDO и условную информацию."""
    frame_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "x1": [0, 0, 0, 0, 1, 1, 1, 1],
            "x2": [0, 1, 0, 1, 0, 1, 0, 1],
            "y": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    ).to_csv(frame_path, index=False)
    output = tmp_path / "report.xlsx"
    main(
        [
            str(frame_path),
            "--y",
            "y",
            "--nadc-x",
            "1",
            "--nadc-y",
            "1",
            "--q",
            "2",
            "--min-dominant",
            "2",
            "--max-dominant",
            "2",
            "--output",
            str(output),
        ]
    )
    book = pd.ExcelFile(output)
    assert EXPECTED_SHEETS <= set(book.sheet_names)
    summary = pd.read_excel(output, sheet_name=SHEET_SUMMARY)
    values = dict(zip(summary["Показатель"], summary["Значение"]))
    assert float(values["MDO"]) + float(values["ODO"]) == pytest.approx(1.0)
    assert "FDO" in values
    assert any(str(name).startswith("VdDO") for name in values)
    assert any(str(name).startswith("I(") for name in values)


@pytest.mark.skipif(
    not PARKINSON_PATH.is_file(),
    reason="Нет reference/parkinsons.csv",
)
def test_parkinson_run(tmp_path: Path):
    """Набор Parkinson считается до конца и даёт допустимые МДО и ОДО."""
    result = run_analysis(
        PARKINSON_PATH,
        "status",
        nadc_x=4,
        nadc_y=6,
        effect_order=2,
    )
    output = write_report(result, tmp_path / "parkinson.xlsx")
    book = pd.ExcelFile(output)
    assert EXPECTED_SHEETS <= set(book.sheet_names)
    assert result.n_obs == 195
    assert len(result.column_names) == 22
    assert 0.0 <= result.mdo <= 1.0
    assert result.mdo + result.odo == pytest.approx(1.0)
    assert "name" not in result.column_names
