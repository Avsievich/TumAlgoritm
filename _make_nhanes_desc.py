"""Собирает описание столбцов подготовленного файла NHANES."""

import re
import shutil
from pathlib import Path

import pandas as pd

SRC_CSV = Path(r"D:\Projects\NHANES\data\out\nhanes_2017_2018_for_analysis.csv")
SRC_DESC = Path(
    r"D:\Projects\NHANES\data\out\opisanie_nhanes_2017_2018_laboratory.md"
)
REF = Path(r"D:\Projects\TumAlgoritm\reference")
DST_CSV = REF / "nhanes_2017_2018_for_analysis.csv"
DST_MD = REF / "opisanie_nhanes_2017_2018_for_analysis.md"

ROLE = {
    "SEQN": "исключить",
    "LBXGH": "Y, выход",
}


def parse_opisanie(text: str) -> dict[str, dict[str, str]]:
    """Читает подписи столбцов из полного описания лаборатории."""
    desc: dict[str, dict[str, str]] = {}
    section = ""
    for line in text.splitlines():
        if line.startswith("## ") and " — " in line:
            section = line[3:].strip()
            continue
        if not line.startswith("| "):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 4:
            continue
        name = parts[0]
        if not re.fullmatch(r"[A-Z0-9_]+", name):
            continue
        desc[name] = {"text": parts[2], "section": section}
    return desc


def clean_text(text: str) -> str:
    """Укорачивает подпись CDC для таблицы ролей."""
    text = text.strip()
    prefix = "Лабораторный результат. Подпись CDC: "
    if text.startswith(prefix):
        text = text[len(prefix):]
    cut = text.find(" В файле ")
    if cut > 0:
        text = text[:cut].rstrip(".")
    return text.replace("|", "/")


def main() -> None:
    """Пишет markdown и переносит CSV с описанием в reference."""
    frame = pd.read_csv(SRC_CSV)
    desc = parse_opisanie(SRC_DESC.read_text(encoding="utf-8"))
    lines = [
        "# Описание файла nhanes_2017_2018_for_analysis.csv",
        "",
        "Подготовленная таблица для расчёта по методике Тюмикова "
        "в программе TumAlgoritm. Источник — лабораторная часть "
        "NHANES 2017–2018 (`nhanes_2017_2018_laboratory.csv`).",
        "",
        f"В файле **{len(frame)}** строк и "
        f"**{len(frame.columns)}** столбцов:",
        "",
        "- `SEQN` — номер участника, роль «исключить»;",
        "- **75** числовых лабораторных факторов, роль Z;",
        "- `LBXGH` — гликированный гемоглобин HbA1c, выход Y.",
        "",
        "Пропусков нет: оставлены только люди и столбцы,",
        "у которых все выбранные анализы заполнены.",
        "",
        "## Как открыть в программе",
        "",
        "1. Открыть этот CSV на экране «Данные».",
        "2. Для `LBXGH` поставить роль Y.",
        "3. `SEQN` оставить «исключить» (программа так и предлагает).",
        "4. Остальные числовые столбцы — факторы Z.",
        "",
        "## Что убрано из исходной лаборатории",
        "",
        "- коды комментариев лаборатории и веса подвыборок;",
        "- опрос о голодании, сессия осмотра, объём мочи,",
        "  тест на беременность;",
        "- вторая единица того же показателя",
        "  (например ммоль/л рядом с мг/дл);",
        "- расчётные величины, уже собранные из других столбцов",
        "  (отношение альбумин/креатинин, абсолютные числа",
        "  лейкоцитов из процентов, насыщение трансферрина);",
        "- анализы редких подвыборок: глюкоза и инсулин натощак,",
        "  токсиканты и surplus-сыворотка — из‑за них таблица",
        "  снова стала бы дырявой.",
        "",
        "## Описание столбцов",
        "",
        "| Столбец | Роль | Раздел CDC | Описание |",
        "| --- | --- | --- | --- |",
    ]
    for name in frame.columns:
        info = desc[name]
        role = ROLE.get(name, "Z, фактор")
        section = info["section"]
        if " — " in section:
            section = section.split(" — ", 1)[0]
        text = clean_text(info["text"])
        lines.append(f"| {name} | {role} | {section} | {text} |")
    lines.extend(
        [
            "",
            "## Источник",
            "",
            "Цикл NHANES 2017–2018, лабораторные файлы CDC.",
            "Полное описание исходных столбцов — в проекте NHANES,",
            "файл `opisanie_nhanes_2017_2018_laboratory.md`.",
            "",
        ]
    )
    REF.mkdir(parents=True, exist_ok=True)
    DST_MD.write_text("\n".join(lines), encoding="utf-8")
    shutil.move(str(SRC_CSV), str(DST_CSV))
    print(f"csv -> {DST_CSV}")
    print(f"md  -> {DST_MD}")
    print(f"rows={len(frame)} cols={len(frame.columns)}")


if __name__ == "__main__":
    main()
