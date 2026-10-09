# Описание файла nhanes_2017_2018_for_analysis.csv

Подготовленная таблица для расчёта по методике Тюмикова в программе TumAlgoritm. Источник — лабораторная часть NHANES 2017–2018 и самоотчёт диагноза диабета из `DIQ_J`.

В файле **3564** строк и **78** столбцов:

- `SEQN` — номер участника, роль «исключить»;
- `status` — диагноз диабета (0/1), обычно выход Y;
- **75** числовых лабораторных факторов, роль Z;
- `LBXGH` — гликированный гемоглобин HbA1c.

Пропусков нет: оставлены только люди и столбцы,
у которых все выбранные анализы заполнены, а ответ
о диабете однозначный (да или нет).

Диагноз взят из `DIQ010`: врач когда-либо говорил,
что есть диабет. Код 1 стал `status = 1`, код 2 —
`status = 0`. Пограничный диабет (код 3) и ответ
«не знает» (код 9) из таблицы убраны: осталось
502 человека с диабетом и 3062 без него.

## Как открыть в программе

1. Открыть этот CSV на экране «Данные».
2. Для `status` поставить роль Y (программа так и предлагает).
3. `SEQN` оставить «исключить».
4. Остальные числовые столбцы — факторы Z.
5. Если выход — диагноз, столбец `LBXGH` лучше исключить:
   HbA1c почти определяет диабет, и связь будет ложной.
   Если выход — сам HbA1c, поставьте Y на `LBXGH`,
   а `status` исключите или оставьте фактором.

## Что убрано из исходной лаборатории

- коды комментариев лаборатории и веса подвыборок;
- опрос о голодании, сессия осмотра, объём мочи,
  тест на беременность;
- вторая единица того же показателя
  (например ммоль/л рядом с мг/дл);
- расчётные величины, уже собранные из других столбцов
  (отношение альбумин/креатинин, абсолютные числа
  лейкоцитов из процентов, насыщение трансферрина);
- анализы редких подвыборок: глюкоза и инсулин натощак,
  токсиканты и surplus-сыворотка — из‑за них таблица
  снова стала бы дырявой.

## Описание столбцов

| Столбец | Роль | Раздел CDC | Описание |
| --- | --- | --- | --- |
| SEQN | исключить | SEQN | Номер участника NHANES. Ключ склейки, в таблице уникален. |
| status | Y, выход | DIQ_J | Диагноз диабета: 1 — врач говорил о диабете, 0 — нет. Из DIQ010, без пограничных и «не знает». |
| URXUMA | Z, фактор | ALB_CR_J | Albumin, urine (ug/mL). |
| URXUCR | Z, фактор | ALB_CR_J | Creatinine, urine (mg/dL). |
| LBDHDD | Z, фактор | HDL_J | Холестерин ЛПВП, прямое измерение, мг/дл. |
| LBXTC | Z, фактор | TCHOL_J | Общий холестерин, мг/дл. |
| LBXWBCSI | Z, фактор | CBC_J | White blood cell count (1000 cells/uL). |
| LBXLYPCT | Z, фактор | CBC_J | Lymphocyte percent (%). |
| LBXMOPCT | Z, фактор | CBC_J | Monocyte percent (%). |
| LBXNEPCT | Z, фактор | CBC_J | Segmented neutrophils percent (%). |
| LBXEOPCT | Z, фактор | CBC_J | Eosinophils percent (%). |
| LBXBAPCT | Z, фактор | CBC_J | Basophils percent (%). |
| LBXRBCSI | Z, фактор | CBC_J | Red blood cell count (million cells/uL). |
| LBXHGB | Z, фактор | CBC_J | Hemoglobin (g/dL). |
| CBC_J__LBXHCT | Z, фактор | CBC_J | Гематокрит, процент |
| LBXMCVSI | Z, фактор | CBC_J | Mean cell volume (fL). |
| LBXMCHSI | Z, фактор | CBC_J | Mean cell hemoglobin (pg). |
| LBXMC | Z, фактор | CBC_J | Mean Cell Hgb Conc. (g/dL). |
| LBXRDW | Z, фактор | CBC_J | Red cell distribution width (%). |
| LBXPLTSI | Z, фактор | CBC_J | Platelet count (1000 cells/uL). |
| LBXMPSI | Z, фактор | CBC_J | Mean platelet volume (fL). |
| LBXNRBC | Z, фактор | CBC_J | Nucleated red blood cells (/100 WBC). |
| LBXCOT | Z, фактор | COT_J | Cotinine, Serum (ng/mL). |
| COT_J__LBXHCT | Z, фактор | COT_J | Гидроксикотинин в сыворотке, нг/мл |
| LBXFER | Z, фактор | FERTIN_J | Ferritin (ng/mL). |
| LBXHSCRP | Z, фактор | HSCRP_J | HS C-Reactive Protein (mg/L). |
| LBXUIB | Z, фактор | FETIB_J | UIBC, Serum (ug/dL). |
| LBDTIBSI | Z, фактор | FETIB_J | Tot Iron Binding Capacity TIBC (umol/L). |
| LBXBPB | Z, фактор | PBCD_J | Blood lead (ug/dL). |
| LBXBCD | Z, фактор | PBCD_J | Blood cadmium (ug/L). |
| LBXTHG | Z, фактор | PBCD_J | Blood mercury, total (ug/L). |
| LBXBSE | Z, фактор | PBCD_J | Blood selenium (ug/L). |
| LBXBMN | Z, фактор | PBCD_J | Blood manganese (ug/L). |
| LBXIHG | Z, фактор | IHGEM_J | Mercury, inorganic (ug/L). |
| LBXBGE | Z, фактор | IHGEM_J | Mercury, ethyl (ug/L). |
| LBXBGM | Z, фактор | IHGEM_J | Mercury, methyl (ug/L). |
| LBXSATSI | Z, фактор | BIOPRO_J | Alanine Aminotransferase (ALT) (U/L). |
| LBXSAL | Z, фактор | BIOPRO_J | Albumin, refrigerated serum (g/dL). |
| LBXSAPSI | Z, фактор | BIOPRO_J | Alkaline Phosphatase (ALP) (IU/L). |
| LBXSASSI | Z, фактор | BIOPRO_J | Aspartate Aminotransferase (AST) (U/L). |
| LBXSC3SI | Z, фактор | BIOPRO_J | Bicarbonate (mmol/L). |
| LBXSBU | Z, фактор | BIOPRO_J | Blood Urea Nitrogen (mg/dL). |
| LBXSCLSI | Z, фактор | BIOPRO_J | Chloride (mmol/L). |
| LBXSCK | Z, фактор | BIOPRO_J | Creatine Phosphokinase (CPK) (IU/L). |
| LBXSCR | Z, фактор | BIOPRO_J | Creatinine, refrigerated serum (mg/dL). |
| LBDSGBSI | Z, фактор | BIOPRO_J | Globulin (g/L). |
| LBXSGL | Z, фактор | BIOPRO_J | Glucose, refrigerated serum (mg/dL). |
| LBXSGTSI | Z, фактор | BIOPRO_J | Gamma Glutamyl Transferase (GGT) (IU/L). |
| LBXSIR | Z, фактор | BIOPRO_J | Iron, refrigerated serum (ug/dL). |
| LBXSOSSI | Z, фактор | BIOPRO_J | Osmolality (mmol/Kg). |
| LBXSPH | Z, фактор | BIOPRO_J | Phosphorus (mg/dL). |
| LBXSKSI | Z, фактор | BIOPRO_J | Potassium (mmol/L). |
| LBXSNASI | Z, фактор | BIOPRO_J | Sodium (mmol/L). |
| LBXSTB | Z, фактор | BIOPRO_J | Total Bilirubin (mg/dL). |
| LBXSCA | Z, фактор | BIOPRO_J | Total Calcium (mg/dL). |
| LBXSTP | Z, фактор | BIOPRO_J | Total Protein (g/dL). |
| LBXSTR | Z, фактор | BIOPRO_J | Triglycerides, refrig serum (mg/dL). |
| LBXSUA | Z, фактор | BIOPRO_J | Uric acid (mg/dL). |
| LBXTFR | Z, фактор | TFR_J | Transferrin receptor (mg/L). |
| LBXALC | Z, фактор | VITAEC_J | alpha-carotene (ug/dL). |
| LBXARY | Z, фактор | VITAEC_J | alpha-crypotoxanthin (ug/dL). |
| LBXBEC | Z, фактор | VITAEC_J | trans-beta-carotene (ug/dL). |
| LBXCBC | Z, фактор | VITAEC_J | cis-beta-carotene (ug/dL). |
| LBXCRY | Z, фактор | VITAEC_J | beta-cryptoxanthin (ug/dL). |
| LBXGTC | Z, фактор | VITAEC_J | gamma-tocopherol (ug/dL). |
| LBXLUZ | Z, фактор | VITAEC_J | Lutein and zeaxanthin (ug/dL). |
| LBXLYC | Z, фактор | VITAEC_J | trans-lycopene (ug/dL). |
| LBXRPL | Z, фактор | VITAEC_J | Retinyl palmitate (ug/dL). |
| LBXRST | Z, фактор | VITAEC_J | Retinyl stearate (ug/dL). |
| LBXLCC | Z, фактор | VITAEC_J | Total Lycopene (ug/dL). |
| LBXVIA | Z, фактор | VITAEC_J | Retinol (ug/dL). |
| LBXVIE | Z, фактор | VITAEC_J | alpha-tocopherol (ug/dL). |
| LBXVIC | Z, фактор | VIC_J | Vitamin C (mg/dL). |
| LBXVIDMS | Z, фактор | VID_J | 25OHD2+25OHD3 (nmol/L). |
| LBXVD2MS | Z, фактор | VID_J | 25OHD2 (nmol/L). |
| LBXVD3MS | Z, фактор | VID_J | 25OHD3 (nmol/L). |
| LBXVE3MS | Z, фактор | VID_J | epi-25OHD3 (nmol/L). |
| LBXGH | Z или исключить | GHB_J | Гликированный гемоглобин HbA1c, процент. При выходе status лучше исключить. |

## Источник

Цикл NHANES 2017–2018: лаборатория CDC и опросник `DIQ_J`.
Полное описание исходных столбцов — в проекте NHANES,
файлы `opisanie_nhanes_2017_2018_laboratory.md` и
`opisanie_nhanes_2017_2018_lab_diagnoses.md`.
