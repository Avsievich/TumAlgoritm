"""Окно автоматизированного рабочего места.

Экраны: данные, параметры, ход расчёта, числа и графики, справка.
Формулы остаются в пакете tumalgoritm.
"""

import ctypes
import io
import os
import queue
import tempfile
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import pandas as pd
from PIL import Image

from arm.charts import (
    FILE_HEATMAP,
    FILE_PDO_HEATMAP,
    FILE_SQUARE_RADAR,
    save_charts,
)
from arm.export_run import export_run
from arm.help_guide import GUIDE_TITLE, guide_for, guide_names
from arm.help_text import BY_NAME, format_help, help_for
from arm.report import SHEET_RES_M, SHEET_SUMMARY, build_frames
from arm.session import (
    apply_range_filter,
    column_overview,
    drop_columns,
    drop_rows,
    load_working_table,
    save_working_csv,
    suggest_role,
)
from arm.y_scan import (
    SHEET_LINKS,
    SHEET_RANK,
    YScanResult,
    build_scan_frames,
    export_y_scan,
    scan_outputs,
    successful_y_names,
    summary_line,
)
from tumalgoritm.constants import (
    DEFAULT_EFFECT_ORDER,
    DEFAULT_ETA0,
    DEFAULT_MAX_DOMINANT,
    DEFAULT_MIN_DOMINANT,
    DEFAULT_NADC_X,
    DEFAULT_NADC_Y,
)
from tumalgoritm.data_io import (
    ROLE_EXCLUDE,
    ROLE_U,
    ROLE_Y,
    build_dataset,
)
from tumalgoritm.pipeline import AnalysisResult, run_frame

# Подписи ролей. В расчёт уходят короткие коды.
ROLE_LABELS = {
    "Z": "Z, неуправляемый вход",
    "U": "U, управляемый вход",
    ROLE_Y: "Y, выход",
    ROLE_EXCLUDE: "исключить",
}
LABEL_TO_ROLE = {label: role for role, label in ROLE_LABELS.items()}

SCREENS = ("Данные", "Параметры", "Расчёт", "Результаты", "Справка")

# Карточки на экране результатов и короткие подписи.
CARD_NAMES = ("MDO", "ODO", "FDO", "R", "H(Y)", "I(X_D; Y)")

# Подписи рисунков. Ключ — имя файла.
CHART_TITLES = {
    "pdo_bars.png": "Столбцы ПДО",
    "radar_pdo.png": "Радар ПДО",
    FILE_PDO_HEATMAP: "Тепловая матрица ПДО",
    FILE_HEATMAP: "Тепловая карта resM",
    "surface_mean.png": "Условное среднее",
    "pair_bars.png": "Парные взаимодействия",
    "network.png": "Сеть факторов",
    "scatter3d.png": "Облако наблюдений",
    FILE_SQUARE_RADAR: "Радарная диаграмма дисперсионных отношений",
}

# Сколько строк таблицы показывать, чтобы окно не зависало.
PREVIEW_ROWS = 2000

# Размер картинки на экране при масштабе 100%. Тепловая карта крупнее.
# Картинка вписывается в прямоугольник с сохранением сторон:
# квадратный радар иначе растягивался в прямоугольник.
HEATMAP_VIEW = (860, 620)
OTHER_CHART_VIEW = (700, 460)

# Шаг кнопок «−» и «+». Ниже минимума подписи не читаются,
# выше максимума рисунок слишком тяжёлый для окна.
CHART_ZOOM_MIN = 0.5
CHART_ZOOM_MAX = 4.0
CHART_ZOOM_STEP = 0.25
CHART_ZOOM_DEFAULT = 1.0

# CF_DIB: рисунок без файлового заголовка BMP, его понимает Word.
_CF_DIB = 8
_GMEM_MOVEABLE = 0x0002


def _fit_image_size(
    image_size: tuple[int, int],
    box: tuple[int, int],
) -> tuple[int, int]:
    """Вписывает картинку в прямоугольник, не меняя пропорции.

    Args:
        image_size: ширина и высота исходного png.
        box: максимальные ширина и высота на экране.

    Returns:
        Размер для показа. Квадрат остаётся квадратом.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return box
    scale = min(box[0] / width, box[1] / height)
    return (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )


def image_to_dib(image: Image.Image) -> bytes:
    """Готовит рисунок для буфера обмена Windows.

    Args:
        image: картинка любого режима Pillow.

    Returns:
        Блок DIB. Первые 14 байт файла BMP отброшены: это заголовок
        файла, а буфер ждёт описание пикселей.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "BMP")
    return buffer.getvalue()[14:]


def copy_image_to_clipboard(image: Image.Image) -> None:
    """Кладёт рисунок в буфер, чтобы его можно было вставить в документ.

    Args:
        image: исходный png, без экранного уменьшения.

    Raises:
        OSError: буфер занят другой программой или системе не хватило памяти.
    """
    _set_windows_dib(image_to_dib(image))


def _set_windows_dib(payload: bytes) -> None:
    """Записывает DIB в буфер обмена.

    После успешной записи память принадлежит Windows, освобождать её
    здесь нельзя. Если запись не удалась, блок освобождается в этой
    функции.

    Args:
        payload: DIB, как его возвращает image_to_dib.

    Raises:
        OSError: буфер не открылся или не принял рисунок.
    """
    if not payload:
        raise OSError("Пустой рисунок")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_int
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.restype = ctypes.c_void_p
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_int
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = ctypes.c_int
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = ctypes.c_int

    if not user32.OpenClipboard(None):
        raise OSError("Буфер обмена занят")
    handle = None
    handed_off = False
    try:
        if not user32.EmptyClipboard():
            raise OSError("Не удалось очистить буфер обмена")
        handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(payload))
        if not handle:
            raise OSError("Не хватило памяти для рисунка")
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            raise OSError("Не удалось подготовить рисунок")
        ctypes.memmove(pointer, payload, len(payload))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(_CF_DIB, handle):
            raise OSError("Не удалось записать рисунок в буфер")
        handed_off = True
    finally:
        if handle and not handed_off:
            kernel32.GlobalFree(handle)
        user32.CloseClipboard()


class RoleChoice:
    """Роль столбца. Метод get нужен экрану и проверкам."""

    def __init__(self, role: str) -> None:
        self.role = role

    def get(self) -> str:
        """Возвращает подпись роли для списка."""
        return ROLE_LABELS[self.role]


class App(ctk.CTk):
    """Главное окно с боковым меню."""

    def __init__(self) -> None:
        super().__init__()
        self.title("TumAlgoritm")
        self.geometry("1180x760")
        self._source: Path | None = None
        self._frame: pd.DataFrame | None = None
        self._role_boxes: dict[str, RoleChoice] = {}
        self._result: AnalysisResult | None = None
        self._journal: list[str] = []
        self._chart_paths: dict[str, Path] = {}
        self._result_frames: dict[str, pd.DataFrame] = {}
        self._chart_photo: ctk.CTkImage | None = None
        self._chart_image: Image.Image | None = None
        self._chart_zoom = CHART_ZOOM_DEFAULT
        self._chart_base: tuple[int, int] = (0, 0)
        self._chart_box: tuple[int, int] = OTHER_CHART_VIEW
        self._chart_size: tuple[int, int] = (0, 0)
        self._y_scan: YScanResult | None = None
        self._scan_y_filter: str | None = None
        self._calculated_frame: pd.DataFrame | None = None
        self._preview_dir = Path(
            tempfile.mkdtemp(prefix="tumalgoritm_charts_")
        )
        self._export_dir: Path | None = None
        self._running = False
        self._ui_queue: queue.Queue = queue.Queue()
        self._poll_job: str | None = None
        self._current: ctk.CTkFrame | None = None
        self._screens: dict[str, ctk.CTkFrame] = {}
        self._build()
        self._poll_ui()

    def load_file(self, path: str | Path) -> None:
        """Читает файл в рабочую таблицу и показывает её.

        Args:
            path: CSV, TXT или Excel. Файл не перезаписывается.
        """
        table = load_working_table(path)
        self._source = Path(path)
        self._frame = table
        self._path_label.configure(text=str(self._source))
        self._assign_roles(table)
        self._refresh_data()
        self.show_screen("Данные")

    def show_screen(self, name: str) -> None:
        """Показывает один экран. Остальные остаются под ним.

        Экраны не снимаются через pack_forget. После такого снятия
        кнопки перестают нажиматься при следующем открытии экрана.

        Args:
            name: пункт бокового меню.
        """
        page = self._screens[name]
        page.tkraise()
        self._current = page
        if name == "Расчёт":
            self._accept_run_click()

    def show_indicator(self, name: str) -> None:
        """Открывает справку по показателю.

        Args:
            name: имя из справочника, например MDO.
        """
        self.show_screen("Справка")
        self._help_pick.set(name)
        self._set_help_text(format_help(help_for(name)))

    def _build(self) -> None:
        """Собирает меню и пять экранов."""
        body = ctk.CTkFrame(self)
        body.pack(fill="both", expand=True)
        side = ctk.CTkFrame(body, width=170)
        side.pack(side="left", fill="y", padx=8, pady=8)
        side.pack_propagate(False)
        for name in SCREENS:
            button = ctk.CTkButton(
                side,
                text=name,
                command=lambda screen=name: self.show_screen(screen),
            )
            button.pack(fill="x", padx=8, pady=4)
        content = ctk.CTkFrame(body)
        content.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self._screens = {
            "Данные": self._build_data(content),
            "Параметры": self._build_params(content),
            "Расчёт": self._build_run(content),
            "Результаты": self._build_results(content),
            "Справка": self._build_help(content),
        }
        # Все экраны лежат в одной клетке. Сверху виден тот, что поднят.
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        for page in self._screens.values():
            page.grid(row=0, column=0, sticky="nsew")
        self.show_screen("Данные")

    def _build_data(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Экран загрузки, ролей и правки таблицы."""
        page = ctk.CTkFrame(parent)
        top = ctk.CTkFrame(page)
        top.pack(fill="x", padx=8, pady=8)
        ctk.CTkButton(
            top, text="Открыть файл", command=self._choose_file
        ).pack(side="left")
        ctk.CTkButton(
            top, text="Сохранить CSV", command=self._save_csv
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            top, text="Инструкция", command=self._open_guide
        ).pack(side="left", padx=8)
        self._path_label = ctk.CTkLabel(top, text="Файл не выбран")
        self._path_label.pack(side="left", padx=8)
        self._size_label = ctk.CTkLabel(page, text="Строк: 0, столбцов: 0")
        self._size_label.pack(anchor="w", padx=12)

        tools = ctk.CTkFrame(page)
        tools.pack(fill="x", padx=8, pady=4)
        self._column_pick = ctk.CTkComboBox(tools, values=[""], width=180)
        self._column_pick.pack(side="left", padx=4)
        ctk.CTkButton(
            tools, text="Удалить столбец", command=self._delete_column
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            tools, text="Удалить строки", command=self._delete_rows
        ).pack(side="left", padx=4)
        self._filter_low = ctk.CTkEntry(
            tools, width=70, placeholder_text="от"
        )
        self._filter_low.pack(side="left", padx=4)
        self._filter_high = ctk.CTkEntry(
            tools, width=70, placeholder_text="до"
        )
        self._filter_high.pack(side="left", padx=4)
        ctk.CTkButton(
            tools, text="Оставить строки", command=self._apply_filter
        ).pack(side="left", padx=4)

        middle = ctk.CTkFrame(page)
        middle.pack(fill="both", expand=True, padx=8, pady=8)
        self._roles = ctk.CTkScrollableFrame(
            middle, label_text="Столбцы", width=460
        )
        self._roles.pack(side="left", fill="y")
        right = ctk.CTkFrame(middle)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._data_tree = self._make_tree(right)
        return page

    def _build_params(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Разрядность, порог и порядок взаимодействий."""
        page = ctk.CTkFrame(parent)
        grid = ctk.CTkFrame(page)
        grid.pack(anchor="w", padx=16, pady=16)
        self._nadc_x = self._param(
            grid, "Биты X, NADCX", str(DEFAULT_NADC_X), 0
        )
        self._nadc_y = self._param(
            grid, "Биты Y, NADCY", str(DEFAULT_NADC_Y), 1
        )
        self._eta0 = self._param(grid, "Порог eta0", str(DEFAULT_ETA0), 2)
        self._order = self._param(
            grid, "Порядок q", str(DEFAULT_EFFECT_ORDER), 3
        )
        self._min_dom = self._param(
            grid, "Минимум доминантных", str(DEFAULT_MIN_DOMINANT), 4
        )
        self._max_dom = self._param(
            grid, "Максимум доминантных", str(DEFAULT_MAX_DOMINANT), 5
        )
        note = (
            "0 в разрядности подбирает биты по числу строк. "
            "Пояснения показателей — на экране «Справка»."
        )
        ctk.CTkLabel(page, text=note, wraplength=700, justify="left").pack(
            anchor="w", padx=16
        )
        return page

    def _build_run(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Полоса хода, журнал и кнопка расчёта."""
        page = ctk.CTkFrame(parent)
        self._step_label = ctk.CTkLabel(page, text="Расчёт ещё не запускался")
        self._step_label.pack(anchor="w", padx=12, pady=(12, 4))
        self._progress = ctk.CTkProgressBar(page)
        self._progress.set(0)
        self._progress.pack(fill="x", padx=12, pady=4)
        buttons = ctk.CTkFrame(page)
        buttons.pack(fill="x", padx=12, pady=8)
        self._run_button = ctk.CTkButton(
            buttons, text="Считать", command=self._start
        )
        self._run_button.pack(side="left")
        self._scan_button = ctk.CTkButton(
            buttons,
            text="Сканировать выходы",
            command=self._start_scan,
            width=160,
        )
        self._scan_button.pack(side="left", padx=8)
        ctk.CTkButton(buttons, text="Выгрузить", command=self._export).pack(
            side="left", padx=8
        )
        ctk.CTkButton(
            buttons, text="Открыть папку", command=self._open_export
        ).pack(side="left")
        self._log = ctk.CTkTextbox(page)
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return page

    def _build_results(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Карточки показателей, таблицы и графики."""
        page = ctk.CTkFrame(parent)
        self._scan_summary = ctk.CTkLabel(
            page,
            text="",
            wraplength=900,
            justify="left",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self._cards_host = ctk.CTkFrame(page)
        self._cards_host.pack(fill="x", padx=8, pady=8)
        self._card_values: dict[str, ctk.CTkLabel] = {}
        for column, name in enumerate(CARD_NAMES):
            box = ctk.CTkFrame(self._cards_host)
            box.grid(row=0, column=column, padx=4, sticky="nsew")
            ctk.CTkLabel(box, text=name).pack(anchor="w", padx=6)
            value = ctk.CTkLabel(box, text="—")
            value.pack(anchor="w", padx=6)
            short = BY_NAME[name].short if name in BY_NAME else ""
            ctk.CTkLabel(
                box, text=short, wraplength=140, justify="left"
            ).pack(anchor="w", padx=6)
            self._card_values[name] = value
            if name == "MDO":
                ctk.CTkButton(
                    box,
                    text="Что это",
                    width=80,
                    command=lambda: self.show_indicator("MDO"),
                ).pack(padx=6, pady=4)
        self._picks_host = ctk.CTkFrame(page, fg_color="transparent")
        self._picks_host.pack(anchor="w", padx=12, pady=4)
        self._table_pick = ctk.CTkComboBox(
            self._picks_host,
            values=[SHEET_SUMMARY],
            command=self._show_table,
            width=220,
        )
        self._table_pick.pack(side="left")
        self._y_filter_label = ctk.CTkLabel(
            self._picks_host, text="Отчёт по Y"
        )
        self._y_filter_pick = ctk.CTkComboBox(
            self._picks_host,
            values=[""],
            command=self._on_scan_y_filter,
            width=220,
        )
        table_host = ctk.CTkFrame(page)
        table_host.pack(fill="x", padx=8)
        self._result_tree = self._make_tree(table_host)
        self._result_tree.configure(height=6)
        charts = ctk.CTkScrollableFrame(page, label_text="График")
        charts.pack(fill="both", expand=True, padx=8, pady=8)
        self._charts_host = charts
        self._chart_pick = ctk.CTkComboBox(
            charts, values=[""], command=self._show_chart, width=420
        )
        self._chart_pick.pack(anchor="w", padx=4, pady=4)
        tools = ctk.CTkFrame(charts, fg_color="transparent")
        tools.pack(anchor="w", padx=4, pady=2)
        self._copy_chart_button = ctk.CTkButton(
            tools,
            text="Копировать",
            width=120,
            command=self._copy_chart,
            state="disabled",
        )
        self._copy_chart_button.pack(side="left", padx=(0, 12))
        ctk.CTkButton(
            tools,
            text="−",
            width=36,
            command=lambda: self._zoom_chart(-CHART_ZOOM_STEP),
        ).pack(side="left")
        self._zoom_reset = ctk.CTkButton(
            tools,
            text="100%",
            width=64,
            fg_color="transparent",
            command=self._reset_chart_zoom,
        )
        self._zoom_reset.pack(side="left", padx=4)
        ctk.CTkButton(
            tools,
            text="+",
            width=36,
            command=lambda: self._zoom_chart(CHART_ZOOM_STEP),
        ).pack(side="left")
        self._chart_label = ctk.CTkLabel(
            charts, text="График появится после расчёта"
        )
        self._chart_label.pack(anchor="w", padx=4, pady=4)
        self._chart_label.bind("<Control-MouseWheel>", self._on_chart_wheel)
        return page

    def _build_help(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Инструкция по разделам и пояснение каждого показателя."""
        page = ctk.CTkFrame(parent)
        names = [*guide_names(), *BY_NAME.keys()]
        self._help_pick = ctk.CTkComboBox(
            page, values=names, command=self._on_help_pick, width=420
        )
        self._help_pick.set(GUIDE_TITLE)
        self._help_pick.pack(anchor="w", padx=12, pady=12)
        self._help_view = ctk.CTkTextbox(
            page, font=("Segoe UI", 14), wrap="word"
        )
        self._help_view.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._set_help_text(guide_for(GUIDE_TITLE))
        return page

    def _open_guide(self) -> None:
        """Открывает инструкцию целиком с экрана «Данные»."""
        self.show_screen("Справка")
        self._help_pick.set(GUIDE_TITLE)
        self._set_help_text(guide_for(GUIDE_TITLE))

    def _set_help_text(self, text: str) -> None:
        """Заменяет текст справки и прокручивает его к началу."""
        self._help_view.delete("1.0", "end")
        self._help_view.insert("1.0", text)
        self._help_view.see("1.0")

    def _param(
        self,
        parent: ctk.CTkFrame,
        caption: str,
        value: str,
        column: int,
    ) -> ctk.CTkEntry:
        """Подпись и поле ввода в строке параметров."""
        ctk.CTkLabel(parent, text=caption).grid(
            row=0, column=column, padx=8, sticky="w"
        )
        entry = ctk.CTkEntry(parent, width=120)
        entry.insert(0, value)
        entry.grid(row=1, column=column, padx=8, pady=4)
        return entry

    def _make_tree(self, parent: ctk.CTkFrame) -> ttk.Treeview:
        """Таблица с вертикальной и горизонтальной прокруткой."""
        scroll_y = ttk.Scrollbar(parent, orient="vertical")
        scroll_x = ttk.Scrollbar(parent, orient="horizontal")
        tree = ttk.Treeview(
            parent,
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
        )
        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        tree.pack(side="left", fill="both", expand=True)
        return tree

    def _choose_file(self) -> None:
        """Диалог выбора файла данных."""
        path = filedialog.askopenfilename(
            filetypes=[
                ("Таблицы", "*.csv *.txt *.xlsx *.xls"),
                ("Все файлы", "*.*"),
            ]
        )
        if path:
            self.load_file(path)

    def _assign_roles(self, table: pd.DataFrame) -> None:
        """Ставит предлагаемые роли. Второй выход не назначается."""
        self._role_boxes = {}
        output_assigned = False
        for name in table.columns:
            role = suggest_role(str(name), table[name])
            if role == ROLE_Y:
                if output_assigned:
                    numeric = pd.api.types.is_numeric_dtype(table[name])
                    role = "Z" if numeric else ROLE_EXCLUDE
                else:
                    output_assigned = True
            self._role_boxes[str(name)] = RoleChoice(role)

    def _refresh_data(self) -> None:
        """Перерисовывает сводку столбцов и строки таблицы."""
        frame = self._frame
        if frame is None:
            return
        self._size_label.configure(
            text=f"Строк: {len(frame)}, столбцов: {frame.shape[1]}"
        )
        names = [str(name) for name in frame.columns]
        self._column_pick.configure(values=names or [""])
        if names:
            self._column_pick.set(names[0])
        self._fill_role_panel(frame)
        self._fill_tree(self._data_tree, frame.head(PREVIEW_ROWS))

    def _fill_role_panel(self, frame: pd.DataFrame) -> None:
        """Список столбцов: тип, пустые ячейки и роль."""
        for child in self._roles.winfo_children():
            child.destroy()
        overview = column_overview(frame)
        labels = list(ROLE_LABELS.values())
        for row, record in overview.iterrows():
            name = str(record["Столбец"])
            info = f"{name}  |  {record['Тип']}, пустых {record['Пустых']}"
            ctk.CTkLabel(self._roles, text=info).grid(
                row=int(row), column=0, sticky="w", padx=6, pady=2
            )
            box = ctk.CTkComboBox(self._roles, values=labels, width=210)
            box.set(self._role_boxes[name].get())
            box.configure(
                command=lambda label, column=name: self._set_role(
                    column, label
                )
            )
            box.grid(row=int(row), column=1, padx=6, pady=2)

    def _set_role(self, name: str, label: str) -> None:
        """Записывает роль, выбранную в списке столбца."""
        self._role_boxes[name].role = LABEL_TO_ROLE[label]

    def _delete_column(self) -> None:
        """Убирает выбранный столбец из рабочей таблицы."""
        if self._frame is None:
            return
        name = self._column_pick.get()
        if name not in self._frame.columns:
            return
        self._frame = drop_columns(self._frame, [name])
        self._role_boxes.pop(name, None)
        self._refresh_data()

    def _delete_rows(self) -> None:
        """Удаляет строки, отмеченные в таблице."""
        if self._frame is None:
            return
        selected = self._data_tree.selection()
        if not selected:
            messagebox.showinfo("TumAlgoritm", "Отметьте строки в таблице")
            return
        indexes = [int(item) for item in selected]
        self._frame = drop_rows(self._frame, indexes)
        self._refresh_data()

    def _apply_filter(self) -> None:
        """Оставляет строки выбранного столбца внутри границ."""
        if self._frame is None:
            return
        column = self._column_pick.get()
        try:
            low = _optional_float(self._filter_low.get())
            high = _optional_float(self._filter_high.get())
            self._frame = apply_range_filter(self._frame, column, low, high)
        except ValueError as exc:
            messagebox.showerror("TumAlgoritm", str(exc))
            return
        self._refresh_data()

    def _save_csv(self) -> None:
        """Пишет рабочую таблицу в новый CSV, не в исходный файл."""
        if self._frame is None or self._source is None:
            messagebox.showwarning("TumAlgoritm", "Сначала откройте файл")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="working.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        try:
            save_working_csv(self._frame, path, source=self._source)
        except ValueError as exc:
            messagebox.showerror("TumAlgoritm", str(exc))
            return
        self._append(f"Сохранено: {path}\n")

    def _start(self) -> None:
        """Запускает расчёт по рабочей таблице в фоновом потоке."""
        if self._frame is None:
            messagebox.showwarning("TumAlgoritm", "Сначала откройте файл")
            return
        if self._running:
            return
        try:
            y_name, controlled, excluded, params = self._read_form()
        except ValueError as exc:
            messagebox.showerror("TumAlgoritm", str(exc))
            return
        self._begin_run("Запуск расчёта")
        frame = self._frame.copy()
        worker = threading.Thread(
            target=self._worker,
            args=(frame, y_name, controlled, excluded, params),
            daemon=True,
        )
        worker.start()

    def _start_scan(self) -> None:
        """Запускает перебор всех числовых столбцов как выходов."""
        if self._frame is None:
            messagebox.showwarning("TumAlgoritm", "Сначала откройте файл")
            return
        if self._running:
            return
        try:
            excluded, params = self._read_scan_form()
        except ValueError as exc:
            messagebox.showerror("TumAlgoritm", str(exc))
            return
        self._begin_run("Запуск сканирования выходов")
        frame = self._frame.copy()
        worker = threading.Thread(
            target=self._scan_worker,
            args=(frame, excluded, params),
            daemon=True,
        )
        worker.start()

    def _begin_run(self, title: str) -> None:
        """Общая подготовка экрана «Расчёт» перед фоновым потоком."""
        self._running = True
        self.show_screen("Расчёт")
        self._set_run_enabled(False)
        self._progress.set(0)
        self._step_label.configure(text=title)
        self._log.delete("1.0", "end")
        self._journal = []

    def _read_form(self) -> tuple[str, list[str], list[str], dict]:
        """Собирает роли и параметры. Нужен ровно один выход."""
        y_names = []
        controlled = []
        excluded = []
        for name, choice in self._role_boxes.items():
            if choice.role == ROLE_Y:
                y_names.append(name)
            elif choice.role == ROLE_U:
                controlled.append(name)
            elif choice.role == ROLE_EXCLUDE:
                excluded.append(name)
        if len(y_names) != 1:
            raise ValueError("Нужен ровно один столбец с ролью Y")
        return y_names[0], controlled, excluded, self._read_params()

    def _read_scan_form(self) -> tuple[list[str], dict]:
        """Для скана нужен только список исключённых и параметры."""
        excluded = [
            name
            for name, choice in self._role_boxes.items()
            if choice.role == ROLE_EXCLUDE
        ]
        return excluded, self._read_params()

    def _read_params(self) -> dict:
        """Читает поля экрана «Параметры»."""
        return {
            "nadc_x": int(self._nadc_x.get()),
            "nadc_y": int(self._nadc_y.get()),
            "eta0": float(self._eta0.get()),
            "effect_order": int(self._order.get()),
            "min_dominant": int(self._min_dom.get()),
            "max_dominant": int(self._max_dom.get()),
        }

    def _worker(
        self,
        frame: pd.DataFrame,
        y_name: str,
        controlled: list[str],
        excluded: list[str],
        params: dict,
    ) -> None:
        """Считает показатели. Окно обновляет только очередь действий."""
        journal: list[str] = []

        def on_progress(name: str, percent: int) -> None:
            line = f"{percent}%  {name}"
            journal.append(line)
            self._post(
                lambda text=line, value=percent: self._show_progress(
                    text, value
                )
            )

        try:
            result = run_frame(
                frame,
                y_name,
                u_names=controlled,
                exclude_names=excluded,
                on_progress=on_progress,
                **params,
            )
            self._post(lambda: self._finish_ok(result, frame, journal))
        except Exception as exc:
            self._post(lambda error=exc: self._finish_error(str(error)))

    def _scan_worker(
        self,
        frame: pd.DataFrame,
        excluded: list[str],
        params: dict,
    ) -> None:
        """Перебирает выходы. Окно обновляет только очередь действий."""
        journal: list[str] = []

        def on_progress(name: str, percent: int) -> None:
            line = f"{percent}%  {name}"
            journal.append(line)
            self._post(
                lambda text=line, value=percent: self._show_progress(
                    text, value
                )
            )

        try:
            scan = scan_outputs(
                frame,
                exclude_names=excluded,
                params=params,
                on_progress=on_progress,
            )
            self._post(lambda: self._finish_scan(scan, frame, journal))
        except Exception as exc:
            self._post(lambda error=exc: self._finish_error(str(error)))

    def _post(self, action) -> None:
        """Кладёт действие в очередь окна.

        Фоновый поток не вызывает Tk напрямую: иначе интерфейс
        падает, если главный цикл ещё не крутится.
        """
        self._ui_queue.put(action)

    def _poll_ui(self) -> None:
        """Выполняет действия расчёта в потоке окна."""
        try:
            while True:
                action = self._ui_queue.get_nowait()
                self._run_ui_action(action)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self._poll_job = self.after(50, self._poll_ui)

    def _run_ui_action(self, action) -> None:
        """Выполняет шаг интерфейса. Ошибка остаётся в журнале.

        Исключение в очереди иначе обрывает опрос окна, и кнопка
        «Считать» больше не включается.
        """
        try:
            action()
        except Exception as exc:
            self._append(f"Ошибка: {exc}\n")
            self._set_run_enabled(True)

    def destroy(self) -> None:
        """Останавливает опрос очереди и закрывает окно."""
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None
        super().destroy()

    def _show_progress(self, line: str, percent: int) -> None:
        """Обновляет полосу и журнал. Вызывается в потоке окна."""
        self._progress.set(percent / 100)
        self._step_label.configure(text=line)
        self._append(line + "\n")

    def _finish_ok(
        self,
        result: AnalysisResult,
        frame: pd.DataFrame,
        journal: list[str],
    ) -> None:
        """Показывает числа и графики после успешного расчёта."""
        self._result = result
        self._y_scan = None
        self._calculated_frame = frame
        self._journal = list(journal)
        for text in result.warnings:
            line = f"Предупреждение: {text}"
            self._journal.append(line)
            self._append(line + "\n")
        try:
            self._fill_results(result)
            self.show_screen("Результаты")
        except Exception as exc:
            self._append(f"Ошибка показа результата: {exc}\n")
        finally:
            self._set_run_enabled(True)

    def _finish_scan(
        self,
        scan: YScanResult,
        frame: pd.DataFrame,
        journal: list[str],
    ) -> None:
        """Показывает рейтинг выходов на экране «Результаты»."""
        self._y_scan = scan
        self._result = None
        self._calculated_frame = frame
        self._journal = list(journal)
        line = summary_line(scan)
        self._journal.append(line)
        self._append(line + "\n")
        try:
            self._fill_scan_results(scan)
            self.show_screen("Результаты")
        except Exception as exc:
            self._append(f"Ошибка показа скана: {exc}\n")
        finally:
            self._set_run_enabled(True)

    def _finish_error(self, message: str) -> None:
        """Оставляет окно открытым и пишет ошибку в журнал."""
        self._append(f"Ошибка: {message}\n")
        self._set_run_enabled(True)

    def _set_run_enabled(self, enabled: bool) -> None:
        """Включает или блокирует кнопки расчёта и скана.

        После смены состояния кнопка иногда теряет нажатие.
        Команда задаётся снова при каждом включении.
        """
        if enabled:
            self._running = False
            self._run_button.configure(
                state="normal", command=self._start
            )
            self._scan_button.configure(
                state="normal", command=self._start_scan
            )
            self._accept_run_click()
            return
        self._run_button.configure(state="disabled")
        self._scan_button.configure(state="disabled")

    def _accept_run_click(self) -> None:
        """Разрешает нажатие кнопок без повторного наведения мыши.

        Кнопка библиотеки окна вызывает команду только если курсор
        уже отмечен как находящийся над ней. После смены экрана эта
        отметка пропадает, и повторное нажатие молча теряется.
        """
        for button in (self._run_button, self._scan_button):
            if hasattr(button, "_mouse_inside"):
                button._mouse_inside = True

    def _fill_results(self, result: AnalysisResult) -> None:
        """Заполняет карточки, таблицы и список графиков."""
        self._set_scan_layout(False)
        frames = build_frames(result)
        self._result_frames = frames
        summary = frames[SHEET_SUMMARY]
        values = dict(zip(summary["Показатель"], summary["Значение"]))
        for name, label in self._card_values.items():
            label.configure(text=_format_cell(values.get(name, "—")))
        sheet_names = list(frames)
        self._table_pick.configure(values=sheet_names)
        self._table_pick.set(SHEET_SUMMARY)
        self._show_table(SHEET_SUMMARY)
        try:
            plots = save_charts(result, self._preview_dir)
        except Exception as exc:
            self._append(f"Ошибка графиков: {exc}\n")
            plots = []
        self._chart_paths = {}
        titles = []
        for path in plots:
            title = CHART_TITLES.get(path.name, path.name)
            self._chart_paths[title] = path
            titles.append(title)
        if titles:
            self._chart_pick.configure(values=titles)
            preferred = CHART_TITLES[FILE_HEATMAP]
            if preferred in self._chart_paths:
                current = preferred
            else:
                current = titles[0]
            self._chart_pick.set(current)
            self._show_chart(current)

    def _fill_scan_results(self, scan: YScanResult) -> None:
        """Показывает рейтинг и отчёты скана на экране результатов."""
        self._set_scan_layout(True)
        self._scan_summary.configure(text=summary_line(scan))
        names = successful_y_names(scan)
        self._scan_y_filter = names[0] if names else None
        if names:
            self._y_filter_pick.configure(values=names)
            self._y_filter_pick.set(self._scan_y_filter)
        else:
            self._y_filter_pick.configure(values=[""])
            self._y_filter_pick.set("")
        self._result_frames = build_scan_frames(
            scan, self._scan_y_filter
        )
        sheet_names = list(self._result_frames)
        self._table_pick.configure(values=sheet_names)
        # Сначала связи Y–X: имена выходов и факторов видны сразу.
        default_sheet = SHEET_LINKS
        if default_sheet not in self._result_frames:
            default_sheet = SHEET_RANK
        self._table_pick.set(default_sheet)
        self._show_table(default_sheet)
        self._show_scan_charts(self._scan_y_filter)

    def _set_scan_layout(self, scan_mode: bool) -> None:
        """Переключает карточки одного Y и панель скана."""
        if scan_mode:
            self._cards_host.pack_forget()
            self._scan_summary.pack(
                fill="x",
                padx=12,
                pady=(12, 4),
                before=self._picks_host,
            )
            self._y_filter_label.pack(side="left", padx=(16, 4))
            self._y_filter_pick.pack(side="left")
            return
        self._scan_summary.pack_forget()
        self._scan_summary.configure(text="")
        self._y_filter_label.pack_forget()
        self._y_filter_pick.pack_forget()
        self._cards_host.pack(
            fill="x", padx=8, pady=8, before=self._picks_host
        )

    def _on_scan_y_filter(self, y_name: str) -> None:
        """Меняет таблицы связей и графики под выбранный выход."""
        if self._y_scan is None or not y_name:
            return
        self._scan_y_filter = y_name
        current = self._table_pick.get()
        self._result_frames = build_scan_frames(
            self._y_scan, y_name
        )
        if current not in self._result_frames:
            current = SHEET_LINKS
        self._table_pick.configure(values=list(self._result_frames))
        self._table_pick.set(current)
        self._show_table(current)
        self._show_scan_charts(y_name)

    def _scan_row_result(self, y_name: str) -> AnalysisResult | None:
        """Полный итог расчёта для выбранного Y в скане."""
        if self._y_scan is None:
            return None
        for row in self._y_scan.rows:
            if row.ok and row.y_name == y_name:
                return row.result
        return None

    def _show_scan_charts(self, y_name: str | None) -> None:
        """Строит или подставляет графики для выбранного выхода скана.

        Рисунки пишутся в подпапку preview по имени Y, чтобы смена
        фильтра не затирала уже показанные картинки.
        """
        if not y_name:
            self._clear_chart_view(
                "Выберите выход Y, чтобы увидеть графики"
            )
            return
        result = self._scan_row_result(y_name)
        if result is None:
            self._clear_chart_view(
                f"Нет расчёта для выхода {y_name}"
            )
            return
        folder = self._preview_dir / "scan" / _safe_folder_name(y_name)
        folder.mkdir(parents=True, exist_ok=True)
        plots = sorted(folder.glob("*.png"))
        if not plots:
            try:
                plots = save_charts(result, folder)
            except Exception as exc:
                self._append(
                    f"Ошибка графиков для {y_name}: {exc}\n"
                )
                self._clear_chart_view(
                    f"Не удалось построить графики для {y_name}"
                )
                return
        self._chart_paths = {}
        titles = []
        for path in plots:
            title = CHART_TITLES.get(path.name, path.name)
            self._chart_paths[title] = path
            titles.append(title)
        if not titles:
            self._clear_chart_view(
                f"Нет рисунков для выхода {y_name}"
            )
            return
        self._chart_pick.configure(values=titles)
        preferred = CHART_TITLES[FILE_HEATMAP]
        if preferred in self._chart_paths:
            current = preferred
        else:
            current = titles[0]
        # Сохраняем тип графика при смене Y, если он есть у нового.
        previous = self._chart_pick.get()
        if previous in self._chart_paths:
            current = previous
        self._chart_pick.set(current)
        self._show_chart(current)

    def _clear_chart_view(self, message: str) -> None:
        """Убирает картинку и показывает пояснение."""
        self._chart_paths = {}
        self._chart_pick.configure(values=[""])
        self._chart_pick.set("")
        self._chart_label.configure(image=None, text=message)
        self._copy_chart_button.configure(state="disabled")
        self._chart_image = None

    def _show_table(self, sheet_name: str) -> None:
        """Показывает выбранный лист результата."""
        frame = self._result_frames.get(sheet_name)
        if frame is None:
            return
        if sheet_name == SHEET_RES_M:
            shown = frame.reset_index()
        else:
            shown = frame
        self._fill_tree(self._result_tree, shown.head(PREVIEW_ROWS))

    def _show_chart(self, title: str) -> None:
        """Показывает рисунок в масштабе 100%. Тепловая карта крупнее."""
        path = self._chart_paths.get(title)
        if path is None:
            return
        with Image.open(path) as opened:
            image = opened.copy()
        if path.name in (FILE_HEATMAP, FILE_PDO_HEATMAP):
            box = HEATMAP_VIEW
        else:
            box = OTHER_CHART_VIEW
        self._chart_image = image
        self._chart_box = box
        self._chart_base = _fit_image_size(image.size, box)
        self._chart_zoom = CHART_ZOOM_DEFAULT
        self._copy_chart_button.configure(state="normal", text="Копировать")
        self._apply_chart_zoom()

    def _apply_chart_zoom(self) -> None:
        """Перерисовывает текущий график в выбранном масштабе.

        Пропорции не меняются: увеличивается вся картинка, квадрат
        остаётся квадратом. Исходный файл при этом не трогается.
        """
        if self._chart_image is None:
            return
        factor = self._chart_zoom
        width = max(1, int(round(self._chart_base[0] * factor)))
        height = max(1, int(round(self._chart_base[1] * factor)))
        self._chart_size = (width, height)
        self._chart_photo = ctk.CTkImage(
            light_image=self._chart_image,
            dark_image=self._chart_image,
            size=self._chart_size,
        )
        self._chart_label.configure(image=self._chart_photo, text="")
        self._zoom_reset.configure(text=f"{int(round(factor * 100))}%")
        self._bind_chart_wheel(self._chart_label)

    def _bind_chart_wheel(self, widget) -> None:
        """Вешает Ctrl+колёсико на подпись и вложенный холст картинки."""
        widget.bind("<Control-MouseWheel>", self._on_chart_wheel)
        for child in widget.winfo_children():
            self._bind_chart_wheel(child)

    def _zoom_chart(self, step: float) -> None:
        """Меняет масштаб на шаг и зажимает его в допустимые пределы.

        Args:
            step: приращение, доля от размера при 100%.
        """
        if self._chart_image is None:
            return
        updated = self._chart_zoom + step
        updated = min(CHART_ZOOM_MAX, max(CHART_ZOOM_MIN, updated))
        snapped = round(updated / CHART_ZOOM_STEP) * CHART_ZOOM_STEP
        self._chart_zoom = float(snapped)
        self._apply_chart_zoom()

    def _reset_chart_zoom(self) -> None:
        """Возвращает масштаб 100%."""
        if self._chart_image is None:
            return
        self._chart_zoom = CHART_ZOOM_DEFAULT
        self._apply_chart_zoom()

    def _on_chart_wheel(self, event) -> str:
        """Ctrl и колёсико меняют масштаб рисунка под указателем."""
        if event.delta > 0:
            self._zoom_chart(CHART_ZOOM_STEP)
        elif event.delta < 0:
            self._zoom_chart(-CHART_ZOOM_STEP)
        return "break"

    def _copy_chart(self) -> None:
        """Копирует исходный рисунок в буфер обмена."""
        if self._chart_image is None:
            return
        try:
            copy_image_to_clipboard(self._chart_image)
        except OSError as exc:
            messagebox.showerror(
                "TumAlgoritm",
                f"Не удалось скопировать рисунок: {exc}",
            )
            return
        self._copy_chart_button.configure(text="Скопировано")
        self.after(
            1500,
            lambda: self._copy_chart_button.configure(text="Копировать"),
        )

    def _on_help_pick(self, name: str) -> None:
        """Показывает раздел инструкции или карточку показателя."""
        if name in guide_names():
            self._set_help_text(guide_for(name))
            return
        self._set_help_text(format_help(help_for(name)))

    def _export(self) -> None:
        """Пишет папку прогона и не трогает исходный файл."""
        root = Path.cwd() / "results"
        if self._y_scan is not None:
            self._export_dir = export_y_scan(
                self._y_scan,
                self._journal,
                root,
            )
            self._append(f"Папка скана: {self._export_dir}\n")
            messagebox.showinfo(
                "TumAlgoritm", f"Сохранено:\n{self._export_dir}"
            )
            return
        if self._result is None or self._calculated_frame is None:
            messagebox.showwarning(
                "TumAlgoritm", "Сначала выполните расчёт или скан"
            )
            return
        self._export_dir = export_run(
            self._calculated_frame,
            self._result,
            self._journal,
            root,
        )
        self._append(f"Папка прогона: {self._export_dir}\n")
        messagebox.showinfo("TumAlgoritm", f"Сохранено:\n{self._export_dir}")

    def _open_export(self) -> None:
        """Открывает папку последнего прогона в проводнике."""
        if self._export_dir is None:
            messagebox.showinfo(
                "TumAlgoritm", "Папка появится после выгрузки"
            )
            return
        os.startfile(self._export_dir)  # noqa: S606

    def _append(self, message: str) -> None:
        """Добавляет строку в журнал расчёта."""
        self._log.insert("end", message)
        self._log.see("end")

    def _fill_tree(self, tree: ttk.Treeview, frame: pd.DataFrame) -> None:
        """Заполняет таблицу окна строками кадра."""
        tree.delete(*tree.get_children())
        columns = [str(name) for name in frame.columns]
        tree["columns"] = columns
        tree["show"] = "headings"
        for name in columns:
            tree.heading(name, text=name)
            tree.column(name, width=110, anchor="w", stretch=False)
        for index, row in frame.iterrows():
            values = [_format_cell(value) for value in row.tolist()]
            tree.insert("", "end", iid=str(index), values=values)


def _optional_float(text: str) -> float | None:
    """Пустая строка границы фильтра означает отсутствие границы."""
    stripped = text.strip()
    if not stripped:
        return None
    return float(stripped.replace(",", "."))


def _format_cell(value: object) -> str:
    """Короткий текст ячейки для таблицы окна."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _safe_folder_name(name: str) -> str:
    """Имя подпапки для графиков одного Y без запрещённых символов."""
    cleaned = []
    for char in str(name):
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    text = "".join(cleaned).strip("._")
    return text or "y"


def factor_count(frame: pd.DataFrame, y_name: str) -> int:
    """Сколько числовых факторов войдёт в расчёт при текущих ролях.

    Args:
        frame: рабочая таблица.
        y_name: столбец выхода.

    Returns:
        Число столбцов X.
    """
    dataset = build_dataset(frame, y_name)
    return int(dataset.x.shape[1])


def launch() -> None:
    """Открывает окно и ждёт действий пользователя."""
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    launch()
