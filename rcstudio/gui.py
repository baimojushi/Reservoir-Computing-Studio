"""PySide6 local experiment workbench; compute work stays in HTTP service."""
import copy
import json
import math
import sys

import numpy as np

try:
    from PySide6.QtCore import Qt, QTimer, QRectF
    from PySide6.QtGui import (
        QColor, QBrush, QFont, QPainter, QPainterPath, QPen,
    )
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
        QFrame, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
        QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
        QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
        QHeaderView, QAbstractItemView,
    )
    import pyqtgraph as pg
except ImportError as exc:
    raise RuntimeError(
        "GUI dependencies unavailable. Install: pip install -e '.[gui]'"
    ) from exc

from .config import DEFAULT
from .client import ServiceError

COLORS = {
    "ink": "#101a2b",
    "panel": "#18273a",
    "panel2": "#203147",
    "text": "#edf3fb",
    "muted": "#9aafc1",
    "teal": "#56c7c3",
    "orange": "#e8ad76",
    "border": "#31475f",
}

QSS = """
QMainWindow, QWidget { background: #101a2b; color: #edf3fb;
    font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 12px; }
QFrame#Header { background: #18273a; border-bottom: 1px solid #31475f; }
QLabel#Hero { font-weight: 700; font-size: 19px; color: #f4f7fd; }
QLabel#Sub { color: #9aafc1; font-size: 11px; }
QGroupBox { border: 1px solid #31475f; border-radius: 8px;
    margin-top: 12px; padding-top: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QSpinBox, QDoubleSpinBox, QComboBox {
    background: #203147; border: 1px solid #3b536c; border-radius: 5px;
    padding: 5px; min-height: 20px; }
QPushButton { background: #2a465c; border: 1px solid #40647a;
    border-radius: 6px; padding: 8px 10px; font-weight: 600; }
QPushButton:hover { background: #365a72; }
QPushButton#Primary { background: #248b8c; border-color: #3cb9b6; }
QPushButton#Primary:hover { background: #2b9fa2; }
QPushButton#Warning { background: #7e4c40; border-color: #c67c63; }
QTabWidget::pane { border: 1px solid #31475f; }
QTabBar::tab { background: #18273a; color: #cad6e4;
    padding: 9px 16px; border: 1px solid #31475f; }
QTabBar::tab:selected { color: #56c7c3; background: #21364b; }
QTableWidget, QTextEdit { background: #152337; alternate-background-color: #1b2c40;
    border: 1px solid #31475f; gridline-color: #304257; }
QHeaderView::section { background: #21354b; padding: 5px;
    border: 0; border-right: 1px solid #31475f; }
QScrollArea { border: none; }
"""

class TopologyCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.units = DEFAULT["model"]["units"]
        self.setMinimumHeight(145)
        self.setMaximumHeight(172)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        x0, x1, x2 = 32, w * 0.50, w - 37
        ys = np.linspace(28, h - 29, 5)
        dots = []
        for i in range(12):
            angle = i * 2.39996
            r = 24 + (i % 4) * 9
            dots.append((x1 + r * math.cos(angle), h/2 + r * math.sin(angle)))
        p.setPen(QPen(QColor("#3e6673"), 1.0))
        for y in ys[::2]:
            for k in (0, 4, 8):
                p.drawLine(int(x0), int(y), int(dots[k][0]), int(dots[k][1]))
        for i, (x, y) in enumerate(dots):
            for j in ((i+3)%12, (i+5)%12):
                p.drawLine(int(x), int(y), int(dots[j][0]), int(dots[j][1]))
            if i % 3 == 0:
                p.drawLine(int(x), int(y), int(x2), int(h/2))
        p.setPen(QPen(QColor("#132e38"), 1.2))
        p.setBrush(QBrush(QColor(COLORS["teal"])))
        for y in ys[::2]:
            p.drawEllipse(QRectF(x0-5, y-5, 10, 10))
        for i, (x, y) in enumerate(dots):
            p.setBrush(QBrush(QColor(
                COLORS["orange"] if i in (1, 7) else "#57a3b0"
            )))
            p.drawEllipse(QRectF(x-5, y-5, 10, 10))
        p.setBrush(QBrush(QColor(COLORS["orange"])))
        p.drawEllipse(QRectF(x2-7, h/2-7, 14, 14))
        p.setPen(QColor(COLORS["muted"]))
        font = QFont("Segoe UI", 8)
        p.setFont(font)
        p.drawText(QRectF(2, h-18, 70, 16),
                   Qt.AlignmentFlag.AlignCenter, "INPUT")
        p.drawText(QRectF(x1-65, h-18, 130, 16),
                   Qt.AlignmentFlag.AlignCenter,
                   f"RESERVOIR · {self.units}")
        p.drawText(QRectF(w-76, h-18, 76, 16),
                   Qt.AlignmentFlag.AlignCenter, "READOUT")
        p.end()

def header_label(label, sub=None):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 2, 0, 2)
    title = QLabel(label)
    title.setStyleSheet("font-size: 14px; font-weight: 650;")
    layout.addWidget(title)
    if sub:
        desc = QLabel(sub)
        desc.setObjectName("Sub")
        layout.addWidget(desc)
    return container

class MainWindow(QMainWindow):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.current_run = None
        self.last_plotted = None
        self.last_status = None
        self.last_runs_signature = None
        self.setWindowTitle("RC Studio · 储备池计算研究工作台")
        self.resize(1280, 820)
        self.setMinimumSize(1020, 650)
        self._build()
        self._refresh_datasets()
        self._refresh()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(1400)

    def _build(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        head = QFrame()
        head.setObjectName("Header")
        head_layout = QHBoxLayout(head)
        head_layout.setContentsMargins(20, 11, 20, 11)
        title_box = header_label(
            "RC Studio", "Local reservoir computing · ESN research"
        )
        title_box.findChildren(QLabel)[0].setObjectName("Hero")
        head_layout.addWidget(title_box)
        head_layout.addStretch()
        self.status_text = QLabel("● 正在连接本地服务")
        self.status_text.setStyleSheet("color: #56c7c3;")
        head_layout.addWidget(self.status_text)
        root_layout.addWidget(head)

        main = QSplitter(Qt.Orientation.Horizontal)
        main.setChildrenCollapsible(False)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_panel = QWidget()
        controls = QVBoxLayout(left_panel)
        controls.setSpacing(10)
        controls.setContentsMargins(12, 12, 8, 12)

        group_data = QGroupBox("01 · 数据来源")
        data_form = QVBoxLayout(group_data)
        self.dataset_box = QComboBox()
        data_form.addWidget(self.dataset_box)
        self.length_box = QSpinBox()
        self.length_box.setRange(150, 20000)
        self.length_box.setValue(DEFAULT["dataset"]["length"])
        self.length_box.setSingleStep(250)
        length_row = QFormLayout()
        length_row.addRow("序列长度", self.length_box)
        data_form.addLayout(length_row)
        import_button = QPushButton("导入 CSV 文件")
        import_button.clicked.connect(self._import_csv)
        data_form.addWidget(import_button)
        controls.addWidget(group_data)

        group_model = QGroupBox("02 · 网络结构")
        model_layout = QVBoxLayout(group_model)
        self.topology = TopologyCanvas()
        model_layout.addWidget(self.topology)
        controls.addWidget(group_model)

        group_config = QGroupBox("03 · ESN 参数")
        form = QFormLayout(group_config)
        form.setSpacing(9)
        self.fields = {}
        specs = [
            ("units", "储备池神经元数", 10, 350, 1, 0),
            ("spectral_radius", "谱半径", 0.01, 2.0, 0.05, 3),
            ("leak_rate", "泄漏率", 0.01, 1.0, 0.05, 3),
            ("density", "连接密度", 0.01, 1.0, 0.01, 3),
            ("input_scaling", "输入缩放", 0.001, 10.0, 0.05, 3),
            ("ridge", "岭回归系数", 0.0, 100.0, 0.00001, 7),
            ("washout", "预热步数", 0, 2000, 10, 0),
            ("horizon", "预测步长", 1, 100, 1, 0),
            ("seed", "随机种子", 0, 2147483647, 1, 0),
        ]
        for key, label, minimum, maximum, step, decimals in specs:
            field = QSpinBox() if decimals == 0 else QDoubleSpinBox()
            field.setRange(minimum, maximum)
            field.setSingleStep(step)
            if decimals:
                field.setDecimals(decimals)
            field.setValue(DEFAULT["model"][key])
            self.fields[key] = field
            form.addRow(label, field)
        self.fields["units"].valueChanged.connect(
            lambda number: self._update_topology(number)
        )
        controls.addWidget(group_config)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("▶ 启动训练")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self._submit)
        buttons.addWidget(self.run_button)
        self.stop_button = QPushButton("■ 停止")
        self.stop_button.setObjectName("Warning")
        self.stop_button.clicked.connect(self._stop)
        buttons.addWidget(self.stop_button)
        controls.addLayout(buttons)
        self.load_button = QPushButton("载入选中实验参数")
        self.load_button.clicked.connect(self._load_config)
        controls.addWidget(self.load_button)
        controls.addStretch()
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(300)
        left_scroll.setMaximumWidth(395)
        main.addWidget(left_scroll)

        right_panel = QSplitter(Qt.Orientation.Vertical)
        right_panel.setChildrenCollapsible(False)
        top_tabs = QTabWidget()
        pg.setConfigOptions(
            antialias=True, background=COLORS["ink"], foreground=COLORS["muted"]
        )
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time step")
        self.plot.setLabel("left", "Signal")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        top_tabs.addTab(self.plot, "预测曲线")
        self.heatmap = pg.ImageView(view=pg.PlotItem())
        top_tabs.addTab(self.heatmap, "神经元状态热力图")
        self.metrics_panel = QTextEdit()
        self.metrics_panel.setReadOnly(True)
        self.metrics_panel.setPlaceholderText("选择实验，查看指标与配置。")
        top_tabs.addTab(self.metrics_panel, "实验指标")
        right_panel.addWidget(top_tabs)

        history = QWidget()
        history_layout = QVBoxLayout(history)
        history_layout.setContentsMargins(10, 6, 10, 10)
        label_row = QHBoxLayout()
        label_row.addWidget(header_label("最近实验", "双击实验可查看预测及内部状态"))
        label_row.addStretch()
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self._refresh)
        label_row.addWidget(refresh_button)
        history_layout.addLayout(label_row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "实验 ID", "状态", "进度", "测试 RMSE", "运行时间 / 秒",
        ])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.cellDoubleClicked.connect(self._select_row)
        self.table.cellClicked.connect(self._select_row)
        history_layout.addWidget(self.table)
        self.log_text = QLabel("模型尚未训练。")
        self.log_text.setWordWrap(True)
        history_layout.addWidget(self.log_text)
        right_panel.addWidget(history)
        right_panel.setSizes([470, 245])
        main.addWidget(right_panel)
        main.setSizes([340, 900])
        root_layout.addWidget(main)
        self.setCentralWidget(root)

    def _update_topology(self, number):
        self.topology.units = number
        self.topology.update()

    def _notify(self, message):
        self.log_text.setText(message)

    def _refresh_datasets(self):
        old = self.dataset_box.currentData()
        old_id = old.get("dataset_id") if isinstance(old, dict) else None
        self.dataset_box.clear()
        self.dataset_box.addItem("合成信号 · Sine", {"kind": "sine"})
        self.dataset_box.addItem("混沌序列 · Mackey–Glass",
                                 {"kind": "mackey_glass"})
        try:
            for item in self.client.workspace.list_datasets():
                self.dataset_box.addItem(
                    f"CSV · {item['name'][:30]}",
                    {"kind": "csv", "dataset_id": item["id"]}
                )
                if item["id"] == old_id:
                    self.dataset_box.setCurrentIndex(self.dataset_box.count()-1)
        except Exception as exc:
            self._notify(f"读取数据集失败: {exc}")

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入数值时间序列", "", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            item = self.client.workspace.import_csv(path)
            self._refresh_datasets()
            for i in range(self.dataset_box.count()):
                info = self.dataset_box.itemData(i)
                if info and info.get("dataset_id") == item["id"]:
                    self.dataset_box.setCurrentIndex(i)
            self._notify(f"已导入: {item['name']} · 请选择一个数值列（自动选择第一列）。")
        except Exception as exc:
            QMessageBox.critical(self, "CSV 导入失败", str(exc))

    def _config(self):
        cfg = copy.deepcopy(DEFAULT)
        data = self.dataset_box.currentData() or {"kind": "sine"}
        cfg["dataset"]["kind"] = data["kind"]
        cfg["dataset"]["length"] = self.length_box.value()
        if data["kind"] == "csv":
            cfg["dataset"]["dataset_id"] = data["dataset_id"]
            cfg["dataset"]["column"] = self.column_box.currentData()
        for key, field in self.fields.items():
            cfg["model"][key] = field.value()
        return cfg

    def _submit(self):
        try:
            row = self.client.submit(self._config())
            self.current_run = row["id"]
            self.last_plotted = None
            self._notify(f"实验已加入队列：{self.current_run}")
            self._refresh()
        except Exception as exc:
            QMessageBox.critical(self, "训练提交失败", str(exc))

    def _stop(self):
        if not self.current_run:
            QMessageBox.information(self, "提示", "请先选择一个实验。")
            return
        try:
            self.client.stop(self.current_run)
            self._notify(f"已请求停止实验：{self.current_run}")
        except Exception as exc:
            QMessageBox.warning(self, "停止失败", str(exc))

    def _select_row(self, row, _column):
        item = self.table.item(row, 0)
        if not item:
            return
        self.current_run = item.text()
        self.last_plotted = None
        self._show_selected()

    def _show_selected(self):
        if not self.current_run:
            return
        try:
            run = self.client.run(self.current_run)
            self.status_text.setText(
                f"● {run['status']} · {round(run['progress']*100)}%"
            )
            if run["metrics"]:
                self.metrics_panel.setPlainText(
                    json.dumps(run["metrics"], indent=2, ensure_ascii=False)
                )
            else:
                self.metrics_panel.setPlainText(
                    json.dumps(run["config"], indent=2, ensure_ascii=False)
                )
            if run["status"] == "finished" and self.last_plotted != run["id"]:
                self._draw(run["id"])
                self.last_plotted = run["id"]
            if run["status"] in ("failed", "interrupted", "cancelled"):
                self._notify(f"{run['status']}: {run.get('error')}")
        except ServiceError as exc:
            self._notify(str(exc))

    def _draw(self, run_id):
        t = self.client.traces(run_id, limit=700, units=32)
        x = np.asarray(t["timestep"], dtype=float)
        truth = np.asarray(t["truth"], dtype=float)
        pred = np.asarray([
            np.nan if v is None else v for v in t["prediction"]
        ], dtype=float)
        tags = np.asarray(t["split"], dtype=int)
        self.plot.clear()
        self.plot.plot(x, truth, pen=pg.mkPen(COLORS["teal"], width=1.7))
        val_pred = np.where(tags == 1, pred, np.nan)
        test_pred = np.where(tags == 2, pred, np.nan)
        self.plot.plot(x, val_pred, pen=pg.mkPen(COLORS["muted"], width=1.6))
        self.plot.plot(x, test_pred, pen=pg.mkPen(COLORS["orange"], width=2.1))
        states = np.asarray(t["states"], dtype=float)
        if states.size:
            # ImageView uses image coordinates for time and unit index.
            self.heatmap.setImage(
                states, autoLevels=False, levels=(-1.0, 1.0)
            )
        self._notify(
            f"实验 {run_id} 完成 · 青色=真实值，灰色=验证预测，橙色=测试预测"
        )

    def _load_config(self):
        if not self.current_run:
            return
        try:
            cfg = self.client.run(self.current_run)["config"]
            data = cfg["dataset"]
            for i in range(self.dataset_box.count()):
                current = self.dataset_box.itemData(i)
                if current["kind"] == data["kind"] and (
                    data["kind"] != "csv" or
                    current.get("dataset_id") == data["dataset_id"]
                ):
                    self.dataset_box.setCurrentIndex(i)
                    break
            self.length_box.setValue(data["length"])
            for key, field in self.fields.items():
                field.setValue(cfg["model"][key])
            self._notify("实验参数已载入，修改后可启动新实验。")
        except Exception as exc:
            QMessageBox.warning(self, "参数载入失败", str(exc))

    def _refresh(self):
        try:
            runs = self.client.runs(40)
            signature = tuple(
                (r["id"], r["status"], int(r["progress"]*100)) for r in runs
            )
            if signature != self.last_runs_signature:
                self.last_runs_signature = signature
                self.table.setRowCount(len(runs))
                for i, run in enumerate(runs):
                    metrics = run.get("metrics") or {}
                    scores = metrics.get("scores", {}).get("test", {})
                    cells = [
                        run["id"], run["status"],
                        f"{int(run['progress']*100)}%",
                        str(scores.get("rmse", "—")),
                        str(metrics.get("runtime_seconds", "—")),
                    ]
                    for j, value in enumerate(cells):
                        self.table.setItem(i, j, QTableWidgetItem(value))
                if runs and not self.current_run:
                    self.current_run = runs[0]["id"]
            if self.current_run:
                self._show_selected()
            else:
                self.status_text.setText("● 服务已连接 · 等待实验")
        except Exception as exc:
            self.status_text.setText("● 服务连接异常")
            self._notify(str(exc))

def start_gui(client):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    window = MainWindow(client)
    window.show()
    return app.exec()
