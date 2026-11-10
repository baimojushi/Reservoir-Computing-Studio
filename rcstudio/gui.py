"""RC Studio Material 3-inspired desktop: local ESN and EBrush research only.

This UI never opens serial ports or cameras and never invokes E-Brush services.
All training is scheduled in the separate RC Studio loopback worker service.
"""
from __future__ import annotations

from .learning_support import anchor as _help_anchor

import copy
import json
import math
import sys

import numpy as np

try:
    from PySide6.QtCore import Qt, QTimer, QRectF, QSettings, QThread, Signal
    from PySide6.QtGui import QColor, QBrush, QFont, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
        QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox,
        QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QTabWidget,
        QTextEdit, QVBoxLayout, QWidget, QHeaderView, QAbstractItemView,
    )
    import pyqtgraph as pg
except ImportError as exc:
    raise RuntimeError("GUI needs PySide6 and pyqtgraph. Install pip install -e '.[gui]'") from exc

from .config import DEFAULT
from .learning_support.integration import install as install_learning, submitted as learning_submitted
from .client import ServiceError
from .ebrush import DEFAULT_EBRUSH, PROJECT_ID, import_source, make_demo

PALETTE = {
    "light": {
        "bg": "#F5F7FA", "panel": "#FFFFFF", "panel_alt": "#EEF2F5",
        "ink": "#202A34", "sub": "#60717D", "stroke": "#D7E0E4",
        "primary": "#006B67", "on_primary": "#FFFFFF", "accent": "#C46F47",
        "blue": "#487EAA", "warning": "#A24E44", "plot": "#FFFFFF",
    },
    "dark": {
        "bg": "#111A20", "panel": "#1B2932", "panel_alt": "#26353E",
        "ink": "#E5EFF0", "sub": "#A5B8BF", "stroke": "#3B4C55",
        "primary": "#8CD5CC", "on_primary": "#133230", "accent": "#F3AA7B",
        "blue": "#8BB7DD", "warning": "#E7A097", "plot": "#152129",
    },
}


def stylesheet(mode):
    p = PALETTE[mode]
    return f"""
    QWidget {{ font-family: 'Microsoft YaHei UI', 'Segoe UI'; color: {p['ink']};
               font-size: 12px; background: {p['bg']}; }}
    QFrame#TopBar {{ background: {p['panel']}; border-bottom: 1px solid {p['stroke']}; }}
    QLabel {{ background: transparent; }}
    QFrame#MetricCard, QFrame#SectionCard {{ border-radius: 13px; background: {p['panel']};
                                           border: 1px solid {p['stroke']}; }}
    QFrame#ProjectNote {{ border-radius: 11px; background: {p['panel_alt']}; }}
    QLabel#Brand {{ font-size: 23px; font-weight: 750; color: {p['ink']}; }}
    QLabel#Caption {{ font-size: 11px; color: {p['sub']}; }}
    QLabel#Metric {{ font-size: 21px; font-weight: 750; color: {p['primary']}; }}
    QLabel#SectionTitle {{ font-size: 15px; font-weight: 700; }}
    QGroupBox {{ border: 1px solid {p['stroke']}; border-radius: 12px;
                 margin-top: 13px; padding: 13px 9px 9px; background: {p['panel']}; font-weight: 650; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 5px;
                        background: {p['panel']}; }}
    QSpinBox, QDoubleSpinBox, QComboBox {{ background: {p['panel_alt']}; border-radius: 9px;
        border: 1px solid {p['stroke']}; padding: 7px 9px; min-height: 21px; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QPushButton {{ border: 1px solid {p['stroke']}; border-radius: 18px;
        padding: 8px 15px; background: {p['panel_alt']}; font-weight: 650; }}
    QPushButton:hover {{ border-color: {p['primary']}; }}
    QPushButton:disabled {{ color: {p['sub']}; }}
    QPushButton#Primary {{ background: {p['primary']}; color: {p['on_primary']};
        border-color: {p['primary']}; }}
    QPushButton#Danger {{ border-color: {p['warning']}; color: {p['warning']}; }}
    QProgressBar {{ background: {p['panel_alt']}; border: none; border-radius: 4px;
        min-height: 7px; max-height: 7px; text-align: center; }}
    QProgressBar::chunk {{ background: {p['primary']}; border-radius: 4px; }}
    QTabWidget::pane {{ border: 1px solid {p['stroke']}; border-radius: 12px;
                       background: {p['panel']}; }}
    QTabBar::tab {{ background: {p['panel_alt']}; color: {p['sub']};
        padding: 11px 17px; margin-right: 5px; border-radius: 9px; }}
    QTabBar::tab:selected {{ background: {p['panel']}; color: {p['primary']};
        font-weight: 700; }}
    QTableWidget, QTextEdit {{ background: {p['panel']}; alternate-background-color: {p['panel_alt']};
        border: 1px solid {p['stroke']}; border-radius: 9px; gridline-color: {p['stroke']}; }}
    QHeaderView::section {{ background: {p['panel_alt']}; padding: 8px;
        border: none; border-bottom: 1px solid {p['stroke']}; font-weight: 650; }}
    QScrollArea {{ border: none; background: {p['bg']}; }}
    QSplitter::handle {{ background: {p['bg']}; }}
    """


class TopologyCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.units = 120
        self.inputs = 1
        self.theme = "light"
        self.setMinimumHeight(147)
        self.setMaximumHeight(180)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        colors = PALETTE[self.theme]
        w, h = self.width(), self.height()
        x0, x1, x2 = 37, w * .51, w-42
        pts = []
        for i in range(13):
            phi = i * 2.39996
            rad = 18 + (i % 5) * 9
            pts.append((x1 + rad*math.cos(phi), h*.46 + rad*math.sin(phi)))
        p.setPen(QPen(QColor(colors["stroke"]), 1.3))
        for y in np.linspace(28, h-43, min(self.inputs, 5)):
            for v in (0, 4, 8):
                p.drawLine(int(x0), int(y), int(pts[v][0]), int(pts[v][1]))
        for i, (x,y) in enumerate(pts):
            for target in ((i+3)%13, (i+5)%13):
                xx, yy = pts[target]
                p.drawLine(int(x), int(y), int(xx), int(yy))
            if i % 4 == 0:
                p.drawLine(int(x), int(y), int(x2), int(h*.46))
        p.setPen(QPen(QColor(colors["panel"]), .8))
        p.setBrush(QBrush(QColor(colors["primary"])))
        for y in np.linspace(28, h-43, min(self.inputs, 5)):
            p.drawEllipse(QRectF(x0-5, y-5, 10, 10))
        for i, (x,y) in enumerate(pts):
            p.setBrush(QBrush(QColor(colors["accent"] if i in (1, 7) else colors["blue"])))
            p.drawEllipse(QRectF(x-5, y-5, 10, 10))
        p.setBrush(QBrush(QColor(colors["accent"])))
        p.drawEllipse(QRectF(x2-7, h*.46-7, 14, 14))
        p.setPen(QColor(colors["sub"]))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(0, h-24, 80, 18), Qt.AlignmentFlag.AlignCenter,
                   f"INPUT · {self.inputs}")
        p.drawText(QRectF(x1-77, h-24, 154, 18), Qt.AlignmentFlag.AlignCenter,
                   f"RESERVOIR · {self.units}")
        p.drawText(QRectF(w-84, h-24, 84, 18), Qt.AlignmentFlag.AlignCenter, "READOUT")
        p.end()


class ImportWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, workspace, path, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self.path = path

    def run(self):
        try:
            self.completed.emit(import_source(self.workspace, self.path))
        except Exception as exc:
            self.failed.emit(str(exc))


def make_caption(text):
    label = QLabel(text)
    label.setObjectName("Caption")
    label.setWordWrap(True)
    return label


def make_metric(label):
    card = QFrame()
    card.setObjectName("MetricCard")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(16, 11, 16, 11)
    lay.setSpacing(2)
    lay.addWidget(make_caption(label))
    value = QLabel("—")
    value.setObjectName("Metric")
    lay.addWidget(value)
    return card, value


class MainWindow(QMainWindow):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.current_run = None
        self.last_plotted = None
        self.last_signature = None
        self.latest_sessions = []
        self.import_worker = None
        self.settings = QSettings("RCStudio", "Desktop")
        mode = self.settings.value("theme", "light")
        self.theme = mode if mode in PALETTE else "light"
        self.setWindowTitle("RC Studio · Reservoir Computing Research")
        self.setMinimumSize(1050, 680)
        self.resize(1360, 840)
        self._build()
        self._set_theme(self.theme)
        self._refresh_datasets()
        self._refresh_sessions()
        self._refresh()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(1800)
        self.learning = install_learning(self)

    def _build(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        top = QFrame()
        top.setObjectName("TopBar")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(21, 12, 24, 12)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        logo = QLabel("◈  RC Studio")
        logo.setObjectName("Brand")
        brand.addWidget(logo)
        brand.addWidget(make_caption("Reservoir Computing  /  本地研究工作台"))
        top_lay.addLayout(brand)
        top_lay.addStretch()
        self.connection = QLabel("● 正在连接")
        top_lay.addWidget(self.connection)
        self.theme_button = _help_anchor(QPushButton("深色模式"), 'ui.theme')
        self.theme_button.clicked.connect(self._toggle_theme)
        top_lay.addWidget(self.theme_button)
        self.learning_button = _help_anchor(QPushButton("帮助 · 新手入门"), "ui.help")
        self.learning_button.clicked.connect(lambda: self.learning.open_center(courses=True))
        top_lay.addWidget(self.learning_button)
        outer.addWidget(top)

        main = QSplitter(Qt.Orientation.Horizontal)
        main.setChildrenCollapsible(False)
        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setMinimumWidth(338)
        sidebar.setMaximumWidth(440)
        controls_widget = QWidget()
        controls = QVBoxLayout(controls_widget)
        controls.setContentsMargins(17, 15, 10, 16)
        controls.setSpacing(13)

        picker = _help_anchor(QGroupBox("研究项目"), 'rc.project')
        picker_box = QVBoxLayout(picker)
        self.project_box = _help_anchor(QComboBox(), 'rc.project')
        self.project_box.addItem("通用 ESN · 时间序列预测", "generic_esn")
        self.project_box.addItem("EBrush Motion Predictor V1", PROJECT_ID)
        self.project_box.currentIndexChanged.connect(self._project_changed)
        picker_box.addWidget(self.project_box)
        self.project_caption = make_caption("单变量信号建模 · NumPy/SciPy ESN")
        picker_box.addWidget(self.project_caption)
        controls.addWidget(picker)

        self.dataset_stack = QStackedWidget()
        self.dataset_stack.addWidget(self._generic_dataset_panel())
        self.dataset_stack.addWidget(self._ebrush_panel())
        controls.addWidget(self.dataset_stack)

        diagram = _help_anchor(QGroupBox("模型拓扑 · ESN"), 'rc.topology')
        diagram_lay = QVBoxLayout(diagram)
        self.topology = TopologyCanvas()
        diagram_lay.addWidget(self.topology)
        controls.addWidget(diagram)

        configuration = _help_anchor(QGroupBox("模型参数"), 'rc.parameters')
        form = QFormLayout(configuration)
        form.setSpacing(9)
        self.fields = {}
        specs = [
            ("units", "储备池神经元", 10, 350, 1, 0),
            ("spectral_radius", "谱半径", .01, 2.0, .05, 3),
            ("leak_rate", "泄漏率", .01, 1.0, .05, 3),
            ("density", "连接密度", .01, 1.0, .01, 3),
            ("input_scaling", "输入缩放", .001, 10., .05, 3),
            ("ridge", "岭回归系数", 0., 100., .00001, 7),
            ("washout", "Washout 预热", 0, 2000, 10, 0),
            ("horizon", "预测步长", 1, 100, 1, 0),
            ("seed", "随机种子", 0, 2147483647, 1, 0),
        ]
        for key, label, low, high, step, decimals in specs:
            field = _help_anchor(QDoubleSpinBox(), 'rc.model.' + key) if decimals else _help_anchor(QSpinBox(), 'rc.model.' + key)
            field.setRange(low, high)
            field.setSingleStep(step)
            if decimals:
                field.setDecimals(decimals)
            field.setValue(DEFAULT["model"][key])
            form.addRow(label, field)
            self.fields[key] = field
        self.fields["units"].valueChanged.connect(self._update_topology)
        controls.addWidget(configuration)
        self.horizon_caption = make_caption("预测步长单位：数据采样点（非毫秒）")
        controls.addWidget(self.horizon_caption)

        action_row = QHBoxLayout()
        self.run_button = _help_anchor(QPushButton("▶  启动训练"), 'rc.run')
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self._submit)
        action_row.addWidget(self.run_button)
        self.stop_button = _help_anchor(QPushButton("■  停止"), 'rc.stop')
        self.stop_button.setObjectName("Danger")
        self.stop_button.clicked.connect(self._stop)
        action_row.addWidget(self.stop_button)
        controls.addLayout(action_row)
        self.load_button = _help_anchor(QPushButton("载入所选实验参数"), 'rc.load')
        self.load_button.clicked.connect(self._load_config)
        controls.addWidget(self.load_button)
        controls.addWidget(make_caption("计算服务：127.0.0.1:8766 · RC Studio 独占\n设备、相机、标定由 E-Brush Studio 独立负责。"))
        controls.addStretch()
        sidebar.setWidget(controls_widget)
        main.addWidget(sidebar)

        right = QWidget()
        right_box = QVBoxLayout(right)
        right_box.setContentsMargins(8, 15, 18, 14)
        right_box.setSpacing(13)
        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.metric_values = {}
        for key, caption in [("rmse", "测试 RMSE"), ("baseline", "持续性基线 RMSE"),
                             ("horizon", "预测时距"), ("records", "独立采集会话")]:
            card, label = make_metric(caption)
            topic = {"rmse": "rc.rmse", "baseline": "rc.baseline",
                     "horizon": "rc.horizon_metric", "records": "rc.records_metric"}[key]
            _help_anchor(card, topic)
            _help_anchor(label, topic)
            self.metric_values[key] = label
            metrics.addWidget(card)
        right_box.addLayout(metrics)

        self.progress = _help_anchor(QProgressBar(), 'rc.progress')
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        right_box.addWidget(self.progress)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        self.tabs = _help_anchor(QTabWidget(), 'ui.tabs')
        self.plot = _help_anchor(pg.PlotWidget(), 'rc.plot')
        self.plot.setLabel("bottom", "Time step")
        self.plot.setLabel("left", "Signal / angular speed")
        self.plot.showGrid(x=True, y=True, alpha=.18)
        self.tabs.addTab(self.plot, "预测曲线")
        self.heatmap = _help_anchor(pg.ImageView(view=pg.PlotItem()), 'rc.states')
        self.tabs.addTab(self.heatmap, "Reservoir states")
        self.metrics_panel = _help_anchor(QTextEdit(), 'rc.diagnostics')
        self.metrics_panel.setReadOnly(True)
        self.metrics_panel.setPlaceholderText("选择实验可查看验证/测试指标与配置。")
        self.tabs.addTab(self.metrics_panel, "实验诊断")
        split.addWidget(self.tabs)

        history = QFrame()
        history.setObjectName("SectionCard")
        hbox = QVBoxLayout(history)
        hrow = QHBoxLayout()
        htitle = QLabel("最近实验")
        htitle.setObjectName("SectionTitle")
        hrow.addWidget(htitle)
        hrow.addStretch()
        refresh = _help_anchor(QPushButton("↻ 刷新"), 'rc.refresh')
        refresh.clicked.connect(self._refresh)
        hrow.addWidget(refresh)
        hbox.addLayout(hrow)
        self.table = _help_anchor(QTableWidget(0, 6), 'rc.history')
        self.table.setHorizontalHeaderLabels(["实验 ID", "项目", "状态", "进度", "Test RMSE", "运行 / s"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._select_row)
        self.table.cellDoubleClicked.connect(self._select_row)
        hbox.addWidget(self.table)
        split.addWidget(history)
        split.setSizes([490, 265])
        right_box.addWidget(split)
        self.log_text = make_caption("等待实验数据。")
        right_box.addWidget(self.log_text)
        main.addWidget(right)
        main.setSizes([370, 980])
        outer.addWidget(main)
        self.setCentralWidget(root)

    def _generic_dataset_panel(self):
        box = _help_anchor(QGroupBox("数据来源 · 通用 ESN"), 'rc.dataset')
        lay = QVBoxLayout(box)
        self.dataset_box = _help_anchor(QComboBox(), 'rc.dataset')
        self.dataset_box.currentIndexChanged.connect(self._dataset_changed)
        lay.addWidget(self.dataset_box)
        form = QFormLayout()
        self.length_box = _help_anchor(QSpinBox(), 'rc.length')
        self.length_box.setRange(150, 20000)
        self.length_box.setSingleStep(250)
        self.length_box.setValue(DEFAULT["dataset"]["length"])
        form.addRow("序列长度", self.length_box)
        self.column_box = _help_anchor(QComboBox(), 'rc.column')
        self.column_box.addItem("自动识别数值列", None)
        form.addRow("CSV 数据列", self.column_box)
        lay.addLayout(form)
        b = _help_anchor(QPushButton("＋ 导入 CSV 数据集"), 'rc.import_csv')
        b.clicked.connect(self._import_csv)
        lay.addWidget(b)
        lay.addWidget(make_caption("CSV 仅用于通用单变量 ESN；电子毛笔六通道数据请使用下方独立项目。"))
        return box

    def _ebrush_panel(self):
        box = _help_anchor(QGroupBox("E-Brush 数据 · 离线研究"), 'rc.ebrush')
        lay = QVBoxLayout(box)
        note = QFrame()
        note.setObjectName("ProjectNote")
        nlay = QVBoxLayout(note)
        nlay.addWidget(make_caption(
            "IMU 六通道 → 未来 50 ms 角速度\n"
            "数据来自采集文件副本或交接清单；不连接笔端、相机或串口。"))
        lay.addWidget(note)
        row = QHBoxLayout()
        self.import_ebrush_button = _help_anchor(QPushButton("＋ 导入采集文件"), 'rc.ebrush_import')
        self.import_ebrush_button.clicked.connect(self._import_ebrush)
        row.addWidget(self.import_ebrush_button)
        demo = _help_anchor(QPushButton("生成模拟样本"), 'rc.demo')
        demo.clicked.connect(self._make_demo)
        row.addWidget(demo)
        lay.addLayout(row)
        self.session_summary = make_caption("尚未导入 E-Brush 数据")
        lay.addWidget(self.session_summary)
        self.session_table = _help_anchor(QTableWidget(0, 3), 'rc.sessions')
        self.session_table.setHorizontalHeaderLabels(["训练", "来源", "帧数"])
        self.session_table.verticalHeader().hide()
        self.session_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.session_table.setMinimumHeight(182)
        self.session_table.setMaximumHeight(260)
        self.session_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.session_table)
        reload = _help_anchor(QPushButton("刷新采集会话"), 'rc.sessions_refresh')
        reload.clicked.connect(self._refresh_sessions)
        lay.addWidget(reload)
        lay.addWidget(make_caption("需 ≥3 份独立记录才能训练/验证/测试；同一长日志切出的片段只算一份记录。"))
        return box

    def _set_theme(self, theme):
        self.theme = theme
        QApplication.instance().setStyleSheet(stylesheet(theme))
        self.settings.setValue("theme", theme)
        self.theme_button.setText("☾ 深色模式" if theme == "light" else "☼ 浅色模式")
        self.topology.theme = theme
        self.topology.update()
        p = PALETTE[theme]
        self.plot.setBackground(p["plot"])
        self.plot.getPlotItem().getAxis("bottom").setTextPen(p["sub"])
        self.plot.getPlotItem().getAxis("left").setTextPen(p["sub"])
        if self.last_plotted and self.last_plotted == self.current_run:
            self._draw(self.current_run)
        if hasattr(self, "learning"):
            self.learning.set_theme(theme)

    def _toggle_theme(self):
        self._set_theme("dark" if self.theme == "light" else "light")

    def _project_changed(self):
        ebrush = self.project_box.currentData() == PROJECT_ID
        self.dataset_stack.setCurrentIndex(1 if ebrush else 0)
        self.topology.inputs = 6 if ebrush else 1
        defaults = DEFAULT_EBRUSH["model"] if ebrush else DEFAULT["model"]
        for key in ("units", "washout", "horizon"):
            self.fields[key].setValue(defaults[key])
        self.topology.update()
        self.project_caption.setText(
            "EBrush · 独立采集记录划分 · 六通道 ESN" if ebrush else
            "单变量观测输入预测 · 合成或 CSV 数据"
        )
        self.horizon_caption.setText(
            "200 Hz 名义采样率：horizon=10 对应 50 ms" if ebrush else
            "预测步长单位：数据采样点（非毫秒）"
        )

    def _update_topology(self, number):
        self.topology.units = number
        self.topology.update()

    def _notify(self, text):
        self.log_text.setText(text)

    def _refresh_datasets(self):
        old = self.dataset_box.currentData() or {}
        old_id = old.get("dataset_id")
        self.dataset_box.blockSignals(True)
        self.dataset_box.clear()
        self.dataset_box.addItem("合成信号 · Sine", {"kind": "sine"})
        self.dataset_box.addItem("混沌信号 · Mackey–Glass", {"kind": "mackey_glass"})
        try:
            for meta in self.client.workspace.list_datasets():
                self.dataset_box.addItem(f"CSV · {meta['name'][:40]}",
                                         {"kind": "csv", "dataset_id": meta["id"],
                                          "columns": meta["columns"]})
                if meta["id"] == old_id:
                    self.dataset_box.setCurrentIndex(self.dataset_box.count() - 1)
        finally:
            self.dataset_box.blockSignals(False)
        self._dataset_changed()

    def _dataset_changed(self):
        info = self.dataset_box.currentData() or {}
        is_csv = info.get("kind") == "csv"
        self.length_box.setEnabled(not is_csv)
        self.column_box.setEnabled(is_csv)
        self.column_box.clear()
        self.column_box.addItem("自动识别数值列", None)
        if is_csv:
            for field in info.get("columns", []):
                self.column_box.addItem(field, field)

    def _import_csv(self):
        name, _ = QFileDialog.getOpenFileName(self, "导入数值信号 CSV", "", "CSV (*.csv)")
        if not name:
            return
        try:
            result = self.client.workspace.import_csv(name)
            self._refresh_datasets()
            for i in range(self.dataset_box.count()):
                if (self.dataset_box.itemData(i) or {}).get("dataset_id") == result["id"]:
                    self.dataset_box.setCurrentIndex(i)
            self._notify(f"已导入 CSV：{result['name']}")
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    def _import_ebrush(self):
        name, _ = QFileDialog.getOpenFileName(self, "导入 E-Brush 采集或移交清单", "",
                                               "EBrush files (*.brlog *.csv *.zip *.json)")
        if not name:
            return
        if self.import_worker and self.import_worker.isRunning():
            return
        self.import_ebrush_button.setEnabled(False)
        self._notify("正在离线校验与解析 E-Brush 数据；不会操作设备。")
        self.import_worker = ImportWorker(self.client.workspace, name, self)
        self.import_worker.completed.connect(self._import_finished)
        self.import_worker.failed.connect(self._import_failed)
        self.import_worker.finished.connect(lambda: self.import_ebrush_button.setEnabled(True))
        self.import_worker.start()

    def _import_finished(self, result):
        self._refresh_sessions()
        self._notify(f"导入完成：{result['imported_episodes']} 个有效片段；"
                     f"独立记录 {result['independent_recordings']} 份。")
        if result["imported_episodes"] and hasattr(self, "learning"):
            self.learning.notify("rc.ebrush.imported")
        if not result["imported_episodes"]:
            QMessageBox.warning(self, "没有可训练片段", "已解析源文件，但没有长度足够的连续 IMU 数据片段。")

    def _import_failed(self, message):
        QMessageBox.warning(self, "E-Brush 离线导入失败", message)
        self._notify(message)

    def _make_demo(self):
        try:
            result = make_demo(self.client.workspace)
            self._refresh_sessions()
            self._notify(f"生成 {len(result['session_ids'])} 个合成采集会话。"
                         "仅用于软件自测，不可替代真实设备结果。")
            if hasattr(self, "learning"):
                self.learning.notify("rc.demo.created")
        except Exception as exc:
            QMessageBox.warning(self, "演示数据失败", str(exc))

    def _refresh_sessions(self):
        try:
            entries = self.client.ebrush_sessions()
        except Exception as exc:
            self.session_summary.setText(f"会话读取失败：{exc}")
            return
        previous = {}
        for i in range(self.session_table.rowCount()):
            item = self.session_table.item(i, 0)
            if item:
                previous[item.data(Qt.ItemDataRole.UserRole)] = item.checkState() == Qt.CheckState.Checked
        self.latest_sessions = entries
        self.session_table.setRowCount(len(entries))
        have_real = any(not m["synthetic"] for m in entries)
        for row, meta in enumerate(entries):
            item = QTableWidgetItem("")
            item.setData(Qt.ItemDataRole.UserRole, meta["id"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = previous.get(meta["id"], not meta["synthetic"] if have_real else True)
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.session_table.setItem(row, 0, item)
            label = meta["source"].split("/")[-1]
            if meta["synthetic"]:
                label += " · DEMO"
            self.session_table.setItem(row, 1, QTableWidgetItem(label[:34]))
            self.session_table.setItem(row, 2, QTableWidgetItem(str(meta["samples"])))
        num_groups = len(set(m["recording_id"] for m in entries))
        self.session_summary.setText(f"{len(entries)} 个片段 · {num_groups} 份独立记录\n"
                                     "勾选会话控制本轮训练数据集。")

    def _selected_sessions(self):
        result = []
        for i in range(self.session_table.rowCount()):
            item = self.session_table.item(i, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                result.append(item.data(Qt.ItemDataRole.UserRole))
        return result

    def _config(self):
        ebrush = self.project_box.currentData() == PROJECT_ID
        cfg = copy.deepcopy(DEFAULT_EBRUSH if ebrush else DEFAULT)
        if ebrush:
            cfg["dataset"]["session_ids"] = self._selected_sessions()
            if not cfg["dataset"]["session_ids"]:
                raise ValueError("请至少勾选三个不同采集文件对应的会话。")
        else:
            info = self.dataset_box.currentData() or {"kind": "sine"}
            cfg["dataset"]["kind"] = info["kind"]
            cfg["dataset"]["length"] = self.length_box.value()
            if info["kind"] == "csv":
                cfg["dataset"]["dataset_id"] = info["dataset_id"]
                cfg["dataset"]["column"] = self.column_box.currentData()
        for key, widget in self.fields.items():
            cfg["model"][key] = widget.value()
        return cfg

    def _submit(self):
        try:
            cfg = self._config()
            row = (self.client.submit_ebrush(cfg) if cfg.get("project") == PROJECT_ID
                   else self.client.submit(cfg))
            self.current_run = row["id"]
            learning_submitted(self, cfg, self.current_run)
            self.last_plotted = None
            self._notify(f"已提交到 RC Studio 的独立训练服务：{self.current_run}")
            self._refresh()
        except Exception as exc:
            QMessageBox.warning(self, "实验提交失败", str(exc))

    def _stop(self):
        if not self.current_run:
            QMessageBox.information(self, "请选择实验", "请先选择一个训练实验。")
            return
        try:
            response = self.client.stop(self.current_run)
            self._notify(f"停止请求已发出：{self.current_run}")
        except Exception as exc:
            QMessageBox.warning(self, "停止失败", str(exc))

    def _select_row(self, row, _column):
        item = self.table.item(row, 0)
        if item:
            self.current_run = item.text()
            self.last_plotted = None
            self._show_selected()

    def _load_config(self):
        if not self.current_run:
            return
        try:
            cfg = self.client.run(self.current_run)["config"]
            ebrush = cfg.get("project") == PROJECT_ID
            self.project_box.setCurrentIndex(1 if ebrush else 0)
            if ebrush:
                wanted = set(cfg["dataset"]["session_ids"])
                if not wanted:
                    wanted = {m["id"] for m in self.latest_sessions}
                for row in range(self.session_table.rowCount()):
                    item = self.session_table.item(row, 0)
                    item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in wanted
                                       else Qt.CheckState.Unchecked)
            else:
                for i in range(self.dataset_box.count()):
                    data = self.dataset_box.itemData(i)
                    if data["kind"] == cfg["dataset"]["kind"] and (
                        data["kind"] != "csv" or data["dataset_id"] == cfg["dataset"]["dataset_id"]
                    ):
                        self.dataset_box.setCurrentIndex(i)
                        break
                self.length_box.setValue(cfg["dataset"]["length"])
                col = cfg["dataset"].get("column")
                for i in range(self.column_box.count()):
                    if self.column_box.itemData(i) == col:
                        self.column_box.setCurrentIndex(i)
                        break
            for key, widget in self.fields.items():
                widget.setValue(cfg["model"][key])
            self._notify("参数已载入，可以修改后创建独立的新实验。")
        except Exception as exc:
            QMessageBox.warning(self, "载入失败", str(exc))

    def _show_selected(self):
        if not self.current_run:
            return
        try:
            row = self.client.run(self.current_run)
            self._learning_last_run = row  # Reuse the normal UI refresh; teaching does not poll HTTP.
            self.progress.setValue(max(0, min(100, int(row["progress"]*100))))
            self.connection.setText(f"● 127.0.0.1:{self.client.port} · {row['status']}")
            metric = row.get("metrics") or {}
            scores = metric.get("scores", {})
            test = scores.get("test", {})
            baseline = scores.get("persistence_baseline_test", {})
            self.metric_values["rmse"].setText(str(test.get("rmse", "—")))
            self.metric_values["baseline"].setText(str(baseline.get("rmse", "—")))
            is_ebrush = row["config"].get("project") == PROJECT_ID
            self.metric_values["horizon"].setText(
                f"{metric.get('nominal_horizon_ms', row['config']['model']['horizon']*5)} ms" if is_ebrush
                else f"{row['config']['model']['horizon']} step"
            )
            self.metric_values["records"].setText(
                str(metric.get("data", {}).get("independent_recordings", "—")) if is_ebrush else "—"
            )
            self.metrics_panel.setPlainText(json.dumps({
                "metrics": metric, "config": row["config"],
                "status": row["status"], "error": row.get("error")},
                ensure_ascii=False, indent=2))
            if row["status"] == "finished" and self.last_plotted != row["id"]:
                self._draw(row["id"])
                self.last_plotted = row["id"]
            if row["status"] in ("failed", "interrupted", "cancelled"):
                self._notify(f"{row['status']}: {row.get('error')}")
        except ServiceError as exc:
            self._notify(str(exc))

    def _draw(self, run_id):
        traces = self.client.traces(run_id, limit=750, units=32)
        x = np.asarray(traces["timestep"], dtype=float)
        true = np.asarray(traces["truth"], dtype=float)
        pred = np.asarray([np.nan if p is None else p for p in traces["prediction"]], dtype=float)
        tags = np.asarray(traces["split"], dtype=int)
        p = PALETTE[self.theme]
        self.plot.clear()
        self.plot.plot(x, true, pen=pg.mkPen(p["primary"], width=1.9), name="Truth")
        self.plot.plot(x, np.where(tags == 1, pred, np.nan), pen=pg.mkPen(p["blue"], width=1.5), name="Validation")
        self.plot.plot(x, np.where(tags == 2, pred, np.nan), pen=pg.mkPen(p["accent"], width=2), name="Test")
        states = np.asarray(traces["states"], dtype=float)
        if states.size:
            self.heatmap.setImage(states, autoLevels=False, levels=(-1.0, 1.0))
        self._notify(f"实验 {run_id}：绿色真实值 · 蓝色验证预测 · 橙色测试预测")

    def _refresh(self):
        try:
            rows = self.client.runs(60)
            signature = tuple((r["id"], r["status"], int(r["progress"]*100)) for r in rows)
            if signature != self.last_signature:
                self.last_signature = signature
                self.table.setRowCount(len(rows))
                for i, row in enumerate(rows):
                    metric = row.get("metrics") or {}
                    value = (metric.get("scores") or {}).get("test", {}).get("rmse", "—")
                    cells = [row["id"], "EBrush" if row["config"].get("project") == PROJECT_ID else "ESN",
                             row["status"], f"{int(row['progress']*100)}%",
                             str(value), str(metric.get("runtime_seconds", "—"))]
                    for j, value in enumerate(cells):
                        self.table.setItem(i, j, QTableWidgetItem(value))
                if rows and not self.current_run:
                    self.current_run = rows[0]["id"]
            if self.current_run:
                self._show_selected()
            else:
                self.connection.setText(f"● 127.0.0.1:{self.client.port} · 就绪")
        except Exception as exc:
            self.connection.setText("● RC 服务连接异常")
            self._notify(str(exc))

    def closeEvent(self, event):
        if hasattr(self, "learning"):
            self.learning.shutdown()
        super().closeEvent(event)


def start_gui(client):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    pg.setConfigOptions(antialias=True)
    window = MainWindow(client)
    window.show()
    return app.exec()
