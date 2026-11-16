"""RC Studio V0.6: one-pool, one-timeline intuitive ESN laboratory (Qt UI)."""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QBrush
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from .learning_support import anchor as _help_anchor
from .intuitive_lab import simulate_sandbox, stable_positions
from .episode_time import align_to_target, episode_breaks
from .sketch_math import interpolate_stroke


LIGHT = {
    "bg": "#F6F8F6", "panel": "#FFFFFF", "soft": "#EEF3F0",
    "ink": "#24343A", "sub": "#6C7E7D", "stroke": "#DCE6E1",
    "positive": "#268E82", "negative": "#C97855", "amber": "#D5A04F",
    "blue": "#5B8EAF", "washout": "#F2E8D8",
}
DARK = {
    "bg": "#111C20", "panel": "#1B2A2E", "soft": "#24373A",
    "ink": "#E6F0EC", "sub": "#A5B8B2", "stroke": "#3B5352",
    "positive": "#6CC6B7", "negative": "#E39A77", "amber": "#E3B86D",
    "blue": "#91BBD5", "washout": "#514335",
}


def _np(values, *, nan=False):
    if values is None:
        return None
    if nan:
        return np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _blend(a: QColor, b: QColor, amount: float) -> QColor:
    t = max(0.0, min(1.0, float(amount)))
    return QColor(round(a.red() + (b.red() - a.red()) * t),
                  round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t))


class InputSketchPad(QWidget):
    """Small hand-drawn input strip; mouse gestures only edit the sandbox signal."""
    changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(190, 54)
        self.setMaximumHeight(62)
        x = np.linspace(0.0, 1.0, 220)
        self.values = 0.55 * np.sin(2.0 * math.pi * x * 2.2)
        self._drawing = False
        self._last_index = None
        self._last_value = None
        self.theme = "light"

    def set_theme(self, theme):
        self.theme = theme if theme in ("light", "dark") else "light"
        self.update()

    def set_values(self, values, emit=False):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) >= 2 and np.isfinite(values).all():
            self.values = np.clip(np.interp(np.linspace(0, 1, 220),
                                            np.linspace(0, 1, len(values)), values), -1, 1)
            self.update()
            if emit:
                self.changed.emit(self.values.copy())

    def _point(self, event):
        rect = QRectF(6, 6, max(1, self.width() - 12), max(1, self.height() - 12))
        pos = event.position()
        index = int(np.clip((pos.x() - rect.left()) / rect.width(), 0, 1) * (len(self.values) - 1))
        value = float(np.clip((rect.center().y() - pos.y()) / (rect.height() * 0.46), -1, 1))
        return index, value

    def _write_point(self, index, value):
        interpolate_stroke(self.values, self._last_index, self._last_value, index, value)
        self._last_index = index
        self._last_value = value
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._last_index = None
            self._last_value = None
            self._write_point(*self._point(event))

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._write_point(*self._point(event))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._write_point(*self._point(event))
            self._drawing = False
            self._last_index = None
            self._last_value = None
            self.changed.emit(self.values.copy())

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = LIGHT if self.theme == "light" else DARK
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(QColor(pal["stroke"]), 1))
        p.setBrush(QColor(pal["panel"]))
        p.drawRoundedRect(rect, 8, 8)
        plot = rect.adjusted(7, 6, -7, -6)
        p.setPen(QPen(QColor(pal["stroke"]), 1, Qt.PenStyle.DotLine))
        p.drawLine(QPointF(plot.left(), plot.center().y()), QPointF(plot.right(), plot.center().y()))
        path = QPainterPath()
        for i, value in enumerate(self.values):
            x = plot.left() + plot.width() * i / max(1, len(self.values) - 1)
            y = plot.center().y() - float(value) * plot.height() * 0.43
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        p.setPen(QPen(QColor(pal["positive"]), 1.8))
        p.drawPath(path)
        p.end()


class ReservoirFieldCanvas(QWidget):
    """One reservoir body with fixed node identity and invariant visual semantics."""
    node_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(440)
        self.payload = None
        self.pos = 0.0
        self.phase = 0.0
        self.lens = "dynamics"
        self.theme = "light"
        self.selected = None
        self.zoom = 1.0
        self.message = "在自由练习场中调整参数，观察一个储备池如何响应输入。"
        self._geometry = None

    @property
    def palette(self):
        return LIGHT if self.theme == "light" else DARK

    def set_payload(self, payload, keep_view=False):
        """Swap the data. ``keep_view`` keeps time, selection and zoom for live parameter edits."""
        previous = self.payload
        self.payload = payload
        same_shape = (keep_view and previous is not None and payload is not None
                      and payload["states"].shape == previous["states"].shape)
        if not same_shape:
            self.pos = 0.0
            self.selected = None
            self.zoom = 1.0
        self.update()

    def set_message(self, message):
        self.message = message
        self.update()

    def set_lens(self, lens):
        self.lens = lens
        self.update()

    def set_theme(self, theme):
        self.theme = theme if theme in ("light", "dark") else "light"
        self.update()

    def set_time(self, pos, phase=None):
        if self.payload is None:
            return
        self.pos = float(np.clip(pos, 0, len(self.payload["states"]) - 1))
        if phase is not None:
            self.phase = float(phase)
        self.update()

    def jump(self, pos):
        self.set_time(pos)

    def _state(self):
        arr = self.payload["states"]
        i0 = int(self.pos)
        i1 = min(len(arr) - 1, i0 + 1)
        bounds = self.payload.get("episode_boundary")
        if self.payload.get("legacy_unsafe") or (bounds is not None and bounds[i1]):
            i1 = i0  # a reset separates two episodes; no cross-episode interpolation
        f = self.pos - i0 if i1 != i0 else 0.0
        return arr[i0] * (1.0 - f) + arr[i1] * f

    def _series_at(self, key):
        arr = self.payload.get(key)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            i0 = int(self.pos)
            i1 = min(len(arr) - 1, i0 + 1)
            bounds = self.payload.get("episode_boundary")
            if self.payload.get("legacy_unsafe") or (bounds is not None and bounds[i1]):
                i1 = i0
            f = self.pos - i0 if i1 != i0 else 0.0
            return arr[i0] * (1.0 - f) + arr[i1] * f
        return None

    @staticmethod
    def _font(size=10, bold=False):
        font = QFont("Microsoft YaHei UI")
        font.setPixelSize(size)
        font.setBold(bold)
        return font

    def _draw_series(self, p, values, rect, color, width=1.7, *, dashed=False, lo=None, hi=None, breaks=None):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        if not len(finite):
            return
        lo = float(np.min(finite)) if lo is None else float(lo)
        hi = float(np.max(finite)) if hi is None else float(hi)
        if hi - lo < 1e-9:
            lo -= 1.0
            hi += 1.0
        path = QPainterPath()
        pen_down = False
        for i, value in enumerate(values):
            if not np.isfinite(value):
                pen_down = False
                continue
            if breaks is not None and breaks[i]:
                pen_down = False
            x = rect.left() + rect.width() * i / max(1, len(values) - 1)
            y = rect.bottom() - (float(value) - lo) / (hi - lo) * rect.height()
            if pen_down:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                pen_down = True
        pen = QPen(QColor(color), width)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def _draw_cursor(self, p, rect, count, top=0):
        x = rect.left() + rect.width() * self.pos / max(1, count - 1)
        p.setPen(QPen(QColor(self.palette["ink"]), 1.1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x, top), QPointF(x, rect.bottom()))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(pal["bg"]))
        if self.payload is None:
            p.setPen(QColor(pal["sub"]))
            p.setFont(self._font(13))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            p.end()
            return

        payload = self.payload
        states = payload["states"]
        n_steps, units = states.shape
        idx = min(n_steps - 1, max(0, int(self.pos)))
        state = self._state()
        positions = np.asarray(payload["positions"], dtype=np.float64)
        src = np.asarray(payload.get("edge_src", []), dtype=np.int64)
        dst = np.asarray(payload.get("edge_dst", []), dtype=np.int64)
        edge_w = np.asarray(payload.get("edge_weights", []), dtype=np.float64)

        # Shared timeline: input above, the single reservoir in the middle, result below.
        input_rect = QRectF(28, 38, max(20, w - 56), 48)
        pool_rect = QRectF(42, 112, max(240, w - 290), max(180, h - 218))
        output_rect = QRectF(28, h - 76, max(20, w - 56), 46)
        p.setFont(self._font(10, True))
        p.setPen(QColor(pal["ink"]))
        p.drawText(QPointF(28, 22), payload.get("input_title", "输入信号"))
        p.drawText(QPointF(pool_rect.left(), 101), "ONE RESERVOIR  ·  一个储备池")
        output_title = payload.get("output_title", "状态与输出")
        if payload.get("kind") == "sandbox":
            output_title = ("示范读出 · 目标未来输入 / 预测值" if self.lens in ("readout", "evaluation")
                            else "状态响应 · 初态差异 / 输入扰动影响")
        p.drawText(QPointF(output_rect.left(), output_rect.top() - 8), output_title)
        p.setFont(self._font(9))
        p.setPen(QColor(pal["sub"]))
        method = payload.get("edge_method", "")
        shown = int(payload.get("edge_shown", len(src)))
        total = int(payload.get("edge_total", len(src)))
        if method == "largest_absolute_weight":
            edge_text = f"显示 |W| 最大的 {shown:,} / {total:,} 条边"
        elif method == "uniform_random_sample":
            edge_text = f"均匀抽样 {shown:,} / {total:,} 条边"
        else:
            edge_text = f"显示 {shown:,} / {total:,} 条非零边"
        p.drawText(QPointF(pool_rect.right() - 215, 101), edge_text)

        # Input trace and shared playhead.
        p.setPen(QPen(QColor(pal["stroke"]), 1))
        p.drawRoundedRect(input_rect, 7, 7)
        inputs = np.asarray(payload.get("inputs", np.zeros(n_steps)), dtype=np.float64)
        if not payload.get("legacy_unsafe"):
            self._draw_series(p, inputs, input_rect.adjusted(5, 7, -5, -7), pal["blue"], 1.8,
                              breaks=payload.get("episode_boundary"))
        self._draw_cursor(p, input_rect, len(inputs), input_rect.top())
        boundaries = payload.get("episode_boundary")
        if boundaries is not None:
            for i in np.flatnonzero(boundaries):
                for strip in (input_rect,):
                    x = strip.left() + strip.width() * i / max(1, n_steps - 1)
                    p.setPen(QPen(QColor(pal["amber"]), 1.2, Qt.PenStyle.DashLine))
                    p.drawLine(QPointF(x, strip.top()), QPointF(x, strip.bottom()))


        # A single field. Positions are independent of topology, node rank and echo depth.
        p.setPen(QPen(QColor(pal["stroke"]), 1.0))
        p.setBrush(QColor(pal["panel"]))
        p.drawRoundedRect(pool_rect, 18, 18)
        center = pool_rect.center()
        rx = pool_rect.width() * 0.43 * self.zoom
        ry = pool_rect.height() * 0.40 * self.zoom
        sx = center.x() + positions[:, 0] * rx
        sy = center.y() + positions[:, 1] * ry
        self._geometry = (sx, sy)

        # Static recurrent weights: line hue encodes sign; width/opacity encode |weight|.
        if len(src):
            scale = float(np.max(np.abs(edge_w))) or 1.0
            for a, b, weight in zip(src.tolist(), dst.tolist(), edge_w.tolist()):
                strength = abs(float(weight)) / scale
                base = pal["positive"] if weight >= 0 else pal["negative"]
                color = QColor(base)
                boost = 2.2 if self.lens == "topology" else 1.0     # the topology lens foregrounds wiring
                color.setAlpha(max(14, min(230, int((18 + 72 * strength) * boost))))
                p.setPen(QPen(color, 0.45 + 1.05 * strength))
                p.drawLine(QPointF(float(sx[a]), float(sy[a])), QPointF(float(sx[b]), float(sy[b])))

        # Input stream is a distinct boundary marker; it is not confused with W_rec edges.
        input_anchor = QPointF(pool_rect.left() + 10, center.y())
        p.setPen(QPen(QColor(pal["blue"]), 1.5))
        p.setBrush(QColor(pal["panel"]))
        p.drawEllipse(input_anchor, 5.5, 5.5)
        p.drawLine(QPointF(input_anchor.x() - 17, input_anchor.y()),
                   QPointF(input_anchor.x() - 7, input_anchor.y()))
        p.drawLine(QPointF(input_anchor.x() - 10, input_anchor.y() - 3),
                   QPointF(input_anchor.x() - 7, input_anchor.y()))
        p.drawLine(QPointF(input_anchor.x() - 10, input_anchor.y() + 3),
                   QPointF(input_anchor.x() - 7, input_anchor.y()))
        p.setFont(self._font(8, True))
        p.setPen(QColor(pal["sub"]))
        p.drawText(QPointF(input_anchor.x() - 16, input_anchor.y() + 19), "INPUT")

        # Readout overlay: node fill remains the state value in every lens.
        readout_nodes = payload.get("readout_nodes")
        contribution = None
        readout_anchor = QPointF(pool_rect.right() - 10, center.y())
        if readout_nodes is not None:
            weights = np.asarray(readout_nodes, dtype=np.float64).reshape(-1)
            if len(weights) == units:
                contribution = weights * state * float(payload.get("output_scale", 1.0))
        if self.lens == "readout" and contribution is not None:
            norm = float(np.max(np.abs(contribution))) or 1.0
            for i, value in enumerate(contribution):
                strength = abs(float(value)) / norm
                if strength < 0.015:
                    continue
                color = pal["positive"] if value >= 0 else pal["negative"]
                p.setPen(QPen(QColor(color), 0.6 + 2.5 * strength))
                p.drawLine(QPointF(float(sx[i]), float(sy[i])), readout_anchor)
            if payload.get("direct_input_available", False):
                raw_inputs = np.asarray(payload.get("inputs", []), dtype=np.float64)
                means = np.asarray(payload.get("input_mean", [0.0]), dtype=np.float64)
                stds = np.asarray(payload.get("input_std", [1.0]), dtype=np.float64)
                bypass = np.asarray(payload.get("bypass", []), dtype=np.float64)
                if idx < len(raw_inputs) and len(means) and len(stds) and len(bypass):
                    normalized = (raw_inputs[idx] - means[0]) / max(abs(stds[0]), 1e-12)
                    direct = float(bypass[0] * normalized * payload.get("output_scale", 1.0))
                    color = pal["positive"] if direct >= 0 else pal["negative"]
                    p.setPen(QPen(QColor(color), 0.8 + min(2.8, abs(direct) / max(norm, 1e-8) * 1.2)))
                    p.drawLine(input_anchor, readout_anchor)
            p.setPen(QPen(QColor(pal["positive"]), 1.5))
            p.setBrush(QColor(pal["panel"]))
            p.drawEllipse(readout_anchor, 7.5, 7.5)
            p.setFont(self._font(8, True))
            p.setPen(QColor(pal["sub"]))
            p.drawText(QPointF(readout_anchor.x() - 22, readout_anchor.y() + 21), "READOUT")

        # Dynamic packets encode the current recurrent drive W_ij x_j(t-1).
        if self.lens == "dynamics" and len(src):
            previous = states[max(0, idx - 1)]
            drive = edge_w * previous[src]
            order = np.argsort(-np.abs(drive))[:min(70, len(drive))]
            for rank, edge_i in enumerate(order.tolist()):
                a, b = int(src[edge_i]), int(dst[edge_i])
                phase = (self.phase * 0.75 + rank * 0.6180339) % 1.0
                x = float(sx[a] + (sx[b] - sx[a]) * phase)
                y = float(sy[a] + (sy[b] - sy[a]) * phase)
                color = pal["positive"] if drive[edge_i] >= 0 else pal["negative"]
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(color))
                p.drawEllipse(QPointF(x, y), 2.1, 2.1)

        # Perturbation ring is an overlay; the node fill continues to encode x_i(t).
        delta = payload.get("delta_states")
        if self.lens == "perturbation" and delta is not None:
            delta = np.asarray(delta, dtype=np.float64)
            if delta.ndim == 2 and len(delta) and delta.shape[1] == units:
                peak = np.max(np.abs(delta), axis=0)
                if payload.get("delta_profile_mode") == "peak":
                    magnitude = peak
                else:
                    di = min(len(delta) - 1, int(self.pos))
                    magnitude = np.abs(delta[di])
                # One scale for the whole response: rings shrink as the perturbation fades.
                # (V0.6 rescaled every frame, which kept rings at full size while the true
                # difference had decayed by orders of magnitude.)
                norm = float(np.percentile(peak, 95))
            else:
                magnitude = np.zeros(units, dtype=np.float64)
                norm = 1.0
            norm = max(norm, 1e-8)
            for i, value in enumerate(magnitude):
                strength = min(1.0, float(value) / norm)
                if strength < 0.025:
                    continue
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(QColor(pal["amber"]), 0.8 + 2.2 * strength))
                p.drawEllipse(QPointF(float(sx[i]), float(sy[i])), 6.5 + 5.0 * strength,
                                     6.5 + 5.0 * strength)

        # Node fill is invariant: signed current state x_i(t), with a fixed zero centre.
        neutral = QColor(pal["soft"])
        positive = QColor(pal["positive"])
        negative = QColor(pal["negative"])
        order = np.argsort(np.sum(positions * positions, axis=1))
        for i in order.tolist():
            value = float(np.clip(state[i], -1.0, 1.0))
            color = _blend(neutral, positive if value >= 0 else negative, abs(value) ** 0.72)
            p.setPen(QPen(QColor(pal["panel"]), 1.0))
            p.setBrush(color)
            p.drawEllipse(QPointF(float(sx[i]), float(sy[i])), 5.0, 5.0)
        if self.selected is not None and 0 <= self.selected < units:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(pal["ink"]), 2.0))
            p.drawEllipse(QPointF(float(sx[self.selected]), float(sy[self.selected])), 9.5, 9.5)

        # Compact invariant legend and model note.
        p.setFont(self._font(9))
        p.setPen(QColor(pal["sub"]))
        p.drawText(QPointF(pool_rect.left() + 12, pool_rect.bottom() - 9),
                   "节点填充 = 当前状态 xᵢ(t)    ·    线色 = 固定权重正/负    ·    外环 = 扰动差 Δx")

        # Output/evaluation strip shares the same time cursor as the input.
        p.setPen(QPen(QColor(pal["stroke"]), 1))
        p.setBrush(QColor(pal["panel"]))
        p.drawRoundedRect(output_rect, 7, 7)
        if payload.get("kind") == "sandbox" and self.lens not in ("readout", "evaluation"):
            maxv = max(float(np.max(payload["initial_gap"])),
                       float(np.max(payload["pulse_effect"])), 1e-6)
            self._draw_series(p, payload["initial_gap"], output_rect.adjusted(5, 6, -5, -6),
                              pal["blue"], 1.5, lo=0, hi=maxv)
            self._draw_series(p, payload["pulse_effect"], output_rect.adjusted(5, 6, -5, -6),
                              pal["amber"], 1.5, lo=0, hi=maxv)
            p.setFont(self._font(8))
            p.setPen(QColor(pal["blue"]))
            p.drawText(QPointF(output_rect.left() + 8, output_rect.top() + 11), "初态差异")
            p.setPen(QColor(pal["amber"]))
            p.drawText(QPointF(output_rect.left() + 78, output_rect.top() + 11), "输入脉冲影响")
        else:
            truth = np.asarray(payload.get("aligned_truth", payload.get("truth")), dtype=np.float64)
            pred = np.asarray(payload.get("aligned_prediction", payload.get("prediction")), dtype=np.float64)
            split = payload.get("split")
            if split is not None:
                split = np.asarray(split, dtype=np.int64)
                n = min(len(split), len(truth))
                if n:
                    starts = np.r_[0, np.flatnonzero(np.diff(split[:n]) != 0) + 1, n]
                    for a, b in zip(starts[:-1], starts[1:]):
                        if split[a] in (1, 2):
                            x0 = output_rect.left() + output_rect.width() * a / max(1, n - 1)
                            x1 = output_rect.left() + output_rect.width() * max(a, b - 1) / max(1, n - 1)
                            color = QColor(pal["blue"] if split[a] == 1 else pal["amber"])
                            color.setAlpha(22)
                            p.fillRect(QRectF(x0, output_rect.top() + 1, max(1, x1 - x0),
                                              output_rect.height() - 2), color)
            # Washout is identified by missing predictions, independently of data split.
            missing_prediction = ~np.isfinite(pred)
            if len(missing_prediction) and np.any(missing_prediction):
                starts = np.r_[0, np.flatnonzero(np.diff(missing_prediction.astype(np.int8)) != 0) + 1,
                               len(missing_prediction)]
                for a, b in zip(starts[:-1], starts[1:]):
                    if missing_prediction[a]:
                        x0 = output_rect.left() + output_rect.width() * a / max(1, len(pred) - 1)
                        x1 = output_rect.left() + output_rect.width() * max(a, b - 1) / max(1, len(pred) - 1)
                        color = QColor(pal["washout"])
                        color.setAlpha(105)
                        p.fillRect(QRectF(x0, output_rect.top() + 1, max(1, x1 - x0),
                                          output_rect.height() - 2), color)
            finite = np.concatenate([truth[np.isfinite(truth)], pred[np.isfinite(pred)]])
            if len(finite):
                lo, hi = float(finite.min()), float(finite.max())
                if hi - lo < 1e-9:
                    lo -= 1
                    hi += 1
                plot = output_rect.adjusted(5, 6, -5, -6)
                self._draw_series(p, truth, plot, pal["ink"], 1.25, lo=lo, hi=hi,
                                  breaks=payload.get("episode_boundary"))
                self._draw_series(p, pred, plot, pal["positive"], 1.8, dashed=True, lo=lo, hi=hi,
                                  breaks=payload.get("episode_boundary"))
                if len(pred) and not np.isfinite(pred[0]):
                    p.setFont(self._font(8))
                    p.setPen(QColor(pal["sub"]))
                    p.drawText(QPointF(plot.left() + 5, plot.bottom() - 4),
                               "预热 / Washout：状态建立中，暂不计入预测")
            p.setFont(self._font(8))
            p.setPen(QColor(pal["sub"]))
            p.drawText(QPointF(output_rect.right() - 155, output_rect.top() + 11),
                       f"预测目标：本段 t + {payload.get('horizon', '?')} 步")
            if payload.get("legacy_unsafe"):
                p.setPen(QColor(pal["warning"]) if "warning" in pal else QColor(pal["negative"]))
                p.drawText(QPointF(output_rect.left() + 8, output_rect.center().y()),
                           "旧录制缺少边界元数据：已隐藏预测，重新训练后可查看")
        self._draw_cursor(p, output_rect, n_steps, output_rect.top())
        if boundaries is not None:
            p.setFont(self._font(8))
            for i in np.flatnonzero(boundaries):
                x = output_rect.left() + output_rect.width() * i / max(1, n_steps - 1)
                p.setPen(QPen(QColor(pal["amber"]), 1.2, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(x, output_rect.top()), QPointF(x, output_rect.bottom()))
                p.setPen(QColor(pal["sub"]))
                p.drawText(QPointF(x + 4, output_rect.top() + 13), "新录制")

        # Current sample details: selected node and actual model diagnostics.
        detail = QRectF(pool_rect.right() + 14, pool_rect.top() + 8,
                        max(120, w - pool_rect.right() - 32), pool_rect.height() - 16)
        p.setPen(QPen(QColor(pal["stroke"]), 1))
        p.setBrush(QColor(pal["panel"]))
        p.drawRoundedRect(detail, 10, 10)
        p.setFont(self._font(10, True))
        p.setPen(QColor(pal["ink"]))
        p.drawText(QPointF(detail.left() + 10, detail.top() + 20),
                   "当前状态" if payload.get("kind") == "run" else "实验观察")
        p.setFont(self._font(9))
        p.setPen(QColor(pal["sub"]))
        lines = []
        if payload.get("kind") == "sandbox":
            lines = [f"谱半径 ρ = {payload['spectral_radius']:.2f}",
                     f"泄漏率 α = {payload['leak_rate']:.2f}",
                     f"输入缩放 = {payload['input_scaling']:.2f}",
                     f"近饱和节点 = {payload['saturation_fraction']:.1%}",
                     f"初态差异末/初 = {payload['initial_gap_ratio']:.3f}",
                     payload["status"]]
            if self.lens == "readout" and contribution is not None:
                reservoir_part = float(np.sum(contribution))
                raw_inputs = np.asarray(payload.get("inputs", []), dtype=np.float64)
                means = np.asarray(payload.get("input_mean", [0.0]), dtype=np.float64)
                stds = np.asarray(payload.get("input_std", [1.0]), dtype=np.float64)
                bypass = np.asarray(payload.get("bypass", []), dtype=np.float64)
                direct_part = 0.0
                if idx < len(raw_inputs) and len(means) and len(stds) and len(bypass):
                    normalized = (raw_inputs[idx] - means[0]) / max(abs(stds[0]), 1e-12)
                    direct_part = float(bypass[0] * normalized * payload.get("output_scale", 1.0))
                base_part = float(payload.get("output_offset", 0.0)
                                  + payload.get("readout_bias", 0.0) * payload.get("output_scale", 1.0))
                lines.extend([f"储备池项 Σwᵢxᵢ = {reservoir_part:+.4g}",
                              f"输入直连项 = {direct_part:+.4g}",
                              f"偏置 + 均值 = {base_part:+.4g}"])
                predicted = np.asarray(payload.get("prediction", []), dtype=np.float64)
                reconstructed = base_part + direct_part + reservoir_part
                if idx < len(predicted) and np.isfinite(predicted[idx]):
                    lines.append(f"读出重建差 = {reconstructed - predicted[idx]:+.3g}")
            if self.lens == "evaluation":
                lines.append(f"留出段 RMSE = {payload.get('test_rmse', float('nan')):.4g}")
                lines.append(f"示范预测时距 = {payload.get('horizon', '?')} 步")
                truth = np.asarray(payload.get("truth", []), dtype=np.float64)
                pred = np.asarray(payload.get("prediction", []), dtype=np.float64)
                if idx < len(truth) and idx < len(pred) and np.isfinite(truth[idx]) and np.isfinite(pred[idx]):
                    lines.append(f"当前预测误差 = {pred[idx] - truth[idx]:+.4g}")
                elif idx < len(pred) and not np.isfinite(pred[idx]):
                    lines.append("当前点处于预热区间")
            lines.append(payload["status_note"])
        else:
            value = float(state[self.selected]) if self.selected is not None else float(np.mean(state))
            lines = [f"实验 {str(payload.get('run_id', ''))[:12]}",
                     f"节点数 {units} · 输入通道 {payload.get('inputs_count', 1)}",
                     f"当前状态 {'x#' + str(self.selected) if self.selected is not None else '平均值'} = {value:+.3f}",
                     f"目标时距 {payload.get('horizon', '?')} 步",
                     payload.get("edge_note", ""),
                     payload.get("probe_note", "")]
            if self.lens == "readout" and contribution is not None:
                reservoir_part = float(np.sum(contribution))
                lines.append(f"储备池读出项 Σwᵢxᵢ = {reservoir_part:+.4g}")
                if payload.get("direct_input_available", False):
                    input_values = np.asarray(payload.get("inputs", []), dtype=np.float64)
                    means = np.asarray(payload.get("input_mean", [0.0]), dtype=np.float64)
                    stds = np.asarray(payload.get("input_std", [1.0]), dtype=np.float64)
                    bypass = np.asarray(payload.get("bypass", []), dtype=np.float64)
                    if idx < len(input_values) and len(bypass) and len(means) and len(stds):
                        normalized_input = (input_values[idx] - means[0]) / max(abs(stds[0]), 1e-12)
                        direct_part = float(bypass[0] * normalized_input * payload.get("output_scale", 1.0))
                        base_part = float(payload.get("output_offset", 0.0)
                                           + payload.get("readout_bias", 0.0) * payload.get("output_scale", 1.0))
                        lines.append(f"输入直连项 = {direct_part:+.4g}")
                        lines.append(f"偏置 + 均值 = {base_part:+.4g}")
                        prediction = np.asarray(payload.get("prediction", []), dtype=np.float64)
                        reconstructed = base_part + direct_part + reservoir_part
                        if idx < len(prediction) and np.isfinite(prediction[idx]):
                            lines.append(f"重建差 = {reconstructed - prediction[idx]:+.3g}")
                else:
                    lines.append("直连项未重建：保存的多通道轨迹不含完整输入向量")
            if self.lens == "evaluation":
                truth = np.asarray(payload.get("truth"), dtype=np.float64)
                pred = np.asarray(payload.get("prediction"), dtype=np.float64)
                if idx < len(truth) and idx < len(pred) and np.isfinite(truth[idx]) and np.isfinite(pred[idx]):
                    lines.append(f"当前误差 = {pred[idx] - truth[idx]:+.4g}")
                elif idx < len(pred) and not np.isfinite(pred[idx]):
                    lines.append("当前点处于预热区间")
        y = detail.top() + 42
        flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop) | int(Qt.TextFlag.TextWordWrap)
        metrics = QFontMetrics(p.font())
        for line in lines:
            if y + 18 > detail.bottom() - 6:
                break
            rect = QRectF(detail.left() + 10, y, detail.width() - 18,
                          min(42, detail.bottom() - y - 6))
            measured = metrics.boundingRect(rect.toRect(), flags, str(line))
            line_height = min(rect.height(), max(18, measured.height()))
            rect.setHeight(line_height)
            p.drawText(rect, flags, str(line))
            y += line_height + 3

        p.setFont(self._font(8))
        p.setPen(QColor(pal["sub"]))
        p.drawText(QPointF(28, h - 8), payload.get("time_label", "时间游标同步定位输入、状态与输出"))
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._geometry is not None:
            sx, sy = self._geometry
            distance = (sx - event.position().x()) ** 2 + (sy - event.position().y()) ** 2
            best = int(np.argmin(distance))
            if distance[best] < 16 ** 2:
                self.selected = None if self.selected == best else best
                if self.selected is not None:
                    self.node_selected.emit(int(self.selected))
                self.update()

    def wheelEvent(self, event):
        # Upper bound keeps every node inside the pool rectangle (0.43 * 1.12 < 0.5).
        self.zoom = float(np.clip(self.zoom * (1.0 + event.angleDelta().y() / 1200.0), 0.72, 1.12))
        self.update()
        event.accept()


class _LabLoader(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, client, run_id, start=0, with_structure=True, parent=None):
        super().__init__(parent)
        self.client = client
        self.run_id = run_id
        self.first_frame = int(start)
        self.with_structure = bool(with_structure)

    def run(self):
        try:
            result = {"run_id": self.run_id}
            if self.with_structure:
                result["structure"] = self.client.reservoir(self.run_id, edges=4000)
            result["frames"] = self.client.reservoir_frames(
                self.run_id, start=self.first_frame, count=1200)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class IntuitiveLabPanel(QWidget):
    """Native fallback: a numerically grounded, openly labeled learning path."""
    chapter_completed = Signal(str, str)
    is_web = False
    SPEEDS = ((0.25, "0.25×"), (0.5, "0.5×"), (1.0, "1×"), (2.0, "2×"), (4.0, "4×"))
    SEGMENTS = (("训练", "train"), ("验证", "validation"), ("测试", "test"))

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self.theme = "light"
        self.run_id = None
        self.run_payload = None
        self.structure = None
        self.frames = None
        self.loader = None
        self._pending = None
        self._request = None
        self.seed = 2026
        self.custom_input = None
        self.playing = True
        self.pos = 0.0
        self.phase = 0.0
        self._clock = time.perf_counter()
        self._pending_knob_event = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)
        self.fallback_notice = QLabel("原生实验室 · 可敲击、选节点、查看读出和调节参数")
        self.fallback_notice.setObjectName("Caption")
        self.fallback_notice.setWordWrap(True)
        root.addWidget(self.fallback_notice)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(7)
        toolbar.addWidget(QLabel("观察对象"))
        self.source_box = _help_anchor(QComboBox(), "rc.echo")
        self.source_box.addItem("自由练习 · 独立示范模型", "sandbox")
        self.source_box.addItem("当前训练实验", "run")
        self.source_box.model().item(1).setEnabled(False)
        self.source_box.currentIndexChanged.connect(self._source_changed)
        toolbar.addWidget(self.source_box)
        toolbar.addWidget(QLabel("观察透镜"))
        self.lens_box = _help_anchor(QComboBox(), "rc.echo")
        for label, key in (("动力学", "dynamics"), ("拓扑", "topology"),
                           ("扰动", "perturbation"), ("读出", "readout"),
                           ("评估", "evaluation")):
            self.lens_box.addItem(label, key)
        self.lens_box.currentIndexChanged.connect(self._lens_changed)
        toolbar.addWidget(self.lens_box)
        toolbar.addWidget(QLabel("解释"))
        self.level_box = _help_anchor(QComboBox(), "ui.help")
        self.level_box.addItem("直觉", "intuition")
        self.level_box.addItem("科研", "science")
        self.level_box.currentIndexChanged.connect(self._update_caption)
        toolbar.addWidget(self.level_box)
        toolbar.addStretch()
        self.segment_box = _help_anchor(QComboBox(), "rc.echo.playback")
        for label, key in self.SEGMENTS:
            self.segment_box.addItem(label, key)
        self.segment_box.setEnabled(False)
        self.segment_box.currentIndexChanged.connect(self._segment_changed)
        toolbar.addWidget(self.segment_box)
        self.play_button = _help_anchor(QPushButton("⏸ 暂停"), "rc.echo.playback")
        self.play_button.clicked.connect(self._toggle_play)
        toolbar.addWidget(self.play_button)
        self.speed_box = _help_anchor(QComboBox(), "rc.echo.playback")
        for factor, label in self.SPEEDS:
            self.speed_box.addItem(label, factor)
        self.speed_box.setCurrentIndex(2)
        toolbar.addWidget(self.speed_box)
        root.addLayout(toolbar)

        self.sandbox_bar = QFrame()
        self.sandbox_bar.setObjectName("SandboxBar")
        controls = QHBoxLayout(self.sandbox_bar)
        controls.setContentsMargins(10, 7, 10, 7)
        controls.setSpacing(9)
        controls.addWidget(QLabel("输入"))
        self.signal_box = _help_anchor(QComboBox(), "rc.dataset")
        for label, key in (("正弦", "sine"), ("脉冲", "pulse"), ("阶跃", "step"),
                           ("变频", "chirp"), ("手绘", "handdrawn")):
            self.signal_box.addItem(label, key)
        self.signal_box.currentIndexChanged.connect(self._signal_changed)
        controls.addWidget(self.signal_box)
        self.pulse_button = _help_anchor(QPushButton("敲一下"), "rc.echo")
        self.pulse_button.clicked.connect(self._poke_native)
        controls.addWidget(self.pulse_button)
        self.readout_button = _help_anchor(QPushButton("查看训练读出"), "rc.echo")
        self.readout_button.clicked.connect(self._show_readout_native)
        controls.addWidget(self.readout_button)
        self.radius_slider, self.radius_value = self._slider_control(
            controls, "谱半径 ρ", 5, 180, 90, "rc.model.spectral_radius")
        self.leak_slider, self.leak_value = self._slider_control(
            controls, "泄漏率 α", 2, 100, 35, "rc.model.leak_rate")
        self.input_slider, self.input_value = self._slider_control(
            controls, "输入缩放", 5, 300, 85, "rc.model.input_scaling")
        self.new_network = _help_anchor(QPushButton("换一组网络"), "rc.model.seed")
        self.new_network.clicked.connect(self._new_network)
        controls.addWidget(self.new_network)
        root.addWidget(self.sandbox_bar)

        sketch_row = QHBoxLayout()
        sketch_row.addWidget(QLabel("在波形区拖动绘制输入："))
        self.sketch_pad = _help_anchor(InputSketchPad(), "rc.dataset")
        self.sketch_pad.changed.connect(self._sketch_changed)
        sketch_row.addWidget(self.sketch_pad, 1)
        self.sketch_row = QWidget()
        self.sketch_row.setLayout(sketch_row)
        self.sketch_row.setVisible(False)
        root.addWidget(self.sketch_row)

        self.canvas = ReservoirFieldCanvas()
        self.canvas.node_selected.connect(self._node_selected_native)
        _help_anchor(self.canvas, "rc.echo")
        root.addWidget(self.canvas, 1)

        timeline = QHBoxLayout()
        self.time_slider = _help_anchor(QSlider(Qt.Orientation.Horizontal), "rc.echo.playback")
        self.time_slider.setRange(0, 219)
        self.time_slider.valueChanged.connect(self._time_changed)
        timeline.addWidget(self.time_slider, 1)
        self.time_label = QLabel("t = 0")
        self.time_label.setObjectName("Caption")
        self.time_label.setMinimumWidth(120)
        timeline.addWidget(self.time_label)
        root.addLayout(timeline)
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setObjectName("Caption")
        root.addWidget(self.caption)

        self._set_slider_label(self.radius_value, self.radius_slider.value() / 100, "ρ")
        self._set_slider_label(self.leak_value, self.leak_slider.value() / 100, "α")
        self._set_slider_label(self.input_value, self.input_slider.value() / 100, "×")
        self._param_timer = QTimer(self)
        self._param_timer.setSingleShot(True)
        self._param_timer.timeout.connect(lambda: self._rebuild_sandbox(keep_time=True))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self._rebuild_sandbox()
        self._update_caption()

    def _slider_control(self, layout, title, low, high, default, topic):
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(1)
        head = QHBoxLayout()
        head.addWidget(QLabel(title))
        value = QLabel("—")
        value.setObjectName("Caption")
        head.addWidget(value)
        col.addLayout(head)
        slider = _help_anchor(QSlider(Qt.Orientation.Horizontal), topic)
        slider.setRange(low, high)
        slider.setValue(default)
        slider.setFixedWidth(115)
        # V0.6 never connected these sliders, so dragging them did nothing.
        slider.valueChanged.connect(self._params_changed)
        col.addWidget(slider)
        layout.addWidget(box)
        return slider, value

    @staticmethod
    def _set_slider_label(label, value, suffix):
        label.setText(f"{value:.2f}{suffix}")

    def set_fallback_reason(self, reason):
        self.fallback_notice.setText("Web 实验室无法启动，已切换可操作的原生界面。原因：" + str(reason))

    def _poke_native(self):
        if self.source_box.currentData() != "sandbox":
            self.source_box.setCurrentIndex(0)
        self.signal_box.setCurrentIndex(self.signal_box.findData("pulse"))
        self._rebuild_sandbox()
        self.chapter_completed.emit("poke", "在原生实验室输入一次脉冲，已计算状态响应")

    def _show_readout_native(self):
        if self.source_box.currentData() != "sandbox":
            self.source_box.setCurrentIndex(0)
        self.lens_box.setCurrentIndex(self.lens_box.findData("readout"))
        self.chapter_completed.emit("readout", "查看示范网络已求解的岭回归读出与贡献")

    def _node_selected_native(self, node):
        if self.source_box.currentData() == "sandbox":
            self.chapter_completed.emit("nodes", f"查看节点 {node + 1} 的状态和连接")

    def _params_changed(self, _value=0):
        """Slider moved: update the labels now, re-simulate shortly (coalesces fast drags)."""
        if not hasattr(self, "input_slider"):
            return                      # still constructing the three sliders
        self._set_slider_label(self.radius_value, self.radius_slider.value() / 100.0, "")
        self._set_slider_label(self.leak_value, self.leak_slider.value() / 100.0, "")
        self._set_slider_label(self.input_value, self.input_slider.value() / 100.0, "×")
        self._pending_knob_event = True
        self._param_timer.start(60)

    def _new_network(self):
        self.seed += 1
        self._rebuild_sandbox()

    def _signal_changed(self):
        custom = self.signal_box.currentData() == "handdrawn"
        self.sketch_row.setVisible(custom and self.source_box.currentData() == "sandbox")
        self._rebuild_sandbox()

    def _sketch_changed(self, values):
        self.custom_input = np.asarray(values, dtype=np.float64)
        if self.signal_box.currentData() == "handdrawn":
            self._rebuild_sandbox()

    def _rebuild_sandbox(self, keep_time=False):
        if self.source_box.currentData() != "sandbox":
            return
        radius = self.radius_slider.value() / 100.0
        leak = self.leak_slider.value() / 100.0
        scaling = self.input_slider.value() / 100.0
        self._set_slider_label(self.radius_value, radius, "")
        self._set_slider_label(self.leak_value, leak, "")
        self._set_slider_label(self.input_value, scaling, "×")
        payload = simulate_sandbox(
            spectral_radius=radius, leak_rate=leak, input_scaling=scaling,
            signal=self.signal_box.currentData(), custom_input=self.custom_input,
            seed=self.seed, units=64, steps=220)
        payload["input_title"] = "输入 u(t) · 可选择预设波形或手绘"
        payload["output_title"] = "示范读出 · 目标未来输入 / 预测值"
        payload["time_label"] = "时间游标同步定位输入、状态与未来输入预测；读出以前 70% 拟合、后 30% 留出"
        payload["probe_note"] = payload["status_note"]
        resume = self.pos if keep_time and getattr(self, "_payload", None) is not None else 0.0
        self.canvas.set_payload(payload, keep_view=keep_time)
        self._payload = payload
        self._sync_timeline()
        self.pos = float(min(resume, len(payload["states"]) - 1))
        if self.pos:                     # a parameter edit must not send the viewer back to t = 0
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(int(self.pos))
            self.time_slider.blockSignals(False)
            self.time_label.setText(self._time_caption(int(self.pos)))
            self.canvas.jump(self.pos)
        self._update_caption()
        if self._pending_knob_event:
            self._pending_knob_event = False
            self.chapter_completed.emit("knobs", "调整旋钮并重新计算储备池状态")

    def _source_changed(self):
        if self.source_box.currentData() == "sandbox":
            self.sandbox_bar.setEnabled(True)
            self.sketch_row.setVisible(self.signal_box.currentData() == "handdrawn")
            self.segment_box.setEnabled(False)
            self._rebuild_sandbox()
        elif self.run_payload is not None:
            self.sandbox_bar.setEnabled(False)
            self.sketch_row.setVisible(False)
            self.segment_box.setEnabled(True)
            self._apply_run_payload()
        else:
            self.source_box.setCurrentIndex(0)

    def _lens_changed(self):
        key = self.lens_box.currentData()
        self.canvas.set_lens(key)
        self._update_caption()

    def _update_caption(self):
        lens = self.lens_box.currentData() or "dynamics"
        level = self.level_box.currentData() or "intuition"
        sandbox = self.source_box.currentData() == "sandbox"
        copy = {
            "dynamics": (
                "每个点都在响应输入：填充色始终表示当前状态，流动的小点表示循环连接正在传递的驱动。",
                "x(t)=(1−α)x(t−1)+α·tanh(W_rec x(t−1)+W_in u(t)+b)。节点填充固定编码 x_i(t)，动态粒子编码 W_ij x_j(t−1)。"),
            "topology": (
                "细线是储备池内部的固定连接。线色区分正负权重，线宽表示权重大小；节点位置只为排版。",
                "W_rec 在一次实验中保持固定。当前显示的边集合及抽样方式见画布标注；几何距离不表达权重或动力学相似度。"),
            "perturbation": (
                "把输入复制一份，在一个时刻轻轻推一下。节点外环显示两次运行的状态差，填充色仍表示原始状态。",
                "Δx_i(t)=x_i^perturbed(t)−x_i^baseline(t)。该响应依赖扰动、输入、初态与观察窗口，不能直接替代记忆容量或 ESP 检验。"),
            "readout": (
                "读出像调音台：每个节点按自己的读出系数贡献一部分输出。线条显示各节点当前贡献的方向与大小。",
                "ŷ(t+h)=b_out+w_uᵀu_norm(t)+w_xᵀx(t)。画布单独显示 w_x,i x_i(t)；多通道原始输入未保存在轨迹时，不重建直连项。"),
            "evaluation": (
                "沿同一条时间轴比较真实值和预测值。开头的空白是预热阶段，模型先建立内部状态再开始计分。",
                "目标是 y(t+h)。Washout 区间的预测值置空；真实值、预测值与数据划分共享同一时间游标。"),
        }
        self.caption.setText(copy[lens][0 if level == "intuition" else 1])
        if sandbox:
            self.caption.setText(self.caption.text() + "  当前为独立示范网络；示范读出以前 70% 输入片段拟合，后 30% 留出观察，不代表正式实验成绩。初态差异探针不构成 ESP 证明。")

    def set_theme(self, theme):
        self.theme = theme if theme in ("light", "dark") else "light"
        self.canvas.set_theme(self.theme)
        self.sketch_pad.set_theme(self.theme)
        pal = LIGHT if self.theme == "light" else DARK
        self.sandbox_bar.setStyleSheet(
            f"QFrame#SandboxBar {{ background: {pal['panel']}; border: 1px solid {pal['stroke']}; border-radius: 10px; }}")

    def show_run(self, run_id, focus=True):
        """``focus=False`` loads quietly: the practice sandbox stays on screen."""
        self._focus_run = bool(focus)
        if not run_id:
            return
        busy = self.loader is not None and self.loader.isRunning()
        if run_id == self.run_id and (self.run_payload is not None or busy or self._pending is not None):
            return
        self.run_id = run_id
        self.run_payload = None
        self.structure = None
        self.frames = None
        self.source_box.model().item(1).setEnabled(False)
        if self._focus_run or self.source_box.currentData() == "run":
            self.canvas.set_payload(None)
            self.canvas.set_message("正在读取真实实验的储备池状态…")
        self._pending = (run_id, 0, True)
        if self.isVisible():
            self._start_pending()

    def _start_pending(self):
        if self._pending is None or (self.loader is not None and self.loader.isRunning()):
            return
        run_id, start, with_structure = self._pending
        self._request = self._pending
        self._pending = None
        self.loader = _LabLoader(self.client, run_id, start, with_structure, self)
        self.loader.completed.connect(self._loaded)
        self.loader.failed.connect(self._load_failed)
        self.loader.finished.connect(self._loader_finished)
        self.loader.start()

    def _loader_finished(self):
        if self.loader is not None:
            self.loader.wait()
        if self._pending is not None:
            self._start_pending()

    def _load_failed(self, message):
        if self._request and self._request[0] == self.run_id:
            self.canvas.set_message(f"实验数据读取失败：{message}")

    def _loaded(self, result):
        if result.get("run_id") != self.run_id:
            return
        if "structure" in result:
            self.structure = result["structure"]
        self.frames = result["frames"]
        starts = self.frames.get("split_starts", {})
        if "structure" in result and starts.get("train") not in (None, 0):
            self._pending = (self.run_id, int(starts["train"]), False)
            return
        self._prepare_run_payload()
        self.source_box.model().item(1).setEnabled(True)
        self._update_segment_enabled()
        if self.source_box.currentData() == "run":
            self.segment_box.setEnabled(True)
            self._apply_run_payload()
        elif getattr(self, "_focus_run", True):
            self.source_box.setCurrentIndex(1)       # triggers _source_changed -> _apply_run_payload
        # otherwise stay in the sandbox; the run is now selectable under "观察对象"

    def _prepare_run_payload(self):
        fr, st = self.frames, self.structure
        if not fr or not st:
            return
        states = np.asarray(fr["states"], dtype=np.float32)
        units = states.shape[1]
        edges = st.get("edges", {})
        readout = st.get("readout", {})
        multi = int(st.get("inputs", 1)) > 1
        h = int(st.get("forecast", {}).get("horizon", 1))
        ebrush = st.get("forecast", {}).get("project") == "ebrush_motion_v1"
        known = bool(fr.get("has_episode_metadata") and "episode_id" in fr)
        legacy_unsafe = bool(ebrush and not known)
        ep = np.asarray(fr["episode_id"] if known else np.zeros(len(states)), dtype=np.int64)
        local = np.asarray(fr["local_step"] if known else
                           (np.asarray(fr.get("timestep", []), dtype=np.int64) - h), dtype=np.int64)
        bounds = episode_breaks(ep) if known else np.zeros(len(states), dtype=bool)
        raw_true = _np(fr.get("truth", []), nan=True)
        raw_pred = _np(fr.get("prediction", []), nan=True)
        aligned_true = (np.full(len(states), np.nan) if legacy_unsafe else
                        align_to_target(raw_true, ep, local, h))
        aligned_pred = (np.full(len(states), np.nan) if legacy_unsafe else
                        align_to_target(raw_pred, ep, local, h))
        self.run_payload = {
            "kind": "run", "run_id": self.run_id, "units": units,
            "episode_id": ep if known else None,
            "episode_local_step": local if known else None,
            "episode_boundary": bounds if known else None,
            "recording_id": fr.get("recording_id", []), "legacy_unsafe": legacy_unsafe,
            "aligned_truth": aligned_true, "aligned_prediction": aligned_pred,
            "positions": stable_positions(units, seed=202611 + units * 13),
            "edge_src": np.asarray(edges.get("src", []), dtype=np.int64),
            "edge_dst": np.asarray(edges.get("dst", []), dtype=np.int64),
            "edge_weights": np.asarray(edges.get("weight", []), dtype=np.float64),
            "edge_total": int(edges.get("total", len(edges.get("weight", [])))),
            "edge_shown": int(edges.get("shown", len(edges.get("weight", [])))),
            "edge_method": "largest_absolute_weight",
            "edge_note": "连接按 |W| 从大到小筛选；完整总数见上方。",
            "input_weights": np.asarray(st.get("input_weights", []), dtype=np.float64),
            "readout_nodes": np.asarray(readout.get("nodes", []), dtype=np.float64),
            "output_scale": float(readout.get("target_std", 1.0)),
            "output_offset": float(readout.get("target_mean", 0.0)),
            "readout_bias": float(readout.get("bias", 0.0)),
            "bypass": np.asarray(readout.get("bypass", []), dtype=np.float64),
            "input_mean": np.asarray(readout.get("input_mean", [0.0]), dtype=np.float64),
            "input_std": np.asarray(readout.get("input_std", [1.0]), dtype=np.float64),
            "direct_input_available": not multi,
            "inputs_count": int(st.get("inputs", 1)),
            "states": states,
            "inputs": _np(fr.get("inputs", []), nan=True),
            "truth": _np(fr.get("truth", []), nan=True),
            "prediction": _np(fr.get("prediction", []), nan=True),
            "split": np.asarray(fr.get("split", []), dtype=np.int64),
            "delta_states": np.asarray((st.get("impulse") or {}).get("states", []), dtype=np.float32),
            "delta_profile_mode": "peak",
            "horizon": int(st.get("forecast", {}).get("horizon", 1)),
            "washout": int(st.get("forecast", {}).get("washout", 0)),
            "input_title": ("输入摘要 · 已保存轨迹记录角速度幅值；六通道原始向量未随轨迹保存"
                            if multi else "输入信号 u(t)"),
            "output_title": "预测目标 y(t+h) · 真实值 / 预测值",
            "time_label": ("独立录制间状态重置、曲线断开；目标映射只在同一录制内；每段预热不计分"
                           if known else "旧 EBrush 轨迹未保存边界，已禁用可能误导的预测曲线" if legacy_unsafe
                           else "时间游标同步输入、状态与预测目标"),
            "probe_note": ("扰动外环显示标准化单次脉冲探针中各节点的峰值响应；多输入时各通道施加相同归一化脉冲。"
                           if multi else "扰动外环显示单次输入脉冲探针中各节点的峰值状态差。"),
        }

    def _apply_run_payload(self):
        if self.run_payload is None:
            return
        self._payload = self.run_payload
        self.canvas.set_payload(self._payload)
        self.canvas.set_lens(self.lens_box.currentData())
        self.pos = 0.0
        self.playing = False
        self.play_button.setText("▶ 播放")
        self._sync_timeline()
        self._update_caption()

    def _update_segment_enabled(self):
        starts = self.frames.get("split_starts", {}) if self.frames else {}
        model = self.segment_box.model()
        for index, (_label, key) in enumerate(self.SEGMENTS):
            item = model.item(index)
            if item is not None:
                item.setEnabled(starts.get(key) is not None)

    def _segment_changed(self):
        if self.source_box.currentData() != "run" or self.frames is None:
            return
        key = self.segment_box.currentData()
        first = self.frames.get("split_starts", {}).get(key)
        if first is None:
            return
        self._pending = (self.run_id, max(0, int(first) - (0 if key == "train" else 40)), False)
        self._start_pending()

    def _time_changed(self, value):
        if getattr(self, "_syncing", False):
            return
        self.pos = float(value)
        self.canvas.jump(self.pos)
        self.time_label.setText(self._time_caption(value))

    def _time_caption(self, value):
        payload = getattr(self, "_payload", None) or {}
        local = payload.get("episode_local_step")
        if local is not None and len(local):
            idx = max(0, min(len(local) - 1, int(value)))
            return f"片段 {int(payload['episode_id'][idx]) + 1} · 本段 t={int(local[idx])}"
        return f"t = {int(value)}"

    def _sync_timeline(self):
        n = len(self._payload["states"]) if getattr(self, "_payload", None) is not None else 0
        self._syncing = True
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, max(0, n - 1))
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)
        self._syncing = False
        self.time_label.setText(self._time_caption(0))

    def _toggle_play(self):
        if getattr(self, "_payload", None) is None:
            return
        if not self.playing and self.pos >= len(self._payload["states"]) - 1:
            self.pos = 0
            self.canvas.jump(0)
        self.playing = not self.playing
        self.play_button.setText("⏸ 暂停" if self.playing else "▶ 播放")

    def _tick(self):
        now = time.perf_counter()
        dt = min(0.1, now - self._clock)
        self._clock = now
        payload = getattr(self, "_payload", None)
        if payload is None:
            return
        if self.playing:
            self.phase += dt             # packets move only while time moves
            previous = self.pos
            self.pos += 30.0 * float(self.speed_box.currentData()) * dt
            bounds = payload.get("episode_boundary")
            if bounds is not None:
                for i in range(int(previous) + 1, min(len(bounds), int(self.pos) + 1)):
                    if bounds[i]:
                        self.pos = float(i)
                        self.playing = False
                        self.play_button.setText("▶ 播放")
                        break
            if self.pos >= len(payload["states"]) - 1:
                if payload.get("kind") == "sandbox":
                    self.pos = 0.0
                else:
                    self.pos = float(len(payload["states"]) - 1)
                    self.playing = False
                    self.play_button.setText("▶ 播放")
        self.canvas.set_time(self.pos, self.phase)
        self._syncing = True
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(self.pos))
        self.time_slider.blockSignals(False)
        self._syncing = False
        self.time_label.setText(self._time_caption(int(self.pos)))

    def showEvent(self, event):
        super().showEvent(event)
        self._clock = time.perf_counter()
        if self._pending is not None:
            self._start_pending()

    def hideEvent(self, event):
        super().hideEvent(event)

    def shutdown(self):
        self.timer.stop()
        self._param_timer.stop()
        if self.loader is not None and self.loader.isRunning():
            self.loader.wait(3000)
