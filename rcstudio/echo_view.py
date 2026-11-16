"""Layered echo view (Qt): input on top, reservoir layers by echo depth, readout at the bottom.

Mid-grey ground with film grain, orthographic projection, nodes strictly black and
white. A resting neuron is a small dim dot; an active neuron above its own mean is a
black blot with a white rim, one below it a white ring with a black rim, and its size
and bloom grow with the activity. Synapses carrying signal are drawn as black
(excitatory) or white (inhibitory) strokes with moving packets. The camera can be
orbited, zoomed and clicked to inspect one neuron. All geometry and activity maths
live in ``reservoir_scene`` (Qt-free and unit-tested); this module only paints.
"""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
    QPixmap, QPolygonF, QRadialGradient,
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from .learning_support import anchor as _help_anchor
from .reservoir_scene import (
    BLACK, DIM, EDGE_LEVELS, GROUND_BOTTOM, GROUND_TOP, REST_LEVEL, WHITE, EchoScene,
    emphasize, grain_tile, impulse_activity, ink_node, replay_activity,
)

SPLIT_NAMES = ("训练", "验证", "测试")
SPLIT_BANDS = {1: (255, 255, 255, 30), 2: (0, 0, 0, 44)}   # validation lighter, test darker
ACTIVE_EDGE_MIN = 0.30          # edge brightness below this only shows as structure
MAX_PULSES = 70


def _nan_array(values):
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def _centered_unit(values):
    """Values centred on their mean and scaled into [-1, 1]; NaN becomes 0."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values))
    centred = np.where(finite, values - np.nanmean(values), 0.0)
    scale = float(np.max(np.abs(centred))) or 1.0
    return centred / scale


def _color(rgb, alpha=255):
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(max(0, min(255, alpha))))


class ReservoirCanvas(QWidget):
    TOP_H = 58
    BOTTOM_H = 78
    DEFAULT_YAW = 0.50
    DEFAULT_PITCH = 0.34

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(380)
        self.scene = None
        self.data = None
        self.pos = 0.0
        self.yaw = self.DEFAULT_YAW
        self.pitch = self.DEFAULT_PITCH
        self.zoom = 1.0
        self.sway = 0.0
        self.selected = None
        self.glow = None
        self.phase = 0.0
        self.message = "完成一次训练后，这里把储备池节点按回声深度分层显示。"
        self._press = None
        self._last = None
        self._moved = False
        self._geometry = None
        self._scope_cache = {}
        self._spark_cache = None

    # ----- data in -------------------------------------------------------
    def set_scene(self, scene):
        self.scene = scene
        self.selected = None
        self._geometry = None
        self.update()

    def set_data(self, data):
        self.data = data
        self.pos = 0.0
        self._scope_cache = {}
        self._spark_cache = None
        if data is not None:
            self.glow = emphasize(np.abs(data["activity"][0]) * data["gain"])
        self.update()

    def set_message(self, text):
        self.message = text
        self.update()

    def set_time(self, pos, dt):
        """Move the playhead and let node glows fade (the afterglow shows the echo)."""
        if self.data is None:
            return
        n = len(self.data["activity"])
        self.pos = float(max(0.0, min(n - 1, pos)))
        state = self._state_at(self.pos)
        decay = math.exp(-max(dt, 0.0) / 0.22)
        fresh = emphasize(np.abs(state) * self.data["gain"])
        self.glow = np.clip(np.maximum(fresh, self.glow * decay), 0.0, 1.0)
        self.phase += max(dt, 0.0)
        self.update()

    def jump(self, pos):
        """Scrub: reset the afterglow so it does not smear across the jump."""
        if self.data is None:
            return
        self.pos = float(pos)
        self.glow = emphasize(np.abs(self._state_at(self.pos)) * self.data["gain"])
        self.update()

    def reset_camera(self):
        self.yaw, self.pitch, self.zoom, self.selected = (
            self.DEFAULT_YAW, self.DEFAULT_PITCH, 1.0, None)
        self.update()

    @property
    def dragging(self):
        return self._press is not None and self._moved

    def _state_at(self, pos):
        arr = self.data["activity"]
        i0 = int(pos)
        i1 = min(len(arr) - 1, i0 + 1)
        f = pos - i0
        return arr[i0] * (1.0 - f) + arr[i1] * f

    def _value_at(self, key, pos):
        arr = self.data[key]
        i0 = int(pos)
        i1 = min(len(arr) - 1, i0 + 1)
        f = pos - i0
        return float(arr[i0] * (1.0 - f) + arr[i1] * f)

    @staticmethod
    def _font(pixels, bold=False):
        font = QFont("Microsoft YaHei UI")
        font.setPixelSize(pixels)
        font.setBold(bold)
        return font

    @staticmethod
    def _segments(sx, sy, a, b):
        return [QLineF(x0, y0, x1, y1) for x0, y0, x1, y1 in zip(
            sx[a].tolist(), sy[a].tolist(), sx[b].tolist(), sy[b].tolist())]

    # ----- painting: mid-grey ground with grain, nodes strictly black and white ---
    _grain = None                       # shared pixmap tile, built on first paint

    @classmethod
    def _grain_pixmap(cls):
        if cls._grain is None:
            tile = grain_tile()
            size = tile.shape[0]
            image = QImage(tile.tobytes(), size, size, size * 4, QImage.Format.Format_RGBA8888)
            cls._grain = QPixmap.fromImage(image.copy())     # copy: detach from the numpy buffer
        return cls._grain

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        ground = QLinearGradient(0, 0, 0, h)
        ground.setColorAt(0.0, _color(GROUND_TOP))
        ground.setColorAt(1.0, _color(GROUND_BOTTOM))
        p.fillRect(self.rect(), QBrush(ground))
        p.fillRect(self.rect(), QBrush(self._grain_pixmap()))      # film grain
        if self.scene is None or self.data is None:
            p.setPen(_color(BLACK, 230))
            p.setFont(self._font(13))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, self.message)
            p.end()
            return
        self._paint_scene(p, w, h)
        self._paint_scopes(p, w, h)
        p.end()

    def _paint_scene(self, p, w, h):
        scene, data = self.scene, self.data
        view_h = max(160, h - self.TOP_H - self.BOTTOM_H)
        yaw = self.yaw + self.sway
        args = (yaw, self.pitch, self.zoom, w, view_h, self.TOP_H)
        state = self._state_at(self.pos)
        gain = data["gain"]
        sx, sy, z, _scale = scene.project(scene.positions, *args)      # orthographic: no size change
        isx, isy, _iz, _ = scene.project(scene.input_positions, *args)
        rsx, rsy, _rz, _ = scene.project(scene.readout_position, *args)
        self._geometry = (sx, sy)
        u = self._value_at("input_activity", self.pos)
        out = self._value_at("output_activity", self.pos)
        p.setPen(Qt.PenStyle.NoPen)

        # layer plates: pale discs, white rim; label with the echo statistics of the layer
        p.setFont(self._font(12))
        for index in range(scene.n_layers):
            rx, ry, _, _ = scene.project(scene.plate_ring(index), *args)
            polygon = QPolygonF([QPointF(a, b) for a, b in zip(rx.tolist(), ry.tolist())])
            p.setPen(QPen(_color(WHITE, 150), 1.2))
            p.setBrush(QBrush(_color(WHITE, 20)))
            p.drawPolygon(polygon)
            row = scene.summary[index]
            k = int(np.argmax(rx))
            p.setPen(QPen(_color(BLACK, 130), 1.0))
            p.drawLine(QPointF(rx[k], ry[k]), QPointF(rx[k] + 22, ry[k]))
            p.setPen(_color(BLACK, 240))
            p.drawText(QPointF(rx[k] + 28, ry[k] + 4),
                       f"L{index + 1} · 回声≈{row['depth']:.1f}步 · {row['count']}节点")
        self._paint_echo_arrow(p, scene, args)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # synapses: faint structure, then strokes for the ones carrying signal now
        if len(scene.w):
            p.setPen(QPen(_color(BLACK, 26), 1.0))
            p.drawLines(self._segments(sx, sy, scene.src, scene.dst))
        strength, sign = scene.edge_activity(state, gain)
        live = strength > ACTIVE_EDGE_MIN
        if live.any():
            level = np.minimum(
                EDGE_LEVELS - 1,
                ((strength - ACTIVE_EDGE_MIN) / (1.0 - ACTIVE_EDGE_MIN) * EDGE_LEVELS).astype(int))
            for direction, tone in ((1, BLACK), (-1, WHITE)):    # black = excitatory, white = inhibitory
                for lv in range(EDGE_LEVELS):
                    pick = np.flatnonzero(live & (sign == direction) & (level == lv))
                    if len(pick):
                        p.setPen(QPen(_color(tone, 90 + 55 * lv), 1.0 + 0.7 * lv))
                        p.drawLines(self._segments(sx, sy, scene.src[pick], scene.dst[pick]))
        # input fan-out and readout fan-in
        weight_scale = float(np.max(np.abs(scene.input_weights))) or 1.0
        drive = min(1.0, abs(u) * 1.2)
        nodes = np.arange(scene.units)
        for j in range(scene.n_inputs):
            share = np.abs(scene.input_weights[:, j]) / weight_scale * drive
            for low, high, alpha in ((0.0, 0.35, 9), (0.35, 0.7, 40), (0.7, 1.01, 95)):
                pick = nodes[(share >= low) & (share < high)]
                if len(pick):
                    p.setPen(QPen(_color(BLACK, alpha), 1.0))
                    p.drawLines([QLineF(isx[j], isy[j], a, b)
                                 for a, b in zip(sx[pick].tolist(), sy[pick].tolist())])
        share = scene.readout_contribution(state, gain)
        for direction, tone in ((1, BLACK), (-1, WHITE)):
            side = share >= 0 if direction == 1 else share < 0
            for low, high, alpha in ((0.0, 0.15, 9), (0.15, 0.4, 50), (0.4, 0.7, 120), (0.7, 1.01, 210)):
                pick = nodes[side & (np.abs(share) >= low) & (np.abs(share) < high)]
                if len(pick):
                    p.setPen(QPen(_color(tone, alpha), 1.0 + (alpha > 100)))
                    p.drawLines([QLineF(a, b, rsx[0], rsy[0])
                                 for a, b in zip(sx[pick].tolist(), sy[pick].tolist())])
        # direct input -> readout path (the readout also sees the raw input)
        if float(np.max(np.abs(scene.readout_bypass))) > 1e-9:
            path = QPainterPath(QPointF(isx[0], isy[0]))
            bulge = -min(170.0, w * 0.22)
            path.cubicTo(QPointF(isx[0] + bulge, isy[0] + 0.25 * (rsy[0] - isy[0])),
                         QPointF(rsx[0] + bulge, isy[0] + 0.75 * (rsy[0] - isy[0])),
                         QPointF(rsx[0], rsy[0]))
            p.setPen(QPen(_color(BLACK, 70), 1.0, Qt.PenStyle.DashLine))
            p.drawPath(path)

        # a selected neuron keeps all its synapses visible
        if self.selected is not None and len(scene.w):
            pick = np.flatnonzero((scene.src == self.selected) | (scene.dst == self.selected))
            if len(pick):
                p.setPen(QPen(_color(WHITE, 240), 3.0))
                p.drawLines(self._segments(sx, sy, scene.src[pick], scene.dst[pick]))
                p.setPen(QPen(_color(BLACK, 240), 1.3))
                p.drawLines(self._segments(sx, sy, scene.src[pick], scene.dst[pick]))

        # neurons, far to near. Resting: small dim dot. Active above own mean: black blot with a
        # white rim and a dark bloom; active below: white ring with a black rim and a white bloom.
        order = np.argsort(z)
        z_min, z_span = float(z.min()), max(1e-9, float(z.max() - z.min()))
        glow = np.clip(self.glow if self.glow is not None else np.zeros(scene.units), 0.0, 1.0)
        for i in order.tolist():
            fog = 0.78 + 0.22 * (float(z[i]) - z_min) / z_span          # depth cue without size change
            act = float(glow[i])
            solid, _level = ink_node(float(state[i]) * gain)
            center = QPointF(float(sx[i]), float(sy[i]))
            if act < REST_LEVEL:                                        # resting
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(_color(DIM, 215 * fog))
                p.drawEllipse(center, 2.0, 2.0)
                continue
            radius = 3.0 + 4.6 * act
            halo = radius * 3.4
            gradient = QRadialGradient(center, halo)
            gradient.setColorAt(0.0, _color(BLACK if solid else WHITE, 150 * act * fog))
            gradient.setColorAt(1.0, _color(BLACK if solid else WHITE, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(gradient))
            p.drawEllipse(center, halo, halo)
            if solid:
                p.setPen(QPen(_color(WHITE, 130 + 110 * act), 1.2))     # white rim keeps black blots legible
                p.setBrush(_color(BLACK, 255))
            else:
                p.setPen(QPen(_color(BLACK, 255), 1.5))
                p.setBrush(_color(WHITE, 255))
            p.drawEllipse(center, radius, radius)
        if self.selected is not None:
            i = int(self.selected)
            ring = 3.0 + 4.6 * float(glow[i]) + 6.0
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(_color(WHITE, 250), 3.2))
            p.drawEllipse(QPointF(float(sx[i]), float(sy[i])), ring, ring)
            p.setPen(QPen(_color(BLACK, 250), 1.4))
            p.drawEllipse(QPointF(float(sx[i]), float(sy[i])), ring, ring)

        # input and readout nodes: black discs with white marks and a white rim
        p.setFont(self._font(12, True))
        for j in range(scene.n_inputs):
            radius = 6.5 + 4.5 * min(1.0, abs(u))
            self._disc(p, isx[j], isy[j], radius, ring=False)
        label = "输入" if scene.n_inputs == 1 else f"输入 ×{scene.n_inputs}"
        p.setPen(_color(BLACK, 245))
        p.drawText(QPointF(float(isx.max()) + 18, float(isy[0]) + 4), label)
        radius = 9.5 + 4.5 * min(1.0, abs(out))
        self._disc(p, rsx[0], rsy[0], radius, ring=True)
        p.setPen(_color(BLACK, 245))
        p.drawText(QPointF(float(rsx[0]) + radius + 10, float(rsy[0]) + 4), "读出")

        # signal packets travelling along the busiest synapses: white drop with a black core
        if live.any():
            busiest = np.flatnonzero(live)
            busiest = busiest[np.argsort(-strength[busiest])][:MAX_PULSES]
            p.setPen(Qt.PenStyle.NoPen)
            for rank, k in enumerate(busiest.tolist()):
                t = (self.phase * 1.4 + rank * 0.6180339) % 1.0
                a, b = int(scene.src[k]), int(scene.dst[k])
                x = float(sx[a] + (sx[b] - sx[a]) * t)
                y = float(sy[a] + (sy[b] - sy[a]) * t)
                p.setBrush(_color(WHITE, 140 + 110 * float(strength[k])))
                p.drawEllipse(QPointF(x, y), 2.6, 2.6)
                p.setBrush(_color(BLACK, 255))
                p.drawEllipse(QPointF(x, y), 1.3, 1.3)
        if self.selected is not None:
            self._paint_info(p, w, h, int(self.selected), state, gain)

    @staticmethod
    def _disc(p, x, y, radius, ring):
        """Black disc with a white rim; the input has a white centre dot, the readout a white ring."""
        center = QPointF(float(x), float(y))
        p.setPen(QPen(_color(WHITE, 235), 1.6))
        p.setBrush(_color(BLACK, 255))
        p.drawEllipse(center, radius, radius)
        if ring:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(_color(WHITE, 245), 1.8))
            p.drawEllipse(center, radius * 0.58, radius * 0.58)
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_color(WHITE, 245))
            p.drawEllipse(center, radius * 0.34, radius * 0.34)

    def _paint_echo_arrow(self, p, scene, args):
        """Left-hand arrow that reads: top = short echo, bottom = long echo."""
        top = scene.project(np.array([[0.0, scene.layer_y[0], 0.0]]), *args)[1][0]
        bottom = scene.project(np.array([[0.0, scene.layer_y[-1], 0.0]]), *args)[1][0]
        x = 44.0
        color = _color(BLACK, 200)
        p.setPen(QPen(color, 1.2))
        p.drawLine(QPointF(x, float(top)), QPointF(x, float(bottom)))
        p.setBrush(QBrush(color))
        p.drawPolygon(QPolygonF([QPointF(x - 5, float(bottom) - 8), QPointF(x + 5, float(bottom) - 8),
                                 QPointF(x, float(bottom))]))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_color(BLACK, 235))
        p.setFont(self._font(12))
        p.drawText(QPointF(x + 10, float(top) + 4), "回声短 · 即时响应")
        p.drawText(QPointF(x + 10, float(bottom) + 4), "回声长 · 长期记忆")

    def _paint_info(self, p, w, h, node, state, gain):
        facts = self.scene.describe(node)
        box = QRectF(14, h - self.BOTTOM_H - 82, 292, 70)
        p.setPen(QPen(_color(WHITE, 200), 1.0))
        p.setBrush(QBrush(_color(BLACK, 228)))
        p.drawRoundedRect(box, 8, 8)
        p.setFont(self._font(12, True))
        p.setPen(_color(WHITE))
        p.drawText(QPointF(box.left() + 10, box.top() + 19),
                   f"节点 #{node} · 第 {facts['layer'] + 1} 层")
        p.setFont(self._font(12))
        p.setPen(_color(WHITE, 205))
        p.drawText(QPointF(box.left() + 10, box.top() + 38),
                   f"回声深度 {facts['depth']:.1f} 步 · 寿命 {facts['lifetime']} 步")
        p.drawText(QPointF(box.left() + 10, box.top() + 57),
                   f"读出权重 {facts['readout_weight']:+.3f} · 当前 {float(state[node]) * gain:+.2f}")
        key = (node, id(self.data))
        if self._spark_cache is None or self._spark_cache[0] != key:
            series = np.asarray(self.data["activity"][:, node], dtype=np.float64)
            self._spark_cache = (key, series)
        series = self._spark_cache[1]
        area = QRectF(box.right() - 112, box.top() + 10, 100, box.height() - 20)
        span = float(np.max(np.abs(series))) or 1.0
        path = QPainterPath()
        xs = area.left() + area.width() * np.arange(len(series)) / max(1, len(series) - 1)
        ys = area.center().y() - series / span * area.height() * 0.5
        path.moveTo(float(xs[0]), float(ys[0]))
        for x, y in zip(xs[1:].tolist(), ys[1:].tolist()):
            path.lineTo(x, y)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_color(WHITE, 240), 1.3))
        p.drawPath(path)
        playhead = area.left() + area.width() * self.pos / max(1, len(series) - 1)
        p.setPen(QPen(_color(WHITE, 130), 1.0))
        p.drawLine(QPointF(playhead, area.top()), QPointF(playhead, area.bottom()))

    # ----- the two oscilloscope strips -----------------------------------
    def _scope_path(self, key, values, rect, lo, hi):
        cache_key = (key, round(rect.width()), round(rect.height()))
        cached = self._scope_cache.get(cache_key)
        if cached is not None:
            return cached
        path = QPainterPath()
        n = len(values)
        span = (hi - lo) or 1.0
        pen_down = False
        for i in range(n):
            v = values[i]
            if not math.isfinite(v):
                pen_down = False
                continue
            x = rect.left() + rect.width() * i / max(1, n - 1)
            y = rect.bottom() - (v - lo) / span * rect.height()
            if pen_down:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                pen_down = True
        self._scope_cache[cache_key] = path
        return path

    def _paint_scopes(self, p, w, h):
        data = self.data
        n = len(data["activity"])
        top = QRectF(24, 20, max(10, w - 48), self.TOP_H - 28)
        bottom = QRectF(24, h - self.BOTTOM_H + 22, max(10, w - 48), self.BOTTOM_H - 32)
        p.setFont(self._font(12))
        p.setPen(_color(BLACK, 245))
        p.drawText(QPointF(24, 14), data["input_title"])
        p.drawText(QPointF(24, h - self.BOTTOM_H + 15), data["output_title"])

        inp = data["input"]
        finite = inp[np.isfinite(inp)]
        lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (-1.0, 1.0)
        p.setBrush(Qt.BrushStyle.NoBrush)
        input_path = self._scope_path("in", inp, top, lo, hi)
        p.setPen(QPen(_color(WHITE, 150), 4.0))                     # light underlay keeps black legible
        p.drawPath(input_path)
        p.setPen(QPen(_color(BLACK, 250), 1.8))
        p.drawPath(input_path)

        values = [data["output"]] + ([data["truth"]] if data.get("truth") is not None else [])
        joined = np.concatenate([v[np.isfinite(v)] for v in values] or [np.zeros(1)])
        lo, hi = (float(joined.min()), float(joined.max())) if len(joined) else (-1.0, 1.0)
        split = data.get("split")
        if split is not None:                       # validation lighter, test darker
            p.setPen(Qt.PenStyle.NoPen)
            start = 0
            for i in range(1, n + 1):
                if i == n or split[i] != split[start]:
                    code = int(split[start])
                    x0 = bottom.left() + bottom.width() * start / max(1, n - 1)
                    x1 = bottom.left() + bottom.width() * (i - 1) / max(1, n - 1)
                    if code in SPLIT_BANDS:
                        p.setBrush(QBrush(QColor(*SPLIT_BANDS[code])))
                        p.drawRect(QRectF(x0, bottom.top() - 4, max(1.0, x1 - x0), bottom.height() + 8))
                    if 0 <= code < len(SPLIT_NAMES):
                        p.setPen(_color(BLACK, 235))
                        p.setFont(self._font(11))
                        p.drawText(QPointF(x0 + 4, bottom.top() + 6), SPLIT_NAMES[code])
                        p.setPen(Qt.PenStyle.NoPen)
                    start = i
        p.setBrush(Qt.BrushStyle.NoBrush)
        if data.get("truth") is not None:
            p.setPen(QPen(_color(BLACK, 250), 1.5))                 # truth: thin black
            p.drawPath(self._scope_path("truth", data["truth"], bottom, lo, hi))
        out_path = self._scope_path("out", data["output"], bottom, lo, hi)
        p.setPen(QPen(_color(BLACK, 120), 4.6))                    # prediction: white with a dark underlay
        p.drawPath(out_path)
        p.setPen(QPen(_color(WHITE, 255), 2.4))
        p.drawPath(out_path)
        x = top.left() + top.width() * self.pos / max(1, n - 1)
        p.setPen(QPen(_color(WHITE, 235), 1.4))
        p.drawLine(QPointF(x, 16), QPointF(x, self.TOP_H - 4))
        p.drawLine(QPointF(x, bottom.top() - 6), QPointF(x, h - 4))

    # ----- mouse: drag = orbit, wheel = zoom, click = select ---------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position()
            self._last = event.position()
            self._moved = False

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        pos = event.position()
        if not self._moved and (pos - self._press).manhattanLength() > 4:
            self._moved = True
        if self._moved:
            delta = pos - self._last
            self.yaw += delta.x() * 0.008
            self.pitch = float(np.clip(self.pitch + delta.y() * 0.006, 0.05, 1.25))
            self.update()
        self._last = pos

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._press is None:
            return
        if not self._moved and self.scene is not None and self._geometry is not None:
            sx, sy = self._geometry
            hit = self.scene.pick(sx, sy, event.position().x(), event.position().y())
            self.selected = None if hit is None or hit == self.selected else hit
            self.update()
        self._press = None
        self._moved = False

    def mouseDoubleClickEvent(self, _event):
        self.reset_camera()

    def wheelEvent(self, event):
        self.zoom = float(np.clip(self.zoom * 1.0015 ** event.angleDelta().y(), 0.5, 2.6))
        self.update()
        event.accept()


class EchoLoader(QThread):
    """Fetches the structure and one frame window without blocking the GUI."""
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, client, run_id, edges, start, count, with_structure, parent=None):
        super().__init__(parent)
        self.client = client
        self.run_id = run_id
        self.edges = edges
        self.first_frame = start        # not "start": that would shadow QThread.start()
        self.frame_count = count
        self.with_structure = with_structure

    def run(self):
        try:
            result = {"run": self.run_id}
            if self.with_structure:
                result["structure"] = self.client.reservoir(self.run_id, edges=self.edges)
            result["frames"] = self.client.reservoir_frames(
                self.run_id, start=self.first_frame, count=self.frame_count)
            self.completed.emit(result)
        except Exception as exc:  # shown in the canvas; the main window keeps running
            self.failed.emit(str(exc))


class EchoPanel(QWidget):
    WINDOW = 360                       # frames fetched per window
    EDGE_FETCH = 2000                  # strongest edges requested once per run
    SEGMENTS = (("训练段", "train"), ("验证段", "validation"), ("测试段", "test"))
    SPEEDS = ((0.25, "0.25×"), (0.5, "0.5×"), (1.0, "1×"), (2.0, "2×"), (4.0, "4×"))
    REPLAY_FPS = 30.0
    IMPULSE_FPS = 10.0

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self.run_id = None
        self.structure = None
        self.scene = None
        self.frames = None
        self.split_starts = {}
        self.loader = None
        self._pending = None
        self._request = None
        self.playing = True
        self.pos = 0.0
        self._clock = time.perf_counter()
        self._started = self._clock

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.mode_box = _help_anchor(QComboBox(), "rc.echo")
        self.mode_box.addItem("数据回放", "replay")
        self.mode_box.addItem("脉冲回声", "impulse")
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        bar.addWidget(self.mode_box)
        self.segment_box = _help_anchor(QComboBox(), "rc.echo.playback")
        for label, key in self.SEGMENTS:
            self.segment_box.addItem(label, key)
        self.segment_box.currentIndexChanged.connect(self._segment_changed)
        bar.addWidget(self.segment_box)
        self.play_button = _help_anchor(QPushButton("⏸ 暂停"), "rc.echo.playback")
        self.play_button.clicked.connect(self._toggle_play)
        bar.addWidget(self.play_button)
        self.speed_box = _help_anchor(QComboBox(), "rc.echo.playback")
        for factor, label in self.SPEEDS:
            self.speed_box.addItem(label, factor)
        self.speed_box.setCurrentIndex(2)
        bar.addWidget(self.speed_box)
        bar.addWidget(QLabel("层数"))
        self.layers_box = _help_anchor(QSpinBox(), "rc.echo.layers")
        self.layers_box.setRange(2, 8)
        self.layers_box.setValue(4)
        self.layers_box.valueChanged.connect(self._layers_changed)
        bar.addWidget(self.layers_box)
        bar.addWidget(QLabel("连线"))
        self.edges_box = _help_anchor(QSpinBox(), "rc.echo.layers")
        self.edges_box.setRange(50, 2000)
        self.edges_box.setSingleStep(50)
        self.edges_box.setValue(450)
        self.edges_box.valueChanged.connect(self._edges_changed)
        bar.addWidget(self.edges_box)
        self.edge_label = QLabel("")
        self.edge_label.setObjectName("Caption")
        bar.addWidget(self.edge_label)
        self.orbit_box = _help_anchor(QCheckBox("自动环绕"), "rc.echo.layers")
        self.orbit_box.setChecked(True)
        bar.addWidget(self.orbit_box)
        bar.addStretch()
        outer.addLayout(bar)

        self.canvas = ReservoirCanvas()
        _help_anchor(self.canvas, "rc.echo")
        outer.addWidget(self.canvas, 1)

        row = QHBoxLayout()
        self.slider = _help_anchor(QSlider(Qt.Orientation.Horizontal), "rc.echo.playback")
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._slider_moved)
        row.addWidget(self.slider, 1)
        self.time_label = QLabel("—")
        self.time_label.setObjectName("Caption")
        self.time_label.setMinimumWidth(150)
        row.addWidget(self.time_label)
        outer.addLayout(row)
        self.hint = QLabel("拖动旋转 · 滚轮缩放 · 点击神经元查看 · 双击复位。节点按脉冲回声深度分层：越靠下，旧输入的回响越久。")
        self.hint.setObjectName("Caption")
        self.hint.setWordWrap(True)
        outer.addWidget(self.hint)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    # ----- run lifecycle -------------------------------------------------
    def show_run(self, run_id):
        busy = self.loader is not None and self.loader.isRunning()
        if run_id == self.run_id and (self.scene is not None or self._pending is not None or busy):
            return                                  # same run again, e.g. after a theme switch
        self.run_id = run_id
        self.structure = self.scene = self.frames = None
        self.canvas.set_scene(None)
        self.canvas.set_data(None)
        self.canvas.set_message("正在读取储备池结构…")
        self._pending = (run_id, True, 0)
        if self.isVisible():
            self._start_pending()

    def showEvent(self, event):
        super().showEvent(event)
        self._clock = time.perf_counter()
        self.timer.start(33)
        if self._pending is not None:
            self._start_pending()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def _start_pending(self):
        if self._pending is None or (self.loader is not None and self.loader.isRunning()):
            return
        run_id, with_structure, start = self._pending
        self._request = self._pending
        self._pending = None
        self.loader = EchoLoader(self.client, run_id, self.EDGE_FETCH, start, self.WINDOW,
                                 with_structure, self)
        self.loader.completed.connect(self._loaded)
        self.loader.failed.connect(self._load_failed)
        self.loader.finished.connect(self._loader_finished)
        self.loader.start()

    def _loader_finished(self):
        if self.loader is not None:
            self.loader.wait()          # finished() fires just before the thread ends
        if self._pending is not None:
            self._start_pending()

    def _load_failed(self, message):
        if self._request and self._request[0] == self.run_id:
            self.canvas.set_message(f"无法读取分层数据：{message}")

    def _loaded(self, result):
        if result["run"] != self.run_id:
            return                                  # the user already picked another run
        if "structure" in result:
            self.structure = result["structure"]
            self.scene = EchoScene(self.structure, self.layers_box.value(), self.edges_box.value())
            self.canvas.set_scene(self.scene)
        self.frames = result["frames"]
        self.split_starts = self.frames.get("split_starts", {})
        first_train = self.split_starts.get("train")
        if "structure" in result and first_train and self.frames["start"] != first_train:
            self._pending = (self.run_id, False, int(first_train))   # open on the training segment
            self._enable_segments()
            return
        self._enable_segments()
        self._apply_mode()

    def _enable_segments(self):
        model = self.segment_box.model()
        for index, (_label, key) in enumerate(self.SEGMENTS):
            item = model.item(index)
            if item is not None:
                item.setEnabled(self.split_starts.get(key) is not None)
        self._update_edge_label()

    # ----- building what the canvas shows --------------------------------
    def _apply_mode(self):
        if self.scene is None:
            return
        mode = self.mode_box.currentData()
        wide = self.structure["inputs"] > 1
        if mode == "impulse":
            imp = self.structure["impulse"]
            activity, gain = impulse_activity(np.asarray(imp["states"], dtype=np.float64))
            steps = len(activity)
            pulse = np.zeros(steps)
            pulse[0] = imp["amplitude"]
            echo = activity @ self.scene.readout_nodes
            data = {
                "activity": activity, "gain": gain, "kind": "impulse",
                "input": pulse, "input_activity": pulse / (imp["amplitude"] or 1.0),
                "output": echo, "output_activity": echo / (float(np.max(np.abs(echo))) or 1.0),
                "truth": None, "split": None,
                "input_title": "输入脉冲 · 六通道同时一次" if wide else "输入脉冲 · 单次 1σ",
                "output_title": "读出端回声 · 这一次脉冲对输出的影响",
                "labels": [f"脉冲后第 {i} 步" for i in range(steps)],
            }
        elif self.frames is not None:
            fr = self.frames
            activity, gain = replay_activity(np.asarray(fr["states"], dtype=np.float64))
            inp, truth, pred = (_nan_array(fr[k]) for k in ("inputs", "truth", "prediction"))
            data = {
                "activity": activity, "gain": gain, "kind": "replay",
                "input": inp, "input_activity": _centered_unit(inp),
                "output": pred, "output_activity": _centered_unit(pred),
                "truth": truth, "split": np.asarray(fr["split"], dtype=np.int64),
                "input_title": "输入 · 角速度幅值（六通道 IMU 的目标量）" if wide else "输入信号 u(t)",
                "output_title": "读出输出 · 真实值(黑) / 预测值(白)",
                "labels": [f"时间步 {t}" for t in fr["timestep"]],
            }
        else:
            return
        self.pos = 0.0
        self.playing = True
        self.play_button.setText("⏸ 暂停")
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(data["activity"]) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.segment_box.setEnabled(mode == "replay")
        self.canvas.set_data(data)
        self._update_label()

    # ----- controls ------------------------------------------------------
    def _mode_changed(self):
        self._apply_mode()

    def _segment_changed(self):
        if self.run_id is None or self.mode_box.currentData() != "replay":
            return
        key = self.segment_box.currentData()
        first = self.split_starts.get(key)
        if first is None:
            return
        lead = 0 if key == "train" else 40            # show the band change into val/test
        self._pending = (self.run_id, False, max(0, int(first) - lead))
        self._start_pending()

    def _toggle_play(self):
        data = self.canvas.data
        if data is None:
            return
        if not self.playing and self.pos >= len(data["activity"]) - 1:
            self.pos = 0.0
            self.canvas.jump(0.0)
        self.playing = not self.playing
        self.play_button.setText("⏸ 暂停" if self.playing else "▶ 播放")

    def _layers_changed(self, value):
        if self.scene is not None:
            self.scene.set_layers(value)
            self.canvas.selected = None
            self.canvas.update()

    def _edges_changed(self, value):
        if self.scene is not None:
            self.scene.set_edge_limit(value)
            self._update_edge_label()
            self.canvas.update()

    def _update_edge_label(self):
        if self.scene is None:
            self.edge_label.setText("")
        else:
            self.edge_label.setText(f"{len(self.scene.w)}/{self.scene.edges_total}")

    def _slider_moved(self, value):
        if self.canvas.data is None:
            return
        self.pos = float(value)
        self.canvas.jump(self.pos)
        self._update_label()

    def _update_label(self):
        data = self.canvas.data
        if data is None:
            self.time_label.setText("—")
            return
        index = int(self.pos)
        self.time_label.setText(f"{data['labels'][index]}  ({index + 1}/{len(data['labels'])})")

    def _tick(self):
        now = time.perf_counter()
        dt = min(0.1, now - self._clock)
        self._clock = now
        canvas = self.canvas
        data = canvas.data
        if data is None:
            return
        if not self.orbit_box.isChecked():
            canvas.sway = 0.0
        elif not canvas.dragging:
            canvas.sway = 0.42 * math.sin((now - self._started) * 0.35)
        n = len(data["activity"])
        if self.playing and n > 1:
            fps = self.REPLAY_FPS if data["kind"] == "replay" else self.IMPULSE_FPS
            self.pos += fps * float(self.speed_box.currentData()) * dt
            if self.pos >= n - 1:
                if data["kind"] == "replay":
                    self.pos = 0.0
                    canvas.jump(0.0)
                else:
                    self.pos = float(n - 1)
                    self.playing = False
                    self.play_button.setText("▶ 播放")
        canvas.set_time(self.pos, dt)
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.pos))
        self.slider.blockSignals(False)
        self._update_label()
