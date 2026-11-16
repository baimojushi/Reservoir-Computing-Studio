"""Layered echo-state scene: geometry and activity maths, no Qt, no IO.

World layout (Y points up)::

    input nodes        y = +INPUT_Y     (top)
    reservoir layer 1  y = +LAYER_TOP   fastest echo   (reacts to the input at once)
    ...
    reservoir layer L  y = -LAYER_TOP   slowest echo   (still ringing long after)
    readout node       y = -INPUT_Y     (bottom)

A node's layer comes from its *echo depth*: the centroid of its impulse
response, in time steps (see ``core.echo_metrics``). Layers therefore draw the
fading-memory property of the reservoir: the deeper the layer, the longer an
input keeps echoing there.
"""
from __future__ import annotations

import math

import numpy as np

GOLDEN_ANGLE = 2.399963229728653
INPUT_Y = 1.30
LAYER_TOP = 0.85
EDGE_LEVELS = 4          # active-edge brightness buckets
PLATE_X = 2.35           # plates are wide ...
PLATE_Z = 0.62           # ... and shallow, so stacked layers stay readable

# Palette (R, G, B): mid-grey ground, nodes strictly black and white.
#   active, above the node's own mean -> solid BLACK blot with a white rim
#   active, below the node's own mean -> hollow WHITE ring with a black rim
#   resting (activity under REST_LEVEL) -> small dim dot, so activity stands out at once
GROUND_TOP = (152, 152, 150)
GROUND_BOTTOM = (126, 126, 124)
BLACK = (8, 8, 8)
WHITE = (250, 250, 248)
DIM = (104, 104, 102)            # resting nodes and faint structure
REST_LEVEL = 0.12                # emphasised activity below this counts as resting
GRAIN_SIZE = 192
GRAIN_STRENGTH = 44              # peak alpha of one grain pixel


def mix(a, b, t):
    """Linear blend of two RGB tuples, t clamped to [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def ink_node(value):
    """(solid, strength) of a signed activity: solid black for >= 0, hollow white below."""
    return bool(value >= 0), min(1.0, abs(float(value)))


def emphasize(level, floor=REST_LEVEL * 0.7, gamma=0.8):
    """Contrast curve: small activity is pushed to zero, strong activity saturates.

    Accepts scalars or arrays of |activity| * gain.
    """
    x = np.clip((np.asarray(level, dtype=np.float64) - floor) / (1.0 - floor), 0.0, 1.0)
    return x ** gamma


def grain_tile(size=GRAIN_SIZE, seed=7, strength=GRAIN_STRENGTH):
    """Tileable film-grain texture as an RGBA uint8 array (size, size, 4).

    Every pixel is black or white with a small random alpha; laid over the grey
    ground it reads as fine noise without shifting the average brightness.
    """
    rng = np.random.default_rng(seed)
    white = rng.random((size, size)) < 0.5
    alpha = (rng.random((size, size)) ** 1.6 * strength).astype(np.uint8)
    tile = np.zeros((size, size, 4), dtype=np.uint8)
    tile[..., :3] = np.where(white[..., None], 255, 0)
    tile[..., 3] = alpha
    return tile


def assign_layers(depth, n_layers):
    """Rank-based quantile layers: balanced counts, layer 0 = smallest depth."""
    depth = np.asarray(depth, dtype=np.float64)
    n = len(depth)
    n_layers = max(1, min(int(n_layers), n))
    order = np.argsort(depth, kind="stable")
    layer = np.empty(n, dtype=np.int64)
    layer[order] = (np.arange(n) * n_layers) // n
    return layer


def layer_summary(depth, lifetime, layer, n_layers):
    """Per-layer statistics shown next to each plate."""
    depth = np.asarray(depth, dtype=np.float64)
    lifetime = np.asarray(lifetime, dtype=np.float64)
    rows = []
    for index in range(n_layers):
        members = np.flatnonzero(layer == index)
        rows.append({
            "layer": index,
            "count": int(len(members)),
            "depth": float(np.median(depth[members])) if len(members) else 0.0,
            "lifetime": float(np.median(lifetime[members])) if len(members) else 0.0,
        })
    return rows


class EchoScene:
    """Geometry of one trained reservoir; positions depend only on the layer count."""

    def __init__(self, structure, n_layers=4, max_edges=900):
        self.units = int(structure["units"])
        self.n_inputs = int(structure["inputs"])
        self.depth = np.asarray(structure["depth"], dtype=np.float64)
        self.peak = np.asarray(structure["peak"], dtype=np.int64)
        self.lifetime = np.asarray(structure["lifetime"], dtype=np.float64)
        edges = structure["edges"]
        self.edges_src_all = np.asarray(edges["src"], dtype=np.int64)
        self.edges_dst_all = np.asarray(edges["dst"], dtype=np.int64)
        self.edges_w_all = np.asarray(edges["weight"], dtype=np.float64)
        self.edges_total = int(edges.get("total", len(self.edges_w_all)))
        self.input_weights = np.asarray(structure["input_weights"], dtype=np.float64)
        readout = structure["readout"]
        self.readout_nodes = np.asarray(readout["nodes"], dtype=np.float64)
        self.readout_bypass = np.asarray(readout["bypass"], dtype=np.float64)
        self.impulse = structure.get("impulse") or {}
        self.set_edge_limit(max_edges)
        self.set_layers(n_layers)

    # ----- configuration -------------------------------------------------
    def set_edge_limit(self, limit):
        """Edges arrive strongest-first, so a prefix is the strongest subset."""
        k = max(0, min(int(limit), len(self.edges_w_all)))
        self.src = self.edges_src_all[:k]
        self.dst = self.edges_dst_all[:k]
        self.w = self.edges_w_all[:k]
        self.w_max = float(np.max(np.abs(self.w))) if k else 1.0

    def set_layers(self, n_layers):
        self.n_layers = max(1, min(int(n_layers), self.units, 8))
        self.layer = assign_layers(self.depth, self.n_layers)
        self.summary = layer_summary(self.depth, self.lifetime, self.layer, self.n_layers)
        self.layer_y = np.array([self._layer_y(i) for i in range(self.n_layers)])
        self.layer_radius = np.zeros(self.n_layers)
        pos = np.zeros((self.units, 3))
        for index in range(self.n_layers):
            members = np.flatnonzero(self.layer == index)
            members = members[np.argsort(self.depth[members], kind="stable")]
            count = max(1, len(members))
            radius = float(np.clip(0.34 + 0.075 * math.sqrt(count), 0.50, 1.00))
            self.layer_radius[index] = radius
            k = np.arange(len(members))
            rad = radius * np.sqrt((k + 0.5) / count)     # fastest nodes sit in the centre
            ang = k * GOLDEN_ANGLE + index * 0.9
            pos[members, 0] = rad * np.cos(ang) * PLATE_X
            pos[members, 1] = self.layer_y[index]
            pos[members, 2] = rad * np.sin(ang) * PLATE_Z
        self.positions = pos
        if self.n_inputs == 1:
            self.input_positions = np.array([[0.0, INPUT_Y, 0.0]])
        else:
            ang = np.arange(self.n_inputs) * 2 * math.pi / self.n_inputs
            self.input_positions = np.column_stack((
                0.30 * PLATE_X * np.cos(ang), np.full(self.n_inputs, INPUT_Y),
                0.30 * PLATE_Z * np.sin(ang)))
        self.readout_position = np.array([0.0, -INPUT_Y, 0.0])

    def plate_ring(self, index, segments=72):
        """Outline of one layer plate as world points (closed by the caller)."""
        angle = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
        radius = self.layer_radius[index]
        return np.column_stack((
            radius * PLATE_X * np.cos(angle),
            np.full(segments, self.layer_y[index]),
            radius * PLATE_Z * np.sin(angle)))

    def _layer_y(self, index):
        if self.n_layers == 1:
            return 0.0
        return LAYER_TOP - 2.0 * LAYER_TOP * index / (self.n_layers - 1)

    # ----- projection ----------------------------------------------------
    @staticmethod
    def project(points, yaw, pitch, zoom, view_w, view_h, top=0.0):
        """Orthographic projection into a (view_w x view_h) rectangle starting at y=top.

        Parallel projection: size does not change with depth, so layers keep the
        same scale however the stack is turned. Returns screen x, screen y,
        camera depth (larger = nearer, used for ordering only) and a scale that
        is always 1 (kept so callers need not special-case the projection).
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        x1 = pts[:, 0] * cy + pts[:, 2] * sy
        z1 = -pts[:, 0] * sy + pts[:, 2] * cy
        y2 = pts[:, 1] * cp - z1 * sp
        z2 = pts[:, 1] * sp + z1 * cp
        unit = view_h / 3.5 * zoom
        sx = view_w * 0.5 + x1 * unit
        sy_ = top + view_h * 0.5 - y2 * unit
        return sx, sy_, z2, np.ones(len(pts))

    # ----- activity ------------------------------------------------------
    def edge_activity(self, state, gain):
        """Brightness (0..1) and sign of each recurrent edge for one state vector."""
        if len(self.w) == 0:
            return np.zeros(0), np.zeros(0)
        drive = state[self.src] * self.w
        strength = (np.abs(self.w) / self.w_max) ** 0.5 * emphasize(np.abs(state[self.src]) * gain)
        return strength, np.sign(drive)

    def readout_contribution(self, state, gain):
        """Signed share of each node in the current readout (weight x state)."""
        contribution = self.readout_nodes * state
        scale = float(np.max(np.abs(self.readout_nodes))) or 1.0
        return np.clip(contribution / scale * gain, -1.0, 1.0)

    def pick(self, sx, sy, x, y, radius=14.0):
        """Index of the node closest to (x, y) within ``radius`` pixels, else None."""
        d2 = (sx - x) ** 2 + (sy - y) ** 2
        best = int(np.argmin(d2))
        return best if d2[best] <= radius * radius else None

    def describe(self, node):
        """Plain-text facts about one node for the info box."""
        return {
            "layer": int(self.layer[node]),
            "depth": float(self.depth[node]),
            "peak": int(self.peak[node]),
            "lifetime": int(self.lifetime[node]),
            "readout_weight": float(self.readout_nodes[node]),
            "input_weight": float(np.sum(self.input_weights[node])),
        }


def display_gain(activity, percentile=95.0, lo=1.0, hi=60.0):
    """Brightness gain so the strong activity uses the full colour range."""
    values = np.abs(np.asarray(activity, dtype=np.float64))
    if values.size == 0:
        return 1.0
    ref = float(np.percentile(values, percentile))
    return float(np.clip(1.0 / max(ref, 1e-6), lo, hi))


def replay_activity(states):
    """Replay activity = state minus the node's own mean over the window.

    A node's resting offset (bias, saturation) carries no news; what the eye
    should follow is how far it swings away from it. Returns (activity, gain).
    """
    states = np.asarray(states, dtype=np.float64)
    activity = states - states.mean(axis=0, keepdims=True)
    return activity, display_gain(activity)


def impulse_activity(response):
    """Impulse-echo activity: the baseline-free response itself, scaled by its peak."""
    response = np.asarray(response, dtype=np.float64)
    return response, display_gain(response, percentile=99.5)
