"""Small, isolated ESN sandbox and stable graph layout for RC Studio V0.6.

The sandbox is deliberately separate from saved experiments. It uses the same
leaky-tanh ESN state equation as the training engine, but it never writes to a
workspace, changes a trained model, or claims that a finite probe proves ESP.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def stable_positions(units: int, seed: int = 202611) -> np.ndarray:
    """Return a deterministic, topology-independent 2-D disk layout.

    Coordinates are a visual arrangement only. They are generated independently
    of weights, impulse response, node rank and time; geometric distance carries
    no model meaning.
    """
    units = max(1, int(units))
    rng = np.random.default_rng(int(seed) + units * 7919)
    points: list[tuple[float, float]] = []
    min_d2 = (0.72 / math.sqrt(units)) ** 2
    attempts = 0
    limit = units * 120
    while len(points) < units and attempts < limit:
        attempts += 1
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        radius = math.sqrt(float(rng.uniform(0.025, 0.98)))
        point = (radius * math.cos(angle), radius * math.sin(angle))
        if all((point[0] - x) ** 2 + (point[1] - y) ** 2 >= min_d2 for x, y in points):
            points.append(point)
    while len(points) < units:
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        radius = math.sqrt(float(rng.uniform(0.025, 0.98)))
        points.append((radius * math.cos(angle), radius * math.sin(angle)))
    return np.asarray(points, dtype=np.float64)


def relaxed_positions(units: int, seed: int = 202611, iterations: int = 260) -> np.ndarray:
    """Evenly spaced, deterministic, topology-independent disk layout (V0.7).

    ``stable_positions`` throws random darts, which leaves clumps and holes and
    forces small node glyphs. This variant starts from the same darts and lets
    the points push each other apart inside the unit disk until spacing is
    close to uniform. It still reads no weights, ranks or responses: distance
    on screen carries no model meaning.
    """
    units = max(1, int(units))
    points = stable_positions(units, seed=seed).copy()
    if units == 1:
        return np.zeros((1, 2), dtype=np.float64)
    iterations = min(int(iterations), max(60, 16000 // units))   # bound cost for large pools
    target = 1.82 / math.sqrt(units)          # near hexagonal spacing for a unit disk
    limit = 0.94
    for step in range(max(1, int(iterations))):
        delta = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=2))
        np.fill_diagonal(dist, np.inf)
        push = np.clip((target - dist) / target, 0.0, None)
        direction = delta / np.maximum(dist, 1e-9)[:, :, None]
        move = np.sum(direction * push[:, :, None], axis=1)
        rate = 0.22 * target * (1.0 - 0.6 * step / max(1, iterations))
        points = points + rate * move
        radius = np.sqrt(np.sum(points * points, axis=1))
        outside = radius > limit
        if np.any(outside):
            points[outside] *= (limit / radius[outside])[:, None]
    return points


def make_signal(kind: str, steps: int = 220, custom: Any = None) -> np.ndarray:
    """Create a bounded one-channel input for hands-on exploration."""
    steps = max(32, int(steps))
    t = np.arange(steps, dtype=np.float64)
    phase = t / max(1, steps - 1)
    if kind == "pulse":
        values = np.zeros(steps, dtype=np.float64)
        start = max(3, steps // 5)
        width = max(2, steps // 32)
        values[start:start + width] = 0.9
    elif kind == "step":
        values = np.where(t < steps * 0.28, -0.15, 0.62).astype(np.float64)
    elif kind == "chirp":
        values = 0.62 * np.sin(2.0 * math.pi * (1.1 * phase + 3.4 * phase * phase))
    elif kind == "handdrawn" and custom is not None:
        raw = np.asarray(custom, dtype=np.float64).reshape(-1)
        if raw.size < 2 or not np.isfinite(raw).all():
            values = 0.65 * np.sin(2.0 * math.pi * phase * 2.3)
        else:
            values = np.interp(np.linspace(0.0, 1.0, steps),
                               np.linspace(0.0, 1.0, raw.size), raw)
    else:
        values = (0.58 * np.sin(2.0 * math.pi * phase * 2.4)
                  + 0.20 * np.sin(2.0 * math.pi * phase * 7.1 + 0.55))
    return np.clip(values, -1.0, 1.0)


def _network(units: int, density: float, seed: int, target_radius: float,
             input_scaling: float):
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(0.0, 1.0, (units, units))
    mask = rng.random((units, units)) < float(density)
    np.fill_diagonal(mask, False)
    raw *= mask
    radius = float(np.max(np.abs(np.linalg.eigvals(raw)))) if np.any(raw) else 0.0
    if radius < 1e-12:
        # A deterministic sparse ring guarantees a nonzero radius for small sizes.
        raw[np.arange(units), (np.arange(units) + 1) % units] = 1.0
        radius = float(np.max(np.abs(np.linalg.eigvals(raw))))
    W = raw * (float(target_radius) / radius)
    win = rng.uniform(-1.0, 1.0, size=units) * float(input_scaling)
    bias = rng.uniform(-0.08, 0.08, size=units)
    measured = float(np.max(np.abs(np.linalg.eigvals(W))))
    return W.astype(np.float64), win.astype(np.float64), bias.astype(np.float64), measured


def run_states(W: np.ndarray, win: np.ndarray, bias: np.ndarray, leak: float,
               inputs: np.ndarray, initial: np.ndarray | None = None) -> np.ndarray:
    """Run the same leaky-tanh state update used by the ESN training engine."""
    W = np.asarray(W, dtype=np.float64)
    win = np.asarray(win, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    inputs = np.asarray(inputs, dtype=np.float64).reshape(-1)
    state = np.zeros(len(win), dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64).copy()
    if state.shape != win.shape:
        raise ValueError("initial state shape does not match input weights")
    leak = float(np.clip(leak, 0.001, 1.0))
    states = np.empty((len(inputs), len(win)), dtype=np.float32)
    for t, value in enumerate(inputs):
        state = (1.0 - leak) * state + leak * np.tanh(W @ state + win * value + bias)
        states[t] = state
    return states


def _edge_sample(W: np.ndarray, max_edges: int, seed: int):
    dst, src = np.nonzero(W)
    weights = W[dst, src]
    total = int(len(weights))
    if total > max_edges:
        rng = np.random.default_rng(seed + 104729)
        selected = np.sort(rng.choice(total, size=max_edges, replace=False))
        method = "uniform_random_sample"
        src, dst, weights = src[selected], dst[selected], weights[selected]
    else:
        method = "all_nonzero_edges"
    return (src.astype(np.int64), dst.astype(np.int64), weights.astype(np.float64),
            total, method)


def simulate_sandbox(*, spectral_radius: float = 0.90, leak_rate: float = 0.35,
                     input_scaling: float = 0.85, signal: str = "sine",
                     custom_input: Any = None, units: int = 64, density: float = 0.10,
                     seed: int = 2026, steps: int = 220, max_edges: int = 1100) -> dict:
    """Run a complete, reproducible what-if experiment on an isolated ESN.

    Three trajectories share exactly the same reservoir parameters:
    1) baseline input from a zero state;
    2) same input from a different initial state (finite ESP probe);
    3) same initial state with one extra input pulse (perturbation response).
    """
    units = max(12, min(160, int(units)))
    spectral_radius = float(np.clip(spectral_radius, 0.05, 1.8))
    leak_rate = float(np.clip(leak_rate, 0.02, 1.0))
    input_scaling = float(np.clip(input_scaling, 0.05, 3.0))
    W, win, bias, measured_radius = _network(
        units, density, seed, spectral_radius, input_scaling)
    raw_inputs = make_signal(signal, steps, custom_input)
    train_boundary = max(36, int(steps * 0.70))
    input_mean = float(np.mean(raw_inputs[:train_boundary]))
    input_std = float(np.std(raw_inputs[:train_boundary])) or 1.0
    normalized_inputs = (raw_inputs - input_mean) / input_std
    baseline = run_states(W, win, bias, leak_rate, normalized_inputs)

    rng = np.random.default_rng(seed + 1)
    alternate_initial = rng.normal(0.0, 0.42, units)
    alternate = run_states(W, win, bias, leak_rate, normalized_inputs, alternate_initial)

    pulse_step = max(4, min(len(raw_inputs) - 4, len(raw_inputs) // 3))
    changed_raw = raw_inputs.copy()
    changed_raw[pulse_step] = np.clip(changed_raw[pulse_step] + 0.72, -1.0, 1.0)
    changed_input = (changed_raw - input_mean) / input_std
    perturbed = run_states(W, win, bias, leak_rate, changed_input)

    initial_gap = np.linalg.norm(baseline - alternate, axis=1) / math.sqrt(units)
    pulse_effect = np.linalg.norm(perturbed - baseline, axis=1) / math.sqrt(units)

    # Fit a tiny, explicit demo readout: predict the future input waveform.
    # The first 70% is used for fitting; the final 30% is held out for display.
    horizon = 6
    washout = 10
    target = np.full(len(raw_inputs), np.nan, dtype=np.float64)
    target[:len(raw_inputs) - horizon] = raw_inputs[horizon:]
    fit_end = max(washout + 12, train_boundary - horizon)
    train_idx = np.arange(washout, fit_end, dtype=np.int64)
    test_idx = np.arange(train_boundary, len(raw_inputs) - horizon, dtype=np.int64)
    features = np.column_stack((np.ones(len(raw_inputs)), normalized_inputs, baseline))
    target_mean = float(np.nanmean(target[train_idx]))
    target_std = float(np.nanstd(target[train_idx])) or 1.0
    normalized_target = (target - target_mean) / target_std
    gram = features[train_idx].T @ features[train_idx]
    penalty = np.eye(gram.shape[0], dtype=np.float64) * 1e-4
    penalty[0, 0] = 0.0
    try:
        readout = np.linalg.solve(gram + penalty,
                                  features[train_idx].T @ normalized_target[train_idx])
    except np.linalg.LinAlgError:
        readout = np.linalg.pinv(gram + penalty) @ (
            features[train_idx].T @ normalized_target[train_idx])
    prediction = np.full(len(raw_inputs), np.nan, dtype=np.float64)
    valid_idx = np.arange(washout, len(raw_inputs) - horizon, dtype=np.int64)
    prediction[valid_idx] = (features[valid_idx] @ readout) * target_std + target_mean
    split = np.where(np.arange(len(raw_inputs)) < train_boundary, 0, 2).astype(np.int64)
    test_rmse = (float(np.sqrt(np.mean((prediction[test_idx] - target[test_idx]) ** 2)))
                 if len(test_idx) else float("nan"))
    start_gap = float(initial_gap[0]) if len(initial_gap) else 0.0
    end_gap = float(initial_gap[-1]) if len(initial_gap) else 0.0
    ratio = end_gap / max(start_gap, 1e-12)
    saturation = float(np.mean(np.abs(baseline) >= 0.97))
    if saturation >= 0.18:
        status = "部分节点接近饱和：输入差异可能被压缩。"
    elif ratio <= 0.05:
        status = "本次探针中，两种初始状态的差异明显衰减。"
    elif ratio >= 1.10:
        status = "本次探针中，初始状态差异没有明显衰减。"
    else:
        status = "本次窗口内，初始状态差异变化温和；可延长窗口或更换输入复测。"
    note = "有限探针只描述当前网络、输入和时间窗口；它不能单独证明回声状态性质（ESP）。"

    src, dst, edge_weights, edge_total, edge_method = _edge_sample(W, max_edges, seed)
    return {
        "kind": "sandbox",
        "units": units,
        "positions": stable_positions(units, seed=seed + 3001),
        "edge_src": src,
        "edge_dst": dst,
        "edge_weights": edge_weights,
        "edge_total": edge_total,
        "edge_shown": len(edge_weights),
        "edge_method": edge_method,
        "W": W,
        "input_weights": win[:, None],
        "readout_nodes": readout[2:].astype(np.float64),
        "bypass": np.array([readout[1]], dtype=np.float64),
        "readout_bias": float(readout[0]),
        "input_mean": np.array([input_mean], dtype=np.float64),
        "input_std": np.array([input_std], dtype=np.float64),
        "output_scale": target_std,
        "output_offset": target_mean,
        "direct_input_available": True,
        "states": baseline,
        "perturbed_states": perturbed,
        "delta_states": perturbed - baseline,
        "alternate_states": alternate,
        "initial_gap": initial_gap,
        "pulse_effect": pulse_effect,
        "inputs": raw_inputs,
        "truth": target,
        "prediction": prediction,
        "split": split,
        "horizon": horizon,
        "washout": washout,
        "test_rmse": test_rmse,
        "pulse_step": pulse_step,
        "spectral_radius": measured_radius,
        "leak_rate": leak_rate,
        "input_scaling": input_scaling,
        "saturation_fraction": saturation,
        "initial_gap_ratio": ratio,
        "status": status,
        "status_note": note,
        "seed": int(seed),
        "signal": signal,
        "readout_task": "future_input_waveform",
        "readout_train_fraction": train_boundary / len(raw_inputs),
    }
