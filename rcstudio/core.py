"""Standalone NumPy/SciPy ESN engine. No notebook, cloud or LLM required."""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .config import normalize_config

class ExperimentCancelled(Exception):
    pass

def build_signal(cfg, workspace):
    d = cfg["dataset"]
    if d["kind"] == "csv":
        values, column = workspace.read_csv_series(d["dataset_id"], d["column"])
        x = np.asarray(values, dtype=np.float64)
        meta = {"kind": "csv", "dataset_id": d["dataset_id"], "column": column}
    elif d["kind"] == "mackey_glass":
        # Delayed Mackey-Glass equation, Euler integration with dt=0.1.
        n = d["length"]
        delay, dt, burn = 170, 0.1, 1800
        y = np.full(n * 10 + burn + delay + 1, 1.2, dtype=np.float64)
        for i in range(delay, len(y) - 1):
            delayed = y[i - delay]
            y[i + 1] = y[i] + dt * (
                0.2 * delayed / (1.0 + delayed**10) - 0.1 * y[i]
            )
        x = y[burn + delay :: 10][:n].copy()
        meta = {"kind": "mackey_glass"}
    else:
        n = d["length"]
        t = np.arange(n, dtype=np.float64)
        x = (np.sin(0.052 * t + 0.000002 * t*t)
             + 0.28 * np.sin(0.135 * t + 0.6))
        meta = {"kind": "sine"}
    if d["kind"] != "csv" and d["noise"] > 0:
        rng = np.random.default_rng(d["seed"])
        x = x + rng.normal(0, d["noise"], size=len(x))
    if len(x) < 150:
        raise ValueError("At least 150 time steps are required")
    if not np.isfinite(x).all():
        raise ValueError("Dataset contains non-finite values")
    return x, meta

def _scores(y, pred):
    if len(y) == 0:
        return {"rmse": None, "mae": None, "r2": None, "n": 0}
    diff = y - pred
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    var_total = float(np.sum((y - y.mean())**2))
    r2 = 1 - float(np.sum(diff**2)) / var_total if var_total > 1e-18 else None
    return {
        "rmse": round(math.sqrt(mse), 8),
        "mae": round(mae, 8),
        "r2": round(r2, 8) if r2 is not None else None,
        "n": int(len(y)),
    }

def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise ExperimentCancelled("Experiment stopped by user")

def train_esn(config, workspace, run_dir, cancel=None, progress=None):
    cfg = normalize_config(config)
    start = time.perf_counter()
    data, data_meta = build_signal(cfg, workspace)
    m = cfg["model"]
    horizon, units = m["horizon"], m["units"]
    num = len(data) - horizon
    tr = int(num * cfg["split"]["train"])
    va = int(num * (cfg["split"]["train"] + cfg["split"]["validation"]))
    if tr <= m["washout"] + 20 or va <= tr + 5 or num <= va + 5:
        raise ValueError("Dataset too short for split/washout/horizon")

    # Fit scaling on train-only time indices. Never use test statistics.
    train_end_raw = tr + horizon
    mu = float(data[:train_end_raw].mean())
    sigma = float(data[:train_end_raw].std())
    if sigma < 1e-12:
        raise ValueError("Training series is constant; cannot fit ESN")
    normalized = (data - mu) / sigma

    rng = np.random.default_rng(m["seed"])
    dense = rng.normal(0, 1, size=(units, units))
    dense[rng.random((units, units)) >= m["density"]] = 0.0
    radius = float(np.max(np.abs(np.linalg.eigvals(dense))))
    if radius < 1e-12:
        raise ValueError("Reservoir initialization has zero spectral radius")
    dense *= m["spectral_radius"] / radius
    W = sparse.csr_matrix(dense)
    win = rng.uniform(-m["input_scaling"], m["input_scaling"], size=units)
    bias = rng.uniform(-0.1, 0.1, size=units)
    leak = m["leak_rate"]
    state = np.zeros(units, dtype=np.float64)
    states = np.empty((num, units), dtype=np.float32)
    for t in range(num):
        _check_cancel(cancel)
        state = (1.0 - leak) * state + leak * np.tanh(
            W @ state + win * normalized[t] + bias
        )
        states[t] = state
        if progress and t % 100 == 0:
            progress(0.05 + 0.66 * (t + 1) / num)
    _check_cancel(cancel)

    # Teacher-forced, observed-input horizon-h forecasting.
    # Features at t contain only observations at or before t.
    X = np.column_stack([
        np.ones(num),
        normalized[:num],
        states.astype(np.float64),
    ])
    y = normalized[horizon:]
    train_x = X[m["washout"]:tr]
    train_y = y[m["washout"]:tr]
    gram = train_x.T @ train_x
    diagonal = np.full(gram.shape[0], m["ridge"], dtype=np.float64)
    diagonal[0] = 0.0  # Do not regularize intercept.
    gram.flat[::gram.shape[0] + 1] += diagonal
    _check_cancel(cancel)
    try:
        readout = np.linalg.solve(gram, train_x.T @ train_y)
    except np.linalg.LinAlgError:
        readout = np.linalg.lstsq(gram, train_x.T @ train_y, rcond=None)[0]
    pred = (X @ readout) * sigma + mu
    pred[:m["washout"]] = np.nan
    true = data[horizon:]
    tags = np.full(num, 2, dtype=np.int8)
    tags[:tr] = 0
    tags[tr:va] = 1

    scores = {
        "train": _scores(true[m["washout"]:tr], pred[m["washout"]:tr]),
        "validation": _scores(true[tr:va], pred[tr:va]),
        "test": _scores(true[va:], pred[va:]),
        "persistence_baseline_test": _scores(true[va:], data[va:num]),
    }
    if progress:
        progress(0.85)
    _check_cancel(cancel)

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "model.npz",
        W=dense, win=win, bias=bias, readout=readout,
        mean=np.array(mu), std=np.array(sigma),
        leak=np.array(leak), horizon=np.array(horizon), washout=np.array(m["washout"]),
        spectral_radius=np.array(m["spectral_radius"]), density=np.array(m["density"]),
        input_scaling=np.array(m["input_scaling"]),
    )
    np.savez_compressed(
        out / "traces.npz",
        timestep=np.arange(horizon, len(data), dtype=np.int32),
        inputs=data[:num].astype(np.float32),
        truth=true.astype(np.float32),
        prediction=pred.astype(np.float32),
        split=tags, states=states,
    )
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestep", "split", "input", "truth", "prediction"])
        labels = ("train", "validation", "test")
        for i in range(num):
            writer.writerow([
                i + horizon, labels[int(tags[i])], float(data[i]),
                float(true[i]), "" if math.isnan(pred[i]) else float(pred[i]),
            ])
    elapsed = time.perf_counter() - start
    summary = {
        "algorithm": "Echo State Network (teacher-forced)",
        "forecast_type": "observed-input h-step forecasting",
        "data": data_meta,
        "samples": int(len(data)),
        "reservoir_units": units,
        "effective_spectral_radius": round(float(m["spectral_radius"]), 8),
        "readout": "ridge regression",
        "scores": scores,
        "runtime_seconds": round(elapsed, 3),
        "split": {"train_end_timestep": tr + horizon - 1,
                  "validation_end_timestep": va + horizon - 1},
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if progress:
        progress(1.0)
    return summary

def sample_traces(run_dir, limit=600, units=24):
    limit = max(20, min(2000, int(limit)))
    units = max(1, min(64, int(units)))
    trace_file = Path(run_dir) / "traces.npz"
    if not trace_file.exists():
        raise FileNotFoundError("Traces not found for this run")
    with np.load(trace_file, allow_pickle=False) as z:
        n, total_units = z["states"].shape
        take = np.unique(np.linspace(0, n - 1, min(n, limit), dtype=int))
        selected = np.unique(np.linspace(
            0, total_units - 1, min(total_units, units), dtype=int
        ))
        def finite_list(vals):
            return [None if not math.isfinite(float(v)) else round(float(v), 7)
                    for v in vals]
        return {
            "timestep": z["timestep"][take].tolist(),
            "truth": finite_list(z["truth"][take]),
            "prediction": finite_list(z["prediction"][take]),
            "split": z["split"][take].tolist(),
            "selected_units": selected.tolist(),
            "states": np.round(
                z["states"][np.ix_(take, selected)].astype(np.float64), 5
            ).tolist(),
        }

# ---------------------------------------------------------------------------
# Echo analysis: gives every reservoir node a data-independent "echo depth" so
# the GUI can stack nodes into layers (input on top, readout at the bottom).
# ---------------------------------------------------------------------------
ECHO_STEPS = 160        # length of the impulse-response analysis, in time steps
ECHO_ANIM_STEPS = 96    # impulse frames sent to the GUI for the echo animation
ECHO_AMPLITUDE = 1.0    # pulse height in normalized (1 sigma) input units
ECHO_VERDICT_STEPS = 2000   # window used to decide whether the echo fades (V0.7.1)
ECHO_THRESHOLD = 0.05       # "faded" means below this fraction of the peak
ECHO_CURVE_MAX = 600        # at most this many points of the echo curve are sent to the GUI


def _load_model_arrays(run_dir):
    model_file = Path(run_dir) / "model.npz"
    if not model_file.exists():
        raise FileNotFoundError("Model not found for this run")
    with np.load(model_file, allow_pickle=False) as z:
        W = np.asarray(z["W"], dtype=np.float64)
        win = np.asarray(z["win"], dtype=np.float64)
        bias = np.asarray(z["bias"], dtype=np.float64)
        readout = np.asarray(z["readout"], dtype=np.float64).reshape(-1)
        leak = float(z["leak"])
    if win.ndim == 1:
        win = win[:, None]
    if W.shape != (len(bias), len(bias)) or win.shape[0] != len(bias):
        raise ValueError("Inconsistent model arrays")
    if len(readout) != 1 + win.shape[1] + len(bias):
        raise ValueError("Readout size does not match the reservoir")
    return W, win, bias, readout, leak


def impulse_response(W, win, bias, leak, steps=ECHO_STEPS, amplitude=ECHO_AMPLITUDE):
    """Response of every node to one input pulse, minus the pulse-free baseline.

    Uses exactly the update rule of train_esn / train_motion. Row t holds the
    state change t steps after the pulse (t = 0 is the step that receives it).
    """
    units, n_in = win.shape
    sparse_w = sparse.csr_matrix(W)
    drive = win @ np.full(n_in, float(amplitude))
    pulse = np.zeros(units)
    base = np.zeros(units)
    out = np.empty((int(steps), units), dtype=np.float64)
    for t in range(int(steps)):
        pulse = (1.0 - leak) * pulse + leak * np.tanh(
            sparse_w @ pulse + (drive if t == 0 else 0.0) + bias)
        base = (1.0 - leak) * base + leak * np.tanh(sparse_w @ base + bias)
        out[t] = pulse - base
    return out


def classify_echo(rms, threshold=ECHO_THRESHOLD):
    """Say what a finite observation of an echo curve does and does not show.

    ``rms[k]`` is the size of the state difference k steps after the pulse.
    Three outcomes, never "it runs forever":

    * ``decayed``          - it fell below ``threshold`` x peak and stayed there for
                             a margin of the window; ``length`` is the step count.
    * ``window_too_short`` - still above the threshold at the end, but shrinking.
    * ``no_decay_seen``    - still above the threshold and not shrinking in this window.

    A finite window can support the first statement only. It never proves the
    echo state property, and the last two are statements about the window.
    """
    rms = np.asarray(rms, dtype=np.float64).reshape(-1)
    window = int(len(rms))
    peak = float(rms.max()) if window else 0.0
    result = {"status": "silent", "length": 0, "window": window,
              "threshold": float(threshold), "peak": peak, "trend": None}
    if window < 8 or peak < 1e-12:
        return result
    last = int(np.flatnonzero(rms > threshold * peak)[-1])
    margin = max(3, window // 20)
    if last < window - margin:
        result.update(status="decayed", length=last + 1)
        return result
    quarter = max(2, window // 4)
    early = float(np.mean(rms[-2 * quarter:-quarter]))
    late = float(np.mean(rms[-quarter:]))
    trend = late / early if early > 1e-15 else 1.0
    result.update(status="window_too_short" if trend < 0.8 else "no_decay_seen",
                  length=None, trend=round(trend, 4))
    return result


def echo_metrics(response):
    """Per-node echo depth (response centroid), peak step and echo lifetime."""
    mag = np.abs(response)
    steps, units = mag.shape
    t = np.arange(steps, dtype=np.float64)[:, None]
    total = mag.sum(axis=0)
    centroid = np.where(total > 1e-12, (mag * t).sum(axis=0) / np.maximum(total, 1e-12), 0.0)
    peak_value = mag.max(axis=0)
    peak_step = mag.argmax(axis=0)
    significant = (mag >= 0.1 * peak_value[None, :]) & (peak_value[None, :] > 1e-12)
    lifetime = np.where(significant, t, 0.0).max(axis=0)
    return centroid, peak_step.astype(int), lifetime.astype(int)


def reservoir_structure(run_dir, max_edges=1200):
    """Static description of a trained reservoir for the unified V0.6 field view."""
    W, win, bias, readout, leak = _load_model_arrays(run_dir)
    max_edges = max(50, min(4000, int(max_edges)))
    units, n_in = win.shape
    model_file = Path(run_dir) / "model.npz"
    with np.load(model_file, allow_pickle=False) as model:
        if "input_mean" in model:
            input_mean = np.asarray(model["input_mean"], dtype=np.float64).reshape(-1)
            input_std = np.asarray(model["input_std"], dtype=np.float64).reshape(-1)
            target_mean = float(model["target_mean"])
            target_std = float(model["target_std"])
            horizon = int(model["horizon"])
            washout = int(model["washout"]) if "washout" in model else 0
            project = str(model["project"]) if "project" in model else "generic_esn"
        else:
            input_mean = np.array([float(model["mean"])], dtype=np.float64)
            input_std = np.array([float(model["std"])], dtype=np.float64)
            target_mean = float(model["mean"])
            target_std = float(model["std"])
            horizon = int(model["horizon"])
            washout = int(model["washout"]) if "washout" in model else 0
            project = "generic_esn"
    # One long run decides whether the echo fades; its first ECHO_STEPS rows are the
    # per-node analysis used since V0.6. Judging from the 96 animation frames alone
    # mislabelled slow but fading models as "never settles".
    long_response = impulse_response(W, win, bias, leak, steps=ECHO_VERDICT_STEPS)
    response = long_response[:ECHO_STEPS]
    echo_rms = np.sqrt(np.mean(long_response * long_response, axis=1))
    echo = classify_echo(echo_rms)
    shown = ECHO_CURVE_MAX if echo["length"] is None else int(min(
        ECHO_CURVE_MAX, max(ECHO_ANIM_STEPS, echo["length"] * 2)))
    echo["curve"] = np.round(echo_rms[:shown], 7).tolist()
    centroid, peak_step, lifetime = echo_metrics(response)

    off_diagonal = W.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    dst, src = np.nonzero(off_diagonal)          # W[i, j] carries node j -> node i
    weight = off_diagonal[dst, src]
    total_edges = int(len(weight))
    order = np.argsort(-np.abs(weight), kind="stable")[:max_edges]
    src, dst, weight = src[order], dst[order], weight[order]   # strongest first

    animated = response[:ECHO_ANIM_STEPS]
    return {
        "units": int(units),
        "inputs": int(n_in),
        "leak": round(leak, 6),
        "echo_steps": int(ECHO_STEPS),
        "depth": np.round(centroid, 4).tolist(),
        "peak": peak_step.tolist(),
        "lifetime": lifetime.tolist(),
        "edges": {
            "src": src.tolist(), "dst": dst.tolist(),
            "weight": np.round(weight, 5).tolist(),
            "total": total_edges, "shown": int(len(weight)),
            "selection": "largest_absolute_weight",
        },
        "input_weights": np.round(win, 4).tolist(),
        "readout": {
            "bias": round(float(readout[0]), 8),
            "bypass": np.round(readout[1:1 + n_in], 8).tolist(),
            "nodes": np.round(readout[1 + n_in:], 8).tolist(),
            "input_mean": np.round(input_mean, 8).tolist(),
            "input_std": np.round(input_std, 8).tolist(),
            "target_mean": round(target_mean, 8),
            "target_std": round(target_std, 8),
        },
        "forecast": {
            "horizon": horizon,
            "washout": washout,
            "project": project,
        },
        "echo": echo,
        "impulse": {
            "amplitude": float(ECHO_AMPLITUDE),
            "steps": int(len(animated)),
            "states": np.round(animated, 5).tolist(),
        },
    }


def reservoir_frames(run_dir, start=0, count=360, stride=1):
    """Contiguous window of full-width reservoir states for animation."""
    count = max(10, min(1200, int(count)))
    stride = max(1, min(20, int(stride)))
    start = max(0, int(start))
    trace_file = Path(run_dir) / "traces.npz"
    if not trace_file.exists():
        raise FileNotFoundError("Traces not found for this run")
    with np.load(trace_file, allow_pickle=False) as z:
        total = int(z["states"].shape[0])
        if start >= total:
            raise ValueError("start is beyond the recorded trace")
        index = np.arange(start, min(total, start + count * stride), stride)
        split = z["split"]
        first = {}
        for code, name in enumerate(("train", "validation", "test")):
            hits = np.flatnonzero(split == code)
            first[name] = int(hits[0]) if len(hits) else None

        def finite_list(vals):
            return [None if not math.isfinite(float(v)) else round(float(v), 6)
                    for v in vals]
        # Generic single-series runs are one episode. EBrush V0.7.1 and older
        # runs do not contain boundary metadata: do not guess their boundaries.
        has_episodes = all(k in z for k in
                           ("episode_id", "episode_start", "local_step",
                            "target_local_step", "episode_recording_ids"))
        episode_meta = {}
        if has_episodes:
            identifiers = z["episode_id"][index].astype(np.int64)
            recorder = z["episode_recording_ids"]
            session = z["episode_session_ids"] if "episode_session_ids" in z else None
            episode_meta = {
                "episode_id": identifiers.tolist(),
                "episode_start": z["episode_start"][index].astype(bool).tolist(),
                "local_step": z["local_step"][index].astype(int).tolist(),
                "target_local_step": z["target_local_step"][index].astype(int).tolist(),
                "recording_id": [str(recorder[i]) for i in identifiers],
            }
            if session is not None:
                episode_meta["session_id"] = [str(session[i]) for i in identifiers]
        return {
            "has_episode_metadata": bool(has_episodes),
            **episode_meta,
            "start": int(index[0]), "count": int(len(index)),
            "stride": stride, "total": total, "split_starts": first,
            "timestep": z["timestep"][index].tolist(),
            "inputs": finite_list(z["inputs"][index]),
            "truth": finite_list(z["truth"][index]),
            "prediction": finite_list(z["prediction"][index]),
            "split": split[index].tolist(),
            "states": np.round(z["states"][index].astype(np.float64), 4).tolist(),
        }

def predict_with_model(model_path, series):
    """Run a saved model on observed input series (without retraining)."""
    data = np.asarray(series, dtype=np.float64)
    if data.ndim != 1 or len(data) < 3 or not np.isfinite(data).all():
        raise ValueError("Expected >=3 finite numeric observations")
    with np.load(model_path, allow_pickle=False) as z:
        W = sparse.csr_matrix(z["W"])
        win = z["win"]
        bias = z["bias"]
        readout = z["readout"]
        mean = float(z["mean"])
        std = float(z["std"])
        leak = float(z["leak"])
        horizon = int(z["horizon"])
    if len(data) <= horizon:
        raise ValueError("Series must exceed forecast horizon")
    s = np.zeros(len(win))
    predicted = []
    for i in range(len(data) - horizon):
        scaled = (data[i] - mean) / std
        s = (1 - leak) * s + leak * np.tanh(W @ s + win * scaled + bias)
        features = np.concatenate(([1.0, scaled], s))
        predicted.append(float(features @ readout * std + mean))
    return predicted
