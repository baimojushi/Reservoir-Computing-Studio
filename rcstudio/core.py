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
        leak=np.array(leak), horizon=np.array(horizon),
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
