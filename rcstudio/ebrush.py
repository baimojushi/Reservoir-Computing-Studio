"""EBrush Motion Predictor V1 — offline research pipeline owned by RC Studio.

No serial, camera, calibration or device-control imports. Source files are read,
validated and copied into RC-managed *derived sessions*. Different windows from
one recording retain one recording_id and MUST NOT cross train/test boundaries.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import re
import struct
import time
import uuid
import zipfile
import zlib
from pathlib import Path

import numpy as np
from scipy import sparse

from .config import ConfigurationError, DEFAULT, normalize_config
from .core import ExperimentCancelled, _check_cancel, _scores

BRLOG_SIGNATURE = b"BRUSH_BRL1\n"
PACKET_HEADER = struct.Struct("<HBBHHI")
DATA_FRAME = struct.Struct("<IIQQQIIhhhhBBBBhhhhhhhffff")
RECORD_HEADER = struct.Struct("<QH")
GYRO_SCALE = 35.0 * 0.001 * math.pi / 180.0
ACCEL_SCALE = 0.122 * 0.001 * 9.80665
FEATURE_NAMES = ["gx_rad_s", "gy_rad_s", "gz_rad_s", "ax_m_s2", "ay_m_s2", "az_m_s2"]
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_RETAINED_PER_RECORDING = 60_000
MAX_EPISODE = 5_000
MIN_EPISODE = 150
MAX_SESSIONS_PER_RUN = 100
PROJECT_ID = "ebrush_motion_v1"

DEFAULT_EBRUSH = {
    "project": PROJECT_ID,
    "dataset": {"kind": "ebrush_sessions", "session_ids": []},
    "model": {**copy.deepcopy(DEFAULT["model"]), "units": 96, "horizon": 10, "washout": 40},
    "split": {"train": 0.67, "validation": 0.17},
}


def normalize_ebrush_config(config=None):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ConfigurationError("EBrush config must be a JSON object")
    cfg = copy.deepcopy(DEFAULT_EBRUSH)
    for key, val in config.items():
        if key == "project":
            if val != PROJECT_ID:
                raise ConfigurationError("Unknown project")
            continue
        if key not in ("model", "split", "dataset") or not isinstance(val, dict):
            raise ConfigurationError(f"Unsupported EBrush configuration section: {key}")
        for field, item in val.items():
            if field not in cfg[key]:
                raise ConfigurationError(f"Unsupported EBrush config field: {key}.{field}")
            cfg[key][field] = item
    if cfg["dataset"]["kind"] != "ebrush_sessions":
        raise ConfigurationError("EBrush dataset.kind must be ebrush_sessions")
    ids = cfg["dataset"]["session_ids"]
    if not isinstance(ids, list) or len(ids) > MAX_SESSIONS_PER_RUN or any(
        not isinstance(i, str) or not re.fullmatch(r"[a-f0-9]{32}", i) for i in ids
    ) or len(set(ids)) != len(ids):
        raise ConfigurationError("session_ids must be a unique list of up to 100 valid IDs")
    validated = normalize_config({"model": cfg["model"], "split": cfg["split"]})
    cfg["model"] = validated["model"]
    cfg["split"] = validated["split"]
    return cfg


def _decode_cobs(data):
    output = bytearray()
    index = 0
    while index < len(data):
        code = data[index]
        if not code or index + code > len(data) + 1:
            raise ValueError("Truncated COBS frame")
        index += 1
        end = index + code - 1
        if end > len(data):
            raise ValueError("Truncated COBS frame")
        output.extend(data[index:end])
        index = end
        if code < 255 and index < len(data):
            output.append(0)
    return output


def _packet_data(wire):
    raw = _decode_cobs(wire)
    if len(raw) < PACKET_HEADER.size + 4:
        raise ValueError("Packet too short")
    magic, version, packet_type, payload_len, flags, packet_seq = PACKET_HEADER.unpack_from(raw)
    if magic != 0x5242 or version != 1 or len(raw) != PACKET_HEADER.size + payload_len + 4:
        raise ValueError("Invalid packet header or size")
    crc, = struct.unpack_from("<I", raw, len(raw) - 4)
    if crc != (zlib.crc32(raw[:-4]) & 0xFFFFFFFF):
        raise ValueError("CRC mismatch")
    if packet_type != 0x01:
        return None
    if payload_len != DATA_FRAME.size:
        raise ValueError("Unexpected DATA frame length")
    return DATA_FRAME.unpack_from(raw, PACKET_HEADER.size)


def _build_sessions(rows, stats):
    sessions = []
    segment = []
    last_time = None
    last_id = None
    retained = 0

    def flush():
        nonlocal segment
        if len(segment) >= MIN_EPISODE:
            sessions.append(np.asarray(segment, dtype=np.float32))
        else:
            stats["short_segments_discarded"] += int(bool(segment))
        segment = []

    for frame_id, sample_us, imu_valid, vals in rows:
        stats["data_frames"] += 1
        if not imu_valid:
            stats["invalid_imu_frames"] += 1
            flush()
            last_time = None
            last_id = None
            continue
        stats["valid_imu_frames"] += 1
        discontinuity = (
            last_time is not None and
            (sample_us <= last_time or sample_us - last_time > 12000 or
             (frame_id - last_id) % (2**32) != 1)
        )
        if discontinuity:
            stats["discontinuities"] += 1
            flush()
        last_time, last_id = sample_us, frame_id
        if retained < MAX_RETAINED_PER_RECORDING:
            # 6 physical-unit channels, target is future ||gyro|| in rad/s.
            segment.append([float(v) for v in vals])
            retained += 1
            if len(segment) >= MAX_EPISODE:
                flush()
        elif segment:
            flush()
    flush()
    stats["retained_frames"] = retained
    stats["episodes_kept"] = len(sessions)
    return sessions


def read_brlog(stream):
    """Read E-Brush BRUSH_BRL1 records without importing any device driver."""
    if stream.read(len(BRLOG_SIGNATURE)) != BRLOG_SIGNATURE:
        raise ValueError("Not an E-Brush BRUSH_BRL1 log")
    stats = {k: 0 for k in (
        "records", "data_frames", "valid_imu_frames", "invalid_imu_frames",
        "crc_or_cobs_errors", "discontinuities", "short_segments_discarded",
    )}

    def rows():
        while True:
            block = stream.read(RECORD_HEADER.size)
            if not block:
                break
            if len(block) != RECORD_HEADER.size:
                raise ValueError("Truncated BRLOG record header")
            host_ns, frame_len = RECORD_HEADER.unpack(block)
            if not frame_len or frame_len > 4096:
                raise ValueError("Invalid BRLOG frame length")
            wire = stream.read(frame_len)
            if len(wire) != frame_len:
                raise ValueError("Truncated BRLOG frame")
            stats["records"] += 1
            try:
                data = _packet_data(wire)
            except ValueError:
                stats["crc_or_cobs_errors"] += 1
                continue
            if data is None:
                continue
            status, drop = int(data[5]), int(data[6])
            imu_valid = bool(status & (1 << 5)) and not bool(drop & (1 << 4))
            gyro = [data[i] * GYRO_SCALE for i in (16, 17, 18)]
            accel = [data[i] * ACCEL_SCALE for i in (19, 20, 21)]
            yield int(data[0]), int(data[2]), imu_valid, gyro + accel

    sessions = _build_sessions(rows(), stats)
    if stats["data_frames"] == 0:
        raise ValueError("No E-Brush DATA frames found")
    return sessions, stats


def read_ebrush_csv(stream):
    """Only accept named, unambiguous physical-unit columns (or raw counts)."""
    reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
    names = set(reader.fieldnames or ())
    physical = set(FEATURE_NAMES)
    raw = {"gx_raw", "gy_raw", "gz_raw", "ax_raw", "ay_raw", "az_raw"}
    if physical <= names:
        keys = FEATURE_NAMES
        scale = [1.] * 6
    elif raw <= names:
        keys = ["gx_raw", "gy_raw", "gz_raw", "ax_raw", "ay_raw", "az_raw"]
        scale = [GYRO_SCALE] * 3 + [ACCEL_SCALE] * 3
    else:
        raise ValueError("CSV requires 6 gx/gy/gz_rad_s & ax/ay/az_m_s2 columns, or *_raw columns")
    time_key = "sample_time_us" if "sample_time_us" in names else "timestamp_us" if "timestamp_us" in names else None
    stats = {k: 0 for k in (
        "records", "data_frames", "valid_imu_frames", "invalid_imu_frames",
        "crc_or_cobs_errors", "discontinuities", "short_segments_discarded",
    )}

    def rows():
        for index, row in enumerate(reader):
            if index >= 1_000_000:
                raise ValueError("CSV exceeds 1,000,000 rows")
            try:
                values = [float(row[k]) * scale[j] for j, k in enumerate(keys)]
                sample_us = int(row[time_key]) if time_key else index * 5000
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Nonfinite data")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid EBrush CSV at row {index+2}: {exc}") from exc
            yield index, sample_us, True, values
    episodes = _build_sessions(rows(), stats)
    return episodes, stats


def _sha256_stream(reader):
    h = hashlib.sha256()
    for chunk in iter(lambda: reader.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def _resolve_source(path):
    source = Path(path).expanduser().resolve(strict=True)
    if source.suffix.lower() == ".json":
        manifest = json.loads(source.read_text(encoding="utf-8"))
        if manifest.get("schema") != "ebrush.handoff.v1":
            raise ValueError("Only ebrush.handoff.v1 manifests are supported")
        entry = manifest.get("source", {})
        candidate = Path(entry.get("absolute_path", "")).expanduser().resolve(strict=True)
        if candidate.suffix.lower() not in (".brlog", ".csv") or not candidate.is_file():
            raise ValueError("Manifest source must be an existing .brlog or .csv")
        if entry.get("size_bytes") != candidate.stat().st_size:
            raise ValueError("Manifest source size differs: file changed")
        with candidate.open("rb") as inp:
            digest = _sha256_stream(inp)
        if entry.get("sha256") != digest:
            raise ValueError("Manifest SHA-256 mismatch: file changed")
        return candidate, {"manifest": str(source), "verified_sha256": digest}
    return source, {}


def _save_episodes(workspace, episodes, source_name, group_id, stats, audit, synthetic=False):
    directory = workspace.root / "ebrush_sessions"
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, feature_data in enumerate(episodes):
        if feature_data.shape[0] < MIN_EPISODE:
            continue
        session_id = uuid.uuid4().hex
        meta = {
            "id": session_id, "project": PROJECT_ID, "recording_id": group_id,
            "source": source_name, "episode_index": index,
            "samples": int(feature_data.shape[0]), "sample_rate_hz_nominal": 200,
            "duration_s_nominal": round(len(feature_data) / 200, 3),
            "features": FEATURE_NAMES, "target": "future_angular_speed_rad_s",
            "synthetic": bool(synthetic), "imported_at": time.time(),
            "quality": stats, "audit": audit,
        }
        np.savez_compressed(directory / f"{session_id}.npz", features=feature_data)
        (directory / f"{session_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(meta)
    return saved


def list_sessions(workspace):
    directory = workspace.root / "ebrush_sessions"
    if not directory.is_dir():
        return []
    results = []
    for path in directory.glob("*.json"):
        if not re.fullmatch(r"[a-f0-9]{32}", path.stem):
            continue
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            if (directory / f"{path.stem}.npz").is_file() and info.get("id") == path.stem:
                results.append(info)
        except (OSError, ValueError):
            pass
    return sorted(results, key=lambda info: (info["imported_at"], info["id"]), reverse=True)


def import_source(workspace, path):
    """Explicit, local, offline import; no E-Brush API calls or device access."""
    source, audit = _resolve_source(path)
    if not source.is_file():
        raise ValueError("Source must be a file")
    if source.suffix.lower() not in (".brlog", ".csv", ".zip"):
        raise ValueError("Accepts BRLOG, EBrush CSV, ZIP or handoff JSON")
    sources = []
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = [m for m in archive.infolist() if not m.is_dir() and
                       Path(m.filename).suffix.lower() == ".brlog"]
            if not members:
                raise ValueError("ZIP does not contain .brlog captures")
            if len(members) > 16 or sum(m.file_size for m in members) > 512 * 1024 * 1024:
                raise ValueError("Too many or too large captures in ZIP")
            for member in members:
                if member.file_size > MAX_SOURCE_BYTES:
                    raise ValueError("Individual log exceeds 256 MiB")
                with archive.open(member) as inp:
                    digest = _sha256_stream(inp)
                with archive.open(member) as inp:
                    episodes, quality = read_brlog(inp)
                sources.append({"name": member.filename, "digest": digest,
                                "episodes": episodes, "quality": quality})
    else:
        if source.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("Source exceeds 256 MiB")
        with source.open("rb") as inp:
            digest = _sha256_stream(inp)
        with source.open("rb") as inp:
            episodes, quality = (read_brlog(inp) if source.suffix.lower() == ".brlog"
                                 else read_ebrush_csv(inp))
        sources.append({"name": source.name, "digest": digest,
                        "episodes": episodes, "quality": quality})
    collected = []
    reports = []
    for item in sources:
        group_id = item["digest"]  # Same recording never splits across train/val/test.
        recordings = _save_episodes(workspace, item["episodes"], item["name"],
                                    group_id, item["quality"], audit)
        collected.extend(recordings)
        reports.append({"source": item["name"], "sha256": group_id,
                        "episodes": len(recordings), "quality": item["quality"]})
    return {"project": PROJECT_ID, "imported_episodes": len(collected),
            "session_ids": [m["id"] for m in collected], "sources": reports,
            "independent_recordings": len({m["recording_id"] for m in collected}),
            "warning": "Training requires 3 independent recordings. Windows cut from one capture remain one recording."}


def make_demo(workspace, count=6, length=650):
    if count < 3 or count > 12 or length < 200:
        raise ValueError("Demo requires 3..12 recordings and >=200 steps")
    results = []
    for index in range(count):
        rng = np.random.default_rng(8020 + index)
        t = np.arange(length, dtype=np.float64) / 200
        base = 0.38 + index * 0.025
        wx = base * np.sin(5.0*t + .17*index) + .05 * rng.normal(size=length)
        wy = (base * 0.65) * np.cos(3.1*t + .12*index) + .04 * rng.normal(size=length)
        wz = .24 * np.sin(6.3*t + .3*index) + .03 * rng.normal(size=length)
        ax = .60 * np.cos(5.0*t) + .12 * rng.normal(size=length)
        ay = .80 * np.sin(3.1*t) + .12 * rng.normal(size=length)
        az = 9.80665 + .25 * np.cos(6.3*t) + .12 * rng.normal(size=length)
        f = np.column_stack((wx, wy, wz, ax, ay, az)).astype(np.float32)
        results.extend(_save_episodes(workspace, [f], f"demo_{index+1}.csv",
                                      "demo-" + uuid.uuid4().hex,
                                      {"data_frames": length, "valid_imu_frames": length,
                                       "synthetic": True}, {"demo": True}, synthetic=True))
    return {"project": PROJECT_ID, "demo": True,
            "session_ids": [r["id"] for r in results], "independent_recordings": count}


def _choose_splits(sessions, model_seed, train_fraction, validation_fraction):
    by_group = {}
    for session in sessions:
        by_group.setdefault(session["recording_id"], []).append(session)
    if len(by_group) < 3:
        raise ValueError("EBrush requires >=3 independent capture files for session-level train/validation/test; "
                         "windows from one .brlog cannot count as independent recordings. "
                         "Use 'project ebrush demo' for synthetic verification.")
    groups = sorted(by_group)
    rng = np.random.default_rng(model_seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, min(n-2, int(round(n*train_fraction))))
    n_val = max(1, min(n-n_train-1, int(round(n*validation_fraction))))
    labels = {g: ("train" if i < n_train else "validation" if i < n_train+n_val else "test")
              for i, g in enumerate(groups)}
    return labels


def preflight_training(config, workspace):
    """Cheap metadata-only check so GUI/CLI fail before scheduling an invalid run."""
    cfg = normalize_ebrush_config(config)
    entries = list_sessions(workspace)
    wanted = set(cfg["dataset"]["session_ids"])
    selected = [e for e in entries if e["id"] in wanted] if wanted else entries
    if wanted and len(selected) != len(wanted):
        raise ValueError("Selected session ID not found")
    if len(selected) > MAX_SESSIONS_PER_RUN:
        raise ValueError("Choose at most 100 sessions")
    eligible = [e for e in selected if e["samples"] >= cfg["model"]["washout"] +
                cfg["model"]["horizon"] + 25]
    _choose_splits(eligible, cfg["model"]["seed"], cfg["split"]["train"],
                   cfg["split"]["validation"])
    if any(e["synthetic"] for e in eligible) and not all(e["synthetic"] for e in eligible):
        raise ValueError("Cannot mix real captures with synthetic demos; select sessions explicitly")
    return {"eligible_sessions": len(eligible),
            "independent_recordings": len({e["recording_id"] for e in eligible}),
            "synthetic": all(e["synthetic"] for e in eligible)}


def train_motion(config, workspace, run_dir, cancel=None, progress=None):
    cfg = normalize_ebrush_config(config)
    started = time.perf_counter()
    metas = list_sessions(workspace)
    wanted = set(cfg["dataset"]["session_ids"])
    selected = [m for m in metas if m["id"] in wanted] if wanted else metas
    if wanted and len(selected) != len(wanted):
        raise ValueError("One or more selected EBrush sessions do not exist")
    if len(selected) > MAX_SESSIONS_PER_RUN:
        raise ValueError("Too many EBrush sessions; select <=100")
    m = cfg["model"]
    horizon, units, washout = m["horizon"], m["units"], m["washout"]
    selected = [entry for entry in selected if entry["samples"] >= washout + horizon + 25]
    labels = _choose_splits(selected, m["seed"], cfg["split"]["train"],
                            cfg["split"]["validation"])
    episodes = []
    for meta in selected:
        path = workspace.root / "ebrush_sessions" / f"{meta['id']}.npz"
        with np.load(path, allow_pickle=False) as z:
            f = z["features"].astype(np.float64)
        if f.shape != (meta["samples"], 6) or not np.isfinite(f).all():
            raise ValueError(f"Corrupt session data: {meta['id']}")
        target = np.linalg.norm(f[:, :3], axis=1)
        episodes.append((meta, f, target))
    if any(e["synthetic"] for e, _, _ in episodes) and not all(
        e["synthetic"] for e, _, _ in episodes
    ):
        raise ValueError("Mixed synthetic demo and real captures are not permitted in one experiment; "
                         "select --config dataset.session_ids explicitly")
    _check_cancel(cancel)
    train_episodes = [(e, f, y) for e, f, y in episodes if labels[e["recording_id"]] == "train"]
    xtrain = np.concatenate([f for _, f, _ in train_episodes], axis=0)
    ytrain = np.concatenate([y for _, _, y in train_episodes], axis=0)
    xmu, xsd = xtrain.mean(0), xtrain.std(0)
    xsd[xsd < 1e-8] = 1.0
    ymu, ysd = float(ytrain.mean()), float(ytrain.std())
    if ysd < 1e-9:
        raise ValueError("Training angular speed is constant")
    del xtrain, ytrain
    rng = np.random.default_rng(m["seed"])
    W = rng.normal(size=(units, units))
    W[rng.random((units, units)) >= m["density"]] = 0.0
    radius = float(np.max(np.abs(np.linalg.eigvals(W))))
    if radius < 1e-10:
        raise ValueError("Reservoir is disconnected; change seed/density")
    W *= m["spectral_radius"] / radius
    sparse_w = sparse.csr_matrix(W)
    win = rng.uniform(-m["input_scaling"], m["input_scaling"], size=(units, 6))
    bias = rng.uniform(-.1, .1, size=units)
    leak = m["leak_rate"]
    features_n = 1 + 6 + units
    gram = np.zeros((features_n, features_n), dtype=np.float64)
    rhs = np.zeros(features_n, dtype=np.float64)

    def scan(f):
        normalized = (f - xmu) / xsd
        state = np.zeros(units, dtype=np.float64)
        states = np.empty((len(f) - horizon, units), dtype=np.float32)
        for i in range(len(states)):
            if i % 128 == 0:
                _check_cancel(cancel)
            state = (1 - leak)*state + leak*np.tanh(sparse_w@state + win@normalized[i] + bias)
            states[i] = state
        X = np.column_stack((np.ones(len(states)), normalized[:len(states)], states))
        return X, states

    for index, (meta, f, target) in enumerate(train_episodes):
        X, states = scan(f)
        yy = (target[horizon:] - ymu) / ysd
        gram += X[washout:].T @ X[washout:]
        rhs += X[washout:].T @ yy[washout:]
        if progress:
            progress(.1 + .25*(index+1) / len(train_episodes))
    diag = np.full(features_n, m["ridge"])
    diag[0] = 0.0
    gram.flat[::features_n + 1] += diag
    _check_cancel(cancel)
    try:
        readout = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        readout = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    if progress:
        progress(.50)

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: [] for k in ("timestep", "inputs", "truth", "prediction", "split", "states")}
    scores_data = {p: {"true": [], "pred": [], "base": []} for p in ("train", "validation", "test")}
    summary_groups = {p: [] for p in ("train", "validation", "test")}
    count = 0
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session_id", "recording_id", "split", "step", "target_step", "truth_rad_s",
                         "prediction_rad_s", "persistence_rad_s"])
        for index, (meta, f, target) in enumerate(episodes):
            _check_cancel(cancel)
            split = labels[meta["recording_id"]]
            if meta["recording_id"] not in summary_groups[split]:
                summary_groups[split].append(meta["recording_id"])
            X, states = scan(f)
            prediction = (X @ readout)*ysd + ymu
            target_out = target[horizon:]
            persist = target[:len(prediction)]
            valid = slice(washout, len(prediction))
            scores_data[split]["true"].append(target_out[valid])
            scores_data[split]["pred"].append(prediction[valid])
            scores_data[split]["base"].append(persist[valid])
            n = len(prediction)
            keep = np.arange(n) + count
            plot_pred = prediction.astype(np.float32)
            plot_pred[:washout] = np.nan
            arrays["timestep"].append(keep + horizon)
            arrays["inputs"].append(target[:n].astype(np.float32))
            arrays["truth"].append(target_out.astype(np.float32))
            arrays["prediction"].append(plot_pred)
            arrays["split"].append(np.full(n, ("train", "validation", "test").index(split), dtype=np.int8))
            arrays["states"].append(states)
            for step in range(washout, n):
                writer.writerow([meta["id"], meta["recording_id"], split, step, step+horizon,
                                 float(target_out[step]), float(prediction[step]), float(persist[step])])
            count += n
            if progress:
                progress(.52 + .43*(index + 1)/len(episodes))
    _check_cancel(cancel)
    np.savez_compressed(out / "model.npz", W=W, win=win, bias=bias, readout=readout,
                        input_mean=xmu, input_std=xsd, target_mean=np.array(ymu),
                        target_std=np.array(ysd), horizon=np.array(horizon), leak=np.array(leak),
                        project=np.array(PROJECT_ID))
    np.savez_compressed(out / "traces.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    scores = {}
    for name, item in scores_data.items():
        truth = np.concatenate(item["true"])
        predicted = np.concatenate(item["pred"])
        baseline = np.concatenate(item["base"])
        scores[name] = _scores(truth, predicted)
        scores[f"persistence_baseline_{name}"] = _scores(truth, baseline)
    summary = {
        "project": PROJECT_ID, "algorithm": "Multi-input ESN / ridge readout",
        "forecast_type": "observed-input, next 50 ms by default; not free-running",
        "target": "angular_speed_rad_s", "input_features": FEATURE_NAMES,
        "horizon_steps": horizon, "nominal_horizon_ms": horizon * 5,
        "data": {"kind": "ebrush_sessions", "sessions": len(episodes),
                 "independent_recordings": len(labels),
                 "synthetic_included": any(meta["synthetic"] for meta, _, _ in episodes)},
        "samples": int(count), "reservoir_units": units,
        "split_protocol": "recording-level disjoint; washout per episode; train-only normalization",
        "recording_groups": summary_groups, "scores": scores,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if progress:
        progress(1.0)
    return summary
