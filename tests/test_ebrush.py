"""Hardware-free EBrush protocol, isolation, metrics and boundary regression."""
import csv
import hashlib
import io
import json
import math
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

import numpy as np

from rcstudio.client import Client, ServiceError
from rcstudio.ebrush import (
    BRLOG_SIGNATURE, DATA_FRAME, PROJECT_ID, _packet_data, _decode_cobs,
    import_source, make_demo, list_sessions, normalize_ebrush_config,
    read_brlog, train_motion,
)
from rcstudio.storage import Workspace
from rcstudio.core import sample_traces


def cobs(raw):
    result = bytearray()
    cursor = 0
    while cursor < len(raw):
        next_zero = raw.find(b"\0", cursor)
        if next_zero < 0:
            next_zero = len(raw)
        while next_zero-cursor >= 254:
            result.append(255)
            result.extend(raw[cursor:cursor+254])
            cursor += 254
        result.append(next_zero-cursor+1)
        result.extend(raw[cursor:next_zero])
        cursor = next_zero+1
    if not raw or raw[-1] == 0:
        result.append(1)
    return bytes(result)


def make_log(start=0, n=260, damaged=False):
    out = bytearray(BRLOG_SIGNATURE)
    for i in range(n):
        x = [0]*26
        x[0] = i
        x[1] = 11
        x[2] = i*5000
        x[4] = i*5000
        x[5] = 1 << 5
        x[16] = int(400*math.sin(.03*i)+start)
        x[17] = int(200*math.cos(.018*i))
        x[18] = int(180*math.sin(.016*i))
        x[19], x[20], x[21] = 400, 100, 8200
        x[22] = 1.0
        payload = DATA_FRAME.pack(*x)
        header = struct.pack("<HBBHHI", 0x5242, 1, 1, len(payload), 0, i)
        packet = header + payload
        crc = zlib.crc32(packet) & 0xffffffff
        if damaged and i == 4:
            crc ^= 1
        wire = cobs(packet+struct.pack("<I",crc))
        out += struct.pack("<QH", i*5_000_000, len(wire))+wire
    return bytes(out)


class TestEBrushOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ws = Workspace(self.root / "workspace")

    def test_brlog_crc_and_units(self):
        blob = make_log(n=280, damaged=True)
        episodes, quality = read_brlog(io.BytesIO(blob))
        self.assertEqual(quality["crc_or_cobs_errors"], 1)
        self.assertEqual(quality["data_frames"], 279)
        self.assertEqual(quality["valid_imu_frames"], 279)
        # CRC hole creates a discontinuity, resulting 1 viable episode.
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].shape[1], 6)
        self.assertTrue(np.isfinite(episodes[0]).all())
        self.assertAlmostEqual(episodes[0][0, 5], 8200*.122*.001*9.80665, places=4)
        self.assertTrue(np.all(np.abs(episodes[0][:, 0]) < 1.0))

    def test_manifest_sha_is_verified_without_touching_source(self):
        original = self.root / "capture.brlog"
        original.write_bytes(make_log(n=310))
        original_bytes = original.read_bytes()
        manifest = self.root / "handoff.json"
        manifest.write_text(json.dumps({
            "schema": "ebrush.handoff.v1",
            "source": {"absolute_path": str(original), "size_bytes": len(original_bytes),
                       "sha256": hashlib.sha256(original_bytes).hexdigest()},
        }), encoding="utf-8")
        result = import_source(self.ws, manifest)
        self.assertEqual(result["imported_episodes"], 1)
        self.assertEqual(original.read_bytes(), original_bytes)
        original.write_bytes(original_bytes + b"x")
        with self.assertRaisesRegex(ValueError, "size differs"):
            import_source(self.ws, manifest)

    def test_zip_and_session_group_isolation(self):
        src = self.root / "captures.zip"
        with zipfile.ZipFile(src, "w") as z:
            z.writestr("logs/a.brlog", make_log(4, 340))
            z.writestr("logs/b.brlog", make_log(99, 350))
            z.writestr("../ignore.exe", b"NOT TO EXTRACT")
        result = import_source(self.ws, src)
        self.assertEqual(result["independent_recordings"], 2)
        self.assertEqual(result["imported_episodes"], 2)
        self.assertEqual(len(list_sessions(self.ws)), 2)
        cfg = {"model": {"units": 16, "washout": 20}}
        with self.assertRaisesRegex(ValueError, "independent capture files"):
            train_motion(cfg, self.ws, self.root / "blocked")
        self.assertFalse((self.root / "ignore.exe").exists())

    def test_demo_training_no_cross_recording_leakage(self):
        info = make_demo(self.ws, count=5, length=280)
        config = normalize_ebrush_config({
            "dataset": {"session_ids": info["session_ids"]},
            "model": {"units": 18, "washout": 24, "horizon": 10, "seed": 9},
        })
        result = train_motion(config, self.ws, self.root / "run")
        self.assertEqual(result["project"], PROJECT_ID)
        self.assertEqual(result["nominal_horizon_ms"], 50)
        self.assertGreater(result["scores"]["test"]["n"], 10)
        groups = result["recording_groups"]
        self.assertFalse(set(groups["train"]) & set(groups["validation"]))
        self.assertFalse(set(groups["train"]) & set(groups["test"]))
        self.assertFalse(set(groups["test"]) & set(groups["validation"]))
        self.assertTrue((self.root / "run" / "model.npz").exists())
        trace = sample_traces(self.root / "run", limit=90, units=8)
        self.assertEqual(len(trace["states"]), 90)
        with np.load(self.root / "run" / "model.npz", allow_pickle=False) as z:
            self.assertEqual(z["win"].shape, (18, 6))
        self.assertTrue(math.isfinite(result["scores"]["test"]["rmse"]))

    def test_csv_requires_explicit_units_and_mixed_demo_guard(self):
        csv_file = self.root / "unnamed_units.csv"
        with csv_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gx","gy","gz","ax","ay","az"])
            for i in range(210):
                writer.writerow([1,2,3,4,5,6])
        with self.assertRaisesRegex(ValueError, "CSV requires 6"):
            import_source(self.ws, csv_file)
        real = self.root / "real.brlog"
        real.write_bytes(make_log(n=350))
        imported = import_source(self.ws, real)
        demo = make_demo(self.ws, count=4, length=220)
        config = {"dataset": {"session_ids": imported["session_ids"] + demo["session_ids"]},
                  "model": {"units": 14, "washout": 15}}
        with self.assertRaisesRegex(ValueError, "Mixed synthetic"):
            train_motion(config, self.ws, self.root / "blocked")


if __name__ == "__main__":
    unittest.main()
