import csv
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rcstudio.config import ConfigurationError, normalize_config
from rcstudio.core import sample_traces, train_esn, predict_with_model
from rcstudio.storage import Workspace

class TestCore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root / "workspace")
        self.config = normalize_config({
            "dataset": {"length": 420, "noise": 0.0},
            "model": {"units": 36, "seed": 7, "washout": 20},
        })

    def test_train_save_reload_and_reproducibility(self):
        a = train_esn(self.config, self.ws, self.root / "a")
        b = train_esn(self.config, self.ws, self.root / "b")
        self.assertGreater(a["scores"]["test"]["n"], 10)
        self.assertTrue(math.isfinite(a["scores"]["test"]["rmse"]))
        self.assertEqual(a["scores"]["test"], b["scores"]["test"])
        self.assertTrue((self.root / "a" / "model.npz").exists())
        self.assertTrue((self.root / "a" / "predictions.csv").exists())
        traces = sample_traces(self.root / "a", limit=75, units=10)
        self.assertEqual(len(traces["states"]), 75)
        self.assertEqual(len(traces["states"][0]), 10)
        with np.load(self.root / "a" / "traces.npz") as original:
            raw = original["inputs"][:120].astype(float)
        pred = predict_with_model(self.root / "a" / "model.npz", raw)
        self.assertEqual(len(pred), 119)
        self.assertTrue(np.isfinite(pred).all())

    def test_reject_invalid_config(self):
        for bad in (
            {"model": {"units": -1}},
            {"model": {"spectral_radius": float("nan")}},
            {"model": {"horizon": 500}},
            {"split": {"train": 0.9, "validation": 0.20}},
            {"dataset": {"kind": "csv", "dataset_id": "etc/passwd"}},
            {"unknown": {}},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfigurationError):
                    normalize_config(bad)

    def test_import_csv_and_train(self):
        path = self.root / "input.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "value"])
            for i in range(330):
                writer.writerow([i, math.sin(i / 10)])
        meta = self.ws.import_csv(path)
        readings, name = self.ws.read_csv_series(meta["id"], "value")
        self.assertEqual(name, "value")
        self.assertEqual(len(readings), 330)
        cfg = normalize_config({
            "dataset": {"kind": "csv", "dataset_id": meta["id"],
                        "column": "value"},
            "model": {"units": 20, "washout": 5}
        })
        result = train_esn(cfg, self.ws, self.root / "csv-run")
        self.assertEqual(result["samples"], 330)
        self.assertLess(result["scores"]["test"]["rmse"], 1.0)

    def test_workspace_releases_sqlite_file(self):
        holder = tempfile.mkdtemp(prefix="rcstudio-lock-")
        self.addCleanup(shutil.rmtree, holder, ignore_errors=True)
        workspace = Workspace(Path(holder) / "workspace")
        workspace.create_run({"probe": 1})
        workspace.list_runs()
        # On Windows a leaked sqlite handle keeps experiments.sqlite3 locked
        # until gc.collect(); the deletion below must succeed without it.
        shutil.rmtree(holder)
        self.assertFalse(Path(holder).exists())

if __name__ == "__main__":
    unittest.main()
