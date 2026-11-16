"""V0.7.1 scientific-fidelity regressions for the lab (review findings P0-1, P0-2 and readout).

Engine-side checks need only NumPy. Page-side checks run the page's own numeric block in
Node.js and are skipped without it. Rendering is covered by scripts/check_lab_page.py.
"""
import json
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib import request

import numpy as np

from rcstudio import lab_web
from rcstudio.core import (
    ECHO_ANIM_STEPS, ECHO_STEPS, ECHO_VERDICT_STEPS, _load_model_arrays, classify_echo,
    impulse_response, reservoir_frames, reservoir_structure, train_esn,
)
from rcstudio.learning_support import integration
from rcstudio.learning_support.model import Catalog
from rcstudio.service import RCHandler, RCService
from rcstudio.storage import Workspace

PAGE = lab_web.page_path().read_text(encoding="utf-8")
ENGINE = re.search(r"/\*ENGINE-BEGIN\*/(.*?)/\*ENGINE-END\*/", PAGE, re.S).group(1)


def _numbers(values):
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


class _Runs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        ws = Workspace(root)
        # slow but fading: the case the review used (48 units, radius 0.90, leak 0.10)
        cls.slow = root / "slow"
        train_esn({"dataset": {"length": 900, "noise": 0.0},
                   "model": {"units": 48, "spectral_radius": 0.90, "leak_rate": 0.10}}, ws, cls.slow)
        cls.chaotic = root / "mg"
        train_esn({"dataset": {"kind": "mackey_glass", "length": 1600, "noise": 0.0},
                   "model": {"units": 64}}, ws, cls.chaotic)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _node(self, script, payloads):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "engine.js").write_text(ENGINE, encoding="utf-8")
            for name, payload in payloads.items():
                (Path(tmp) / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            (Path(tmp) / "main.js").write_text("const E = require('./engine.js');\n" + script, encoding="utf-8")
            done = subprocess.run(["node", "main.js"], cwd=tmp, capture_output=True, text=True, check=True)
        return json.loads(done.stdout)


class TestEchoWindow(_Runs):
    """P0-2: a truncated window must never be reported as an echo that never stops."""

    def _rms(self, steps):
        W, win, bias, _readout, leak = _load_model_arrays(self.slow)
        response = impulse_response(W, win, bias, leak, steps=steps)
        return np.sqrt(np.mean(response * response, axis=1))

    def test_same_response_seen_through_96_and_160_steps(self):
        rms = self._rms(ECHO_STEPS)
        short = classify_echo(rms[:ECHO_ANIM_STEPS])
        full = classify_echo(rms)
        self.assertEqual(short["status"], "window_too_short")       # still shrinking, not yet below 5%
        self.assertIsNone(short["length"])
        self.assertEqual(full["status"], "decayed")
        self.assertTrue(ECHO_ANIM_STEPS < full["length"] < ECHO_STEPS, full)

    def test_structure_reports_the_long_window_verdict(self):
        st = reservoir_structure(self.slow)
        echo = st["echo"]
        self.assertEqual((echo["status"], echo["window"]), ("decayed", ECHO_VERDICT_STEPS))
        self.assertEqual(echo["length"], classify_echo(self._rms(ECHO_STEPS))["length"])
        self.assertGreaterEqual(len(echo["curve"]), echo["length"])
        self.assertEqual(st["impulse"]["steps"], ECHO_ANIM_STEPS)     # V0.6 field kept for the native panel
        self.assertEqual(len(st["lifetime"]), st["units"])

    def test_verdict_vocabulary_on_known_curves(self):
        k = np.arange(400, dtype=float)
        self.assertEqual(classify_echo(np.exp(-k / 10))["status"], "decayed")
        self.assertEqual(classify_echo(np.exp(-k / 400))["status"], "window_too_short")
        self.assertEqual(classify_echo(1 + 0.2 * np.sin(k))["status"], "no_decay_seen")
        self.assertEqual(classify_echo(np.zeros(50))["status"], "silent")
        # dropping below the threshold only at the very end is not enough support
        late = np.r_[np.ones(395), np.zeros(5)]
        self.assertNotEqual(classify_echo(late)["status"], "decayed")

    @unittest.skipUnless(shutil.which("node"), "Node.js not available")
    def test_page_uses_the_verdict_and_handles_old_payloads_honestly(self):
        st = reservoir_structure(self.slow)
        frames = reservoir_frames(self.slow, start=0, count=200)
        old = {k: v for k, v in st.items() if k != "echo"}               # what a V0.7.0 service sent
        out = self._node(
            "const now = E.prepareRun(require('./now.json')).echo, old = E.prepareRun(require('./old.json')).echo;\n"
            "const rms = require('./rms.json');\n"
            "console.log(JSON.stringify({now: [now.status, now.len, now.window], old: [old.status, old.window, old.settled],\n"
            "  js96: E.classifyEcho(Float64Array.from(rms.slice(0, 96)), 0).status, js160: E.classifyEcho(Float64Array.from(rms), 0)}));",
            {"now": {"structure": st, "frames": frames, "runId": "a"},
             "old": {"structure": old, "frames": frames, "runId": "b"}, "rms": self._rms(ECHO_STEPS).tolist()})
        self.assertEqual(out["now"], ["decayed", st["echo"]["length"], ECHO_VERDICT_STEPS])
        self.assertEqual(out["old"], ["window_too_short", ECHO_ANIM_STEPS, False])
        self.assertEqual(out["js96"], "window_too_short")
        self.assertEqual((out["js160"]["status"], out["js160"]["length"]), ("decayed", st["echo"]["length"]))
        for phrase in ("窗口不够长，不能判定", "这只是这段窗口内的表现"):
            self.assertIn(phrase, PAGE)
        self.assertNotIn("<b>不停</b>", PAGE)


class TestRunFidelity(_Runs):
    """P0-1 and readout: what the page draws for a real run is the model's own arithmetic."""

    def setUp(self):
        self.st = reservoir_structure(self.chaotic, max_edges=lab_web.EDGE_COUNT)
        self.fr = reservoir_frames(self.chaotic, start=0, count=600)
        ro = self.st["readout"]
        self.u_model = (_numbers(self.fr["inputs"]) - ro["input_mean"][0]) / ro["input_std"][0]
        self.pred_norm = (_numbers(self.fr["prediction"]) - ro["target_mean"]) / ro["target_std"]

    def test_saved_prediction_is_the_sum_of_the_readout_pieces(self):
        ro, X = self.st["readout"], np.asarray(self.fr["states"])
        total = ro["bias"] + ro["bypass"][0] * self.u_model + X @ np.asarray(ro["nodes"])
        seen = np.isfinite(self.pred_norm)
        self.assertGreater(int(seen.sum()), 400)
        # states are stored with 4 decimals, so the match is close rather than exact
        self.assertLess(float(np.max(np.abs(total[seen] - self.pred_norm[seen]))), 0.02)

    @unittest.skipUnless(shutil.which("node"), "Node.js not available")
    def test_page_drives_input_lines_with_the_standardised_input(self):
        out = self._node(
            "const d = E.prepareRun(require('./run.json'));\n"
            "console.log(JSON.stringify({model: Array.from(d.uModel), shown: Array.from(d.u), w: Array.from(d.w), win: d.win,\n"
            "  P: Array.from(d.parts.P), N: Array.from(d.parts.N), Dr: Array.from(d.parts.Dr), net: Array.from(d.parts.net),\n"
            "  bias: d.parts.bias, residual: d.parts.residual, inferred: d.parts.inferred, ext: d.parts.ext}));",
            {"run": {"structure": self.st, "frames": self.fr, "runId": "x"}})
        model, shown = np.asarray(out["model"]), np.asarray(out["shown"])
        self.assertLess(float(np.max(np.abs(model - self.u_model))), 1e-9)          # exactly what the engine feeds in
        self.assertTrue(np.all(np.sign(shown) == np.sign(model)))                   # drawn value never flips its sign
        self.assertLess(float(np.max(np.abs(shown / model - shown[0] / model[0]))), 1e-9)   # one positive factor
        # V0.7.0 rescaled to the window's min/max; on this data that flipped signs, which is what the review caught
        raw = _numbers(self.fr["inputs"])
        old = (raw - raw.min()) / (raw.max() - raw.min()) * 1.8 - 0.9
        self.assertGreater(int(np.sum(np.sign(old) != np.sign(model))), 0)
        # weights are passed through in model units
        self.assertEqual(out["w"], self.st["edges"]["weight"])
        self.assertEqual(out["win"], [row[0] for row in self.st["input_weights"]])
        # signed pieces add up to the prediction the page shows, and match the saved one
        P, N, Dr, net = (np.asarray(out[k]) for k in ("P", "N", "Dr", "net"))
        self.assertTrue(np.all(P >= 0) and np.all(N <= 0))
        self.assertLess(float(np.max(np.abs(P + N + Dr + out["bias"] - net))), 1e-9)
        self.assertFalse(out["inferred"])
        self.assertLess(out["residual"], 0.02)
        self.assertGreater(float(np.mean(P)), float(np.mean(np.abs(net))))          # pieces cancel: that is the lesson

    @unittest.skipUnless(shutil.which("node"), "Node.js not available")
    def test_multi_input_runs_never_pretend_to_know_the_model_input(self):
        st = json.loads(json.dumps(self.st))
        st["inputs"] = 6
        st["input_weights"] = [row * 6 for row in st["input_weights"]]
        st["readout"]["bypass"] = st["readout"]["bypass"] * 6
        out = self._node(
            "const d = E.prepareRun(require('./run.json'));\n"
            "console.log(JSON.stringify({model: d.uModel, win: d.win, inferred: d.parts.inferred}));",
            {"run": {"structure": st, "frames": self.fr, "runId": "x"}})
        self.assertEqual(out, {"model": None, "win": None, "inferred": True})


class TestCourseUsesWhatHappenedInTheLab(unittest.TestCase):
    def test_lab_course_is_gated_by_lab_observations(self):
        catalog = Catalog.load(Path(integration.__file__).parent)
        course = catalog.courses["rc_lab"]
        self.assertEqual(next(iter(catalog.courses)), "rc_lab")                       # listed first
        predicates = [s["predicate"] for s in course["steps"] if s["gate"] == "predicate"]
        self.assertEqual(predicates, ["rc.lab.poke", "rc.lab.nodes", "rc.lab.readout", "rc.lab.knobs"])
        self.assertTrue(all(s["target"] in catalog.entries for s in course["steps"]))

    def test_already_done_chapters_pass_and_quote_the_evidence(self):
        window = SimpleNamespace(lab=SimpleNamespace(is_web=True), lab_progress={})
        ok, why = integration.evaluate(window, "rc.lab.poke", None)
        self.assertFalse(ok)
        self.assertIn("做完后这里会自动通过", why)
        window.lab_progress["poke"] = "余波约 33 步后平息"
        ok, why = integration.evaluate(window, "rc.lab.poke", None)
        self.assertTrue(ok)
        self.assertIn("余波约 33 步后平息", why)
        self.assertIn("不需要重做", why)
        self.assertFalse(integration.evaluate(window, "rc.lab.readout", None)[0])
        native = SimpleNamespace(lab=SimpleNamespace(), lab_progress={})
        self.assertIn("原生实验室也能记录", integration.evaluate(native, "rc.lab.poke", None)[1])

    def test_help_documents_are_generated_from_the_catalog(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        try:
            import sync_help_docs
        finally:
            sys.path.pop(0)
        self.assertEqual(sync_help_docs.sync(write=False), [])


class TestParentProjectContract(unittest.TestCase):
    """E-Brush Studio only ever probes GET /v1/health and checks these two fields."""

    def test_health_is_recognised_by_the_ebrush_read_only_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = RCService(Workspace(Path(temp)))
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), RCHandler)
            httpd.rcservice = backend
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                req = request.Request(f"http://127.0.0.1:{httpd.server_port}/v1/health",
                                      headers={"Accept": "application/json"}, method="GET")
                with request.urlopen(req, timeout=5) as response:      # no token: the probe has none
                    payload = json.loads(response.read(8192))
            finally:
                httpd.shutdown()
                httpd.server_close()
                backend.pool.shutdown(wait=True)
        self.assertEqual(payload.get("app"), "RCStudio")
        self.assertEqual(payload.get("port_role"), "rc_training_only")
        self.assertEqual(payload.get("version"), "0.7.2")


if __name__ == "__main__":
    unittest.main()
