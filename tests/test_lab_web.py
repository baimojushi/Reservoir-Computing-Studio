"""V0.7 web lab: embedded data, host helpers, numeric parity and static GUI checks.

None of these tests need PySide6. The parity test needs Node.js and is skipped
without it. Pixel-level checks live in docs/V07_LAB_CN.md (headless Chromium).
"""
import ast
import builtins
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rcstudio import lab_web
from rcstudio.intuitive_lab import relaxed_positions, run_states, stable_positions

ROOT = Path(__file__).resolve().parents[1]
PAGE = lab_web.page_path().read_text(encoding="utf-8")


def _block(begin, end):
    return re.search(re.escape(begin) + r"(.*?)" + re.escape(end), PAGE, re.S).group(1)


NETS = json.loads(_block("/*NETWORKS-BEGIN*/", "/*NETWORKS-END*/"))
ENGINE = _block("/*ENGINE-BEGIN*/", "/*ENGINE-END*/")


def _min_spacing(points):
    d = np.hypot(points[:, None, 0] - points[None, :, 0], points[:, None, 1] - points[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return float(d.min())


class TestRelaxedLayout(unittest.TestCase):
    def test_is_deterministic_inside_the_disk_and_evenly_spaced(self):
        for units in (12, 48, 96, 350):
            a = relaxed_positions(units, seed=5)
            self.assertTrue(np.array_equal(a, relaxed_positions(units, seed=5)))
            self.assertEqual(a.shape, (units, 2))
            self.assertLessEqual(float(np.hypot(a[:, 0], a[:, 1]).max()), 0.94 + 1e-9)
            # at least 1.3 / sqrt(N) even for the largest pools: room for glyphs ~2x the V0.6 size
            self.assertGreaterEqual(_min_spacing(a) * math.sqrt(units), 1.3)
            self.assertGreater(_min_spacing(a), 1.8 * _min_spacing(stable_positions(units, seed=5)))

    def test_layout_reads_only_node_count_and_seed(self):
        import inspect
        self.assertEqual(list(inspect.signature(relaxed_positions).parameters), ["units", "seed", "iterations"])
        self.assertFalse(np.array_equal(relaxed_positions(48, seed=1), relaxed_positions(48, seed=2)))


class TestEmbeddedPage(unittest.TestCase):
    def test_page_is_self_contained(self):
        self.assertNotRegex(PAGE, r"<script[^>]+src=")
        self.assertNotRegex(PAGE, r"<link\b")
        self.assertNotRegex(PAGE, r"https?://")
        self.assertNotIn("localStorage", PAGE)
        self.assertLess(len(PAGE.encode("utf-8")), 400_000)

    def test_embedded_pools_have_unit_spectral_radius_and_still_rest_state(self):
        self.assertGreaterEqual(len(NETS), 4)
        for net in NETS:
            n = net["n"]
            W = np.zeros((n, n))
            W[net["dst"], net["src"]] = net["w"]
            self.assertTrue(all(a != b for a, b in zip(net["src"], net["dst"])), "no self loops")
            self.assertAlmostEqual(float(np.max(np.abs(np.linalg.eigvals(W)))), 1.0, places=5)
            self.assertTrue(all(v == 0.0 for v in net["bias"]), "teaching pools rest at exactly zero")
            self.assertLessEqual(max(abs(v) for v in net["win"]), 1.0)
            pos = np.asarray(net["pos"])
            self.assertEqual(pos.shape, (n, 2))
            self.assertGreaterEqual(_min_spacing(pos) * math.sqrt(n), 1.4)
            # with no input and no bias the pool must stay silent
            self.assertEqual(float(np.abs(run_states(W * 0.9, np.asarray(net["win"]), np.zeros(n), 0.35,
                                                     np.zeros(50))).max()), 0.0)

    def test_embedded_pools_match_the_generator_script(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import build_lab_networks
        finally:
            sys.path.pop(0)
        self.assertEqual(json.loads(json.dumps(build_lab_networks.build_networks())), NETS)

    def test_copy_keeps_the_misconception_guards(self):
        for phrase in ("训练时一条也不改", "不能当作回声状态性质（ESP）的证明", "成绩不代表你的正式实验",
                       "远近不代表连接强弱", "它是只读的"):
            self.assertIn(phrase, PAGE)


@unittest.skipUnless(shutil.which("node"), "Node.js not available")
class TestEngineParity(unittest.TestCase):
    def _node(self, script):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "engine.js").write_text(ENGINE, encoding="utf-8")
            (Path(tmp) / "main.js").write_text("const E = require('./engine.js');\n" + script, encoding="utf-8")
            done = subprocess.run(["node", "main.js"], cwd=tmp, capture_output=True, text=True, check=True)
        return json.loads(done.stdout)

    def test_state_update_matches_the_python_engine(self):
        net = NETS[0]
        n = net["n"]
        W = np.zeros((n, n))
        W[net["dst"], net["src"]] = net["w"]
        rho, leak, gain = 1.1, 0.4, 1.3
        u = np.clip(np.random.default_rng(3).normal(0, 0.5, 120), -1, 1)
        expected = run_states(W * rho, np.asarray(net["win"]) * gain, np.asarray(net["bias"]), leak, u)
        got = self._node(
            f"const net = E.makeNet({json.dumps(net)});\n"
            f"const X = E.run(net, {{rho: {rho}, leak: {leak}, gain: {gain}}}, Float64Array.from({json.dumps(u.tolist())}), {{preroll: 0}});\n"
            "console.log(JSON.stringify(Array.from(X)));")
        self.assertLess(float(np.max(np.abs(np.asarray(got).reshape(120, n) - expected))), 1e-6)

    def test_echo_fades_or_persists_and_readout_beats_persistence(self):
        out = self._node(
            f"const net = E.makeNet({json.dumps(NETS[0])}); const T = 120, L = 300;\n"
            "const echo = p => { const b = new Float64Array(T + L), a = E.withPokes(b, [12]);\n"
            "  const e = E.echo(E.run(net, p, a), E.run(net, p, b), T + L, net.n, 12); return [e.len, e.settled]; };\n"
            "const u = E.signal('wave', 240), X = E.run(net, {rho: .9, leak: .35, gain: .85}, u);\n"
            "const f = E.fitReadout(X, u, 240, net.n, {horizon: 6, washout: 20, trainEnd: 168, ridge: 1e-4});\n"
            "console.log(JSON.stringify({d: echo({rho: .9, leak: .35, gain: .85}), s: echo({rho: .3, leak: .6, gain: .85}),\n"
            "  w: echo({rho: 1.55, leak: .5, gain: .85}), rmse: f.rmse, base: f.baseline}));")
        self.assertTrue(out["d"][1] and 15 <= out["d"][0] <= 80, out)          # default: fades in tens of steps
        self.assertTrue(out["s"][1] and out["s"][0] < out["d"][0] / 2, out)     # "forgetful": much shorter
        self.assertFalse(out["w"][1], out)                                        # "can't stop": no decay within the window
        self.assertLess(out["rmse"] * 5, out["base"], out)                        # readout beats "stay put" 5x+


class TestHostHelpers(unittest.TestCase):
    def test_js_call_is_json_only_and_nan_safe(self):
        code = lab_web.js_call("loadFrames", {"truth": [1.0, float("nan")], "note": "</script>\u2028“引号”"})
        self.assertTrue(code.startswith("window.RCLab && window.RCLab.loadFrames("))
        self.assertIn("null", code)
        self.assertNotIn("NaN", code)
        self.assertNotIn("\u2028", code)

    def test_host_messages(self):
        self.assertEqual(lab_web.parse_host_message('rclab:{"n":1,"type":"segment","key":"test"}')["key"], "test")
        for bad in ("储备池实验室", "rclab:{oops", 'rclab:["x"]', 'rclab:{"n":1}', None):
            self.assertIsNone(lab_web.parse_host_message(bad))

    def test_segment_start_and_payload(self):
        frames = {"split_starts": {"train": 0, "validation": 1399, "test": None}}
        self.assertEqual(lab_web.segment_start(frames, "train"), 0)
        self.assertEqual(lab_web.segment_start(frames, "validation"), 1359)
        self.assertIsNone(lab_web.segment_start(frames, "test"))
        payload = lab_web.run_payload({"units": 24}, {"count": 10}, 7)
        self.assertEqual((len(payload["positions"]), payload["runId"]), (24, "7"))

    def test_cli_knows_the_lab_command(self):
        from rcstudio.cli import parser
        self.assertEqual(parser().parse_args(["lab"]).cmd, "lab")


class TestGuiSourceStatically(unittest.TestCase):
    """Cheap guards for the two V0.6 defects that only show up when Qt actually runs."""
    MODULES = ("gui.py", "intuitive_view.py", "lab_web_view.py", "echo_view.py")

    def test_every_qt_name_used_is_imported(self):
        for name in self.MODULES:
            tree = ast.parse((ROOT / "rcstudio" / name).read_text(encoding="utf-8"))
            known = set(dir(builtins))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    known.update((a.asname or a.name).split(".")[0] for a in node.names)
                elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    known.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    known.add(node.id)
            used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            missing = sorted(n for n in used if re.fullmatch(r"Q[A-Z]\w+|Qt|Signal", n) and n not in known)
            self.assertEqual(missing, [], f"{name} uses Qt names it never imports")

    def test_native_sandbox_sliders_are_wired_and_keep_time(self):
        source = (ROOT / "rcstudio" / "intuitive_view.py").read_text(encoding="utf-8")
        self.assertIn("slider.valueChanged.connect(self._params_changed)", source)
        self.assertIn("self._rebuild_sandbox(keep_time=True)", source)
        self.assertIn("norm = float(np.percentile(peak, 95))", source)      # one ring scale for the whole response

    def test_only_a_user_picked_run_may_take_over_the_lab(self):
        source = (ROOT / "rcstudio" / "gui.py").read_text(encoding="utf-8")
        self.assertIn("self.lab.show_run(run_id, focus=picked)", source)
        self.assertNotIn("self.lab.show_run(run_id)\n", source)


if __name__ == "__main__":
    unittest.main()
