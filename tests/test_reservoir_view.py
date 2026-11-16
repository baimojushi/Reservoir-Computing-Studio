"""Echo analysis, layered scene geometry and the two read-only reservoir endpoints."""
import math
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from rcstudio.client import Client, ServiceError
from rcstudio.core import (
    echo_metrics, impulse_response, reservoir_frames, reservoir_structure, train_esn,
)
from rcstudio.ebrush import PROJECT_ID, make_demo, normalize_ebrush_config, train_motion
from rcstudio.reservoir_scene import (
    EchoScene, assign_layers, display_gain, impulse_activity, layer_summary,
    replay_activity, ink_node, mix, emphasize, grain_tile, BLACK, WHITE, DIM, GROUND_TOP,
    GROUND_BOTTOM, REST_LEVEL, GRAIN_STRENGTH, INPUT_Y,
)
from rcstudio.service import RCHandler, RCService
from rcstudio.storage import Workspace


def train(workspace, folder, **model):
    config = {"dataset": {"length": 700, "noise": 0.0}, "model": {"units": 48, **model}}
    train_esn(config, workspace, folder)
    return folder


class TestEchoAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.ws = Workspace(Path(cls.temp.name))
        cls.run_dir = train(cls.ws, Path(cls.temp.name) / "base")
        cls.structure = reservoir_structure(cls.run_dir, max_edges=400)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_structure_shape_and_ordering(self):
        st = self.structure
        n = st["units"]
        self.assertEqual((n, st["inputs"]), (48, 1))
        for key in ("depth", "peak", "lifetime"):
            self.assertEqual(len(st[key]), n)
        edges = st["edges"]
        self.assertEqual(len(edges["src"]), edges["shown"])
        self.assertLessEqual(edges["shown"], 400)
        self.assertGreaterEqual(edges["total"], edges["shown"])
        strength = np.abs(edges["weight"])
        self.assertTrue(np.all(np.diff(strength) <= 1e-12), "edges must come strongest first")
        self.assertTrue(all(a != b for a, b in zip(edges["src"], edges["dst"])), "no self loops")
        self.assertEqual(len(st["readout"]["nodes"]), n)
        self.assertEqual(len(st["readout"]["bypass"]), 1)
        self.assertEqual(np.asarray(st["input_weights"]).shape, (n, 1))
        self.assertEqual(np.asarray(st["impulse"]["states"]).shape, (st["impulse"]["steps"], n))
        self.assertEqual(st["edges"]["selection"], "largest_absolute_weight")
        self.assertEqual(st["forecast"]["horizon"], 1)
        self.assertEqual(st["forecast"]["washout"], 50)
        self.assertEqual(len(st["readout"]["input_mean"]), 1)
        self.assertGreater(st["readout"]["target_std"], 0)

    def test_analysis_is_deterministic(self):
        again = reservoir_structure(self.run_dir, max_edges=400)
        self.assertEqual(again["depth"], self.structure["depth"])
        self.assertEqual(again["edges"]["src"], self.structure["edges"]["src"])

    def test_impulse_response_fades(self):
        """Echo state property: a single pulse leaves no lasting trace."""
        from rcstudio.core import _load_model_arrays
        W, win, bias, _readout, leak = _load_model_arrays(self.run_dir)
        response = impulse_response(W, win, bias, leak, steps=160)
        early = np.abs(response[:10]).max()
        late = np.abs(response[-20:]).max()
        self.assertGreater(early, 0.01)
        self.assertLess(late, 0.05 * early)

    def test_metrics_on_known_response(self):
        steps = np.arange(30, dtype=float)
        fast = np.exp(-steps / 2.0)
        slow = np.exp(-steps / 10.0)
        centroid, peak, lifetime = echo_metrics(np.column_stack((fast, slow, np.zeros(30))))
        self.assertLess(centroid[0], centroid[1])
        self.assertEqual(list(peak[:2]), [0, 0])
        self.assertLess(lifetime[0], lifetime[1])
        self.assertEqual((centroid[2], lifetime[2]), (0.0, 0))

    def test_slower_leak_and_larger_radius_deepen_the_echo(self):
        """The layers must react to the parameters the user can change."""
        def median_depth(**model):
            folder = Path(self.temp.name) / ("m%d" % len(list(Path(self.temp.name).iterdir())))
            return float(np.median(reservoir_structure(train(self.ws, folder, **model))["depth"]))
        fast, slow = median_depth(leak_rate=0.9), median_depth(leak_rate=0.1)
        self.assertGreater(slow, 2.0 * fast)
        small, large = median_depth(spectral_radius=0.5), median_depth(spectral_radius=1.2)
        self.assertGreater(large, 2.0 * small)

    def test_edge_cap_is_enforced(self):
        capped = reservoir_structure(self.run_dir, max_edges=60)
        self.assertEqual(capped["edges"]["shown"], 60)
        self.assertEqual(capped["edges"]["total"], self.structure["edges"]["total"])

    def test_missing_model_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            reservoir_structure(Path(self.temp.name) / "absent")


class TestFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.run_dir = train(Workspace(Path(cls.temp.name)), Path(cls.temp.name) / "run")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_window_slicing_and_split_markers(self):
        frames = reservoir_frames(self.run_dir, start=0, count=100)
        self.assertEqual((frames["start"], frames["count"]), (0, 100))
        self.assertEqual(np.asarray(frames["states"]).shape, (100, 48))
        starts = frames["split_starts"]
        self.assertEqual(starts["train"], 0)
        self.assertLess(starts["validation"], starts["test"])
        later = reservoir_frames(self.run_dir, start=starts["test"], count=40)
        self.assertEqual(set(later["split"]), {2})

    def test_washout_predictions_become_none(self):
        frames = reservoir_frames(self.run_dir, start=0, count=30)
        self.assertIsNone(frames["prediction"][0])          # inside the washout
        self.assertIsNotNone(frames["inputs"][0])

    def test_stride_and_clamping(self):
        frames = reservoir_frames(self.run_dir, start=10, count=50, stride=5)
        self.assertEqual(frames["timestep"][1] - frames["timestep"][0], 5)
        self.assertEqual(reservoir_frames(self.run_dir, count=1)["count"], 10)      # lower clamp

    def test_start_beyond_trace_is_rejected(self):
        with self.assertRaises(ValueError):
            reservoir_frames(self.run_dir, start=10_000_000)


class TestScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        run = train(Workspace(Path(cls.temp.name)), Path(cls.temp.name) / "run")
        cls.structure = reservoir_structure(run, max_edges=500)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_layers_are_balanced_and_ordered_by_depth(self):
        depth = np.asarray(self.structure["depth"])
        for count in range(1, 9):
            layer = assign_layers(depth, count)
            sizes = np.bincount(layer, minlength=count)
            self.assertLessEqual(sizes.max() - sizes.min(), 1)
            order = np.argsort(depth, kind="stable")
            self.assertTrue(np.all(np.diff(layer[order]) >= 0), "deeper nodes never sit above shallower ones")
        ties = assign_layers(np.zeros(10), 4)               # all equal depth still splits evenly
        self.assertEqual(sorted(np.bincount(ties)), [2, 2, 3, 3])

    def test_summary_depth_increases_with_layer(self):
        scene = EchoScene(self.structure, n_layers=4)
        depths = [row["depth"] for row in scene.summary]
        self.assertEqual(depths, sorted(depths))
        self.assertEqual(sum(row["count"] for row in scene.summary), scene.units)
        rows = layer_summary(scene.depth, scene.lifetime, scene.layer, 4)
        self.assertEqual(rows, scene.summary)

    def test_input_on_top_readout_at_bottom(self):
        for count in (2, 3, 5, 8):
            scene = EchoScene(self.structure, n_layers=count)
            self.assertTrue(np.all(scene.input_positions[:, 1] == INPUT_Y))
            self.assertLess(scene.readout_position[1], scene.positions[:, 1].min())
            self.assertGreater(scene.input_positions[0, 1], scene.positions[:, 1].max())
            layer_y = [scene.positions[scene.layer == i, 1][0] for i in range(count)]
            self.assertEqual(layer_y, sorted(layer_y, reverse=True), "layer 1 is the highest")
            # the same ordering survives the default camera on screen (smaller y = higher)
            _sx, sy, _z, _p = scene.project(scene.positions, 0.5, 0.34, 1.0, 1000, 420)
            _ix, iy, _, _ = scene.project(scene.input_positions, 0.5, 0.34, 1.0, 1000, 420)
            _rx, ry, _, _ = scene.project(scene.readout_position, 0.5, 0.34, 1.0, 1000, 420)
            means = [float(sy[scene.layer == i].mean()) for i in range(count)]
            self.assertEqual(means, sorted(means))
            self.assertLess(iy[0], min(means))
            self.assertGreater(ry[0], max(means))

    def test_projection_stays_inside_the_view(self):
        scene = EchoScene(self.structure, n_layers=4)
        for yaw in (0.0, 0.5, 1.5, 3.0):
            sx, sy, _z, persp = scene.project(scene.positions, yaw, 0.34, 1.0, 1000, 420, top=58)
            self.assertTrue(np.all(np.isfinite(sx)) and np.all(np.isfinite(sy)))
            self.assertTrue(np.all(persp > 0))
            self.assertGreater(sy.min(), 58 - 20)
            self.assertLess(sy.max(), 58 + 420 + 20)

    def test_edge_limit_keeps_the_strongest(self):
        scene = EchoScene(self.structure, n_layers=4, max_edges=500)
        scene.set_edge_limit(40)
        self.assertEqual(len(scene.w), 40)
        self.assertAlmostEqual(scene.w_max, float(np.max(np.abs(self.structure["edges"]["weight"]))))
        scene.set_edge_limit(10 ** 6)
        self.assertEqual(len(scene.w), len(self.structure["edges"]["weight"]))

    def test_edge_activity_is_bounded_and_signed(self):
        scene = EchoScene(self.structure, n_layers=4)
        rng = np.random.default_rng(3)
        state = rng.normal(0, 0.5, scene.units)
        strength, sign = scene.edge_activity(state, gain=3.0)
        self.assertTrue(np.all((strength >= 0) & (strength <= 1)))
        expected = np.sign(state[scene.src] * scene.w)
        self.assertTrue(np.array_equal(sign, expected))
        silent, _ = scene.edge_activity(np.zeros(scene.units), gain=3.0)
        self.assertEqual(float(silent.max()), 0.0)

    def test_pick_chooses_nearest_node_within_radius(self):
        scene = EchoScene(self.structure, n_layers=4)
        sx, sy, _z, _p = scene.project(scene.positions, 0.5, 0.34, 1.0, 1000, 420)
        self.assertEqual(scene.pick(sx, sy, sx[7] + 2, sy[7] - 1), 7)
        self.assertIsNone(scene.pick(sx, sy, -500, -500))
        facts = scene.describe(7)
        self.assertEqual(facts["layer"], int(scene.layer[7]))

    def test_activity_helpers(self):
        states = np.random.default_rng(1).normal(0.4, 0.2, (80, 10))
        activity, gain = replay_activity(states)
        self.assertTrue(np.allclose(activity.mean(axis=0), 0.0, atol=1e-12))
        self.assertGreaterEqual(gain, 1.0)
        fading = np.exp(-np.arange(50) / 8.0)[:, None] * np.ones((1, 4)) * 0.05
        act, g = impulse_activity(fading)
        self.assertTrue(np.array_equal(act, fading))
        self.assertGreater(g * float(np.abs(fading[0]).max()), 0.5)     # first frames use the colour range
        self.assertEqual(display_gain(np.zeros((0, 3))), 1.0)

    def test_palette_is_neutral_and_nodes_are_black_and_white(self):
        """No hue anywhere; black and white both contrast with the grey ground."""
        for colour in (BLACK, WHITE, DIM, GROUND_TOP, GROUND_BOTTOM):
            self.assertTrue(all(0 <= channel <= 255 for channel in colour))
            self.assertLess(max(colour) - min(colour), 6, "neutral grey, no colour cast")
        ground = (sum(GROUND_TOP) + sum(GROUND_BOTTOM)) / 6.0
        self.assertTrue(90 < ground < 170, "mid-grey ground")
        self.assertGreater(ground - BLACK[0], 100)          # black reads against the ground
        self.assertGreater(WHITE[0] - ground, 80)           # so does white
        self.assertEqual(mix(BLACK, WHITE, 0.0), BLACK)
        self.assertEqual(mix(BLACK, WHITE, 2.0), WHITE)     # clamped

    def test_contrast_curve_separates_rest_from_activity(self):
        levels = np.linspace(0, 1, 101)
        curve = emphasize(levels)
        self.assertEqual(float(curve[0]), 0.0)
        self.assertAlmostEqual(float(curve[-1]), 1.0)
        self.assertTrue(np.all(np.diff(curve) >= 0), "monotonic")
        self.assertEqual(float(emphasize(0.05)), 0.0)        # faint wobble stays at rest
        self.assertLess(float(emphasize(REST_LEVEL * 0.7 + 0.001)), REST_LEVEL)
        self.assertGreater(float(emphasize(0.5)), 0.4)       # real activity is clearly lit

    def test_grain_tile(self):
        tile = grain_tile()
        self.assertEqual(tile.shape, (192, 192, 4))
        self.assertEqual(tile.dtype, np.uint8)
        self.assertTrue(np.array_equal(tile, grain_tile()), "deterministic")
        self.assertFalse(np.array_equal(tile, grain_tile(seed=8)))
        self.assertLessEqual(int(tile[..., 3].max()), GRAIN_STRENGTH)
        self.assertGreater(float(tile[..., 3].mean()), 4.0)  # visible
        self.assertLess(float(tile[..., 3].mean()), 30.0)    # but fine
        light = float((tile[..., 0] == 255).mean())
        self.assertTrue(0.4 < light < 0.6, "black and white grains balance, so brightness is unchanged")

    def test_projection_is_orthographic(self):
        """Depth changes order, never size or position of the screen image."""
        scene = EchoScene(self.structure, n_layers=4)
        for yaw, pitch in ((0.0, 0.0), (0.7, 0.0), (1.2, 0.0)):
            near = np.array([[0.4, 0.3, 0.9]])
            # slide along the viewing ray (with pitch 0 it is horizontal: -sin(yaw), 0, cos(yaw))
            ray = np.array([[-math.sin(yaw), 0.0, math.cos(yaw)]])
            a = scene.project(near, yaw, pitch, 1.0, 1000, 420)
            b = scene.project(near + 1.3 * ray, yaw, pitch, 1.0, 1000, 420)
            self.assertAlmostEqual(float(a[0][0]), float(b[0][0]), places=6)
            self.assertAlmostEqual(float(a[1][0]), float(b[1][0]), places=6)
            self.assertGreater(float(b[2][0]), float(a[2][0]))      # nearer in depth order
        _sx, _sy, _z, scale = scene.project(scene.positions, 0.5, 0.34, 1.0, 1000, 420)
        self.assertTrue(np.all(scale == 1.0))

    def test_ink_node_splits_by_sign(self):
        self.assertEqual(ink_node(0.4), (True, 0.4))
        self.assertEqual(ink_node(0.0), (True, 0.0))
        self.assertEqual(ink_node(-0.4), (False, 0.4))
        self.assertEqual(ink_node(-7.0), (False, 1.0))               # strength is capped


class TestEBrushStructure(unittest.TestCase):
    def test_six_input_model_gets_six_input_nodes(self):
        with tempfile.TemporaryDirectory() as temp:
            ws = Workspace(Path(temp))
            info = make_demo(ws, count=5, length=280)
            config = normalize_ebrush_config({
                "dataset": {"session_ids": info["session_ids"]},
                "model": {"units": 24, "washout": 24, "horizon": 10, "seed": 9}})
            run = Path(temp) / "run"
            train_motion(config, ws, run)
            structure = reservoir_structure(run, max_edges=300)
            self.assertEqual(structure["inputs"], 6)
            scene = EchoScene(structure, n_layers=3)
            self.assertEqual(scene.input_positions.shape, (6, 3))
            self.assertTrue(np.all(scene.input_positions[:, 1] > scene.positions[:, 1].max()))
            self.assertEqual(len(structure["readout"]["bypass"]), 6)
            self.assertEqual(len(structure["readout"]["input_mean"]), 6)
            self.assertEqual(structure["forecast"]["project"], PROJECT_ID)
            self.assertEqual(structure["forecast"]["washout"], config["model"]["washout"])
            self.assertGreater(structure["readout"]["target_std"], 0)
            frames = reservoir_frames(run, start=0, count=60)
            self.assertEqual(np.asarray(frames["states"]).shape, (60, 24))
            self.assertIn("train", frames["split_starts"])


class TestReservoirEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.workspace = Workspace(Path(cls.temp.name))
        cls.backend = RCService(cls.workspace)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), RCHandler)
        cls.httpd.rcservice = cls.backend
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.client = Client(port=cls.httpd.server_port, workspace=cls.workspace)
        row = cls.client.submit({"dataset": {"length": 420, "noise": 0},
                                 "model": {"units": 30, "washout": 20}})
        for _ in range(200):
            row = cls.client.run(row["id"])
            if row["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert row["status"] == "finished", row
        cls.run_id = row["id"]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.backend.pool.shutdown(wait=True)
        cls.temp.cleanup()

    def test_structure_and_frames_over_http(self):
        structure = self.client.reservoir(self.run_id, edges=120)
        self.assertEqual(structure["units"], 30)
        self.assertLessEqual(structure["edges"]["shown"], 120)
        frames = self.client.reservoir_frames(self.run_id, start=50, count=40)
        self.assertEqual((frames["start"], frames["count"]), (50, 40))
        self.assertEqual(len(frames["states"][0]), 30)

    def test_read_capability_is_enough(self):
        reader = Client(port=self.httpd.server_port, workspace=self.workspace, permission="read")
        self.assertEqual(reader.reservoir(self.run_id)["units"], 30)

    def test_errors_use_the_existing_status_codes(self):
        with self.assertRaisesRegex(ServiceError, "404"):
            self.client.reservoir("does-not-exist")
        with self.assertRaisesRegex(ServiceError, "400"):
            self.client.reservoir_frames(self.run_id, start=10 ** 7)
        queued = self.workspace.create_run({"dataset": {}, "model": {}})
        with self.assertRaisesRegex(ServiceError, "409"):
            self.client.reservoir(queued)


if __name__ == "__main__":
    unittest.main()
